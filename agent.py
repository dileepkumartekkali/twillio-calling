import asyncio
import os

from dotenv import load_dotenv
from loguru import logger

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.frames.frames import EndFrame, LLMRunFrame, LLMTextFrame, TranscriptionFrame, TTSSpeakFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.processors.idle_frame_processor import IdleFrameProcessor
from pipecat.runner.types import RunnerArguments
from pipecat.runner.utils import create_transport
from pipecat.services.groq.llm import GroqLLMService
from pipecat.services.sarvam.stt import SarvamSTTService
from pipecat.services.sarvam.tts import SarvamTTSService
from pipecat.transports.websocket.fastapi import FastAPIWebsocketParams

from lang_router import DEFAULT_LANGUAGE, detect_target_language

load_dotenv(override=True)

# Twilio has no documented hard cutoff for Media Streams (unlike Exotel's ~60min
# plan cap) — TimeLimit is account/config-specific. Keep the same wrap-up window
# as a UX choice: nobody wants a 55-minute phone-bot call anyway.
MAX_CALL_SECONDS = 55 * 60
# Silence handling: nudge once, then hang up rather than hold a dead line open.
IDLE_NUDGE_SECONDS = 15


class LanguageRouterProcessor(FrameProcessor):
    """Sits between `llm` and `tts`. Picks the TTS language from the LLM's OWN
    reply text as it streams out, not from the caller's transcript.

    Why the reply and not the question: TTS needs to match what's actually about
    to be spoken. Deriving language from the caller's input assumes the LLM will
    mirror it exactly; if it ever drifts, TTS would be locked to the wrong
    language for text that doesn't match. Detecting on the reply itself is
    self-consistent by construction, and costs nothing extra — same one LLM call,
    same one TTS call per turn, this is just a stdlib script-count over text that
    was already flowing through here.

    Not done via an inline LLM-emitted tag (e.g. "[te-IN] reply..."): a tag would
    get split across streamed token frames and is unreliable to parse mid-stream.
    """

    def __init__(self, tts: SarvamTTSService):
        super().__init__()
        self._tts = tts
        self._reply_so_far = ""

    async def process_frame(self, frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if isinstance(frame, TranscriptionFrame):
            self._reply_so_far = ""  # new user turn starting -> next reply is fresh
        elif isinstance(frame, LLMTextFrame):
            self._reply_so_far += frame.text
            lang = detect_target_language(self._reply_so_far)
            # ponytail: mutates the TTS service's settings object directly — Sarvam's
            # Pipecat TTS wrapper has no public set_language(). If replies stop
            # switching language after a pipecat-ai upgrade, check the field name here:
            #   python -c "from pipecat.services.sarvam.tts import SarvamTTSService as S; print(vars(S.Settings()))"
            self._tts._settings.language = lang
        await self.push_frame(frame, direction)


SYSTEM_PROMPT = """You are a helpful voice assistant answering a phone call in India.

The caller may speak in English, a single Indian language, or a mix of an Indian
language and English in the same sentence (e.g. Telugu+English, Hindi+English).

Always reply in the SAME language or language-mix the caller just used. If they
mixed languages, mix your reply the same way — don't switch to pure English or
pure Hindi unless they did. Write Indian-language words in their NATIVE SCRIPT
(Devanagari for Hindi, Telugu script for Telugu, etc.) exactly as the caller's
own transcript appears to you — never Romanized/Latin transliteration (no
"kaise ho", write "कैसे हो"). Keep replies short and conversational: this is a
phone call, not a chat window, so avoid lists, markdown, or long paragraphs."""


async def bot(runner_args: RunnerArguments):
    transport = await create_transport(
        runner_args,
        {
            # TwilioFrameSerializer is built automatically by create_transport and
            # reads TWILIO_ACCOUNT_SID / TWILIO_AUTH_TOKEN from the environment —
            # required because auto_hang_up=True (the default) calls Twilio's REST
            # API to end the call, unlike Exotel which needed no credentials here.
            "twilio": lambda: FastAPIWebsocketParams(
                audio_in_enabled=True,
                audio_out_enabled=True,
                audio_in_sample_rate=8000,
                audio_out_sample_rate=8000,
            ),
        },
    )

    stt = SarvamSTTService(
        api_key=os.getenv("SARVAM_API_KEY"),
        mode="codemix",  # needs sarvamai>=0.1.25 pinned in requirements.txt, see pipecat-ai/pipecat#3783
        settings=SarvamSTTService.Settings(model="saaras:v3", language="unknown"),
    )
    tts = SarvamTTSService(
        api_key=os.getenv("SARVAM_API_KEY"),
        settings=SarvamTTSService.Settings(
            model="bulbul:v3",
            # "shubh" is Sarvam's own documented default speaker for bulbul:v3 —
            # no published naturalness ranking exists, but shipping it as the
            # default is the closest signal to "most tuned" available. A/B test
            # against "anand" (the previous voice, also valid) if this doesn't
            # sound right on real calls.
            voice="shubh",
            language=DEFAULT_LANGUAGE,  # overwritten per-turn by LanguageRouterProcessor
            pace=1.0,
        ),
    )
    # Confirmed post-VAD-fix: Sarvam's own LLM is still too slow for a live call
    # (user-verified, not just the earlier stacked-call artifact). Gemini
    # 2.5 Flash-Lite was tried next but its free tier's request-per-minute cap
    # is too tight for a live phone line. Settled on Groq's LPU inference for
    # speed. Known risk, unresolved: no confirmed evidence Llama reliably
    # writes native Devanagari/Telugu/Tamil script instead of Romanized
    # transliteration — watch real calls for this; if TTS keeps landing on
    # en-IN regardless of what the caller spoke, that's the symptom.
    llm = GroqLLMService(
        api_key=os.getenv("GROQ_API_KEY"),
        settings=GroqLLMService.Settings(model="llama-3.3-70b-versatile"),
    )

    context = LLMContext([{"role": "system", "content": SYSTEM_PROMPT}])
    # Without a VAD analyzer, TurnAnalyzerUserTurnStopStrategy never sees a real
    # VADUserStoppedSpeakingFrame and falls back to firing "turn stopped" on a
    # blind ~1s timeout after EVERY transcript — so one utterance can trigger
    # multiple stacked LLM calls with growing context, which is what caused the
    # repeated/overlapping replies when a caller tried to interrupt. Silero VAD
    # restores real silence-based turn detection (and lets Sarvam STT's own
    # VAD-gated flush() fire at the right time too).
    context_aggregator = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(vad_analyzer=SileroVADAnalyzer(params=VADParams(stop_secs=0.2))),
    )
    lang_router = LanguageRouterProcessor(tts)

    idle_retries = 0

    async def on_idle(processor):
        nonlocal idle_retries
        idle_retries += 1
        if idle_retries <= 2:
            await processor.push_frame(TTSSpeakFrame("Hello? Are you still there?"))
        else:
            await processor.push_frame(TTSSpeakFrame("I'll end the call here. Goodbye!"))
            await processor.push_frame(EndFrame())

    # types=[TranscriptionFrame]: only the caller actually speaking resets the idle
    # timer. Left at default (monitors all frames) this would never fire, since
    # audio/control frames flow through the pipeline constantly regardless of silence.
    idle_guard = IdleFrameProcessor(callback=on_idle, timeout=IDLE_NUDGE_SECONDS, types=[TranscriptionFrame])

    pipeline = Pipeline(
        [
            transport.input(),
            stt,
            idle_guard,
            context_aggregator.user(),
            llm,
            lang_router,
            tts,
            transport.output(),
            context_aggregator.assistant(),
        ]
    )

    # `allow_interruptions` was removed from PipelineParams in pipecat-ai 1.5.0 (moved to
    # per-aggregator turn strategies) — barge-in is on by default without setting anything here.
    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            audio_in_sample_rate=8000,
            audio_out_sample_rate=8000,
        ),
    )

    async def end_call_before_provider_cutoff():
        await asyncio.sleep(MAX_CALL_SECONDS)
        await task.queue_frames(
            [TTSSpeakFrame("We're close to the time limit for this call, ending here now."), EndFrame()]
        )

    call_timer = asyncio.create_task(end_call_before_provider_cutoff())

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        # Verified against Sarvam's own Twilio+Pipecat reference: a one-off system
        # message plus LLMRunFrame is what reliably produces an opening greeting
        # (pushing the bare system-prompt context frame risks an empty/odd first
        # turn since there's no user message yet for the LLM to respond to).
        context.add_messages([{"role": "system", "content": "Greet the caller and briefly introduce yourself."}])
        await task.queue_frames([LLMRunFrame()])

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        call_timer.cancel()
        await task.cancel()

    runner = PipelineRunner(handle_sigint=False)
    try:
        await runner.run(task)
    except Exception:
        logger.exception("Pipeline crashed mid-call")
    finally:
        call_timer.cancel()
