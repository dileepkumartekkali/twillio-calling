import asyncio
import os
import time

from dotenv import load_dotenv
from loguru import logger

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    EndFrame,
    LLMRunFrame,
    LLMTextFrame,
    TranscriptionFrame,
    TTSAudioRawFrame,
    TTSSpeakFrame,
)
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
from pipecat.turns.user_start.min_words_user_turn_start_strategy import MinWordsUserTurnStartStrategy
from pipecat.turns.user_turn_strategies import UserTurnStrategies

from lang_router import DEFAULT_LANGUAGE, detect_target_language

load_dotenv(override=True)

# Twilio has no documented hard cutoff for Media Streams (unlike Exotel's ~60min
# plan cap) — TimeLimit is account/config-specific. Keep the same wrap-up window
# as a UX choice: nobody wants a 55-minute phone-bot call anyway.
MAX_CALL_SECONDS = 55 * 60
# Silence handling: nudge once at 5s, hang up at 10s rather than hold a dead line open.
IDLE_NUDGE_SECONDS = 5

# Native-script text for the two idle-timeout messages, keyed by the same Sarvam
# language codes lang_router.py produces, so the nudge/goodbye speaks in whatever
# language the call was last using (falls back to English if somehow unset).
# NOTE: these are machine-drafted translations, not reviewed by a native speaker
# per language — spot-check with someone fluent before fully trusting on live calls.
STILL_THERE_MESSAGES = {
    "en-IN": "Are you there?",
    "hi-IN": "क्या आप वहाँ हैं?",
    "bn-IN": "আপনি কি সেখানে আছেন?",
    "pa-IN": "ਕੀ ਤੁਸੀਂ ਉੱਥੇ ਹੋ?",
    "gu-IN": "શું તમે ત્યાં છો?",
    "or-IN": "ଆପଣ କଣ ସେଠାରେ ଅଛନ୍ତି?",
    "ta-IN": "நீங்கள் அங்கே இருக்கிறீர்களா?",
    "te-IN": "మీరు అక్కడ ఉన్నారా?",
    "kn-IN": "ನೀವು ಅಲ್ಲಿ ಇದ್ದೀರಾ?",
    "ml-IN": "നിങ്ങൾ അവിടെ ഉണ്ടോ?",
}
GOODBYE_MESSAGES = {
    "en-IN": "I'll end the call here. Goodbye!",
    "hi-IN": "मैं यहाँ कॉल समाप्त कर रहा हूँ। अलविदा!",
    "bn-IN": "আমি এখানে কল শেষ করছি। বিদায়!",
    "pa-IN": "ਮੈਂ ਇੱਥੇ ਕਾਲ ਖਤਮ ਕਰ ਰਿਹਾ ਹਾਂ। ਅਲਵਿਦਾ!",
    "gu-IN": "હું અહીં કૉલ સમાપ્ત કરી રહ્યો છું. આવજો!",
    "or-IN": "ମୁଁ ଏଠାରେ କଲ୍ ସମାପ୍ତ କରୁଛି। ବିଦାୟ!",
    "ta-IN": "நான் இங்கே அழைப்பை முடிக்கிறேன். போய் வருகிறேன்!",
    "te-IN": "నేను ఇక్కడ కాల్ ముగిస్తున్నాను. వీడ్కోలు!",
    "kn-IN": "ನಾನು ಇಲ್ಲಿ ಕರೆಯನ್ನು ಮುಗಿಸುತ್ತಿದ್ದೇನೆ. ವಿದಾಯ!",
    "ml-IN": "ഞാൻ ഇവിടെ കോൾ അവസാനിപ്പിക്കുന്നു. വിട!",
}


class LanguageHintProcessor(FrameProcessor):
    """Sits between `stt` and the context aggregator. Tags each caller
    transcript with its script-detected language before it reaches the LLM,
    instead of leaving language purely to the LLM's own inference from raw
    text. Reuses the same detect_target_language() already used for TTS
    voice selection — no extra API call, just another pass over text that's
    already flowing through here.

    Tagging the transcript itself (not a separate system message) keeps the
    hint attached to the exact turn it's about, so it can't go stale or get
    lost among earlier system messages on a long call.
    """

    async def process_frame(self, frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if isinstance(frame, TranscriptionFrame) and frame.text:
            lang = detect_target_language(frame.text)
            frame.text = f"[Caller's language: {lang}] {frame.text}"
            # High-signal marker for reading latency out of Render logs: every
            # real caller turn logs here, so "time from this line to the next
            # Bot started speaking line" is the actual end-to-end turn latency,
            # without having to piece it together from internal DEBUG lines.
            logger.info(f"TURN START: {frame.text}")
        await self.push_frame(frame, direction)


class IdleResetProcessor(FrameProcessor):
    """Calls reset_callback() on real activity (caller speech or bot speech
    start/stop). Needed because IdleFrameProcessor's own retry counter (in
    on_idle) has no way to know the difference between "silence right after
    a fresh reset" and "silence that's actually the Nth bout of the whole
    call" — without this, one legitimate idle firing (e.g. while waiting on
    a slow LLM response) permanently bumps the counter past the nudge
    threshold, so the NEXT unrelated idle firing anywhere later in the call
    hangs up immediately instead of nudging first. Confirmed in a real call
    log: retry #2 fired 0.8s after the bot finished a normal reply and ended
    the call outright, because retry #1 had already fired earlier for an
    unrelated reason.
    """

    def __init__(self, reset_callback, watch_types):
        super().__init__()
        self._reset_callback = reset_callback
        self._watch_types = tuple(watch_types)

    async def process_frame(self, frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if isinstance(frame, self._watch_types):
            self._reset_callback()
        await self.push_frame(frame, direction)


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
            if not self._reply_so_far:
                # First token of this reply — pairs with LanguageHintProcessor's
                # "TURN START" log to give LLM time-to-first-token straight from
                # Render logs, without cross-referencing internal metric lines.
                logger.info("LLM FIRST TOKEN")
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


class AudioGapMonitor(FrameProcessor):
    """Sits between `tts` and `transport.output()`. Logs a warning whenever
    the gap between consecutive TTS audio chunks exceeds a threshold that
    would be audible as a stutter/dropout — pipecat itself doesn't expose
    any native signal for this (confirmed: no underrun/buffer/resample
    logging anywhere in the transport or TTS service source), so this is
    the concrete thing to grep for if choppy voice ever recurs, instead of
    re-diagnosing from scratch.
    """

    GAP_WARNING_THRESHOLD_SECS = 0.3

    def __init__(self):
        super().__init__()
        self._last_audio_time = None

    async def process_frame(self, frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if isinstance(frame, TTSAudioRawFrame):
            now = time.monotonic()
            if self._last_audio_time is not None:
                gap = now - self._last_audio_time
                if gap > self.GAP_WARNING_THRESHOLD_SECS:
                    logger.warning(f"AUDIO GAP: {gap:.3f}s between TTS audio chunks — likely audible stutter")
            self._last_audio_time = now
        elif isinstance(frame, (BotStartedSpeakingFrame, BotStoppedSpeakingFrame)):
            # Silence BETWEEN utterances is normal, not a gap to flag — only
            # measure gaps within a single continuous bit of speech.
            self._last_audio_time = None
        await self.push_frame(frame, direction)


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
        # bulbul:v3 defaults to generating at 24kHz, which then has to be
        # downsampled to Twilio's required 8kHz on every streamed chunk —
        # an extra resampling stage on small, frequent chunks is a known
        # source of choppy/robotic-sounding audio. Generating natively at
        # 8000 (a Sarvam-documented supported rate) skips that step entirely.
        sample_rate=8000,
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

    async def warm_up_groq():
        # Confirmed via a real call log: the FIRST request on a fresh Groq
        # client took 9.8s (TLS/connection-pool setup), a later request in
        # the same call took 0.1s. A new GroqLLMService is created per call
        # (no cross-call reuse), so every call pays that cost once unless we
        # pay it here first — fired as a background task so it runs
        # concurrently with Sarvam STT/TTS connecting, not blocking either.
        try:
            await llm._client.chat.completions.create(
                model=llm._settings.model,
                messages=[{"role": "user", "content": "hi"}],
                max_tokens=1,
            )
            logger.info("Groq connection warmed up")
        except Exception:
            logger.exception("Groq warm-up call failed (non-fatal, continuing)")

    asyncio.create_task(warm_up_groq())

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
        user_params=LLMUserAggregatorParams(
            vad_analyzer=SileroVADAnalyzer(params=VADParams(stop_secs=0.2)),
            # Default start strategy (VADUserTurnStartStrategy) triggers a barge-in
            # on ANY audio energy, including noise or the bot's own voice leaking
            # back on the line (no echo cancellation on this call) — confirmed in
            # a real call log: false triggers repeatedly cancelled the greeting
            # mid-sentence and restarted the turn cycle, producing "how can I help
            # you" 2-3 times in a row. MinWordsUserTurnStartStrategy requires 2+
            # actually-transcribed words to interrupt the bot while it's speaking
            # (only 1 word when it isn't, so normal turns still start immediately) —
            # filters out noise/echo blips without slowing down real conversation.
            user_turn_strategies=UserTurnStrategies(start=[MinWordsUserTurnStartStrategy(min_words=2)]),
        ),
    )
    lang_hint = LanguageHintProcessor()
    lang_router = LanguageRouterProcessor(tts)
    audio_gap_monitor = AudioGapMonitor()

    idle_retries = 0

    def reset_idle_retries():
        nonlocal idle_retries
        if idle_retries:
            logger.info("IDLE: real activity resumed, resetting retry counter")
        idle_retries = 0

    async def on_idle(processor):
        nonlocal idle_retries
        idle_retries += 1
        logger.info(f"IDLE FIRED: retry #{idle_retries} (timeout={IDLE_NUDGE_SECONDS}s)")
        # tts._settings.language is kept current by LanguageRouterProcessor from
        # the last real turn, so these messages land in whatever language the
        # call was actually using — not hardcoded English regardless of caller.
        current_lang = tts._settings.language
        if idle_retries == 1:
            await processor.push_frame(
                TTSSpeakFrame(STILL_THERE_MESSAGES.get(current_lang, STILL_THERE_MESSAGES[DEFAULT_LANGUAGE]))
            )
        else:
            logger.info("IDLE: ending call")
            await processor.push_frame(
                TTSSpeakFrame(GOODBYE_MESSAGES.get(current_lang, GOODBYE_MESSAGES[DEFAULT_LANGUAGE]))
            )
            await processor.push_frame(EndFrame())

    # Also watch Bot{Started,Stopped}SpeakingFrame (pushed upstream by the output
    # transport, so they do reach this position) — without them, the idle timer
    # only resets on caller speech, so a 5s timeout fires WHILE the bot is still
    # generating/speaking its own reply (no new caller transcript arrives during
    # that gap), misreading normal turn latency as caller silence and cutting
    # the call mid-conversation. Now it resets on either side's activity, so it
    # only fires on genuine dead air from both parties.
    idle_guard = IdleFrameProcessor(
        callback=on_idle,
        timeout=IDLE_NUDGE_SECONDS,
        types=[TranscriptionFrame, BotStartedSpeakingFrame, BotStoppedSpeakingFrame],
    )
    idle_reset = IdleResetProcessor(
        reset_idle_retries,
        watch_types=[TranscriptionFrame, BotStartedSpeakingFrame, BotStoppedSpeakingFrame],
    )

    pipeline = Pipeline(
        [
            transport.input(),
            stt,
            lang_hint,
            idle_guard,
            idle_reset,
            context_aggregator.user(),
            llm,
            lang_router,
            tts,
            audio_gap_monitor,
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
            enable_metrics=True,  # per-turn TTFB/processing time in logs, for real latency numbers
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
        # Explicitly default to English here: with no caller input yet to mirror,
        # the LLM was defaulting the greeting to Hindi on its own (likely biased
        # by "a phone call in India" in SYSTEM_PROMPT), then correctly switching
        # once real input arrived — so only the greeting itself needed pinning.
        context.add_messages(
            [
                {
                    "role": "system",
                    "content": (
                        "Greet the caller briefly in English — you don't know their "
                        "language yet. Once they reply, switch to match them as usual."
                    ),
                }
            ]
        )
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
