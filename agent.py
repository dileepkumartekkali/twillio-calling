"""Pipecat-free voice agent: Twilio Media Streams <-> Sarvam STT/TTS <-> Groq LLM.

Why no pipecat: on Render's free tier the pipecat stack (onnxruntime, local
Silero VAD, Smart Turn, worker/bus machinery) burned the container's CPU
quota per call, and the resulting cgroup throttle debt made the bot deaf for
~7-12s at call start. This rewrite keeps ZERO local models: Sarvam's
server-side VAD (vad_signals) does all speech detection, so per-call setup
is just one STT websocket connect (~1s).

Wire protocols verified against the installed sarvamai SDK and pipecat's
production-proven Sarvam integration (same audio encoding combination that
ran successfully for weeks: raw PCM16 @ 8kHz, encoding "audio/wav").

Lessons from the pipecat build, all carried over:
- Barge-in is transcript-gated (>=2 real words), never raw-VAD-gated: line
  noise/echo used to cancel replies mid-sentence and repeat the greeting.
- Idle nudge (5s) / goodbye (10s) in the CALLER'S language, and the timer
  never counts while the caller's turn is open, a reply is generating, or
  bot audio is still playing.
- end_call LLM tool requires a CLEAR explicit request (Llama over-triggers
  tools on a casual "okay"/"bye").
- Transcripts are tagged "[xx-IN]" (single token — longer tags broke word
  counting) so the LLM gets an explicit language signal per turn.
- TTS language comes from the LLM's OWN reply text, not the caller's input.
- The greeting is fixed text, pre-synthesized at server startup and cached
  as raw mu-law — an LLM round-trip for a constant sentence cost 1.5-2.5s.
"""

import asyncio
import audioop
import base64
import json
import os
import re
import time
import wave
from io import BytesIO

import httpx
from dotenv import load_dotenv
from loguru import logger
from openai import AsyncOpenAI
from sarvamai import AsyncSarvamAI
from sarvamai.core.events import EventType

from lang_router import DEFAULT_LANGUAGE, detect_target_language

load_dotenv(override=True)

SAMPLE_RATE = 8000  # Twilio Media Streams: 8kHz mu-law, both directions

# Twilio has no documented hard cutoff for Media Streams; keep a wrap-up
# window as a UX choice — nobody wants a 55-minute phone-bot call anyway.
MAX_CALL_SECONDS = 55 * 60
IDLE_NUDGE_SECONDS = 5      # silence before "are you there?"
IDLE_HANGUP_SECONDS = 10    # silence before goodbye + hangup
# Quiet time after the last transcript before the LLM turn fires. Measured
# transcript lag after END_SPEECH is 0.1-0.7s; 0.45 balances snappy replies
# against splitting one utterance into two turns when transcripts arrive in
# parts. Raise this first if turns start double-firing.
TURN_SETTLE_SECONDS = 0.45
MAX_CONCURRENT_CALLS = 3    # ponytail: free tier is one tiny instance; shed load beyond this

GREETING_TEXT = "Hello! How can I help you today?"

LLM_MODEL = "llama-3.3-70b-versatile"
TTS_MODEL = "bulbul:v3"
TTS_SPEAKER = "shubh"
STT_MODEL = "saaras:v3"

# Native-script text for the idle-timeout messages, keyed by the same Sarvam
# language codes lang_router.py produces. NOTE: machine-drafted translations —
# spot-check with a fluent speaker before fully trusting on live calls.
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

SYSTEM_PROMPT = """You are a helpful voice assistant answering a phone call in India.

The caller may speak in English, a single Indian language, or a mix of an Indian
language and English in the same sentence (e.g. Telugu+English, Hindi+English).
Each caller message is prefixed with a detected-language tag like "[te-IN]".

Always reply in the SAME language or language-mix the caller just used. If they
mixed languages, mix your reply the same way — don't switch to pure English or
pure Hindi unless they did. Write Indian-language words in their NATIVE SCRIPT
(Devanagari for Hindi, Telugu script for Telugu, etc.) exactly as the caller's
own transcript appears to you — never Romanized/Latin transliteration (no
"kaise ho", write "कैसे हो"). This is a live phone call, not a chat window:
reply in AT MOST 1-2 short sentences (under about 15 words each), like a
human speaking naturally. Never use lists, markdown, or long paragraphs.
If there is a lot to say, give the most important part and ask if they
want more.

Call the end_call function ONLY when the caller CLEARLY and EXPLICITLY asks
to end the call — in any language or mix ("end the call", "call cut chey",
"कॉल काट दो", "band karo", "hang up now"). Do NOT call it for casual
acknowledgments, a passing "okay"/"bye" mid-conversation, unclear or
garbled speech, or anything you are not sure about — when in doubt, keep
talking and ask instead of hanging up. Never call it on your own
initiative. When you do call it, the goodbye is spoken automatically."""

END_CALL_TOOL = {
    "type": "function",
    "function": {
        "name": "end_call",
        "description": (
            "End the phone call. Use ONLY when the caller clearly and explicitly "
            "asks to end/cut/stop/hang up the call (any language or mix). Never "
            "use for casual acknowledgments, a passing okay/bye, or unclear "
            "speech — if unsure, do not call this."
        ),
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
}

# Sentence boundaries incl. Devanagari danda and common Indic terminators.
_SENTENCE_END = re.compile(r"(?<=[.!?।؟…])\s+")
_CLAUSE_END = re.compile(r"[,;:]\s")
# Measured on a real call: Sarvam TTS REST took ~5s for a 135-char sentence
# vs ~1.3s for short ones — synthesis time scales with length, so long
# sentences must be chunked at clause boundaries to keep first-audio fast.
MAX_TTS_CHUNK_CHARS = 60


def _split_speech_chunks(text: str) -> tuple[list[str], str]:
    """Split streamed LLM text into TTS-ready chunks. Returns (chunks, rest):
    complete sentences always split; an overlong sentence-in-progress is cut
    at a clause boundary (comma/semicolon) so TTS never waits on a 5s synth."""
    chunks = []
    rest = text
    while True:
        m = _SENTENCE_END.search(rest)
        if m:
            piece = rest[: m.start()].strip()
            if piece:
                chunks.append(piece)
            rest = rest[m.end():]
            continue
        if len(rest) > MAX_TTS_CHUNK_CHARS:
            cut = None
            for cm in _CLAUSE_END.finditer(rest[: MAX_TTS_CHUNK_CHARS + 20]):
                cut = cm.end()
            if cut and cut > 20:
                chunks.append(rest[:cut].strip())
                rest = rest[cut:]
                continue
        break
    return chunks, rest

# Module-level shared clients: connection pools persist ACROSS calls, so no
# call after the first pays TLS/DNS setup to Groq or Sarvam's REST API (the
# per-call Groq cold connection measured 9.8s on its first request once).
# "unset" placeholders: the SDKs reject None at CONSTRUCTION, which would
# crash the whole server at import when an env var is missing — with a
# placeholder the failure surfaces at the first API call with a clear 401
# instead of taking down /health and /voice with it.
groq = AsyncOpenAI(
    api_key=os.getenv("GROQ_API_KEY") or "unset",
    base_url="https://api.groq.com/openai/v1",
)
sarvam = AsyncSarvamAI(api_subscription_key=os.getenv("SARVAM_API_KEY") or "unset")

active_calls = 0
_greeting_ulaw: bytes | None = None


def _wav_or_raw_to_ulaw(audio_bytes: bytes) -> bytes:
    """Convert Sarvam TTS output (WAV container or raw PCM16 @ 8kHz mono) to
    raw mu-law bytes for Twilio. Sniffs for a RIFF header rather than assuming:
    Sarvam's REST TTS documents WAV output, but sniffing keeps us correct if a
    codec/container variant returns headerless PCM.
    """
    if audio_bytes[:4] == b"RIFF":
        with wave.open(BytesIO(audio_bytes), "rb") as w:
            rate, width, channels = w.getframerate(), w.getsampwidth(), w.getnchannels()
            pcm = w.readframes(w.getnframes())
        logger.debug(f"TTS WAV: {rate}Hz, {width * 8}-bit, {channels}ch, {len(pcm)} bytes")
        if channels == 2:
            pcm = audioop.tomono(pcm, width, 0.5, 0.5)
        if width != 2:
            pcm = audioop.lin2lin(pcm, width, 2)
        if rate != SAMPLE_RATE:
            # audioop.ratecv has NO anti-aliasing filter — downsampling here
            # produces harsh/metallic voice. We request 8kHz so this should
            # never run; if this warning appears in logs, THAT is the voice
            # clarity culprit and the fix is making Sarvam honor 8kHz output.
            logger.warning(f"TTS CLARITY RISK: Sarvam returned {rate}Hz, resampling to {SAMPLE_RATE}Hz without filtering")
            pcm, _ = audioop.ratecv(pcm, 2, 1, rate, SAMPLE_RATE, None)
    else:
        logger.debug(f"TTS raw (no RIFF header): {len(audio_bytes)} bytes, assuming PCM16 @ 8kHz")
        pcm = audio_bytes  # trust requested format: PCM16 @ 8kHz mono
    return audioop.lin2ulaw(pcm, 2)


async def synthesize_ulaw(text: str, language: str) -> bytes:
    """Sarvam TTS REST -> raw mu-law 8kHz bytes ready for Twilio."""
    resp = await sarvam.text_to_speech.convert(
        text=text,
        target_language_code=language,
        speaker=TTS_SPEAKER,
        model=TTS_MODEL,
        speech_sample_rate=SAMPLE_RATE,
        output_audio_codec="linear16",
        enable_preprocessing=True,
        pace=1.0,
    )
    audio = b"".join(base64.b64decode(a) for a in resp.audios)
    return _wav_or_raw_to_ulaw(audio)


async def warm_up():
    """Run once at server startup (NOT per call): pre-synthesize the greeting
    and open warm connections to Groq and Sarvam so the first call is fast.
    Failures are non-fatal — everything falls back to on-demand.
    """
    global _greeting_ulaw
    try:
        _greeting_ulaw = await synthesize_ulaw(GREETING_TEXT, "en-IN")
        logger.info(f"Greeting pre-synthesized: {len(_greeting_ulaw)} bytes mu-law")
    except Exception:
        logger.exception("Greeting pre-synthesis failed (will synthesize on demand)")
    try:
        await groq.chat.completions.create(
            model=LLM_MODEL, messages=[{"role": "user", "content": "hi"}], max_tokens=1
        )
        logger.info("Groq connection warmed up")
    except Exception:
        logger.exception("Groq warm-up failed (non-fatal)")


async def _twilio_hangup(call_sid: str):
    """End the call via Twilio's REST API (same mechanism pipecat's serializer used)."""
    sid = os.getenv("TWILIO_ACCOUNT_SID", "").strip()
    token = os.getenv("TWILIO_AUTH_TOKEN", "").strip()
    url = f"https://api.twilio.com/2010-04-01/Accounts/{sid}/Calls/{call_sid}.json"
    try:
        async with httpx.AsyncClient() as client:
            r = await client.post(url, data={"Status": "completed"}, auth=(sid, token))
        logger.info(f"Twilio hangup for {call_sid}: HTTP {r.status_code}")
    except Exception:
        logger.exception("Twilio REST hangup failed")


class CallSession:
    """One phone call: owns the Twilio websocket and all per-call tasks."""

    def __init__(self, websocket, stream_sid: str, call_sid: str):
        self.ws = websocket
        self.stream_sid = stream_sid
        self.call_sid = call_sid

        self.messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        self.current_lang = DEFAULT_LANGUAGE  # last language the bot SPOKE in

        # Turn state (timestamps instead of frame plumbing)
        self.transcript_buffer: list[str] = []
        self.last_transcript_time = 0.0
        self._last_end_speech = 0.0
        self.user_speaking = False            # Sarvam server VAD START/END_SPEECH
        self.last_activity = time.monotonic() # caller-side activity only
        self.nudged = False
        self.call_ending = False

        # Bot output state
        self.marks_pending = 0                # Twilio echoes marks when audio has PLAYED
        self.gen_task: asyncio.Task | None = None
        self._mark_counter = 0
        self._recent_bot_text = ""            # rolling tail of what the bot said (echo guard)
        self._out_q: asyncio.Queue = asyncio.Queue()  # paced outbound audio/marks
        self._sender_started = False

        self._stt_ctx = None
        self._stt_socket = None
        self._bg_tasks: list[asyncio.Task] = []
        # Strong refs for fire-and-forget tasks: asyncio only weakly references
        # running tasks, so an unreferenced create_task() can be garbage
        # collected mid-flight and silently drop STT messages.
        self._msg_tasks: set[asyncio.Task] = set()

    # ---------------- Twilio output ----------------

    async def send_ulaw(self, ulaw: bytes):
        """Queue raw mu-law audio for PACED delivery to Twilio, followed by a
        playback mark. Real-time pacing (not instant dumping) is what pipecat
        did and what the first cut of this rewrite got wrong: dumping whole
        sentences into Twilio's buffer meant any barge-in `clear` chopped
        seconds of audio mid-word — heard as terrible voice breaking."""
        self._ensure_sender()
        CHUNK = 1600  # 200ms per media message
        for i in range(0, len(ulaw), CHUNK):
            await self._out_q.put(("audio", ulaw[i:i + CHUNK]))
        self._mark_counter += 1
        self.marks_pending += 1
        await self._out_q.put(("mark", f"m{self._mark_counter}"))

    def _ensure_sender(self):
        if not self._sender_started:
            self._sender_started = True
            self._bg_tasks.append(asyncio.create_task(self._output_sender()))

    async def _output_sender(self):
        """Send queued audio at real-time rate with a small jitter cushion.
        Keeps at most ~LEAD+0.2s buffered at Twilio, so an interruption stops
        the voice almost immediately and event-loop hiccups don't stutter."""
        LEAD = 0.4  # how far ahead of real time we allow Twilio's buffer to run
        next_time = 0.0
        while True:
            kind, payload = await self._out_q.get()
            if kind == "mark":
                await self.ws.send_text(json.dumps({
                    "event": "mark",
                    "streamSid": self.stream_sid,
                    "mark": {"name": payload},
                }))
                continue
            now = time.monotonic()
            # A schedule far in the past just means the stream was idle
            # between utterances — restart it silently (the old warning here
            # fired a misleading "STALL" at every utterance start).
            next_time = max(next_time, now - LEAD)
            if next_time > now:
                target = next_time
                await asyncio.sleep(target - now)
                late = time.monotonic() - target
                if late > 0.5:
                    # We overslept a scheduled send: genuine event-loop/CPU stall.
                    logger.warning(f"OUTBOUND AUDIO STALL: send {late:.2f}s late (CPU starvation)")
            await self.ws.send_text(json.dumps({
                "event": "media",
                "streamSid": self.stream_sid,
                "media": {"payload": base64.b64encode(payload).decode()},
            }))
            next_time += len(payload) / float(SAMPLE_RATE)  # mu-law: 1 byte/sample

    async def clear_twilio_audio(self):
        """Barge-in: drop our queued audio, wipe Twilio's (small) buffer.
        Twilio returns pending marks after a clear; we also zero the counter
        so bot_speaking state can't get stuck if it doesn't."""
        try:
            while True:
                self._out_q.get_nowait()
        except asyncio.QueueEmpty:
            pass
        await self.ws.send_text(json.dumps({"event": "clear", "streamSid": self.stream_sid}))
        self.marks_pending = 0

    async def speak(self, text: str, language: str):
        self._recent_bot_text = (self._recent_bot_text + " " + text)[-400:]
        try:
            t0 = time.monotonic()
            ulaw = await synthesize_ulaw(text, language)
            if not ulaw:
                logger.warning(f"TTS returned EMPTY audio for: {text[:60]!r} ({language})")
                return
            logger.info(
                f"TTS synth {time.monotonic() - t0:.2f}s: {len(text)} chars ({language}) "
                f"-> {len(ulaw) / SAMPLE_RATE:.1f}s audio"
            )
            await self.send_ulaw(ulaw)
        except Exception:
            logger.exception(f"TTS failed for: {text[:60]!r} ({language})")

    def _looks_like_bot_echo(self, transcript: str) -> bool:
        """True when a 'caller' transcript is mostly words the bot itself just
        spoke — i.e. the phone line echoed the bot's voice back into STT (no
        acoustic echo cancellation exists on this path). Without this guard,
        the echo transcript trips barge-in and the bot cuts ITSELF off
        mid-word — heard as broken/unclear speech."""
        if not self._recent_bot_text:
            return False
        recent = set(re.findall(r"\w+", self._recent_bot_text.lower()))
        words = re.findall(r"\w+", transcript.lower())
        if not words:
            return False
        overlap = sum(w in recent for w in words) / len(words)
        return overlap >= 0.7

    @property
    def bot_speaking(self) -> bool:
        return self.marks_pending > 0

    # ---------------- STT ----------------

    async def connect_stt(self):
        self._stt_ctx = sarvam.speech_to_text_streaming.connect(
            model=STT_MODEL,
            language_code="unknown",
            mode="codemix",
            sample_rate=str(SAMPLE_RATE),
            # Server-side VAD: Sarvam detects speech start/end and finalizes
            # utterances itself — no local VAD model, no flush choreography
            # (the local-VAD flush chain was the source of the 14s transcript
            # spikes and the CPU cost of Silero/SmartTurn on the old build).
            vad_signals="true",
            # high_vad_sensitivity deliberately NOT set: enabling it was an
            # unverified guess in the first cut, and higher sensitivity means
            # more false speech events + junk transcripts on an echoey phone
            # line — each one a chance to chop the bot's reply mid-sentence.
        )
        self._stt_socket = await self._stt_ctx.__aenter__()
        self._stt_socket.on(EventType.MESSAGE, self._on_stt_message_sync)
        # STT socket dying silently was a real failure mode on the old build —
        # surface error/close explicitly so the log names it.
        self._stt_socket.on(EventType.ERROR, lambda e: logger.error(f"STT SOCKET ERROR: {e}"))
        self._stt_socket.on(EventType.CLOSE, lambda e: logger.warning(f"STT SOCKET CLOSED: {e}"))
        self._bg_tasks.append(asyncio.create_task(self._stt_listener()))
        logger.info("Sarvam STT connected (server-side VAD)")

    async def _stt_listener(self):
        try:
            await self._stt_socket.start_listening()
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("STT listener died")

    def _on_stt_message_sync(self, message):
        # SDK callbacks are sync; hop back into async land.
        t = asyncio.create_task(self._on_stt_message(message))
        self._msg_tasks.add(t)
        t.add_done_callback(self._msg_tasks.discard)

    async def _on_stt_message(self, message):
        try:
            if message.type == "events":
                signal = message.data.signal_type
                if signal == "START_SPEECH":
                    self.user_speaking = True
                    self.last_activity = time.monotonic()
                    # The caller's VOICE resets the idle escalation immediately
                    # (spec: return between 5-10s => silence counter back to 0).
                    # Waiting for the transcript to reset was too late — STT
                    # delivery lags seconds behind the voice itself.
                    self.nudged = False
                    logger.info("VAD(server): user started speaking")
                elif signal == "END_SPEECH":
                    self.user_speaking = False
                    self.last_activity = time.monotonic()
                    self._last_end_speech = time.monotonic()
                    logger.info("VAD(server): user stopped speaking")
            elif message.type == "data":
                transcript = (message.data.transcript or "").strip()
                if not transcript:
                    return  # empty/silence transcript: never trigger a turn (old known gap)
                lang = detect_target_language(transcript)
                tagged = f"[{lang}] {transcript}"
                # STT delivery lag: END_SPEECH -> transcript arrival. The old
                # build's chronic latency lived exactly in this gap (14s spikes).
                lag = time.monotonic() - self._last_end_speech if self._last_end_speech else -1
                logger.info(f"TURN START: {tagged} (transcript +{lag:.2f}s after END_SPEECH)")
                self.last_activity = time.monotonic()
                self.nudged = False

                # Barge-in: transcript-gated, >=2 real words while bot output
                # is in flight (raw VAD blips used to falsely cancel replies).
                if self.bot_speaking or (self.gen_task and not self.gen_task.done()):
                    if self._looks_like_bot_echo(transcript):
                        logger.info(f"Ignoring echo of bot's own speech: {transcript[:50]!r}")
                        return
                    if len(transcript.split()) >= 2:
                        logger.info("BARGE-IN: interrupting bot output")
                        if self.gen_task and not self.gen_task.done():
                            self.gen_task.cancel()
                        await self.clear_twilio_audio()
                    else:
                        logger.info("Ignoring 1-word blip while bot speaking (noise guard)")
                        return

                self.transcript_buffer.append(tagged)
                self.last_transcript_time = time.monotonic()
        except Exception:
            logger.exception("Error handling STT message")

    async def feed_audio(self, ulaw_b64: str):
        """Twilio media payload (base64 mu-law) -> PCM16 -> Sarvam STT."""
        if not self._stt_socket:
            return
        pcm = audioop.ulaw2lin(base64.b64decode(ulaw_b64), 2)
        # Same proven combination pipecat ran in production: raw PCM16 bytes,
        # encoding "audio/wav", sample_rate 8000.
        await self._stt_socket.transcribe(
            audio=base64.b64encode(pcm).decode(),
            encoding="audio/wav",
            sample_rate=SAMPLE_RATE,
        )

    # ---------------- LLM turn ----------------

    async def _turn_manager(self):
        """Fire an LLM turn once transcripts have settled and the caller isn't
        mid-speech. Timestamp loop — no frame machinery to deadlock."""
        while True:
            await asyncio.sleep(0.15)
            if not self.transcript_buffer or self.user_speaking or self.call_ending:
                continue
            if time.monotonic() - self.last_transcript_time < TURN_SETTLE_SECONDS:
                continue
            if self.gen_task and not self.gen_task.done():
                continue
            user_text = " ".join(self.transcript_buffer)
            self.transcript_buffer.clear()
            logger.info(f"TURN FIRED: {user_text[:80]!r}")
            self.gen_task = asyncio.create_task(self._run_llm_turn(user_text))

    async def _run_llm_turn(self, user_text: str):
        self.messages.append({"role": "user", "content": user_text})
        reply_so_far = ""   # everything the LLM produced (language detection + history)
        buffer = ""         # text not yet sent to TTS
        end_call_requested = False
        try:
            t0 = time.monotonic()
            stream = await groq.chat.completions.create(
                model=LLM_MODEL,
                messages=self.messages,
                tools=[END_CALL_TOOL],
                stream=True,
                max_tokens=300,
            )
            first_token = True
            first_audio = True
            async for chunk in stream:
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta
                if delta.tool_calls:
                    for tc in delta.tool_calls:
                        if tc.function and tc.function.name == "end_call":
                            end_call_requested = True
                if delta.content:
                    if first_token:
                        logger.info(f"LLM FIRST TOKEN (+{time.monotonic() - t0:.2f}s)")
                        first_token = False
                    reply_so_far += delta.content
                    buffer += delta.content
                    # Speak each completed sentence/clause while the rest streams.
                    chunks, buffer = _split_speech_chunks(buffer)
                    for piece in chunks:
                        await self._speak_reply_sentence(piece, reply_so_far)
                        if first_audio:
                            logger.info(f"FIRST REPLY AUDIO QUEUED (+{time.monotonic() - t0:.2f}s from turn fire)")
                            first_audio = False
            if buffer.strip():
                await self._speak_reply_sentence(buffer.strip(), reply_so_far)
                buffer = ""
            if reply_so_far.strip():
                self.messages.append({"role": "assistant", "content": reply_so_far.strip()})
            if end_call_requested:
                logger.info("END CALL: caller asked to end the call (LLM tool call)")
                await self.end_call()
        except asyncio.CancelledError:
            # Barge-in: keep whatever was already SPOKEN in context so the
            # LLM knows it was cut off mid-reply.
            spoken = reply_so_far[: len(reply_so_far) - len(buffer)].strip()
            if spoken:
                self.messages.append({"role": "assistant", "content": spoken + " —"})
            raise
        except Exception:
            logger.exception("LLM turn failed")

    async def _speak_reply_sentence(self, sentence: str, reply_so_far: str):
        # TTS language from the LLM's OWN reply so voice matches the words
        # actually spoken (detecting on caller input drifts when the LLM does).
        lang = detect_target_language(reply_so_far)
        self.current_lang = lang
        await self.speak(sentence, lang)

    # ---------------- Idle watchdog ----------------

    async def _idle_watchdog(self):
        """Nudge at 5s of genuine dead air, goodbye+hangup at 10s.
        Never counts while: caller is mid-speech, a transcript is buffered, a
        reply is generating, or bot audio is still playing — every one of
        those was a premature-hangup bug on the old build."""
        while True:
            await asyncio.sleep(0.5)
            if self.call_ending:
                continue
            busy = (
                self.user_speaking
                or self.transcript_buffer
                or (self.gen_task and not self.gen_task.done())
                or self.bot_speaking
            )
            if busy:
                continue
            idle = time.monotonic() - self.last_activity
            if idle >= IDLE_HANGUP_SECONDS and self.nudged:
                logger.info("IDLE: ending call")
                await self.end_call()
            elif idle >= IDLE_NUDGE_SECONDS and not self.nudged:
                self.nudged = True
                logger.info("IDLE: nudging caller")
                await self.speak(
                    STILL_THERE_MESSAGES.get(self.current_lang, STILL_THERE_MESSAGES[DEFAULT_LANGUAGE]),
                    self.current_lang if self.current_lang in STILL_THERE_MESSAGES else DEFAULT_LANGUAGE,
                )
                # The nudge does NOT reset last_activity — only real caller
                # activity does, so goodbye stays anchored to the same silence.

    async def _call_length_limit(self):
        await asyncio.sleep(MAX_CALL_SECONDS)
        logger.info("MAX CALL LENGTH reached, wrapping up")
        await self.end_call()

    async def end_call(self):
        if self.call_ending:
            return  # fire exactly once (old bug: goodbye+hangup queued repeatedly)
        self.call_ending = True
        lang = self.current_lang if self.current_lang in GOODBYE_MESSAGES else DEFAULT_LANGUAGE
        await self.speak(GOODBYE_MESSAGES[lang], lang)
        # Give Twilio time to play the goodbye out before killing the call
        # (paced delivery means a ~3s goodbye takes ~3s to drain).
        for _ in range(80):  # up to 8s
            if not self.bot_speaking:
                break
            await asyncio.sleep(0.1)
        await asyncio.sleep(0.3)
        await _twilio_hangup(self.call_sid)

    # ---------------- Main loop ----------------

    async def run(self):
        """Drive the call: greeting immediately, then relay Twilio media."""
        try:
            # Greeting first — cached bytes, hearable before STT even connects.
            if _greeting_ulaw:
                await self.send_ulaw(_greeting_ulaw)
                self.messages.append({"role": "assistant", "content": GREETING_TEXT})
                self._recent_bot_text = GREETING_TEXT  # echo guard covers the greeting too
                logger.info("Greeting sent from cache")
            else:
                self._bg_tasks.append(asyncio.create_task(self._greet_on_demand()))

            await self.connect_stt()
            self.last_activity = time.monotonic()

            self._bg_tasks.append(asyncio.create_task(self._turn_manager()))
            self._bg_tasks.append(asyncio.create_task(self._idle_watchdog()))
            self._bg_tasks.append(asyncio.create_task(self._call_length_limit()))

            media_frames = 0
            first_media = True
            rate_window_start = time.monotonic()
            while True:
                raw = await self.ws.receive_text()
                msg = json.loads(raw)
                event = msg.get("event")
                if event == "media":
                    if first_media:
                        logger.info("First caller audio frame received")
                        first_media = False
                    media_frames += 1
                    # Inbound audio health: Twilio sends ~50 frames/s; a low
                    # rate here means the caller's audio isn't reaching us
                    # steadily (network / Twilio side), which STT hears as
                    # gaps — a mishearing/clarity culprit outside our code.
                    if time.monotonic() - rate_window_start >= 15:
                        logger.info(f"INBOUND AUDIO: {media_frames} frames in last 15s (~750 expected)")
                        media_frames = 0
                        rate_window_start = time.monotonic()
                    await self.feed_audio(msg["media"]["payload"])
                elif event == "mark":
                    self.marks_pending = max(0, self.marks_pending - 1)
                    logger.debug(f"PLAYBACK DONE ({msg.get('mark', {}).get('name')}, pending={self.marks_pending})")
                    # Bot audio just finished PLAYING: the silence clock starts
                    # now, not at the caller's previous utterance — otherwise a
                    # long bot reply eats the whole idle window and the nudge/
                    # hangup fires the moment the bot stops, mid-conversation.
                    self.last_activity = time.monotonic()
                elif event == "stop":
                    logger.info("Twilio sent stop — call over")
                    break
        except Exception as e:
            # WebSocketDisconnect lands here too — normal caller hangup.
            logger.info(f"Call loop ended: {type(e).__name__}: {e}")
        finally:
            await self._cleanup()

    async def _greet_on_demand(self):
        global _greeting_ulaw
        try:
            _greeting_ulaw = await synthesize_ulaw(GREETING_TEXT, "en-IN")
            await self.send_ulaw(_greeting_ulaw)
            self.messages.append({"role": "assistant", "content": GREETING_TEXT})
            logger.info("Greeting synthesized on demand")
        except Exception:
            logger.exception("On-demand greeting failed")

    async def _cleanup(self):
        for t in self._bg_tasks:
            t.cancel()
        if self.gen_task and not self.gen_task.done():
            self.gen_task.cancel()
        if self._stt_ctx and self._stt_socket:
            try:
                await self._stt_ctx.__aexit__(None, None, None)
            except Exception:
                pass
        logger.info(f"Call {self.call_sid} cleaned up")


async def handle_twilio_ws(websocket):
    """Entry point from server.py: parse Twilio's handshake, run the session."""
    global active_calls
    stream_sid = call_sid = None
    # Twilio sends "connected" then "start"; media only after that. A caller
    # abandoning during this handshake just ends the coroutine quietly.
    try:
        while True:
            msg = json.loads(await websocket.receive_text())
            if msg.get("event") == "start":
                stream_sid = msg["start"]["streamSid"]
                call_sid = msg["start"]["callSid"]
                break
            if msg.get("event") == "stop":
                return
    except Exception as e:
        logger.info(f"Caller left during handshake: {type(e).__name__}")
        return

    if active_calls >= MAX_CONCURRENT_CALLS:
        logger.warning("Call rejected: concurrency cap reached")
        await websocket.close()
        return

    active_calls += 1
    logger.info(f"Call started: {call_sid} (stream {stream_sid}), active={active_calls}")
    try:
        await CallSession(websocket, stream_sid, call_sid).run()
    except Exception:
        logger.exception("Call session crashed")
    finally:
        active_calls -= 1
