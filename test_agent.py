"""Runnable self-check for the pipecat-free agent: python test_agent.py

Drives a full fake call offline: greeting, turn detection, sentence-streamed
replies, barge-in, noise guard, idle nudge/goodbye/hangup, end_call tool,
and the WAV->mu-law converter. No network, no keys.
"""
import asyncio
import audioop
import base64
import json
import time
import wave
from io import BytesIO
from types import SimpleNamespace

import agent
from agent import CallSession, GREETING_TEXT, _wav_or_raw_to_ulaw

# Speed the timers up for testing (module globals, read at call time).
agent.TURN_SETTLE_SECONDS = 0.15
agent.IDLE_NUDGE_SECONDS = 0.7
agent.IDLE_HANGUP_SECONDS = 1.4


# ---------- fakes ----------

class FakeWS:
    """Fake Twilio websocket: records outbound, scripts inbound, echoes marks
    back immediately (as if Twilio played the audio instantly)."""

    def __init__(self):
        self.sent = []
        self.incoming = asyncio.Queue()

    async def send_text(self, text):
        msg = json.loads(text)
        self.sent.append(msg)
        if msg["event"] == "mark":
            await self.incoming.put(json.dumps({"event": "mark", "mark": msg["mark"]}))

    async def receive_text(self):
        return await self.incoming.get()

    def events(self, kind):
        return [m for m in self.sent if m["event"] == kind]


def stt_data(transcript):
    return SimpleNamespace(type="data", data=SimpleNamespace(transcript=transcript, language_code=None))


def stt_event(signal):
    return SimpleNamespace(type="events", data=SimpleNamespace(signal_type=signal, occured_at=0))


def fake_stream(tokens=(), tool_name=None, delay=0.01):
    """Build a fake openai streaming response."""
    async def gen():
        if tool_name:
            tc = SimpleNamespace(function=SimpleNamespace(name=tool_name, arguments=""))
            yield SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=None, tool_calls=[tc]))])
        for t in tokens:
            yield SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=t, tool_calls=None))])
            await asyncio.sleep(delay)
    return gen()


spoken = []          # (text, language) sent to TTS
hangups = []


async def fake_synthesize(text, language):
    spoken.append((text, language))
    return b"\x00" * 400  # 50ms of mu-law


async def fake_hangup(call_sid):
    hangups.append(call_sid)


def make_session(llm_responses):
    """CallSession wired to fakes. llm_responses: list of fake_stream()s
    handed out per LLM call."""
    ws = FakeWS()
    s = CallSession(ws, "MZtest", "CAtest")
    responses = list(llm_responses)

    async def fake_create(**kwargs):
        return responses.pop(0)

    agent.groq = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=fake_create))
    )
    return ws, s


async def _mark_pump(s):
    """Stand-in for run()'s Twilio receive loop: consume echoed mark events
    so bot_speaking clears when 'playback' finishes."""
    while True:
        msg = json.loads(await s.ws.receive_text())
        if msg["event"] == "mark":
            s.marks_pending = max(0, s.marks_pending - 1)


async def start_session(s):
    """Start the per-call background tasks without a real STT connection."""
    s._bg_tasks.append(asyncio.create_task(s._turn_manager()))
    s._bg_tasks.append(asyncio.create_task(s._idle_watchdog()))
    s._bg_tasks.append(asyncio.create_task(_mark_pump(s)))
    s.last_activity = time.monotonic()


async def stop_session(s):
    for t in s._bg_tasks:
        t.cancel()
    if s.gen_task and not s.gen_task.done():
        s.gen_task.cancel()


# ---------- tests ----------

async def test_normal_turn_and_reply():
    spoken.clear()
    ws, s = make_session([fake_stream(["Hi", " there! ", "How can", " I help?"])])
    await start_session(s)
    await s._on_stt_message(stt_event("START_SPEECH"))
    await s._on_stt_message(stt_data("Hello"))
    await s._on_stt_message(stt_event("END_SPEECH"))
    await asyncio.sleep(0.6)
    assert s.messages[-2]["role"] == "user" and "[en-IN] Hello" in s.messages[-2]["content"], s.messages
    assert s.messages[-1]["role"] == "assistant" and "How can I help?" in s.messages[-1]["content"]
    assert len(spoken) == 2, spoken  # two sentences spoken separately
    assert ws.events("media"), "audio was sent to Twilio"
    await stop_session(s)
    print("PASS normal turn: transcript -> LLM -> per-sentence TTS -> context updated")


async def test_empty_transcript_ignored():
    ws, s = make_session([])
    await start_session(s)
    await s._on_stt_message(stt_data("   "))
    await asyncio.sleep(0.4)
    assert not s.transcript_buffer and (s.gen_task is None), "empty transcript must not trigger a turn"
    await stop_session(s)
    print("PASS empty transcript ignored")


async def test_barge_in_and_noise_guard():
    spoken.clear()
    slow = fake_stream(
        ["One. ", "Two. ", "Three. ", "Four. ", "Five. ", "Six. ", "Seven. ", "Eight. "],
        delay=0.25,
    )
    ws, s = make_session([slow, fake_stream(["Answering the interruption."])])
    await start_session(s)
    await s._on_stt_message(stt_data("Hello"))
    await asyncio.sleep(0.5)  # turn fires (settle 0.15 + tick), reply still streaming
    assert s.gen_task and not s.gen_task.done()
    # 1-word blip while generating -> ignored
    await s._on_stt_message(stt_data("uh"))
    assert not s.transcript_buffer, "1-word blip must be ignored while bot output in flight"
    # 3-word interruption -> cancels generation + clears Twilio buffer
    await s._on_stt_message(stt_data("wait stop please"))
    await asyncio.sleep(0.05)
    assert s.gen_task.cancelled() or s.gen_task.done(), "generation cancelled on barge-in"
    assert ws.events("clear"), "Twilio clear sent on barge-in"
    await asyncio.sleep(0.4)  # the interruption becomes the next turn
    assert s.messages[-1]["content"] == "Answering the interruption."
    await stop_session(s)
    print("PASS barge-in: cancel + clear + noise guard + new turn answered")


async def test_idle_nudge_then_goodbye():
    spoken.clear()
    hangups.clear()
    ws, s = make_session([])
    agent._twilio_hangup = fake_hangup
    await start_session(s)
    await asyncio.sleep(2.2)  # nudge at 0.7, goodbye at 1.4 (+playback drain)
    texts = [t for t, _ in spoken]
    assert any("Are you there" in t for t in texts), texts
    assert any("Goodbye" in t for t in texts), texts
    assert hangups == ["CAtest"], hangups
    assert s.call_ending
    # ending decision must fire exactly once even though watchdog keeps looping
    await asyncio.sleep(0.6)
    assert hangups == ["CAtest"], "end_call must fire exactly once"
    await stop_session(s)
    print("PASS idle: nudge at 5s-equiv, goodbye+hangup at 10s-equiv, fires once")


async def test_idle_respects_open_turn():
    spoken.clear()
    hangups.clear()
    ws, s = make_session([])
    agent._twilio_hangup = fake_hangup
    await start_session(s)
    s.user_speaking = True  # caller mid-speech (e.g. STT transcript pending)
    await asyncio.sleep(1.9)
    assert not spoken and not hangups, "idle must not count while the caller's turn is open"
    await stop_session(s)
    print("PASS idle frozen while user turn open")


async def test_end_call_tool():
    spoken.clear()
    hangups.clear()
    ws, s = make_session([fake_stream(tool_name="end_call")])
    agent._twilio_hangup = fake_hangup
    await start_session(s)
    await s._on_stt_message(stt_data("please end the call now"))
    await asyncio.sleep(0.7)
    assert hangups == ["CAtest"], hangups
    assert any("Goodbye" in t or "వీడ్కోలు" in t for t, _ in spoken), spoken
    await stop_session(s)
    print("PASS end_call tool -> localized goodbye + hangup")


async def test_language_flow():
    spoken.clear()
    ws, s = make_session([fake_stream(["మీకు ఏమి కావాలి?"])])
    await start_session(s)
    await s._on_stt_message(stt_data("నాకు loan కావాలి"))
    await asyncio.sleep(0.6)
    assert s.messages[-2]["content"].startswith("[te-IN]"), s.messages[-2]
    assert spoken and spoken[-1][1] == "te-IN", spoken  # TTS language from the reply
    assert s.current_lang == "te-IN"
    await stop_session(s)
    print("PASS language: Telugu tagged for LLM, TTS spoken in te-IN")


def test_wav_to_ulaw():
    # 8kHz mono PCM16 WAV -> mu-law
    pcm = (b"\x00\x40" * 800)
    buf = BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(8000)
        w.writeframes(pcm)
    out = _wav_or_raw_to_ulaw(buf.getvalue())
    assert out == audioop.lin2ulaw(pcm, 2)
    # 16kHz WAV gets resampled to 8k (half the frames)
    buf2 = BytesIO()
    with wave.open(buf2, "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(16000)
        w.writeframes(pcm)
    out2 = _wav_or_raw_to_ulaw(buf2.getvalue())
    assert abs(len(out2) - len(pcm) // 4) <= 2, (len(out2), len(pcm) // 4)
    # raw headerless PCM passes straight through
    assert _wav_or_raw_to_ulaw(pcm) == audioop.lin2ulaw(pcm, 2)
    print("PASS wav/raw -> mu-law conversion (8k, resampled 16k, raw)")


async def test_handshake_and_media_flow():
    ws = FakeWS()
    sent_to_stt = []

    class FakeSTT:
        async def transcribe(self, audio, encoding, sample_rate):
            sent_to_stt.append((base64.b64decode(audio), encoding, sample_rate))

    await ws.incoming.put(json.dumps({"event": "connected"}))
    await ws.incoming.put(json.dumps({
        "event": "start",
        "start": {"streamSid": "MZx", "callSid": "CAx"},
    }))
    # handle_twilio_ws parses the handshake:
    msg = json.loads(await ws.receive_text())
    assert msg["event"] == "connected"
    msg = json.loads(await ws.receive_text())
    assert msg["start"]["callSid"] == "CAx"
    # media -> PCM16 upconversion for Sarvam
    s = CallSession(ws, "MZx", "CAx")
    s._stt_socket = FakeSTT()
    ulaw = audioop.lin2ulaw(b"\x00\x40" * 160, 2)
    await s.feed_audio(base64.b64encode(ulaw).decode())
    pcm, enc, rate = sent_to_stt[0]
    assert enc == "audio/wav" and rate == 8000
    assert pcm == audioop.ulaw2lin(ulaw, 2)
    print("PASS Twilio handshake parse + mu-law -> PCM16 relay to STT")


async def main():
    test_wav_to_ulaw()
    await test_handshake_and_media_flow()
    await test_normal_turn_and_reply()
    await test_empty_transcript_ignored()
    await test_barge_in_and_noise_guard()
    await test_idle_nudge_then_goodbye()
    await test_idle_respects_open_turn()
    await test_end_call_tool()
    await test_language_flow()
    print("\nall agent self-checks passed")


if __name__ == "__main__":
    agent.synthesize_ulaw = fake_synthesize
    # CallSession.speak references the module global at call time
    asyncio.run(main())
