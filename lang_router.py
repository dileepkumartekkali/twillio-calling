"""Maps a caller's transcript to the Sarvam TTS language code that should answer it.

Detection is by Unicode script, not by asking the LLM to tag its own output —
the LLM's reply streams token-by-token to TTS, so an inline tag like "[te-IN]"
would get split across frames and is unreliable to parse. Script detection runs
once on the STT's finalized transcript, before the LLM turn starts.
"""

# ponytail: Devanagari covers both Hindi and Marathi; we always resolve it to
# hi-IN. Upgrade to a real language-ID model only if Marathi callers turn out
# to be common enough that Hindi TTS reading their replies sounds wrong.
SCRIPT_RANGES = [
    (0x0900, 0x097F, "hi-IN"),  # Devanagari (Hindi, Marathi)
    (0x0980, 0x09FF, "bn-IN"),  # Bengali
    (0x0A00, 0x0A7F, "pa-IN"),  # Gurmukhi (Punjabi)
    (0x0A80, 0x0AFF, "gu-IN"),  # Gujarati
    (0x0B00, 0x0B7F, "or-IN"),  # Odia
    (0x0B80, 0x0BFF, "ta-IN"),  # Tamil
    (0x0C00, 0x0C7F, "te-IN"),  # Telugu
    (0x0C80, 0x0CFF, "kn-IN"),  # Kannada
    (0x0D00, 0x0D7F, "ml-IN"),  # Malayalam
]

DEFAULT_LANGUAGE = "en-IN"


def detect_target_language(text: str) -> str:
    """Dominant Indic script in `text` -> Sarvam language code. Falls back to en-IN
    for pure-English/Latin text (also covers empty/whitespace-only input)."""
    counts = {}
    for ch in text:
        cp = ord(ch)
        for start, end, code in SCRIPT_RANGES:
            if start <= cp <= end:
                counts[code] = counts.get(code, 0) + 1
                break
    if not counts:
        return DEFAULT_LANGUAGE
    return max(counts, key=counts.get)
