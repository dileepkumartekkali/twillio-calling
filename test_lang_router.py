"""Runnable self-check: python test_lang_router.py"""
from lang_router import detect_target_language

assert detect_target_language("Hello, how are you?") == "en-IN"
assert detect_target_language("") == "en-IN"
assert detect_target_language("नमस्ते, आप कैसे हैं?") == "hi-IN"
assert detect_target_language("మీరు ఎలా ఉన్నారు?") == "te-IN"
# Telugu+English code-mix: Telugu script dominates -> te-IN, Bulbul handles the embedded English.
assert detect_target_language("Sir నాకు ఒక appointment కావాలి please") == "te-IN"
assert detect_target_language("வணக்கம், எப்படி இருக்கீங்க?") == "ta-IN"

print("all lang_router checks passed")
