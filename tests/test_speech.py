import pytest
from scipy.signal import resample_poly

from liberational import config as cfg
from liberational.speech import kokoro, sherpa

SPOKEN = {
    "Sun sets at 20:45 tonight.": "Sun sets at twenty forty five tonight.",
    "It's 11:28 PM.": "It's eleven twenty eight p m.",
    "Meet at 9:05 a.m. or 9:00.": "Meet at nine oh five a m or nine o'clock.",
    "Lights out at 24:00.": "Lights out at twenty four hundred.",
    "Nearest coffee is 200 meters ahead.": "Nearest coffee is two hundred meters ahead.",
    "That costs $3.50, or $1,200 total.": "That costs three dollars fifty, or one thousand two hundred dollars total.",
    "It was $5.5 each.": "It was five dollars fifty each.",
    "Battery at 45%.": "Battery at forty five percent.",
    "Your 21st birthday was in 2019.": "Your twenty first birthday was in twenty nineteen.",
    "There were 2019 people.": "There were two thousand and nineteen people.",
    "Back in the 1990s.": "Back in the nineteen nineties.",
    "Music of the 2000s.": "Music of the two thousands.",
    "Early 1900s houses.": "Early nineteen hundreds houses.",
    "Update to v2.0.1 now.": "Update to v two point oh point one now.",
    "Python 3.12.4 is out.": "Python three point twelve point four is out.",
    "Gas is $1.005 a liter.": "Gas is one point zero zero five dollars a liter.",
    "Pi is about 3.14.": "Pi is about three point one four.",
    "Room 101, floor 3.": "Room one hundred and one, floor three.",
    "Up 3 to 5 floors.": "Up three to five floors.",
    "Call 555-1234.": "Call five five five one two three four.",
    "It's 54°F in Oakland.": "It's fifty four degrees Fahrenheit in Oakland.",
    "partly cloudy, +12°C.": "partly cloudy, plus twelve degrees Celsius.",
    "London is -3°C": "London is minus three degrees Celsius",
    "The bike was blue.": "The bike was blue.",
}


@pytest.mark.parametrize("text, expected", SPOKEN.items(), ids=list(SPOKEN))
def test_speakable_spells_out_numbers_and_symbols(text, expected):
    assert kokoro.speakable(text) == expected


def test_speakable_reads_a_number_too_long_for_words_digit_by_digit():
    assert kokoro.speakable("code " + "12" * 200 + ".") == "code " + " ".join(["one two"] * 200) + "."


@pytest.fixture
def fresh_has_mlx():
    cfg.has_mlx.cache_clear()
    yield
    cfg.has_mlx.cache_clear()


def test_mlx_only_on_apple_silicon(monkeypatch, fresh_has_mlx):
    monkeypatch.setattr(cfg, "platform_key", lambda: "linux-x86_64")
    assert not cfg.has_mlx()


def _models_ready():
    try:
        kokoro.espeak_config()
    except Exception:
        return False
    return all(cfg.model_path(n).exists() for n in ("tts_kokoro", "tts_kokoro_voices", "asr_parakeet"))


@pytest.fixture(scope="module")
def models():
    """The real weights, and the phonemizer backend they phonemize through released at the end: espeak keeps global
    state, so phonemizer copies the library into a temp dir of its own and deletes it only when the backend goes. Left
    to the interpreter's exit, that copy outlives the suite's own leak check."""
    from phonemizer.phonemize import clear_backends_cache

    yield kokoro.Kokoro(), sherpa.SherpaParakeet()
    clear_backends_cache()


# Loads the real models: catches wiring bugs the unit tests can't.
@pytest.mark.skipif(not _models_ready(), reason="models or espeak-ng missing (mise run models; apt install espeak-ng)")
@pytest.mark.parametrize(
    "text, heard", [("The bike is at the north rack.", "north rack"), ("Hi Luca, Dana is at the door.", "dana")]
)
def test_kokoro_to_parakeet_round_trip(models, text, heard):
    tts, asr = models
    audio, sr = tts.synth(text)
    assert sr == 24000
    assert heard in asr(resample_poly(audio, 2, 3).astype("float32")).lower()
