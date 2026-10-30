"""Text normalisation: cp1251 Kazakh look-alikes, homoglyphs, tokenisation, cheap stemming."""

import re
import unicodedata

# The game client is cp1251, so on kz servers Kazakh letters arrive as look-alike
# cp1251 characters (table from the brief / data README).
KK_FROM_CP1251 = {
    "ќ": "қ", "Ќ": "Қ",
    "ѓ": "ғ", "Ѓ": "Ғ", "ґ": "ө", "Ґ": "Ө",
    "ѕ": "ң", "Ѕ": "Ң",
    "є": "ә", "Є": "Ә",
    "ј": "ө", "Ј": "Ө",
    "ў": "ұ", "Ў": "Ұ",
    "ї": "ү", "Ї": "Ү",
}
KK_TO_CP1251 = {
    "қ": "ќ", "Қ": "Ќ", "ғ": "ѓ", "Ғ": "Ѓ", "ң": "ѕ", "Ң": "Ѕ", "ә": "є", "Ә": "Є",
    "ө": "ј", "Ө": "Ј", "ұ": "ў", "Ұ": "Ў", "ү": "ї", "Ү": "Ї", "һ": "h", "Һ": "H",
}
# Characters that only appear in cp1251-encoded Kazakh (never in Russian or Uzbek).
_KK_CP1251_MARKERS = set("ќЌѓЃѕЅјЈ")  # є ї і ґ also occur in Ukrainian, so they alone do not mean Kazakh
# Latin letters typed inside Cyrillic words (keyboard layout mix): normalise to Cyrillic.
_HOMOGLYPHS = str.maketrans({
    "a": "а", "c": "с", "e": "е", "o": "о", "p": "р", "x": "х", "y": "у", "k": "к",
    "A": "А", "C": "С", "E": "Е", "O": "О", "P": "Р", "X": "Х", "B": "В", "H": "Н",
    "K": "К", "M": "М", "T": "Т",
})
_CYR = re.compile(r"[а-яёәғқңөұүһіўҳ]", re.I)
_LAT = re.compile(r"[a-z]", re.I)
_WORD = re.compile(r"[^\W_]+(?:['ʻ’‘`][^\W_]+)*", re.U)
_ZERO_WIDTH = re.compile(r"[​-‏⁠﻿]")
_UZ_APOS = re.compile(r"(?<=[oOgG])[ʻ’‘`´]")


def has_kk_cp1251(text: str) -> bool:
    return any(ch in _KK_CP1251_MARKERS for ch in text)


def fix_kazakh(text: str, force: bool = False) -> str:
    """Map cp1251 look-alikes back to real Kazakh letters.

    ў is a legitimate Uzbek Cyrillic letter, so it is only remapped when the text
    is known to be Kazakh (force=True, e.g. kz server) or other kk markers are present.
    """
    if not text:
        return text
    if not (force or has_kk_cp1251(text)):
        return text
    out = "".join(KK_FROM_CP1251.get(ch, ch) for ch in text)
    # Latin s/j typed for ң/ө inside an otherwise Cyrillic word.
    out = re.sub(r"(?<=[а-яәғқңөұүһі])s|s(?=[а-яәғқңөұүһі])", "ң", out)
    out = re.sub(r"(?<=[а-яәғқңөұүһі])j|j(?=[а-яәғқңөұүһі])", "ө", out)
    return out


def to_game_encoding(text: str) -> str:
    """Inverse mapping so a Kazakh draft can be shown by the cp1251 game client."""
    return "".join(KK_TO_CP1251.get(ch, ch) for ch in text)


def _fix_mixed_word(word: str) -> str:
    if _CYR.search(word) and _LAT.search(word):
        return word.translate(_HOMOGLYPHS)
    return word


def normalize(text: str, kazakh: bool = False) -> str:
    """Clean text for analysis (the original text is never altered in outputs)."""
    if not text:
        return ""
    t = unicodedata.normalize("NFC", str(text))
    t = _ZERO_WIDTH.sub("", t)
    t = fix_kazakh(t, force=kazakh)
    t = _UZ_APOS.sub("'", t)
    t = re.sub(r"\S+", lambda m: _fix_mixed_word(m.group(0)), t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def tokens(text: str) -> list[str]:
    return [w.lower().replace("ё", "е") for w in _WORD.findall(text or "")]


def stem(word: str, n: int = 5) -> str:
    """Prefix stemmer: crude but language-agnostic and good enough for keyword matching."""
    return word[:n] if len(word) > n else word


def stems(text: str, n: int = 5) -> list[str]:
    return [stem(w, n) for w in tokens(text)]


def char_ngrams(text: str, n: int = 3) -> set[str]:
    t = re.sub(r"\s+", " ", (text or "").lower()).strip()
    if len(t) < n:
        return {t} if t else set()
    return {t[i:i + n] for i in range(len(t) - n + 1)}


def jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def letters_count(text: str) -> int:
    return sum(1 for ch in text or "" if ch.isalpha())
