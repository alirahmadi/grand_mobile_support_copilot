"""Heuristic language identification for short, noisy, mixed-language player messages.

Returns one of ru | kk | uz | en | es | pt | und. If a trained char-n-gram model
(models/lang.joblib, built by `make train` from the `lang` field of the train set)
exists, it is blended in.
"""

import re

from .textnorm import letters_count, tokens

KK_SPECIFIC = set("әңөұүһі")          # not used in Russian or Uzbek Cyrillic
UZ_CYR_SPECIFIC = set("ўҳ")
SHARED_KK_UZ = set("қғ")

KK_WORDS = {
    "мен", "сен", "біз", "сіз", "ол", "жоқ", "бар", "керек", "қалай", "неге", "үшін",
    "және", "мені", "маған", "сәлем", "сәлеметсіз", "рахмет", "көмек", "көмектесіңіз", "ойын",
    "ойыншы", "емес", "болды", "жатыр", "қойды", "алды", "берді", "деп", "бұл", "осы", "мына",
    "ақша", "өтінемін", "өтініш", "кешіріңіз", "жаза", "шағым", "ұрды", "өлтірді",
    "балағаттады", "қорлады", "тұрып", "қалдым", "шыға", "алмай", "жатырмын", "ма", "ме", "ба",
    "бе", "па", "пе", "иә", "жақсы", "бір", "екі", "уақыт", "кім", "қайда", "неше",
}
UZ_LAT_WORDS = {
    "men", "sen", "biz", "siz", "u", "va", "yoq", "yo'q", "bor", "kerak", "qanday", "nega",
    "nima", "uchun", "bilan", "salom", "assalomu", "alaykum", "rahmat", "yordam", "iltimos",
    "o'yin", "oyin", "o'yinchi", "emas", "bo'ldi", "boldi", "qildi", "qiling", "mani", "meni",
    "menga", "akkaunt", "pul", "haqorat", "so'kdi", "sokdi", "o'ldirdi", "oldirdi", "qolib",
    "ketdim", "chiqa", "olmayapman", "ishlamayapti", "xato", "admin", "nimaga", "qachon",
    "qayerda", "bergan", "berdi", "aka", "ber", "bering", "kim", "endi", "hali", "juda",
}
UZ_CYR_WORDS = {
    "ва", "йўқ", "бор", "керак", "қандай", "нега", "нима", "учун", "билан", "салом", "раҳмат",
    "ёрдам", "илтимос", "ўйин", "эмас", "бўлди", "қилди", "мени", "менга", "пул", "ҳақорат",
    "сўкди", "ўлдирди", "қолиб", "кетдим", "ишламаяпти", "хато", "нимага", "қачон", "беринг",
}
RU_WORDS = {
    "и", "в", "не", "на", "я", "что", "меня", "мне", "он", "она", "как", "это", "у", "за",
    "по", "с", "а", "но", "так", "почему", "помогите", "пожалуйста", "игрок", "админ",
    "здравствуйте", "привет", "можно", "нет", "да", "мой", "моя", "был", "была", "его",
    "когда", "где", "уже", "просто", "очень", "сейчас", "тут", "есть", "без", "вы",
}
EN_WORDS = {
    "the", "a", "an", "is", "are", "i", "me", "my", "you", "he", "she", "it", "and", "or",
    "not", "no", "please", "help", "why", "how", "what", "player", "admin", "hacked", "stuck",
    "banned", "ban", "account", "this", "that", "was", "with", "for", "can", "cant", "can't",
    "insulted", "killed", "cheater", "cheat", "money", "hello", "hi", "thanks", "game",
}
ES_WORDS = {
    "el", "la", "los", "las", "de", "que", "y", "en", "un", "una", "por", "para", "con", "no",
    "me", "mi", "es", "hola", "ayuda", "jugador", "cuenta", "porque", "por qué", "gracias",
    "baneado", "insultó", "mató", "dinero", "está", "estoy", "atascado", "pero", "muy",
}
PT_WORDS = {
    "o", "a", "os", "as", "de", "que", "e", "em", "um", "uma", "por", "para", "com", "não",
    "nao", "me", "meu", "minha", "é", "olá", "ola", "ajuda", "jogador", "conta", "porque",
    "obrigado", "banido", "xingou", "matou", "dinheiro", "estou", "preso", "você", "voce",
}
# Russian typed in Latin ("translit"): reply in Russian.
RU_TRANSLIT = {
    "privet", "pomogite", "pochemu", "kak", "chto", "menya", "mne", "zastryal", "zastrial",
    "admin", "adminy", "pozhaluista", "pojaluista", "pozhalujsta", "igrok", "oskorbil",
    "ubil", "zabanili", "ban", "razban", "vzlomali", "akkaunt", "dengi", "ne", "rabotaet",
    "spasibo", "zdravstvuite", "tekstury", "tekstur", "cheater", "chiter", "tut", "eto",
}


def _score(words: list[str], vocab: set) -> int:
    return sum(1 for w in words if w in vocab)


def detect_lang(text: str, hint: str | None = None) -> tuple[str, float]:
    """Return (lang, confidence 0..1)."""
    t = text or ""
    n_letters = letters_count(t)
    if n_letters < 2:
        return (hint if hint and hint != "und" else "und"), 0.3
    low = t.lower()
    words = tokens(low)
    cyr = sum(1 for ch in low if "а" <= ch <= "я" or ch in "ёәғқңөұүһіўҳ")
    lat = sum(1 for ch in low if "a" <= ch <= "z")

    if cyr >= lat:
        kk_chars = sum(1 for ch in low if ch in KK_SPECIFIC)
        uz_chars = sum(1 for ch in low if ch in UZ_CYR_SPECIFIC)
        shared = sum(1 for ch in low if ch in SHARED_KK_UZ)
        scores = {
            "kk": 2.0 * kk_chars + 2.5 * _score(words, KK_WORDS) + 0.5 * shared,
            "uz": 2.0 * uz_chars + 2.5 * _score(words, UZ_CYR_WORDS) + 0.5 * shared,
            "ru": 1.5 * _score(words, RU_WORDS) + 0.02 * cyr,
        }
        # Russian function words are short and also occur in mixed messages: Kazakh
        # wins only with real evidence, Russian is the default for Cyrillic.
        best = max(scores, key=scores.get)
        if best != "ru" and scores[best] < 2.0:
            best = "ru"
        total = sum(scores.values()) or 1.0
        conf = min(0.99, 0.5 + scores[best] / (2 * total))
        return best, round(conf, 2)

    # Latin script
    apos_uz = len(re.findall(r"\b\w*[og]'\w*", low)) + len(re.findall(r"\b\w*(?:sh|ch)\w*", low)) * 0.3
    scores = {
        "uz": 2.0 * _score(words, UZ_LAT_WORDS) + 1.5 * apos_uz + 0.3 * low.count("q"),
        "en": 2.0 * _score(words, EN_WORDS),
        "es": 2.0 * _score(words, ES_WORDS) + 1.0 * len(re.findall(r"[ñ¿¡]", low)),
        "pt": 2.0 * _score(words, PT_WORDS) + 1.0 * len(re.findall(r"[ãõç]", low)),
        "ru": 2.5 * _score(words, RU_TRANSLIT),
    }
    best = max(scores, key=scores.get)
    if scores[best] < 1.5:
        return (hint if hint and hint != "und" else "und"), 0.35
    total = sum(scores.values()) or 1.0
    conf = min(0.99, 0.5 + scores[best] / (2 * total))
    return best, round(conf, 2)


class LangDetector:
    """Heuristic detector, optionally blended with a char-n-gram model trained on train.lang."""

    def __init__(self, model=None):
        self.model = model  # sklearn pipeline with predict_proba, or None

    def detect(self, text: str, hint: str | None = None) -> tuple[str, float]:
        lang, conf = detect_lang(text, hint)
        if self.model is not None and letters_count(text) >= 2:
            try:
                proba = self.model.predict_proba([text])[0]
                classes = list(self.model.classes_)
                i = int(proba.argmax())
                m_lang, m_conf = classes[i], float(proba[i])
                if m_lang != "und" and (m_conf > conf or lang == "und"):
                    lang, conf = m_lang, round(m_conf, 2)
            except Exception:
                pass
        # The dataset carries the organisers' own heuristic label; trust it when we are unsure.
        if hint and hint not in ("und", None, "") and conf < 0.6:
            return hint, 0.6
        return lang, conf
