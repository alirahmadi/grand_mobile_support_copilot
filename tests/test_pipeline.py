"""Contract and safety tests. Run: make test (uses demo fixtures if data/ is empty)."""

import json

import pytest

from copilot.pipeline import Copilot
from copilot.templates import translate_punishment, uz_latin_to_cyrillic
from copilot.textnorm import fix_kazakh, normalize

CONTRACT_KEYS = {"lang", "category", "priority", "is_spam_or_duplicate", "extracted", "draft_reply", "needs_human",
                 "confidence", "reasons"}


@pytest.fixture(scope="module")
def cop():
    return Copilot(use_llm=False)


def test_contract_example(cop):
    req = {"kind": "complaint", "messages": [{"role": "player", "text": "Arman_Petrov оскорблял меня и мою семью в чате, скрины есть, 2.1 ОП", "attachments": 2}]}
    r = cop.triage(req)
    assert CONTRACT_KEYS <= set(r)
    assert set(r["extracted"]) == {"nickname", "rule_refs", "has_evidence", "server"}
    assert set(r["draft_reply"]) == {"lang", "text", "cited"}
    assert r["lang"] == "ru"
    assert r["extracted"]["nickname"] == "Arman_Petrov"
    assert r["extracted"]["has_evidence"] is True
    assert r["priority"] in cop.tax.priority_ids
    assert r["category"] in cop.tax.by_id
    assert all(c in cop.book.by_id for c in r["draft_reply"]["cited"])
    assert r["needs_human"] is True  # punishment must be confirmed by an admin
    json.dumps(r, ensure_ascii=False)


def test_kazakh_cp1251_normalisation():
    assert fix_kazakh("текстураѓа тўрып ќалдым кјмектесіѕіз") == "текстураға тұрып қалдым көмектесіңіз"
    # Uzbek ў is left alone when there are no Kazakh markers
    assert fix_kazakh("ўйин") == "ўйин"
    assert normalize("Cпacибo") == "Спасибо"  # Latin homoglyphs inside a Cyrillic word


@pytest.mark.parametrize("text,lang", [
    ("застрял в текстурах помогите", "ru"),
    ("текстураѓа тўрып ќалдым кјмектесіѕіз", "kk"),
    ("teksturada qolib ketdim yordam bering", "uz"),
    ("my account was hacked, password changed", "en"),
])
def test_language_and_reply_language(cop, text, lang):
    r = cop.triage({"kind": "report", "messages": [{"role": "player", "text": text}]})
    assert r["lang"] == lang
    assert r["draft_reply"]["lang"] == lang


def test_always_human_topics(cop):
    for text in ("задонатил 500 рублей а донат не пришел", "меня взломали, сменили пароль", "разбаньте пожалуйста, бан несправедливый"):
        r = cop.triage({"kind": "report", "messages": [{"role": "player", "text": text}]})
        assert r["needs_human"] is True, text


def test_need_evidence_is_auto_and_safe(cop):
    r = cop.triage({"kind": "complaint", "reported_nick": "Oleg_Smirnov",
                    "messages": [{"role": "player", "text": "он меня послал матом и обозвал"}]})
    assert r["extracted"]["has_evidence"] is False
    assert r["meta"]["suggested_verdict"] == "need_evidence"
    assert r["needs_human"] is False


def test_duplicates_within_hour(cop):
    base = {"kind": "complaint", "reporter_hash": "dup-test", "reported_nick": "Kirill_Lebedev", "images_count": 1}
    a = cop.triage({**base, "created_at": "2026-03-14T10:00:00", "messages": [{"role": "player", "text": "обманул при обмене машины"}]})
    b = cop.triage({**base, "created_at": "2026-03-14T10:20:00", "messages": [{"role": "player", "text": "обманул при обмене машины!!"}]})
    c = cop.triage({**base, "created_at": "2026-03-14T13:00:00", "messages": [{"role": "player", "text": "обманул при обмене машины"}]})
    assert not a["is_spam_or_duplicate"] and b["is_spam_or_duplicate"] and not c["is_spam_or_duplicate"]


def test_spam(cop):
    r = cop.triage({"kind": "report", "messages": [{"role": "player", "text": "заходите на наш сервер t.me/promo промокод"}]})
    assert r["is_spam_or_duplicate"] is True


def test_admin_messages_are_ignored(cop):
    req = {"kind": "complaint", "reported_nick": "A_B", "messages": [
        {"role": "player", "text": "оскорбил меня"}, {"role": "admin", "text": "жалоба одобрена, мут 30 минут"}]}
    assert "одобрена" not in cop.prepare(req)["t"]["player_text"]


class FakeLLM:
    """Simulates a misbehaving model: invented rule, unban promise."""
    model = "fake"

    def __init__(self, payload):
        self.payload = payload

    def call_json(self, *a, **k):
        return self.payload, {"cost_usd": 0.0003, "latency_ms": 5, "error": None}


def test_guard_removes_invented_rules_and_flags_promises(cop):
    fake = FakeLLM({"category": cop.tax.categories[0].id, "priority": "p2", "suggested_verdict": "answer",
                    "draft_text": "Здравствуйте! По пункту 99.9 правил мы вас разбаним сегодня.", "cited": ["99.9"],
                    "needs_human": False, "confidence": 0.95, "reasons": ["ok"]})
    old, cop.llm = cop.llm, fake
    try:
        r = cop.triage({"kind": "report", "messages": [{"role": "player", "text": "разбаньте меня"}]})
    finally:
        cop.llm = old
    assert "99.9" not in r["draft_reply"]["cited"]
    assert "99.9" not in r["draft_reply"]["text"]
    assert r["needs_human"] is True
    assert r["meta"]["cost_usd"] == pytest.approx(0.0003)


def test_llm_valid_answer_is_used(cop):
    rid = "2.1" if "2.1" in cop.book.by_id else cop.book.rules[0].id
    fake = FakeLLM({"category": cop.tax.categories[0].id, "priority": "p3", "suggested_verdict": "need_evidence",
                    "draft_text": f"Здравствуйте! Приложите, пожалуйста, скриншоты нарушения пункта {rid}.",
                    "cited": [rid], "needs_human": False, "confidence": 0.9, "reasons": ["нет доказательств"]})
    old, cop.llm = cop.llm, fake
    try:
        r = cop.triage({"kind": "complaint", "reported_nick": "Oleg_Smirnov",
                        "messages": [{"role": "player", "text": "Oleg_Smirnov оскорбил меня"}]})
    finally:
        cop.llm = old
    assert r["meta"]["mode"] == "llm"
    assert r["draft_reply"]["cited"] == [rid]


def test_punishment_translation_and_uz_script():
    assert translate_punishment("мут на 30 минут", "kk") == "мут (чатқа тыйым) 30 минутқа"
    assert translate_punishment("бан на 3 дня", "en") == "ban for 3 days"
    assert translate_punishment("что-то непонятное", "uz") is None
    assert uz_latin_to_cyrillic("Assalomu alaykum! Rahmat.") == "Ассалому алайкум! Раҳмат."
