"""Contract and safety tests (demo fixtures, no trained models). Run: make test."""

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
    # the example from the brief
    req = {"kind": "complaint", "messages": [{"role": "player", "text": "Arman_Petrov оскорблял меня и мою семью в чате, скрины есть, 2.1 ОП", "attachments": 2}]}
    r = cop.triage(req)
    assert CONTRACT_KEYS <= set(r)
    assert set(r["extracted"]) == {"nickname", "rule_refs", "has_evidence", "server"}
    assert set(r["draft_reply"]) == {"lang", "text", "cited"}
    assert r["lang"] == "ru"
    assert r["category"] == "complaint.insult"
    assert r["priority"] == "p2"
    assert r["extracted"]["nickname"] == "Arman_Petrov"
    assert r["extracted"]["rule_refs"] == ["2.1"]
    assert r["extracted"]["has_evidence"] is True
    assert r["draft_reply"]["cited"] == ["2.1"]
    assert "2.1 ОП" in r["draft_reply"]["text"]
    assert r["needs_human"] is True  # punishment must be confirmed by an admin
    json.dumps(r, ensure_ascii=False)


def test_rule_references(cop):
    b = cop.book
    assert b.parse_refs("2.1оп")[0] == ["2.1"]
    assert b.parse_refs("1.1 gz")[0] == ["GZ 1.1"]
    assert b.parse_refs("ЖБ 1.2.7")[0] == ["ЖБ 1.2.7"]
    assert b.refs_in_text("п. 1.2.7 правил подачи жалоб") == ["ЖБ 1.2.7"]
    assert b.refs_in_text("(shikoyat berish qoidalarining 1.2.9-bandi)") == ["ЖБ 1.2.9"]
    assert b.parse_refs("9.9") == ([], ["9.9"])


def test_kazakh_cp1251_normalisation():
    assert fix_kazakh("текстураѓа тўрып ќалдым кјмектесіѕіз") == "текстураға тұрып қалдым көмектесіңіз"
    assert fix_kazakh("ўйин") == "ўйин"  # Uzbek ў is left alone without Kazakh markers
    assert normalize("Cпacибo") == "Спасибо"  # Latin homoglyphs inside a Cyrillic word


@pytest.mark.parametrize("text,lang", [
    ("застрял в текстурах помогите", "ru"),
    ("текстураѓа тўрып ќалдым кјмектесіѕіз", "kk"),
    ("teksturada qolib ketdim yordam bering", "uz"),
    ("my account was hacked, password changed", "en"),
    ("адміністратор він має мене в'язати", "ru"),
])
def test_language_and_reply_language(cop, text, lang):
    r = cop.triage({"kind": "report", "messages": [{"role": "player", "text": text}]})
    assert r["lang"] == lang or (lang == "ru" and r["draft_reply"]["lang"] == "ru")
    assert r["draft_reply"]["lang"] == lang


def test_always_human_topics(cop):
    for kind, text in (("report", "задонатил 500 рублей а донат не пришел"), ("report", "меня взломали, сменили пароль"),
                       ("complaint", "не согласен с наказанием, прошу снять бан")):
        r = cop.triage({"kind": kind, "reported_nick": "A_B", "messages": [{"role": "player", "text": text}]})
        assert r["needs_human"] is True, text
    r = cop.triage({"kind": "complaint", "reported_nick": "A_B", "messages": [{"role": "player", "text": "не согласен с наказанием, прошу снять бан"}]})
    assert r["category"] == "complaint.admin"


def test_profanity_complaint_cites_complaint_rules(cop):
    r = cop.triage({"kind": "complaint", "reported_nick": "Oleg_Smirnov", "images_count": 1,
                    "messages": [{"role": "player", "text": "этот пидор меня убил нахуй"}]})
    assert r["meta"]["suggested_verdict"] == "invalid_format"
    assert r["draft_reply"]["cited"] == ["ЖБ 1.2.7"]


def test_help_report_mimics_admins(cop):
    r = cop.triage({"kind": "report", "messages": [{"role": "player", "text": "помогите"}]})
    assert r["category"] == "report.offtopic"
    assert r["draft_reply"]["text"] == "Приветствую, слежу за вами."
    assert r["needs_human"] is True
    r = cop.triage({"kind": "report", "messages": [{"role": "player", "text": "+"}]})
    assert r["draft_reply"]["text"] == "Приветствую, уточните ваш вопрос или жалобу."


def test_duplicates_within_hour(cop):
    base = {"kind": "complaint", "reporter_hash": "dup-test", "reported_nick": "Kirill_Lebedev", "images_count": 1}
    a = cop.triage({**base, "created_at": "2026-03-14T10:00:00", "messages": [{"role": "player", "text": "обманул при обмене машины"}]})
    b = cop.triage({**base, "created_at": "2026-03-14T10:20:00", "messages": [{"role": "player", "text": "обманул при обмене машины!!"}]})
    c = cop.triage({**base, "created_at": "2026-03-14T13:00:00", "messages": [{"role": "player", "text": "обманул при обмене машины"}]})
    assert not a["is_spam_or_duplicate"] and b["is_spam_or_duplicate"] and not c["is_spam_or_duplicate"]
    assert b["draft_reply"]["cited"] == ["ЖБ 1.2.9"]


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
    fake = FakeLLM({"category": "report.question", "priority": "p2", "suggested_verdict": "answer",
                    "draft_text": "Приветствую, вас разбанят сегодня. По пункту 99.9 ОП это разрешено.", "cited": ["99.9", "ЖБ 7.7"],
                    "needs_human": False, "confidence": 0.95, "reasons": ["ok"]})
    old, cop.llm = cop.llm, fake
    try:
        r = cop.triage({"kind": "report", "messages": [{"role": "player", "text": "разбаньте меня"}]})
    finally:
        cop.llm = old
    assert r["draft_reply"]["cited"] == []
    assert "99.9" not in r["draft_reply"]["text"]
    assert r["needs_human"] is True
    assert r["meta"]["cost_usd"] == pytest.approx(0.0003)


def test_llm_valid_answer_is_used(cop):
    fake = FakeLLM({"category": "complaint.insult", "priority": "p2", "suggested_verdict": "need_evidence",
                    "draft_text": "Уважаемый игрок, благодарим вас за жалобу. Приложите скриншоты нарушения п. 2.1 ОП.",
                    "cited": ["2.1"], "needs_human": False, "confidence": 0.9, "reasons": ["нет доказательств"]})
    old, cop.llm = cop.llm, fake
    try:
        r = cop.triage({"kind": "complaint", "reported_nick": "Oleg_Smirnov",
                        "messages": [{"role": "player", "text": "Oleg_Smirnov оскорбил меня"}]})
    finally:
        cop.llm = old
    assert r["meta"]["mode"] == "llm"
    assert r["draft_reply"]["cited"] == ["2.1"]


def test_punishment_translation_and_uz_script():
    assert translate_punishment("мут на 30 минут", "kk") == "мут (чатқа тыйым) 30 минутқа"
    assert translate_punishment("бан на 3 дня", "en") == "a ban for 3 days"
    assert translate_punishment("блокировкой чата на 30 минут", "ru") == "мут на 30 минут"
    assert translate_punishment("что-то непонятное", "uz") is None
    assert uz_latin_to_cyrillic("Assalomu alaykum! Rahmat.") == "Ассалому алайкум! Раҳмат."


@pytest.mark.parametrize("req", [
    {}, {"kind": "complaint"}, {"kind": "report", "messages": []}, {"kind": "report", "messages": "помогите"},
    {"kind": "report", "messages": [{"role": "player", "text": None}]}, {"kind": "report", "messages": ["помогите", None]},
    {"kind": "complaint", "images_count": "abc", "messages": [{"role": "player", "text": "оск", "attachments": "много"}]},
    {"kind": "report", "messages": [{"role": "player", "text": 12345}]}, {"kind": "ticket", "region": 5, "server_lang": ["uz"]},
    {"kind": "complaint", "rule_refs_raw": ["2.1", 5.2], "messages": [{"role": "player", "text": "{nick} {0} %s оск"}]},
    {"_bad_line": "{oops", "ticket_id": "x"}, "not a dict", None,
])
def test_malformed_inputs_never_crash(cop, req):
    r = cop.safe_finish(cop.safe_prepare(req))
    assert r["category"] in cop.tax.by_id and r["priority"] in cop.tax.priority_ids
    assert r["draft_reply"]["text"] and "{" not in r["draft_reply"]["text"]
    if isinstance(req, dict) and "_bad_line" in req:
        assert r["needs_human"] is True and r["meta"]["mode"] == "fallback"
