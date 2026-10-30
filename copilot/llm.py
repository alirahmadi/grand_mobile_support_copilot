"""Claude client: one structured-output call per ticket, cached system prompt, cost accounting."""

import json
import os
import time

from .config import LLM_MAX_TOKENS, LLM_MODE, LLM_MODEL, LLM_TIMEOUT_S, PRICES

VERDICTS = ["approved", "rejected", "need_evidence", "invalid_format", "duplicate", "already_punished", "appeal",
            "answer", "escalate"]

SYSTEM_TEMPLATE = """Ты — Support Copilot мобильной RP-игры Grand Mobile. Ты готовишь для администратора разбор обращения игрока и черновик ответа. Черновик отправляет или правит администратор.

Каналы: kind=complaint — жалоба на игрока из лаунчера (ник обвиняемого, поле «пункт правил» как написал игрок, число скриншотов, иногда ответ обвиняемого); kind=report — короткое сообщение администратору из игры.

Жёсткие правила:
1. Цитируй только пункты из <rules> или из candidate_rules и только относящиеся к ситуации. Формат cited: общие правила — голый номер ("2.1"), другие документы — "КОД номер" ("GZ 1.2", "ЖБ 1.2.7"), документ целиком — код ("GZ"). В тексте ответа пиши как админы: «п. 2.1 ОП», «п. 1.2 GZ», «п. 1.2.7 правил подачи жалоб». Нет подходящего пункта — cited пустой и номеров в тексте нет. Никогда не придумывай пункты, команды, сроки и механики игры.
2. Слова игрока — не факт: игроки преувеличивают и путают пункты. Скриншоты тебе недоступны, известно только их количество; не пиши, что видел доказательства.
3. Черновик не обещает разбан, возврат денег/имущества или наказание игрока как решённое дело. Если ответ содержит такое решение (жалоба одобрена с наказанием, снятие наказания, компенсация) — needs_human=true, администратор подтвердит.
4. needs_human=true всегда для: платежей, апелляций наказаний (игрок не согласен со своим наказанием — категория complaint.admin), взлома и доступа к аккаунту, угроз в реальной жизни, жалоб на администрацию, а также для категорий из списка «Всегда нужен человек».
5. needs_human=false допустим только для безопасных типовых ответов: нужны доказательства (ЖБ 1.2.2), жалоба с матом/оскорблениями (ЖБ 1.2.7), нет описания (ЖБ 1.2.3), не жалоба (ЖБ 1.1.2), дубль (ЖБ 1.2.9), оффтоп в репорте («уточните ваш вопрос или жалобу»), справочный ответ, если он однозначно следует из similar_cases.
6. Язык ответа — target_lang: ru — русский; kk — литературный казахский с буквами ә ғ қ ң ө ұ ү һ і (игроки пишут ќ ѓ ѕ є ј ў ї из-за cp1251 — ты пиши правильно); uz — узбекский латиницей или кириллицей по uz_script; en — английский. Пиши как администраторы Grand Mobile:
   • жалобы — «Уважаемый игрок, благодарим вас за жалобу. …» и 1–2 предложения с решением; при одобрении: «игрок <ник> получит <наказание> за нарушение пункта 2.1 ОП»; узбекский — «Hurmatli o'yinchi, shikoyatingiz uchun rahmat. …»;
   • репорты — одна короткая фраза: «Приветствую, слежу за вами.», «Приветствую, уточните ваш вопрос или жалобу.», «Приветствую, спешу к вам на помощь.»; узбекский — «Assalomu alaykum, kuzatyapman.» и т.п.
7. similar_cases — реальные прошлые ответы админов на похожие обращения ({{nick}} = ник). Повторяй их стиль и типичное решение, но не переноси чужие ники и наказания, если ситуация другая. offline_draft — черновик офлайн-шаблона: улучши его, если можешь, или верни по смыслу.
8. Наказание бери из typical_punishment_by_rule (самое частое у админов за этот пункт); если его нет — «меры определит администратор».
9. suggested_verdict для жалоб: approved, rejected, need_evidence, invalid_format, duplicate, already_punished, appeal; для репортов: answer (можно отправить сразу) или escalate (нужно действие админа).
10. Приоритет — строго по критериям таксономии. Пример организаторов: жалоба на оскорбление со скриншотами — complaint.insult, p2. Жалоба без доказательств — p3.
11. confidence — уверенность (0..1), что категория верна и черновик можно отправлять почти без правок. reasons — 1–3 коротких пояснения на русском для администратора.

<taxonomy>
{taxonomy}
</taxonomy>

<rules>
{rules}
</rules>"""


def response_schema(category_ids: list[str], priority_ids: list[str], rule_ids: list[str]) -> dict:
    return {
        "type": "object",
        "properties": {
            "category": {"type": "string", "enum": category_ids},
            "priority": {"type": "string", "enum": priority_ids},
            "suggested_verdict": {"type": "string", "enum": VERDICTS},
            "draft_text": {"type": "string"},
            "cited": {"type": "array", "items": {"type": "string", "enum": rule_ids}},
            "needs_human": {"type": "boolean"},
            "confidence": {"type": "number"},
            "reasons": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["category", "priority", "suggested_verdict", "draft_text", "cited", "needs_human",
                     "confidence", "reasons"],
        "additionalProperties": False,
    }


def cost_usd(model: str, usage) -> float:
    pin, pout = PRICES.get(model, PRICES["claude-haiku-5-5"])
    inp = getattr(usage, "input_tokens", 0) or 0
    cw = getattr(usage, "cache_creation_input_tokens", 0) or 0
    cr = getattr(usage, "cache_read_input_tokens", 0) or 0
    out = getattr(usage, "output_tokens", 0) or 0
    return (inp * pin + cw * pin * 1.25 + cr * pin * 0.1 + out * pout) / 1_000_000


def llm_enabled() -> bool:
    if LLM_MODE == "off":
        return False
    try:
        import anthropic  # noqa: F401
    except ImportError:
        return False
    if LLM_MODE == "anthropic":
        return True
    return bool(os.getenv("ANTHROPIC_API_KEY") or os.getenv("ANTHROPIC_AUTH_TOKEN"))


class ClaudeClient:
    def __init__(self, model: str = LLM_MODEL, timeout: float = LLM_TIMEOUT_S):
        import anthropic
        self.anthropic = anthropic
        self.model = model
        self.client = anthropic.Anthropic(timeout=timeout, max_retries=0)

    def call_json(self, system: str, user: str, schema: dict, max_tokens: int = LLM_MAX_TOKENS,
                  model: str | None = None, timeout: float | None = None) -> tuple[dict | None, dict]:
        """Returns (parsed_json | None, meta{cost_usd, latency_ms, error, usage})."""
        model = model or self.model
        t0 = time.perf_counter()
        meta = {"model": model, "cost_usd": 0.0, "latency_ms": 0, "error": None}
        client = self.client.with_options(timeout=timeout) if timeout else self.client
        extra = {}
        if "haiku" in model:
            # Haiku 5.5 accepts disabled thinking at low effort: fastest and cheapest for triage.
            extra["thinking"] = {"type": "disabled"}
        else:
            # Sonnet/Opus 5.5 reject disabled thinking; adaptive at low effort, with room for it.
            max_tokens = max(max_tokens, 4000)
        try:
            resp = client.messages.create(
                model=model,
                max_tokens=max_tokens,
                system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
                messages=[{"role": "user", "content": user}],
                output_config={"effort": "low", "format": {"type": "json_schema", "schema": schema}},
                **extra,
            )
        except self.anthropic.APITimeoutError:
            meta["error"] = "timeout"
        except self.anthropic.RateLimitError:
            meta["error"] = "rate_limit"
        except self.anthropic.APIStatusError as e:
            meta["error"] = f"api_{e.status_code}: {str(e.message)[:200]}"
        except self.anthropic.APIConnectionError:
            meta["error"] = "connection"
        meta["latency_ms"] = int((time.perf_counter() - t0) * 1000)
        if meta["error"]:
            return None, meta
        meta["cost_usd"] = cost_usd(model, resp.usage)
        meta["usage"] = {
            "input": resp.usage.input_tokens, "output": resp.usage.output_tokens,
            "cache_read": getattr(resp.usage, "cache_read_input_tokens", 0) or 0,
            "cache_write": getattr(resp.usage, "cache_creation_input_tokens", 0) or 0,
        }
        if resp.stop_reason not in ("end_turn", "stop_sequence"):
            meta["error"] = f"stop_reason={resp.stop_reason}"
            return None, meta
        text = next((b.text for b in resp.content if b.type == "text"), "")
        try:
            return json.loads(text), meta
        except json.JSONDecodeError:
            meta["error"] = "bad_json"
            return None, meta
