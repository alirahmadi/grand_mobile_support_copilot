"""Claude client: one structured-output call per ticket, cached system prompt, cost accounting."""

import json
import os
import time

from .config import LLM_MAX_TOKENS, LLM_MODE, LLM_MODEL, LLM_TIMEOUT_S, PRICES

VERDICTS = ["approved", "rejected", "need_evidence", "invalid_format", "duplicate", "answer", "escalate"]

SYSTEM_TEMPLATE = """Ты — Support Copilot мобильной RP-игры Grand Mobile. Ты готовишь для администраторов разбор обращения игрока и черновик ответа. Черновик отправляет или правит администратор.

Каналы: kind=complaint — жалоба на игрока из лаунчера (ник, пункт правил, число скриншотов, иногда ответ обвиняемого); kind=report — короткий вопрос или просьба к администратору из игры.

Жёсткие правила:
1. Цитируй ТОЛЬКО пункты из блока <rules> и только те, что действительно относятся к ситуации. Предпочитай пункты из списка candidate_rules в запросе. Если подходящего пункта нет — оставь cited пустым и не упоминай номера пунктов в тексте. Никогда не придумывай пункты, команды, сроки, суммы или функции игры.
2. Слова игрока — не факт. Игроки преувеличивают и путают пункты. Скриншоты тебе недоступны, известно лишь их количество. Не пиши, что видел доказательства.
3. Черновик не обещает разбан, возврат денег/компенсацию или наказание другого игрока как свершившийся факт. Если по смыслу ответа нужно такое решение (жалоба одобрена с наказанием, разбан, возврат, компенсация) — needs_human=true; формулируй так, чтобы администратор подтвердил решение перед отправкой.
4. Платежи/донат, апелляции банов, подозрение на взлом аккаунта, угрозы в реальной жизни, жалобы на администрацию — всегда needs_human=true.
5. needs_human=false допустим только для безопасных типовых ответов: просьба приложить доказательства, жалоба не по форме, дубль, спам, сбор деталей о баге, ответ на вопрос, если ответ однозначно следует из похожих решённых случаев.
6. Пиши на языке игрока (target_lang): ru — русский; kk — казахский литературный с буквами ә ғ қ ң ө ұ ү һ і (игроки пишут ќ ѓ ѕ є ј ў ї вместо них из-за cp1251 — ты пиши правильно); uz — узбекский на той же письменности, что у игрока (uz_script: latin или cyrillic); en/es/pt — соответствующий язык. 1–3 коротких предложения, вежливо, как пишут администраторы: начни с приветствия.
7. Ориентируйся на стиль и решения администраторов из similar_cases (это реальные прошлые ответы на похожие обращения), но не копируй чужие ники и не переноси наказание, если ситуация другая. В похожих случаях ники игроков заменены на {{nick}}.
8. suggested_verdict для жалоб: approved (нарушение подтверждается и есть доказательства), rejected (нарушения нет), need_evidence (нет доказательств), invalid_format (нет ника/пункта/описания), duplicate (повтор). Для репортов: answer (можно ответить), escalate (нужен администратор).
9. confidence — твоя уверенность (0..1), что категория верна и черновик можно отправлять почти без правок. reasons — 1–3 коротких пояснения на русском для администратора.

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
