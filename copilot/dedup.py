"""Spam heuristics and duplicate detection (same player, same problem, within an hour)."""

import re
import threading
from collections import defaultdict, deque
from datetime import datetime, timedelta

from .config import DEDUP_SIMILARITY, DEDUP_WINDOW_MIN
from .textnorm import char_ngrams, jaccard, letters_count, tokens

AD_RE = re.compile(r"t\.me/|discord\.gg|vk\.com/(?!video)|промокод|заходи(те)? на (наш )?сервер|продам|куплю вирт|вирты за", re.I)
REPEAT_RE = re.compile(r"(.)\1{5,}")
VOWELS = set("аеёиоуыэюяaeiouyәөұүіoʻ")


def spam_check(text: str, kind: str, images: int = 0) -> tuple[bool, str]:
    t = (text or "").strip()
    n = letters_count(t)
    if n == 0 and images == 0:
        return True, "пустое обращение без текста"
    if AD_RE.search(t):
        return True, "реклама/ссылки на сторонние ресурсы"
    words = tokens(t)
    if n >= 6 and REPEAT_RE.search(t) and len(set(words)) <= 2:
        return True, "флуд повторяющимися символами"
    if n >= 8 and len(words) <= 2:
        letters = [ch for ch in t.lower() if ch.isalpha()]
        vow = sum(1 for ch in letters if ch in VOWELS) / max(1, len(letters))
        if vow < 0.12:
            return True, "бессмысленный набор символов"
    return False, ""


class DedupStore:
    """In-memory sliding window. Keys: reporter hash (preferred) or reported nick."""

    def __init__(self, window_min: int = DEDUP_WINDOW_MIN, threshold: float = DEDUP_SIMILARITY):
        self.window = timedelta(minutes=window_min)
        self.threshold = threshold
        self.events = defaultdict(deque)  # key -> deque[(time, grams, nick, category, ticket_id)]
        self.lock = threading.Lock()

    def check_and_add(self, *, reporter: str | None, nick: str | None, text: str, concept: str | None,
                      ts: datetime | None, ticket_id=None) -> tuple[bool, str | None, str]:
        if not reporter:
            return False, None, ""
        now = ts or datetime.now()
        grams = char_ngrams(text, 3)
        found, dup_of, why = False, None, ""
        with self.lock:
            q = self.events[reporter]
            while q and abs(now - q[0][0]) > self.window and q[0][0] < now:
                q.popleft()
            for (t0, g0, nick0, concept0, tid0) in q:
                if abs(now - t0) > self.window:
                    continue
                sim = jaccard(grams, g0)
                same_target = nick and nick0 and nick.lower() == nick0.lower()
                if sim >= self.threshold:
                    found, dup_of, why = True, tid0, f"повтор текста ({sim:.2f}) за {DEDUP_WINDOW_MIN} мин"
                    break
                if same_target and concept and concept == concept0:
                    found, dup_of, why = True, tid0, f"повторная жалоба на {nick} по той же теме"
                    break
                if not nick and concept and concept == concept0 and sim >= self.threshold * 0.6:
                    found, dup_of, why = True, tid0, "повтор той же проблемы"
                    break
            q.append((now, grams, nick, concept, ticket_id))
        return found, dup_of, why
