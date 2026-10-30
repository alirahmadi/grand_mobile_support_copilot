# Support Copilot — Grand Mobile x Decentrathon 6.0 (кейс 03)
# Windows без make: python -m copilot.cli <команда> (см. README)
PY ?= python
IN ?= data/tickets_valid.jsonl
OUT ?= pred.jsonl
PORT ?= 8000
N ?= 3000

.PHONY: install check train train-final label predict predict-offline eval eval-offline judge serve test demo

install:
	$(PY) -m pip install -r requirements.txt

check:            ## проверить, что данные на месте и таксономия сопоставилась
	$(PY) -m copilot.cli check-data

train:            ## майнинг train: база знаний, наказания, модели вердикта/языка/категорий
	$(PY) -m copilot.cli train

train-final:      ## то же + валидация в базе знаний (перед финальным прогоном на скрытом тесте)
	$(PY) -m copilot.cli train --include-valid

label:            ## LLM-псевдоразметка категорий/приоритетов (нужен ANTHROPIC_API_KEY)
	$(PY) -m copilot.cli label --n $(N)

predict:          ## make predict IN=tickets.jsonl OUT=pred.jsonl
	$(PY) -m copilot.cli predict --in $(IN) --out $(OUT)

predict-offline:
	$(PY) -m copilot.cli predict --in $(IN) --out $(OUT) --offline

eval:             ## метрики на валидации + reports/REPORT.md
	$(PY) -m copilot.cli eval

eval-offline:
	$(PY) -m copilot.cli eval --offline

judge:            ## eval + LLM-судья 1-5 на 100 черновиках
	$(PY) -m copilot.cli eval --judge

serve:            ## HTTP: POST /triage, админка на http://localhost:$(PORT)/
	$(PY) -m copilot.cli serve --port $(PORT)

test:
	$(PY) -m pytest -q tests

demo:             ## офлайн-прогон на демо-фикстуре (без данных и без ключа)
	$(PY) -m copilot.cli predict --in samples/tickets_demo.jsonl --out reports/demo_pred.jsonl --offline
