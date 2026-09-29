.PHONY: help install seed dev api web backtest demo-check test lint clean reseed

VENV      := .venv
PY        := $(VENV)/bin/python
PIP       := $(VENV)/bin/pip
UVICORN   := $(VENV)/bin/uvicorn
WEB_DIR   := apps/web

help:
	@echo "VAYU — Verifiable Airshed Intelligence & Enforcement"
	@echo ""
	@echo "  make install     create venv + install python & node deps"
	@echo "  make seed        fetch/refresh bundled data -> data/samples -> DuckDB"
	@echo "  make reseed      force re-download of all sample data"
	@echo "  make dev         run API (:8000) + web (:3000) together"
	@echo "  make api         run FastAPI only"
	@echo "  make web         run Next.js only"
	@echo "  make test        pytest for vayu_core + API smoke tests"
	@echo "  make backtest    regenerate docs/evaluation.md        (Phase 2)"
	@echo "  make calibrate   re-derive attribution scales + IITM cross-check"
	@echo "  make demo-check  playwright golden-flow walk          (Phase 6)"

$(VENV):
	python3 -m venv $(VENV)
	$(PIP) install --quiet --upgrade pip

install: $(VENV)
	$(PIP) install --quiet -r requirements.txt
	@test -d $(WEB_DIR)/node_modules || (cd $(WEB_DIR) && npm install --silent)
	@echo "✓ dependencies installed"

# Idempotent: reuses anything already in data/samples/. Network optional once seeded.
seed: install
	$(PY) -m services.pipeline.seed
	$(PY) -m services.pipeline.score

reseed: install
	$(PY) -m services.pipeline.seed --force
	$(PY) -m services.pipeline.score --retrain

api: install
	$(UVICORN) services.api.main:app --reload --port 8000

web:
	cd $(WEB_DIR) && npm run dev

dev: install
	@echo "→ API http://localhost:8000/docs   WEB http://localhost:3000"
	@trap 'kill 0' EXIT INT TERM; \
	$(UVICORN) services.api.main:app --reload --port 8000 & \
	(cd $(WEB_DIR) && npm run dev) & \
	wait

test: install
	$(VENV)/bin/pytest -q tests

# `lint` was declared .PHONY but had no recipe, so `make lint` silently did
# nothing. The web side is checked with tsc rather than eslint: Next 16 removed
# `next lint`, and this repo has no eslint config, so `npm run lint` errors out.
lint: install
	$(VENV)/bin/ruff check vayu_core services tests
	cd $(WEB_DIR) && npx tsc --noEmit

backtest: install
	$(PY) -m vayu_core.forecast.backtest

# Scheduled jobs: refresh = hourly ingest+score; retrain = weekly retrain behind promotion gate.
refresh: install
	$(PY) -m services.jobs.refresh

retrain-gated: install
	$(PY) -m services.jobs.retrain

calibrate: install
	$(PY) -m vayu_core.attribution.calibrate

demo-check:
	cd $(WEB_DIR) && npx playwright test

clean:
	rm -rf data/vayu.duckdb data/cache/* $(WEB_DIR)/.next
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
