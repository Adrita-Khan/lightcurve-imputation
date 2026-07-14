# ============================================================
# Makefile for lightcurve-imputation thesis repository
# ============================================================
# Usage:
#   make install      — Install Python dependencies
#   make reproduce    — Run complete experiment pipeline
#   make test         — Run unit tests with coverage
#   make lint         — Run code style checks
#   make format       — Auto-format code with black + isort
#   make clean        — Remove generated outputs
#   make docker-build — Build Docker image
#   make docker-run   — Run full pipeline in Docker
#   make docs         — Build HTML documentation

PYTHON      := python3
PIP         := pip3
CONFIG      := configs/experiment.yml
FAST_CONFIG := configs/fast_test.yml
DOCKER_TAG  := lightcurve-imputation:latest

.PHONY: all install reproduce fast-test test lint format clean \
        docker-build docker-run docs help

all: reproduce

# ─── Installation ────────────────────────────────────────────────────────────

install:
	@echo "Installing dependencies …"
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	$(PIP) install -e .
	@echo "Installation complete."

install-dev: install
	$(PIP) install pytest pytest-cov black flake8 isort mypy jupyter

# ─── Experiments ─────────────────────────────────────────────────────────────

reproduce:
	@echo "Running complete experiment pipeline (this may take a while) …"
	$(PYTHON) run_all.py --config $(CONFIG)

fast-test:
	@echo "Running fast-test pipeline (reduced seeds/epochs) …"
	$(PYTHON) run_all.py --config $(FAST_CONFIG)

dry-run:
	$(PYTHON) run_all.py --config $(CONFIG) --dry-run

# ─── Testing ─────────────────────────────────────────────────────────────────

test:
	@echo "Running unit tests …"
	$(PYTHON) -m pytest tests/ -v --tb=short --cov=src --cov-report=term-missing

test-fast:
	$(PYTHON) -m pytest tests/ -v --tb=short -x -q

# ─── Code quality ────────────────────────────────────────────────────────────

lint:
	@echo "Linting …"
	$(PYTHON) -m flake8 src/ tests/ run_all.py --max-line-length=100 \
	    --extend-ignore=E203,W503
	@echo "Lint OK."

format:
	@echo "Formatting with black + isort …"
	$(PYTHON) -m black src/ tests/ run_all.py --line-length=100
	$(PYTHON) -m isort src/ tests/ run_all.py --profile=black --line-length=100

type-check:
	$(PYTHON) -m mypy src/ --ignore-missing-imports

# ─── Clean ───────────────────────────────────────────────────────────────────

clean:
	@echo "Removing generated outputs …"
	rm -rf data/results/*.csv
	rm -rf figures/*.pdf figures/*.png figures/*.svg
	rm -rf tables/*.tex tables/*.csv
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
	find . -name ".coverage" -delete 2>/dev/null || true
	rm -rf htmlcov/ .mypy_cache/ dist/ build/ *.egg-info/
	@echo "Clean done."

# ─── Docker ──────────────────────────────────────────────────────────────────

docker-build:
	docker build -t $(DOCKER_TAG) .

docker-run:
	docker run --rm \
	    -v "$(PWD)/data:/app/data" \
	    -v "$(PWD)/figures:/app/figures" \
	    -v "$(PWD)/tables:/app/tables" \
	    $(DOCKER_TAG)

# ─── Documentation ───────────────────────────────────────────────────────────

docs:
	@echo "Documentation is in docs/ (Markdown). Open docs/index.md to start."

# ─── Help ────────────────────────────────────────────────────────────────────

help:
	@echo ""
	@echo "lightcurve-imputation Makefile"
	@echo "────────────────────────────────────────────"
	@echo "  make install      Install all dependencies"
	@echo "  make reproduce    Run full experiment pipeline"
	@echo "  make fast-test    Run pipeline with reduced settings"
	@echo "  make dry-run      Validate config without running"
	@echo "  make test         Unit tests with coverage"
	@echo "  make lint         Flake8 lint check"
	@echo "  make format       Auto-format (black + isort)"
	@echo "  make clean        Remove generated outputs"
	@echo "  make docker-build Build Docker image"
	@echo "  make docker-run   Run pipeline in Docker"
	@echo "  make docs         Open documentation"
	@echo ""
