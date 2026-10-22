OUTPUT_DIR ?= outputs
RELEASE_NAME ?= latest
N_RUNS ?= 10
DATA_DIR ?=
DATA_ARG = $(if $(DATA_DIR),--data-dir $(DATA_DIR),)

.PHONY: continue help install install-colab install-prod install-dev test run run-quick run-colab run-prod report-only clean clean-builds clean-all

help:
	@echo "Open Bandit A/B Testing Makefile"
	@echo ""
	@echo "Usage:"
	@echo "  make install          Install pipeline dependencies"
	@echo "  make install-colab    Install Colab-specific dependencies"
	@echo "  make install-prod     Install pinned production dependencies"
	@echo "  make install-dev      Install dependencies plus pytest"
	@echo "  make test             Run tests"
	@echo "  make run              Run pipeline (auto-detect mode) and promote to releases/latest"
	@echo "  make run-quick        Fast run: no tuning, no time split, 2 bootstrap runs"
	@echo "  make run-colab        Run pipeline in Colab mode"
	@echo "  make run-prod         Run pipeline in production mode"
	@echo "  make report-only      Rebuild the report from RESUME=<build dir>"
	@echo "  make continue         Continue the latest interrupted build in OUTPUT_DIR"
	@echo "  make clean            Clean build and release files"
	@echo "  make clean-all        Clean all outputs"
	@echo ""
	@echo "Variables: DATA_DIR=<full dataset folder>, OUTPUT_DIR=$(OUTPUT_DIR), N_RUNS=$(N_RUNS), RESUME=<build dir>"

install:
	pip install -r requirements/colab.txt

install-colab:
	pip install -r requirements/colab.txt

install-prod:
	pip install -r requirements/prod.txt

install-dev:
	pip install -r requirements/dev.txt

test:
	pytest tests/ -v --tb=short

run:
	python cli.py --output-dir $(OUTPUT_DIR) $(DATA_ARG) --n-runs $(N_RUNS) --continue-build auto --promote-release --release-name $(RELEASE_NAME)

run-quick:
	python cli.py --output-dir $(OUTPUT_DIR) $(DATA_ARG) --model-trials 0 --no-time-split --n-runs 2 --n-sim 100000

run-colab:
	python cli.py --mode colab --output-dir $(OUTPUT_DIR) $(DATA_ARG)

run-prod:
	python cli.py --mode production --output-dir $(OUTPUT_DIR) $(DATA_ARG) --n-runs 30 --promote-release --release-name $(RELEASE_NAME)

continue:
	python cli.py --output-dir $(OUTPUT_DIR) --continue-build latest

report-only:
	python cli.py --output-dir $(OUTPUT_DIR) $(DATA_ARG) --only-stage 6 --resume-build $(RESUME)

clean: clean-builds

clean-builds:
	rm -rf $(OUTPUT_DIR)/builds/
	rm -rf $(OUTPUT_DIR)/releases/
	rm -rf __pycache__/ src/__pycache__/ src/obp_lite/__pycache__/ tests/__pycache__/ .pytest_cache/
	rm -f *.log

clean-all: clean-builds
	rmdir $(OUTPUT_DIR) 2>/dev/null || true
