# Common developer tasks (Poetry-based).
# Usage: `make help`

POETRY ?= poetry
PYTEST_OPTS ?= -q --tb=short
RUFF_PATHS = pattern_recognition tests

.PHONY: help test test-v test-speller format format-check lint check install

help: ## Show available targets
	@awk 'BEGIN {FS = ":.*##"; printf "\nTargets:\n"} \
		/^[a-zA-Z0-9_-]+:.*?##/ { printf "  %-14s %s\n", $$1, $$2 }' $(MAKEFILE_LIST)
	@printf "\nExamples:\n  make test\n  make format\n  make check\n\n"

install: ## Install package + dev deps via Poetry
	$(POETRY) install

test: ## Run full test suite (quiet)
	$(POETRY) run pytest tests/ $(PYTEST_OPTS)

test-v: ## Run full test suite (verbose)
	$(POETRY) run pytest tests/ -v --tb=short

test-speller: ## Run speller tests only
	$(POETRY) run pytest tests/speller/ $(PYTEST_OPTS)

format: ## Format + apply safe Ruff lint autofixes (imports, etc.)
	$(POETRY) run ruff check --fix $(RUFF_PATHS)
	$(POETRY) run ruff format $(RUFF_PATHS)

format-check: ## Check formatting without writing
	$(POETRY) run ruff format --check $(RUFF_PATHS)

lint: ## Lint package + tests with Ruff
	$(POETRY) run ruff check $(RUFF_PATHS)

check: format-check lint test ## Format check + lint + tests
