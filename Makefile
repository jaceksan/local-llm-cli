# Folders / paths
VENV_DIR = .venv

# Tools definition with versions pinned
TOOL_RUFF = uv run ruff
TOOL_PYRIGHT = uv run pyright

# Ensure uv always uses the same Python version
export UV_PROJECT_ENVIRONMENT=$(VENV_DIR)

# Prepare env for the development
.PHONY: dev
dev:
	uv venv
	uv sync --group dev --frozen


# Update the lockfile after new or updated deps
.PHONY: update
update:
	uv lock


# Run the demo
.PHONY: run
run:
	uv run python run.py

#
# Testing
#

# Run unit tests (default; excludes tests/integration)
.PHONY: test
test:
	@$(MAKE) test-unit

# Run unit tests only (exclude slow integration suite under tests/integration)
.PHONY: test-unit
test-unit:
	uv run python -m pytest tests --ignore=tests/integration

# Run integration tests (explicit opt-in; forces offline mode to prevent downloads)
.PHONY: test-integration
test-integration:
	TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 RUN_LLM_INTEGRATION_TESTS=1 uv run python -m pytest tests/integration


### MISC / HELPERS

# Clean everything, including the virtual environment
.PHONY: ultra-clean
ultra-clean:
	[ -d $(VENV_DIR) ] && rm -rf $(VENV_DIR) || true
	[ -d .pytest_cache ] && rm -rf .pytest_cache || true
	[ -d .ruff_cache ] && rm -rf .ruff_cache || true
	[ -d __pycache__ ] && rm -rf __pycache__ || true
	[ -d build ] && rm -rf build || true
	[ -d dist ] && rm -rf dist || true

# Validate linting
.PHONY: lint
lint:
	$(TOOL_RUFF) check .

# Fix linting
.PHONY: lint-fix
lint-fix:
	$(TOOL_RUFF) check . --fix

# Validate formatting
.PHONY: format
format:
	$(TOOL_RUFF) format --check .

# Fix formatting
.PHONY: format-fix
format-fix:
	$(TOOL_RUFF) format .
	$(TOOL_RUFF) check . --fix --fixable I

# Validate typing
.PHONY: types
types:
	$(TOOL_PYRIGHT) -p .

# Run all validation checks with auto-fix
.PHONY: validate
validate: format-fix lint-fix types
