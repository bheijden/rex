lint_check:
	# Lint the code
	uv run ruff check --fix

format_check:
	# Reformat using black in check mode
	uv run ruff format --check

format:
	# Reformat using black
	uv run ruff format

# Make sure you install the docs requirements first
# In the project directory (../rex) run `uv pip install -r docs/requirements.txt`
build_docs:
	# Build the documentation
	# JUPYTER_PLATFORM_DIRS=1 uv run mkdocs build --strict  # https://github.com/danielfrg/mkdocs-jupyter/issues/154
	uv run mkdocs build --strict # Add -v for verbose output

serve_docs:
	# Serve the documentation # Note! Does not run twice..., so not the same as build_docs.
	#JUPYTER_PLATFORM_DIRS=1 uv run mkdocs serve  # https://github.com/danielfrg/mkdocs-jupyter/issues/154
	uv run mkdocs serve

run_tests:
	# Run tests
	uv run pytest tests \
			--cov=rex \
			--cov-report=html \
			--cov-report=xml \
			--cov-report=term \
			--cov-config=pyproject.toml \
			-v --color=yes

run_integration_tests:
	# Run tests
	uv run pytest tests/integration

run_unit_tests:
	# Run tests
	uv run pytest tests \
			--ignore tests/integration \
			--cov=rex \
			--cov-report=html \
			--cov-report=xml \
			--cov-report=term \
			--cov-config=pyproject.toml \
			-v --color=yes

TEST_FILE ?= tests/unit/test_jax_utils.py
run_test:
	# Run specific test
	uv run pytest $(TEST_FILE) \
			--ignore tests/integration \
			--cov=rex \
			--cov-report=html \
			--cov-report=xml \
			--cov-report=term \
			--cov-config=pyproject.toml \
			-v --color=yes

.PHONY: check-codestyle
