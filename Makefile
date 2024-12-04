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
	uv run mkdocs build
	# twice, see https://github.com/patrick-kidger/pytkdocs_tweaks
	uv run mkdocs build

run_tests:
	# Run tests
	uv run pytest tests \
			--cov=rex \
			--cov-report=html \
			--cov-report=xml \
			--cov-report=term \
			--cov-config=pyproject.toml \
			-v --color=yes

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
