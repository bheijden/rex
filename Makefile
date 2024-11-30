#PACKAGE_NAME=rex
#
#SHELL=/bin/bash
#LINT_PATHS=${PACKAGE_NAME}/
#
#coverage-badge:
#	# Generate coverage badge
#	poetry run python ./scripts/generate_coverage_badge.py

#check-codestyle:
#	# Reformat using black
#	poetry run black --check -l 127 ${LINT_PATHS}
#
#codestyle:
#	# Reformat using black
#	poetry run black -l 127 ${LINT_PATHS}
#
#lint:
#	# stop the build if there are Python syntax errors or undefined names
#	# see https://lintlyci.github.io/Flake8Rules/
#	poetry run flake8 ${LINT_PATHS} --count --select=E9,F63,F7,F82 --show-source --statistics
#	# exit-zero treats all errors as warnings.
#	poetry run flake8 ${LINT_PATHS} --count --exit-zero --statistics

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

.PHONY: check-codestyle
