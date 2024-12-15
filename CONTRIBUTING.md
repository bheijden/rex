# Contributing

Contributions (pull requests) are very welcome! Here's how to get started.

---
**Pre-requisites**

We use [uv](https://docs.astral.sh/uv) to manage our development environment.
Follow the installation instructions from [here](https://docs.astral.sh/uv/getting-started/installation/#configuring-installation), or install it globally with:
```bash
pip install uv
```

**Getting started**

First fork the library on GitHub and clone your fork:

```bash
git clone https://github.com/your-username-here/rex.git
````

Then navigate to the directory, and create a virtual environment:

```bash
cd rex
uv venv --python 3.9
source .venv/bin/activate
```

Then, install the library including all extras and the development dependencies:
```bash
uv sync --all-extras --dev --group tests
```
or, with GPU support:
```bash
uv sync --all-extras --dev --group tests --group gpu
```

Then install the pre-commit hooks to apply code formatting and linting using [ruff](https://docs.astral.sh/ruff):
```bash
pre-commit install  # Make sure .venv is activated, or use `uv run pre-commit install`
```

---

**If you're making changes to the code:**

Now make your changes. Make sure to include additional tests if necessary.

Next verify the tests all pass:

```bash
pytest tests  # Make sure .venv is activated, or use `uv run pytest tests`
```

Then push your changes back to your fork of the repository:

```bash
git push
```

Finally, open a pull request on GitHub!

---

**If you're making changes to the documentation:**

Make your changes. You can then build the documentation from the root directory (where `mkdocs.yml` is located) by doing

```bash
uv pip install -r docs/requirements.txt  
mkdocs serve  # Make sure .venv is activated, or use `uv run mkdocs serve`
```

[//]: # (Then doing `Control-C`, and running:)

[//]: # (```bash)

[//]: # (mkdocs serve  # Make sure .venv is activated, or use `uv run mkdocs serve`)

[//]: # (```)

[//]: # (&#40;So you run `mkdocs serve` twice.&#41;)

You can then see your local copy of the documentation by navigating to `localhost:8000` in a web browser.
