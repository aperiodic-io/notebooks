# Repository Instructions

- Do not create or commit notebook HTML exports in `html/`; CI generates those later.
- Never commit, print, or otherwise reveal the API key or other secrets from `.env`.
- Use `uv pip` for dependency installation and `uv` to execute notebooks/scripts.
- Load environment variables from `.env` if that file is present, but do not use `load_dotenv` in notebook code.
- Run `ruff format .` before committing.
- Run `ruff check .` and fix reported issues before committing.
- Do not run `ruff` on notebook files in `notebooks/*.py` or `notebooks/*.ipynb`; those paths are excluded in `pyproject.toml`.
