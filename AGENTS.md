# Repository Instructions

- Execute the notebooks before committing changes to them.
- Preserve graph outputs in the committed `.ipynb` files.
- Do not create or commit notebook HTML exports in `html/`; CI generates those later.
- Never commit, print, or otherwise reveal the API key or other secrets from `.env`.
- Run `ruff format .` before committing.
- Run `ruff check .` and fix reported issues before committing.
- Do not run `ruff` on notebook files in `notebooks/*.py` or `notebooks/*.ipynb`; those paths are excluded in `pyproject.toml`.
