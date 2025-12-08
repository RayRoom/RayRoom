# Build and Publish Instructions

## 1. Build the Package

First, ensure you have `build` installed and run the build command:

```bash
./venv/bin/pip install build
./venv/bin/python -m build
```

This creates the distribution files in the `dist/` folder:
- `rayroom-0.1.0.tar.gz` (Source archive)
- `rayroom-0.1.0-py3-none-any.whl` (Wheel file)

## 2. Check the Artifacts

It's good practice to check the created artifacts with `twine` before uploading:

```bash
./venv/bin/pip install twine
./venv/bin/python -m twine check dist/*
```

## 3. Upload to PyPI

To upload, you will need a PyPI account. 

**Using an API Token (Recommended):**
1. Go to [PyPI Account Settings](https://pypi.org/manage/account/).
2. Scroll down to "API tokens" and add a new token (scope: "Entire account" for new projects).
3. Copy the token (starts with `pypi-`).

Run the upload command:
```bash
./venv/bin/python -m twine upload dist/*
```

When prompted:
- **Username**: `__token__`
- **Password**: Paste your API token (it will be hidden).

## 4. Verify Installation

After uploading, wait a minute and try installing it in a fresh environment:

```bash
pip install rayroom
```

## 5. Build the Documentation

To build the HTML documentation, navigate to the `docs/` directory and run:

```bash
cd docs/
make html
```

The generated documentation can be found in `docs/_build/html/`.
