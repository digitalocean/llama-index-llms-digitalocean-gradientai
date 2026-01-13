---
skill: python-packaging-pyproject
category: packaging
tags: [python, pyproject, setuptools, build]
updated: 2026-01-13
---
# Python Packaging with pyproject.toml

## Overview
Guide to modern Python packaging using `pyproject.toml` for the LlamaIndex Gradient AI integration.

## Quick Reference

```bash
# Bump version: Edit `version` in `[project]` section of pyproject.toml
# Add dependency: Add to `dependencies` array in pyproject.toml
# Build package: python -m build
# Test install: pip install -e ".[dev]"
# Publish: twine upload dist/*
```

## Project Structure

### pyproject.toml Location
The `pyproject.toml` file should be at the root of the project:
```
llama-index-llms-gradient/
├── pyproject.toml
├── setup.py
├── README.md
├── llama_index/
│   └── llms/
│       └── digitalocean/
│           └── gradientai/
│               ├── __init__.py
│               └── base.py
└── tests/
    └── test_gradient_llm.py
```

## Build System Configuration

### Basic Build System
```toml
[build-system]
requires = ["setuptools>=61.0", "wheel"]
build-backend = "setuptools.build_meta"
```

## Project Metadata

### Core Project Information
```toml
[project]
name = "llama-index-llms-digitalocean-gradientai"
version = "0.1.1"
description = "LlamaIndex integration for DigitalOcean Gradient AI"
readme = "README.md"
requires-python = ">=3.8"
license = {text = "MIT"}
license-files = ["LICENSE"]
authors = [
    {name = "Narasimha Badrinath", email = "bnarasimha21@gmail.com"},
]
keywords = ["llamaindex", "gradient", "gradientai", "llm", "rag", "digitalocean"]
```

### Classifiers
```toml
classifiers = [
    "Development Status :: 3 - Alpha",
    "Intended Audience :: Developers",
    "Operating System :: OS Independent",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.8",
    "Programming Language :: Python :: 3.9",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
]
```

## Dependencies

### Runtime Dependencies
```toml
dependencies = [
    "llama-index-core>=0.10.0",
    "gradient>=3.8.0",
]
```

### Optional Dependencies (Dev)
```toml
[project.optional-dependencies]
dev = [
    "pytest>=7.0.0",
    "pytest-asyncio>=0.21.0",
    "pytest-cov>=4.0.0",
    "black>=23.0.0",
    "ruff>=0.1.0",
    "mypy>=1.0.0",
    "pre-commit>=3.0.0",
    "python-dotenv>=1.0.0",
]
```

Install dev dependencies:
```bash
pip install -e ".[dev]"
```

## Project URLs

```toml
[project.urls]
Homepage = "https://github.com/bnarasimha21/llamaindex-digitalocean-gradientai"
Documentation = "https://github.com/bnarasimha21/llamaindex-digitalocean-gradientai#readme"
Repository = "https://github.com/bnarasimha21/llamaindex-digitalocean-gradientai"
Issues = "https://github.com/bnarasimha21/llamaindex-digitalocean-gradientai/issues"
```

## Package Discovery

### Automatic Package Discovery
```toml
[tool.setuptools.packages.find]
where = ["."]
include = ["llama_index*"]
namespaces = true
```

This automatically finds packages matching `llama_index*` pattern.

### Manual Package Specification (Alternative)
```toml
[tool.setuptools]
packages = ["llama_index.llms.digitalocean.gradientai"]
```

## Code Quality Tools Configuration

### Black Configuration
```toml
[tool.black]
line-length = 100
target-version = ['py38', 'py39', 'py310', 'py311', 'py312']
```

### MyPy Configuration
```toml
[tool.mypy]
python_version = "3.8"
warn_return_any = true
warn_unused_configs = true
ignore_missing_imports = true
exclude = [
    "tests/",
    "build/",
    "dist/",
]
```

### Ruff Configuration
```toml
[tool.ruff]
line-length = 100
target-version = "py38"
select = [
    "E",   # pycodestyle errors
    "W",   # pycodestyle warnings
    "F",   # pyflakes
    "I",   # isort
    "B",   # flake8-bugbear
    "C4",  # flake8-comprehensions
    "UP",  # pyupgrade
]
ignore = [
    "E501",  # line too long (handled by black)
]

[tool.ruff.per-file-ignores]
"__init__.py" = ["F401"]  # Allow unused imports in __init__.py
```

## Testing Configuration

### Pytest Configuration
```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py"]
python_classes = ["Test*"]
python_functions = ["test_*"]
addopts = [
    "-v",
    "--strict-markers",
    "--tb=short",
    "--cov=llama_index.llms.digitalocean.gradientai",
    "--cov-report=term-missing",
    "--cov-report=html",
    "--cov-report=xml",
    "--asyncio-mode=auto",
]
markers = [
    "integration: marks tests as integration tests",
    "slow: marks tests as slow",
]
```

### Coverage Configuration
```toml
[tool.coverage.run]
source = ["llama_index"]
omit = [
    "*/tests/*",
    "*/test_*.py",
]

[tool.coverage.report]
exclude_lines = [
    "pragma: no cover",
    "def __repr__",
    "raise AssertionError",
    "raise NotImplementedError",
    "if __name__ == .__main__.:",
    "if TYPE_CHECKING:",
    "@abstractmethod",
]
```

## Building the Package

### Install in Development Mode
```bash
pip install -e .
```

### Build Distribution
```bash
# Build wheel and source distribution
python -m build

# Output in dist/:
# - llama_index_llms_digitalocean_gradientai-0.1.1-py3-none-any.whl
# - llama_index_llms_digitalocean_gradientai-0.1.1.tar.gz
```

### Install from Build
```bash
pip install dist/llama_index_llms_digitalocean_gradientai-0.1.1-py3-none-any.whl
```

## Version Management

### Version Bumping
Update version in `pyproject.toml`:
```toml
[project]
version = "0.1.2"  # Increment version
```

### Semantic Versioning
- **MAJOR.MINOR.PATCH** (e.g., 1.2.3)
- **MAJOR**: Breaking changes
- **MINOR**: New features, backward compatible
- **PATCH**: Bug fixes, backward compatible

## Publishing to PyPI

### Prerequisites
```bash
pip install build twine
```

### Build and Upload
```bash
# Build distributions
python -m build

# Upload to TestPyPI (for testing)
twine upload --repository testpypi dist/*

# Upload to PyPI (production)
twine upload dist/*
```

### PyPI Credentials
Store credentials in `~/.pypirc`:
```ini
[distutils]
index-servers =
    pypi
    testpypi

[pypi]
username = __token__
password = pypi-...

[testpypi]
username = __token__
password = pypi-...
```

## Additional Files

### MANIFEST.in
Include additional files in source distribution:
```
include README.md
include LICENSE
include CHANGELOG.md
recursive-include tests *.py
```

### setup.py (Legacy Support)
If needed for compatibility:
```python
from setuptools import setup

setup()
```

## Namespace Packages

### LlamaIndex Namespace Structure
```
llama_index/
├── __init__.py  # May be empty for namespace packages
└── llms/
    └── digitalocean/
        └── gradientai/
            ├── __init__.py
            └── base.py
```

### Namespace Package Configuration
```toml
[tool.setuptools.packages.find]
namespaces = true  # Important for namespace packages
```

## LlamaIndex Namespace Package Gotchas

### Critical Configuration
```toml
[tool.setuptools.packages.find]
where = ["."]
include = ["llama_index*"]
namespaces = true  # CRITICAL for namespace packages
```

**Why this matters**: LlamaIndex uses namespace packages to allow multiple integration packages to coexist. Without `namespaces = true`, your package won't be discoverable after installation.

### Directory Structure Requirements
- Each directory must have `__init__.py` (can be empty for namespace packages)
- Package path must match import path: `llama_index.llms.digitalocean.gradientai`
- Underscores in directory names, not hyphens

## Common Commands

```bash
# Install package in development mode
pip install -e .

# Install with dev dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Run with coverage
pytest --cov

# Format code
black .

# Lint code
ruff check .

# Type check
mypy llama_index

# Build package
python -m build

# Check build
twine check dist/*

# Upload to PyPI
twine upload dist/*
```

## Troubleshooting

### Package Not Found After Install
Most common issue: missing `namespaces = true` in `[tool.setuptools.packages.find]`

### Import Errors in Namespace Packages
```bash
# After install, if this fails:
from llama_index.llms.digitalocean.gradientai import GradientAI

# Check:
pip show llama-index-llms-digitalocean-gradientai  # Is it installed?
python -c "import llama_index; print(llama_index.__path__)"  # Namespace configured?
```

