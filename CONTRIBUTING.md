# Contributing to llama-index-llms-digitalocean-gradientai

Thank you for your interest in contributing! This document provides guidelines and instructions for contributing.

## Development Setup

1. **Fork and clone the repository:**
   ```bash
   git clone https://github.com/your-username/llama-index-llms-digitalocean-gradientai.git
   cd llama-index-llms-digitalocean-gradientai
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install development dependencies:**
   ```bash
   pip install -e ".[dev]"
   ```

4. **Install pre-commit hooks:**
   ```bash
   pre-commit install
   ```

## Development Workflow

1. **Create a branch:**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes:**
   - Write code following the existing style
   - Add tests for new functionality
   - Update documentation as needed

3. **Run tests:**
   ```bash
   pytest tests/
   ```

4. **Run linting:**
   ```bash
   black llama_index/ tests/
   ruff check llama_index/ tests/
   mypy llama_index/
   ```

5. **Commit your changes:**
   ```bash
   git add .
   git commit -m "Description of your changes"
   ```
   Pre-commit hooks will run automatically.

6. **Push and create a pull request:**
   ```bash
   git push origin feature/your-feature-name
   ```

## Code Style

- Follow [PEP 8](https://pep8.org/)
- Use `black` for formatting (line length: 100)
- Use `ruff` for linting
- Type hints are encouraged
- Docstrings should follow Google style

## Testing

- Write tests for all new functionality
- Ensure all tests pass before submitting a PR
- For integration tests, set `MODEL_ACCESS_KEY` and `GRADIENT_WORKSPACE_ID` environment variables
- Tests should be deterministic and not depend on external state

## Pull Request Process

1. Update CHANGELOG.md with your changes
2. Ensure all CI checks pass
3. Request review from maintainers
4. Address any feedback
5. Once approved, maintainers will merge

## Reporting Issues

When reporting issues, please include:
- Python version
- Package version
- Steps to reproduce
- Expected vs actual behavior
- Error messages or logs

## Questions?

Feel free to open an issue for questions or discussions!


