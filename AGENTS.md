# AGENTS.md - Development Guidelines

## Build/Test Commands
- **Install deps**: `uv sync --group dev`
- **Run all tests**: `pytest`
- **Run single test**: `pytest tests/test_vit.py::test_vision_transformer_inference`
- **Lint**: `ruff check --fix`
- **Format**: `ruff format`
- **Type check**: No dedicated command (use ruff)

## Code Style
- **Line length**: 200 characters
- **Imports**: Standard library first, then third-party (jax, flax, jaxtyping), then local (jimm.*)
- **Quotes**: Double quotes for strings
- **Indentation**: 4 spaces
- **Type hints**: Required for all function parameters and returns (use jaxtyping for arrays)
- **Naming**: snake_case for functions/variables, PascalCase for classes
- **Docstrings**: Google style for classes and public methods

## Error Handling
- Use descriptive error messages
- Prefer ValueError for input validation
- Include context in exception messages