# Contributing to PromptDrifter

This document provides guidelines for contributing to PromptDrifter.

## Project Overview

PromptDrifter is a tool for testing and monitoring LLM prompt consistency across different models and providers. It helps catch "prompt drift" - changes in model outputs for the same prompt over time.

## How to Contribute

- Add support for new LLM providers/adapters
- Improve documentation
- Report bugs
- Fix bugs or implement new features
- Add tests
- Share your experience using PromptDrifter

## Contribution Process

1. Fork this repository
2. Create an [issue](https://github.com/Code-and-Sorts/PromptDrifter/issues)
3. Make your changes
4. Run the linter and tests
5. Submit a pull request
6. Address any feedback

## Development Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/promptdrifter.git
   cd promptdrifter
   ```

2. Install dependencies using uv:
   ```bash
   pip install uv
   uv pip install -e .
   uv pip install -e ".[dev]"
   ```

## Code Standards

We use:
- **Ruff**: For code formatting and linting
- **Pytest**: For testing
- **Type Annotations**: All functions must include type annotations
- **Pydantic**: For data modeling and validation
- **Async/Await**: For asynchronous programming

Commands:
```bash
make lint    # Run linter
make test-unit    # Run tests
```

## Schema Versioning

PromptDrifter uses a versioned schema system to ensure backward compatibility. For detailed information, see [Schema Versioning Guide](schema-versioning.md).

## Working with Adapters

When adding a new adapter:

1. Create a new adapter class that inherits from `Adapter` base class
2. Implement the required methods
3. Add the adapter type to the schema
4. Create unit tests for the adapter
5. Add documentation and update README adapter table

## Documentation Style

- Self-documenting code through clear naming
- Minimal comments, only when necessary
- README files for major components

## Contact

- Open an issue or discussion on GitHub
