# Contributing to PropDAG

We welcome contributions! Please follow these guidelines to ensure smooth collaboration.

## Development Setup

### Setting Up Development Environment

```bash
# Clone your fork
git clone https://github.com/YOUR_USERNAME/propdag.git
cd propdag

# Install in editable mode with dev dependencies
pip install -e ".[dev]"

# Verify setup
pytest tests/ -v
ruff check src/propdag tests
```

## Development Workflow

### Branch Naming Conventions

Use descriptive branch names with prefixes:
- `feature/` - New features (e.g., `feature/add-multivariate-bounds`)
- `fix/` - Bug fixes (e.g., `fix/cache-clearing-issue`)
- `refactor/` - Code refactoring (e.g., `refactor/simplify-sorting`)
- `docs/` - Documentation updates (e.g., `docs/improve-api-examples`)
- `test/` - Test improvements (e.g., `test/add-edge-cases`)

### Commit Message Format

Write clear, concise commit messages:

```
<type>: <short summary in present tense>

<optional detailed description>

<optional footer with issue references>
```

**Types:**
- `feat:` - New feature
- `fix:` - Bug fix
- `refactor:` - Code refactoring (no behavior change)
- `test:` - Add or update tests
- `docs:` - Documentation changes
- `style:` - Code style/formatting (ruff, whitespace)
- `perf:` - Performance improvements
- `chore:` - Maintenance tasks (dependencies, config)

**Examples:**
```
feat: Add support for multi-input DAG nodes

Implements handling for nodes with multiple input branches,
enabling more complex graph topologies like inception modules.

Fixes #42
```

```
fix: Correct cache clearing in backward propagation

The backward cache wasn't being cleared for leaf nodes,
causing memory accumulation in deep networks.
```

### Push Workflow

```bash
# Create feature branch
git checkout -b feature/my-new-feature

# Make changes and commit
git add <files>
git commit -m "feat: Add my new feature"

# Run pre-push checks (recommended)
pytest tests/ -v
ruff check src/propdag tests
python -m mypy

# Push to your fork
git push origin feature/my-new-feature

# Create PR on GitHub
```

## Code Quality Standards

### Linting

```bash
ruff check src/propdag tests
```

### Formatting

```bash
ruff format src/propdag tests
```

### Type Checking

```bash
python -m mypy
```

### Code Style

PropDAG follows strict code quality standards:

- **Formatter**: `ruff format` (100 char line length)
- **Linter**: `ruff check` (comprehensive ruleset)
- **Type checker**: `mypy`
- **Docstrings**: PEP 257 style with type hints

## Pull Request Guidelines

1. **Before creating a PR:**
   ```bash
   # Run all tests
   pytest tests/ -v

   # Run linting
   ruff check src/propdag tests
   ruff format src/propdag tests

   # Run type checking
   python -m mypy
   ```

2. **Create PR with:**
   - **Clear title**: Follow commit message format
   - **Description**: Explain what changes and why
   - **Tests**: Add tests for new features/fixes
   - **Documentation**: Update README/docstrings if needed
   - **Changelog**: Note breaking changes if any

3. **PR template:**
   ```markdown
   ## Summary
   Brief description of changes

   ## Type of Change
   - [ ] Bug fix (non-breaking change which fixes an issue)
   - [ ] New feature (non-breaking change which adds functionality)
   - [ ] Breaking change (fix or feature that would cause existing functionality to not work as expected)
   - [ ] Documentation update

   ## Testing
   - [ ] All tests pass (`pytest tests/ -v`)
   - [ ] Linting passes (`ruff check src/propdag tests`)
   - [ ] Type checking passes (`mypy`)
   - [ ] Added tests for new features/fixes

   ## Checklist
   - [ ] Code follows project style guidelines
   - [ ] Self-reviewed my own code
   - [ ] Commented complex logic
   - [ ] Updated documentation
   - [ ] No new warnings introduced
   ```

4. **Review process:**
   - Maintainers will review within 48-72 hours
   - Address feedback by pushing to your PR branch
   - Once approved, maintainers will merge

## Testing

Run the test suite to verify functionality:

```bash
# Run all tests
pytest tests/ -v

# Run specific test category
pytest tests/test_dag_topologies.py -v

# Run with coverage
pytest tests/ --cov=src/propdag --cov-report=term-missing -v
```

## CI/CD

### Running CI Locally

Before pushing, run the same checks as GitHub Actions:

```bash
# Full CI simulation
pytest tests/ --cov=src/propdag --cov-report=term-missing --cov-report=xml -v
ruff check src/propdag tests
ruff format --check src/propdag tests
python -m mypy
```

### Pre-commit Hooks (Optional)

Install pre-commit hooks to automatically check code before commits:

```bash
pip install pre-commit
pre-commit install

# Hooks will run automatically on git commit
# Or run manually:
pre-commit run --all-files
```

## Documentation

- All public classes/functions must have docstrings
- Use reStructuredText format (Sphinx-compatible)
- Include type hints for all parameters and return values

## Release Process

Releases are managed by maintainers:
1. Version bump in `pyproject.toml`
2. Update `__version__` in `src/propdag/__init__.py`
3. Create annotated git tag: `git tag -a v2026.1.0 -m "Release v2026.1.0"`
4. Push tag: `git push origin v2026.1.0`
5. GitHub Actions will run tests and create release

## Getting Help

- **Questions**: Open a GitHub Discussion
- **Bug reports**: Open an Issue with reproducible example
- **Feature requests**: Open an Issue with use case description

Thank you for contributing to PropDAG!
