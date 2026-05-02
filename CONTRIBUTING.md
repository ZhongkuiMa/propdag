# Contributing to PropDAG

Shared conventions (imports, type hints, docstrings, line length) are in the
[root CONTRIBUTING.md](../CONTRIBUTING.md). This file covers propdag-specific workflow only.

## Setup

```bash
cd propdag
pip install -e ".[dev]"
pytest tests/ -v
```

## Checks

```bash
pre-commit run --all-files
```

## Workflow

1. Create branch from `main`
2. Make changes
3. Run checks (above)
4. Commit and push
5. Open PR

## Adding a New Node Type

1. Choose template: `template/` (forward/backward modes) or `template2/` (reversed-graph backward only)
2. Create a subpackage under `src/propdag/` (see `toy/` or `toy2/` as examples)
3. Subclass `TNode`/`T2Node`, implement abstract methods: `forward()`, `build_rlx()`, `init_symbnd()`, `fwdprop_symbnd()`, `cal_and_update_cur_node_bnd()`
4. Subclass `TCache`/`T2Cache` with your storage fields
5. Subclass `TArgument`/`T2Argument` as a frozen dataclass
6. Export from your subpackage `__init__.py` and from `src/propdag/__init__.py`
7. Add tests in `tests/` covering topology, sorting, cache, and error handling
8. Verify: `pytest tests/ -v`

## Constraints

| Rule | Details |
|------|---------|
| No runtime dependencies | `dependencies = []` in pyproject.toml |
| Absolute imports only | `from propdag.template._node import TNode` (no relative) |
| `__docformat__` + `__all__` | Required in every module |
| Frozen dataclasses for arguments | `@dataclass(frozen=True, slots=True)` |
| McCabe complexity <= 10 | Enforced by ruff C90 |
| Single input/output constraint | `T2Model` requires exactly one input and one output node |
