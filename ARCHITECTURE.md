# PropDAG Architecture

DAG execution engine for bound propagation in neural network verification. Pure Python, zero dependencies.

## Package Tree

```
src/propdag/
├── template/          Forward-graph ABCs (modify when: adding forward propagation features)
│   ├── _node.py       TNode ABC — forward/backward methods
│   ├── _model.py      TModel ABC — topo sort + run loop + cache clearing
│   ├── _cache.py      TCache ABC — shared bounds/relaxation storage
│   ├── _arguments.py  TArgument — frozen config dataclass
│   └── _sort.py       Topological sort: BFS, DFS, backward
├── template2/         Reversed-graph ABCs (modify when: adding backward-only propagation)
│   ├── _node.py       T2Node ABC — reversed-graph forward() = backward propagation
│   ├── _model.py      T2Model ABC — reverse_dag + run loop
│   ├── _cache.py      T2Cache ABC — simplified cache
│   ├── _arguments.py  T2Argument — frozen config (no prop_mode)
│   └── _sort.py       Topological sort for reversed graphs
├── toy/               Example: template/ implementation (modify when: updating template/ tests)
│   ├── _forward_node.py   ForwardToyNode
│   └── _backward_node.py  BackwardToyNode
├── toy2/              Example: template2/ implementation (modify when: updating template2/ tests)
│   └── _node.py       Toy2Node
├── custom_types.py    TypeVars: CacheType, ArgumentType, NodeType
└── utils.py           PropMode enum (FORWARD/BACKWARD)
```

## Modification Map

| Intent | Primary Modify | Follow-ups | Avoid | Constraints | Failure Signal |
|--------|---------------|------------|-------|-------------|----------------|
| Add propagation algorithm | New subpackage under `src/propdag/` | Export in `__init__.py`, add tests | Editing `template/` or `template2/` internals | Must subclass ABCs (enforced) | `TypeError` on instantiation |
| Change graph traversal logic | `template/_sort.py` or `template2/_sort.py` | Update `_model.py` if signature changes | `_node.py` | Single input/output (enforced) | `ValueError` from `reverse_dag` |
| Add abstract method to node | `template/_node.py` or `template2/_node.py` | Implement in `toy/`, `toy2/`, and all consumers | Removing existing methods | ABC enforcement (enforced) | `TypeError` on instantiation |
| Change cache lifecycle | `template/_model.py` or `template2/_model.py` | Update `clear_*_cache` functions | Node implementations | Reference counting logic (observed) | Memory leak or stale data |
| Add configuration option | `template/_arguments.py` or `template2/_arguments.py` | Update consumer nodes that read it | `utils.py` | Frozen dataclass (enforced) | `FrozenInstanceError` |

## Dependency Rules

| Rule | Source | Failure |
|------|--------|---------|
| No runtime dependencies (pure Python) | pyproject.toml `dependencies = []` (enforced) | Import error in consumers |
| Absolute imports only | ruff TID `ban-relative-imports = "all"` (enforced) | `ruff check` failure |
| `template2/` does not import from `template/` | Module structure (observed) | Circular import |
| `toy*/` imports only from corresponding `template*/` | Module structure (observed) | Coupling violation |

## Common Mistakes

| Mistake | Detection Signal | Fix |
|---------|-----------------|-----|
| Adding relative imports | `ruff check` TID error | Use `from propdag.template._node import TNode` |
| Forgetting `__all__` in new module | `ruff check` F401 in `__init__.py` | Add `__all__` listing all public names |
| Mutating frozen argument dataclass | `FrozenInstanceError` at runtime | Create new instance with `replace()` |
| Building multi-input/output graph for T2Model | `ValueError` from `reverse_dag` | Ensure exactly one root and one leaf |

## Dependency Diagram

```
toy/ ──────► template/
toy2/ ─────► template2/
template/ ──► custom_types.py, utils.py
template2/ ─► (standalone, no cross-dependency to template/)
```

## Conventions

- Every module: `__docformat__ = "restructuredtext"` and `__all__`
- Arguments: `@dataclass(frozen=True, slots=True)`
- Node methods raise `RuntimeError` if not overridden (not `NotImplementedError`)
- Graph reversal is in-place (`pre_nodes`/`next_nodes` swap)

## Related Documents

- [README.md](README.md) -- usage examples and API overview
- [CONTRIBUTING.md](CONTRIBUTING.md) -- development setup and workflow
- [Root ARCHITECTURE.md](../ARCHITECTURE.md) -- rover system architecture
