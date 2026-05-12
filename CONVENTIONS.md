# Propdag Conventions

This file defines style and documentation conventions for the propdag package.
Use it as a **checklist** — when writing or reviewing code, check each item below
one by one.

---

## 1. Module Docstrings

Every `.py` file begins with a module docstring.

### Rules

| # | Rule | Pass/Fail |
|---|------|-----------|
| 1.1 | **First line**: short summary of the module's purpose (one sentence) | ☐ |
| 1.2 | **Extended description** (optional): 1-2 paragraphs after a blank line, covering the module's role or key design decisions | ☐ |
| 1.3 | **Format**: ReST plain text; `**bold**` section headers permitted for multi-paragraph docstrings | ☐ |
| 1.4 | Always followed by `__docformat__ = "restructuredtext"` | ☐ |
| 1.5 | **No author, date, or version lines** — git history is authoritative | ☐ |
| 1.6 | **No non-ASCII characters** in docstrings — use ASCII equivalents for symbols | ☐ |

### Patterns

| File type | Style | Example |
|-----------|-------|---------|
| Package `__init__.py` | Summary + `Main components` bullet list | See `propdag/__init__.py` |
| ABC module (`_node.py`, `_model.py`) | One line describing the abstract contract | `"""Abstract base class for computational graph nodes."""` |
| Concrete implementation (`toy/_forward_node.py`) | One line naming the concrete class | `"""Forward node for toy model with verbose logging."""` |
| Utility module (`utils.py`) | One line | `"""Utility functions for graph traversal and sorting."""` |
| Constants/enums (`_constants.py`, `_enums.py`) | One line | `"""Package-level constants for propdag."""` |

---

## 2. Class Docstrings

propdag is an OOP framework — every class must have a docstring.

### 2.1 Structure

```python
class TNode(ABC, Generic[CacheType, ArgumentType]):
    """
    Short summary of what the class represents.

    Extended description (optional) — responsibilities, graph semantics,
    or key design decisions.

    **Key responsibilities:**
    - First responsibility.
    - Second responsibility.

    **Graph structure:**
    - Description of pre/post node relationships.

    :param Generic_param: Description of the type parameter's role.
    """
```

### 2.2 Rules

| # | Rule | Pass/Fail |
|---|------|-----------|
| 2.1 | **First line**: describes what the class represents, ends with period | ☐ |
| 2.2 | `**Key responsibilities:**` section for ABCs describing the contract subclasses must fulfill | ☐ |
| 2.3 | `**Graph structure:**` section for node classes describing pre/post node relationships | ☐ |
| 2.4 | Constructor parameters documented in class docstring when `__init__` is defined on the class | ☐ |
| 2.5 | `__init__` may have its own docstring for parameter details; use `:raises RuntimeError:` for abstract methods that raise instead of using `@abstractmethod` | ☐ |
| 2.6 | **No `@abstractmethod` decorator** — propdag uses `raise RuntimeError("Must be instantiated in {type(self).__name__}")` for clearer error messages | ☐ |
| 2.7 | Type parameters (`Generic[CacheType, ArgumentType]`) documented in the class docstring | ☐ |
| 2.8 | Use `::` for code examples (not `.. code-block:: python`) | ☐ |

### 2.3 Good examples

```python
class TNode(ABC, Generic[CacheType, ArgumentType]):
    """
    Abstract base class for computational graph nodes.

    Each node represents a layer or operation in a neural network DAG.
    Nodes must implement forward/backward propagation, bound calculation,
    and cache management.

    **Key responsibilities:**
    - Build relaxations for non-linear operations.
    - Compute intermediate bounds during propagation.
    - Manage per-node cache entries.

    **Graph structure:**
    - Input nodes: no predecessors, bounds provided externally.
    - Intermediate nodes: one or more predecessors, one or more successors.
    - Output nodes: no successors, loss computed here.

    :param CacheType: The cache type used by this node family.
    :param ArgumentType: The argument/configuration type for this node family.
    """
```

```python
class ToyCache:
    """Cache implementation for toy model with verbose debug logging.

    Stores forward/backward bounds, intermediate relaxations, and
    propagation statistics. All operations log to a shared logger
    for educational traceability.
    """
```

---

## 3. Method/Function Docstrings

### 3.1 Structure

```python
def forward(self, mode: PropMode) -> None:
    """
    Short imperative description of what the method computes.

    Extended description (optional) — the algorithm or propagation logic.

    :param mode: Propagation mode controlling which bounds to compute.
    :return: Description of return value (capitalized, ends with period).
    :raises RuntimeError: When the method is called on the ABC directly.
    """
```

### 3.2 Rules

| # | Rule | Pass/Fail |
|---|------|-----------|
| 3.1 | **First line**: imperative mood, describes what the method computes, ends with period | ☐ |
| 3.2 | Use `:param name:`, `:return:`, and `:raises ExceptionType:` tags — no `:type:` tags | ☐ |
| 3.3 | `:param` descriptions: **capitalized, end with period**, describe semantics not types | ☐ |
| 3.4 | `:return` description: **capitalized, end with period**; use "Tuple of" for multi-returns | ☐ |
| 3.5 | `:raises` descriptions: **capitalized, end with period**; describe the condition | ☐ |
| 3.6 | ABC methods that raise `RuntimeError` must document this in `:raises RuntimeError:` | ☐ |
| 3.7 | Private methods (`_` prefix) may use a single-line docstring without `:param:` tags | ☐ |
| 3.8 | Static methods use the same docstring format as instance methods | ☐ |

---

## 4. Inline Comments

| # | Rule | Pass/Fail |
|---|------|-----------|
| 4.1 | Comment **why**, not what — the code already says what | ☐ |
| 4.2 | Only add comments when the reasoning is non-obvious (graph reversal semantics, memory management) | ☐ |
| 4.3 | `# NOTE:` for important design notes that future readers need | ☐ |
| 4.4 | No commented-out code — delete it | ☐ |
| 4.5 | `# TODO:` comments require an associated issue reference (enforced by ruff TD001) | ☐ |
| 4.6 | Section divider comments use `# ----------` (10 dashes, 1 blank line before) for grouping related methods | ☐ |

---

## 5. Naming Conventions

| # | Rule | Pass/Fail |
|---|------|-----------|
| 5.1 | **ABC classes**: `T` prefix — `TNode`, `TModel`, `TCache`, `TArgument` | ☐ |
| 5.2 | **Template2 ABCs**: `T2` prefix — `T2Node`, `T2Model`, `T2Cache`, `T2Argument` | ☐ |
| 5.3 | **Toy implementations**: `Toy` prefix for single-node toys (`ToyModel`, `ToyCache`, `ToyArgument`); directional variants use `ForwardToyNode`, `BackwardToyNode`; `Toy2` prefix for template2-based variants | ☐ |
| 5.4 | **Methods/functions**: snake_case — `forward`, `backward`, `clear_fwd_cache`, `topo_sort_forward_bfs` | ☐ |
| 5.5 | **Private methods**: `_` prefix — `_build_relaxation`, `_update_cache` | ☐ |
| 5.6 | **Private modules**: `_` prefix — `_node.py`, `_cache.py`, `_model.py`, `_arguments.py`, `_sort.py` | ☐ |
| 5.7 | **Constants**: UPPER_CASE — `DEFAULT_MAX_ITER`, `CACHE_CLEANUP_THRESHOLD` | ☐ |
| 5.8 | **Type aliases**: PascalCase — `NodeType`, `CacheType`, `ArgumentType` | ☐ |
| 5.9 | **Graph edges**: `pre_nodes` (incoming), `next_nodes` (outgoing) — never `predecessors`/`successors` | ☐ |
| 5.10 | **Graph reversal** (template2): after `reverse_dag()`, `pre_nodes` and `next_nodes` swap semantics. Document this with `**CRITICAL OPERATION**` or `**SEMANTIC SHIFT**` in method docstrings | ☐ |
| 5.11 | **`_t2` suffix**: Functions specific to template2 use `_t2` suffix — `clear_bwd_cache_t2`, `topo_sort_forward_bfs_t2` | ☐ |

---

## 6. ABC and Generic Conventions

| # | Rule | Pass/Fail |
|---|------|-----------|
| 6.1 | ABCs inherit from `ABC` and `Generic[CacheType, ArgumentType]` | ☐ |
| 6.2 | Use `raise RuntimeError("Must be instantiated in {type(self).__name__}")` instead of `@abstractmethod` | ☐ |
| 6.3 | `TModel` is the top-level orchestrator: owns nodes, cache, arguments; delegates to `TNode` for per-node work | ☐ |
| 6.4 | `TNode` holds `pre_nodes: list[TNode]` and `next_nodes: list[TNode]` (bidirectional). `T2Node` follows the same convention with graph edges reversed by `reverse_dag()` | ☐ |
| 6.5 | `TCache` is an empty `@dataclass(slots=True)` serving as a type bound for the `CacheType` TypeVar. Concrete caches (e.g., `ToyCache`, `Toy2Cache`) store bounds/relaxations keyed by node name as `dict[str, tuple]`. Reference counting lives in module-level `clear_fwd_cache()` / `clear_bwd_cache()` functions, not in the cache class | ☐ |
| 6.6 | `TArgument` is a frozen dataclass (`@dataclass(frozen=True, slots=True)`) holding per-node configuration. Document fields with `:ivar name:` or `:param name:` in the class docstring | ☐ |
| 6.7 | Abstract methods that subclasses must override document the contract in their docstring — what they compute, not how | ☐ |
| 6.8 | `template/` and `template2/` provide ABCs with `raise RuntimeError` stubs — not concrete implementations. `toy/` and `toy2/` provide concrete working implementations with verbose logging for education | ☐ |

---

## 7. Code Style

| # | Rule | Pass/Fail |
|---|------|-----------|
| 7.1 | **100-char line length** (enforced by ruff) | ☐ |
| 7.2 | **Double quotes** for strings and docstrings | ☐ |
| 7.3 | **Absolute imports only** — `from propdag._enums import PropMode` | ☐ |
| 7.4 | `__docformat__ = "restructuredtext"` after module docstring, before imports | ☐ |
| 7.5 | `__all__` in every module, alphabetically sorted, listing all public names | ☐ |
| 7.6 | **Import order**: stdlib → first-party (`propdag.*`). No third-party dependencies. | ☐ |
| 7.7 | **No external dependencies** — propdag is pure stdlib (`abc`, `dataclasses`, `enum`, `collections.abc`, `typing`) | ☐ |
| 7.8 | `from __future__ import annotations` at top of files using forward references | ☐ |
| 7.9 | **McCabe complexity ≤ 10** (enforced by ruff C90) | ☐ |
| 7.10 | **Only import what you use** — clean up unused imports (enforced by ruff F401) | ☐ |
| 7.11 | **No string annotations** when type is already imported — write `-> TNode` not `-> "TNode"` | ☐ |

---

## 8. Frozen Dataclass Conventions

| # | Rule | Pass/Fail |
|---|------|-----------|
| 8.1 | Arguments/configuration classes use `@dataclass(frozen=True, slots=True)` | ☐ |
| 8.2 | Mutable cache classes use `@dataclass(slots=True)` (no `frozen`) | ☐ |
| 8.3 | Every field has an explicit type annotation | ☐ |
| 8.4 | Default values use `field(default=...)` for mutable defaults | ☐ |
| 8.5 | Class docstring describes what the dataclass holds; `:param name:` tags for each field | ☐ |

---

## 9. Cache and Memory Management

| # | Rule | Pass/Fail |
|---|------|-----------|
| 9.1 | `cache_counter: dict[NodeType, int]` tracks reference counts per node object for cache cleanup | ☐ |
| 9.2 | `clear_fwd_cache()` / `clear_bwd_cache()` decrement counters and clear when ≤ 0 | ☐ |
| 9.3 | Concrete caches (e.g., `ToyCache`) store entries keyed by node name as `dict[str, tuple]` | ☐ |
| 9.4 | Concrete caches may use a `cur_node` field tracking the currently executing node | ☐ |

---

## 10. Enum Conventions

| # | Rule | Pass/Fail |
|---|------|-----------|
| 10.1 | Enums live in `_enums.py` at the package root | ☐ |
| 10.2 | Use `IntEnum` with `@unique` decorator | ☐ |
| 10.3 | Enum class docstring describes what the enum represents | ☐ |
| 10.4 | Enum member names: UPPER_CASE — `FORWARD`, `BACKWARD`, `BOTH` | ☐ |

---

## 11. Template vs Toy Package Conventions

| # | Rule | Pass/Fail |
|---|------|-----------|
| 11.1 | `template/` and `template2/` provide abstract/production-ready implementations | ☐ |
| 11.2 | `toy/` and `toy2/` provide educational implementations with verbose logging | ☐ |
| 11.3 | Toy classes inherit from the corresponding template ABCs | ☐ |
| 11.4 | Each subpackage (`template/`, `template2/`, `toy/`, `toy2/`) has its own `__init__.py` re-exporting its public classes | ☐ |
| 11.5 | New reference implementations go in a new `template<N>/` or `toy<N>/` subpackage | ☐ |
| 11.6 | `# STEP N:` comments document algorithm phases in multi-step methods (e.g., `reverse_dag()` in `template2/_model.py`) | ☐ |

---

## 12. Cross-Cutting Patterns

| # | Rule | Pass/Fail |
|---|------|-----------|
| 12.1 | **`custom_types.py`**: Dedicated module for `TypeVar` and type aliases. Not re-exported via `__init__.py` — acts as a private type-definition module imported by other modules under `TYPE_CHECKING` | ☐ |
| 12.2 | **`utils.py` as re-export shim**: May re-export public symbols from private modules when a simple public API surface is desired | ☐ |
| 12.3 | **`__version__` attribute**: Module-level `__version__ = "YYYY.MINOR.PATCH"` in root `__init__.py` for package identification | ☐ |
| 12.4 | **`create_cache_counter` pattern**: Both `TModel.run()` and `T2Model.run()` locally construct `cache_counter` dicts as `{node: len(node.next_nodes) for node in self._nodes}` | ☐ |
| 12.5 | **Method name abbreviations**: Permitted for well-known propagation terms — `fwdprop_symbnd`, `bwdprop_symbnd`, `init_symbnd`, `cal_and_update_cur_node_bnd` | ☐ |
| 12.6 | **`AssertionError` for invariants**: Use `assert` for internal invariants that indicate bugs; use `raise ValueError` for user-facing input validation | ☐ |

---

## 13. Test Style

### 13.1 Directory Layout

```
tests/
├── _mirrors_exempt.txt       # files excluded from mirror-symlink checks
├── test_units/
│   ├── _helpers.py           # shared across test subpackages
│   ├── test_template/
│   │   ├── __init__.py
│   │   ├── conftest.py
│   │   ├── _helpers.py
│   │   └── test_<concern>.py
│   └── test_template2/
│       ├── __init__.py
│       ├── conftest.py
│       ├── _helpers.py
│       └── test_<concern>.py
```

### 13.2 Rules

| # | Rule | Pass/Fail |
|---|------|-----------|
| 13.1 | **Test file naming**: `test_<concern>.py` — `test_sorting_algorithms.py`, `test_error_handling.py` | ☐ |
| 13.2 | **Test class naming**: `Test<Behavior>` — `TestTopoSort`, `TestErrorHandling` | ☐ |
| 13.3 | **Topology builders** in `_helpers.py`: return `(model, cache, nodes)` tuple for consistent test setup | ☐ |
| 13.4 | **Golden sequence testing**: use `capture_golden_sequences.py` for expected propagation order verification | ☐ |
| 13.5 | `_t2` suffix convention for template2-specific test files and helpers | ☐ |
| 13.6 | `conftest.py` at test subpackage level for fixtures shared within that subpackage | ☐ |
| 13.7 | `__init__.py` at leaf `test_<pkg>/` level only (collision avoidance) | ☐ |
| 13.8 | **No pytest markers** except `@pytest.mark.parametrize` | ☐ |
| 13.9 | Test module docstrings: 1-3 lines max summarizing what the file validates | ☐ |
| 13.10 | **Default test suite**: `pytest` runs `tests/test_units/` by default. Benchmark and integration tests are opt-in | ☐ |
| 13.11 | **No `@pytest.mark.skip`** in committed code — use conditional early return with `[REVIEW]` comment | ☐ |

---

## 14. Enum Conventions

| # | Rule | Pass/Fail |
|---|------|-----------|
| 14.1 | **IntEnum with `@unique`**: All enums use `IntEnum` with `@unique` decorator. `StrEnum` for user-facing string values | ☐ |
| 14.2 | **Placement**: Subpackage-local enums in `<subfolder>/_enums.py` (e.g., `propdag/_enums.py` for `PropMode`) | ☐ |
| 14.3 | **Class naming**: PascalCase with categorical suffix — `Mode` (behavioral), `Type` (variant), `Status` (state), `Strategy` (algorithm). Never suffix with `Enum` | ☐ |
| 14.4 | **Member naming**: `UPPER_SNAKE_CASE`, 1-3 words. Must be unique within the class | ☐ |
| 14.5 | **Custom `__repr__`**: IntEnum classes define `__repr__` returning `f"{self.name}"` | ☐ |
| 14.6 | **Member docstrings**: Every enum member has a one-line ReST docstring after the value assignment | ☐ |
| 14.7 | **Module boilerplate**: `__docformat__ = "restructuredtext"`, `__all__` alphabetically sorted listing enum classes | ☐ |

---

## 15. Constants Conventions

| # | Rule | Pass/Fail |
|---|------|-----------|
| 15.1 | **Naming**: `UPPER_SNAKE_CASE`, 2-4 words. Use prefixes (`DEFAULT_`, `MAX_`, `MIN_`) and suffixes (`_DIR`, `_NAME`, `_MB`) for clarity | ☐ |
| 15.2 | **Scope levels**: Place at narrowest scope — function-level → file-level → subfolder `_constants.py` → package-level. Promote when a second consumer at broader scope appears | ☐ |
| 15.3 | **Extraction trigger**: Extract a literal when it appears 2+ times. Never duplicate a constant across files | ☐ |
| 15.4 | **When NOT to extract**: Self-documenting single-use values, test data, function defaults already named by the parameter, `0`/`1`/`-1` for indexing | ☐ |
| 15.5 | **Type annotations**: Annotate only when the type is not obvious from the literal | ☐ |
| 15.6 | **Frozen collections**: Use `frozenset` or `tuple` for constant collections — never mutable `list` or `set` | ☐ |
| 15.7 | **File-level private constants**: Use `_` prefix + UPPER_CASE — `_DEFAULT_MATMUL_BOUND` | ☐ |

