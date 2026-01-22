# Changelog

All notable changes to PropDAG will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [2026.1.3] - 2026-01-23

### Added
- Backward substitution infrastructure in T2Model for tighter bound propagation
- `_all_backward_sorts` dictionary in T2Model to track backward topological orders
- Integration with `topo_sort_backward` for reversed graph back-substitution

### Technical Details
- T2Model now computes backward sorts during initialization
- Enables future back-substitution from any node back to graph input (user's output)
- No API changes; internal infrastructure enhancement

## [2026.1.1] - 2026-01-03

### Added
- Template2 module with reversed graph semantics for intuitive backward propagation
- T2Node, T2Model, T2Cache, T2Argument classes for reversed graph operations
- Toy2 example implementations demonstrating Template2 usage patterns
- Automatic graph reversal via `reverse_dag()` function
- Comprehensive test suite with 119 tests and 95% code coverage

### Changed
- Expanded test suite to cover Template2 functionality
- Enhanced documentation with Template2 usage examples
- Improved module organization with separate template/ and template2/ directories

### Fixed
- Improved cache management with progressive cleanup strategies
- Better error messages for graph constraint violations

## [2026.1.0] - 2025-12-19

### Added
- Initial release of PropDAG framework for neural network verification
- Core template classes: TNode, TModel, TCache, TArgument for forward graph
- Toy example implementations demonstrating abstract template usage
- Support for forward and backward bound propagation through DAGs
- BFS and DFS topological sorting strategies
- Zero-dependency pure Python 3.11+ implementation
- Comprehensive test suite with 119 tests
- MIT License

### Features
- DAG-native architecture naturally supporting residual/skip connections
- Flexible bidirectional propagation for verification algorithms
- Customizable node implementations via template pattern
- Memory-efficient cache management with reference counting
