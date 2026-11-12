# Repository Guidelines

## Project Structure & Module Organization

This repository contains an educational, standard-library-only Python implementation of k-means clustering. Keep the file count small without sacrificing readability.

- `kmeans.py`: Lloyd’s algorithm (`kmeans`), immutable `KMeansResult`, input validation, Gaussian sample generation (`make_blobs`), and a runnable demonstration.
- Add automated tests in a root-level `test_kmeans.py`; no test module currently exists.
- No assets, dependency manifest, or build configuration are currently present. Generate test data in memory instead of committing datasets.

## Build, Test, and Development Commands

Run commands from the repository root using Python 3.10 or newer. No installation or build step is required.

- `python kmeans.py`: run the seeded demonstration and print centers, cluster sizes, inertia, and convergence information.
- `python -m unittest discover -v`: discover and run standard-library tests after they are added. Check the reported test count; zero tests does not validate the algorithm.
- `python -m py_compile kmeans.py`: check source syntax.

## Coding Style & Naming Conventions

Use four-space indentation, PEP 8 naming, and type annotations for function interfaces. Use `snake_case` for functions and variables, `PascalCase` for classes, and a leading underscore for private helpers. Preserve the existing Russian-language documentation and validation messages when extending the module.

Keep random state local through `random.Random(seed)` and avoid modifying caller-owned inputs. Prefer standard-library solutions. No formatter or linter is configured.

## Testing Guidelines

Use `unittest.TestCase` with methods named `test_<behavior>`. Cover known cluster assignments, centroid means, inertia, convergence, iteration limits, duplicate points, empty clusters, invalid inputs, and multiple dimensions. Test `make_blobs` separately, including zero spread and seed reproducibility.

Use fixed seeds for randomized cases and approximate comparisons for floating-point values. Compare cluster membership independently of arbitrary label numbering. No numerical coverage threshold is configured; include normal, boundary, and failure cases for each public function.

## Commit & Pull Request Guidelines

No usable Git history is available in this checkout, so existing commit conventions cannot be established. Use concise imperative subjects such as `Add deterministic clustering tests`.

Pull requests should explain the change, relevant algorithm decisions, and verification commands with results. Link an issue when applicable. Document changes to parameters, return values, or empty-cluster behavior.
