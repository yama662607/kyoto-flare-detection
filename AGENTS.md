# AGENTS.md

This document provides guidelines for AI agents to work efficiently on this project.

## Justfile Guide

This project uses **Justfile** to standardize quality checks and CI/CD for the Python codebase.

### Basic Commands

| Command              | Description                              | When to Use            |
| -------------------- | ---------------------------------------- | ---------------------- |
| `just` / `just check` | Run full quality checks (format + lint + test) | Before starting work, before PR |
| `just fix`           | Auto-fix (format + lint --fix)           | First response to errors |
| `just setup`         | Environment setup (install deps)         | New environment, after dependency updates |

### Testing

| Command                    | Description          | Example                             |
| -------------------------- | -------------------- | ----------------------------------- |
| `just test`                | Run all tests        | -                                   |
| `just test path/to/file.py` | Run tests for a file | `just test tests/test_detector.py` |

### Individual Tasks

| Command         | Description                  | Purpose               |
| --------------- | ---------------------------- | --------------------- |
| `just fmt`      | Apply formatting             | Manual formatting     |
| `just fmt-check`| Check formatting             | CI check              |
| `just lint`     | Run static analysis          | Code quality check    |
| `just lint-fix` | Auto-fix lint errors         | Minor lint issues     |
| `just clean`    | Remove caches/artifacts      | Cleanup               |

### Development/Operations

| Command     | Description          | Notes        |
| ----------- | -------------------- | ------------ |
| `just dev`  | Start dev server     | Not configured |
| `just build`| Production build     | Not configured |

## Agent Workflow

### 1. Start of work

```bash
just check  # Check current quality state
```

### 2. After code changes

```bash
just fix
just check
```

### 3. Before PR

```bash
just check && just test
```

### 4. When issues are found

```bash
just fix
# If not resolved, fix manually
```

## Project Structure

```
kyoto-flare-detection/
├── justfile              # Task runner
├── pyproject.toml        # Python project settings
├── src/                  # Source code
│   ├── base_flare_detector.py  # Base class
│   ├── flarepy_DS_Tuc_A.py     # DS Tuc A
│   ├── flarepy_EK_Dra.py       # EK Dra
│   └── flarepy_V889_Her.py     # V889 Her
├── tests/                # Tests
├── docs/                 # Documentation
├── notebooks/            # Jupyter notebooks
│   ├── flare_create_graphs.ipynb # Main analysis/graph generation
│   └── flare_detect_*.ipynb      # Per-star detection/analysis
├── outputs/              # Output files (figures, etc.)
│   └── debug/            # Per-run debug output (YYYYMMDD_HHMMSS)
└── tools/                # Utility scripts
```

## Tooling

- **Package manager**: uv
- **Formatter**: Ruff format
- **Linter**: Ruff check
- **Tests**: pytest (minimal smoke test)
- **Target directory**: `src/`

## CI/CD Alignment

Justfile tasks align with CI/CD:

- **Quality gate**: `just check`
- **Auto-fix**: `just fix`
- **Standardized commands** for all agents

## Notes

- **Argument pass-through**: `just test` allows args
- **Env vars**: `.env` is loaded automatically
- **Error handling**: tasks are expected to handle errors properly
- **Unconfigured tasks**: `dev` / `build` are placeholders
```
