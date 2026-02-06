# =============================================================================
# ⚙️ Configuration & Variables
# =============================================================================

set dotenv-load := true
set shell := ["bash", "-c"]

# Python project variables
src_dir := "src"
test_dir := "tests"

# =============================================================================
# 🤖 Standard Interface (AI Agent Protocol)
# =============================================================================

# Default: run read-only full checks
default: check

# Environment setup: install dependencies and toolchain
setup:
    @echo "📦 Setting up environment..."
    uv sync --all-extras

# Full quality verification without code changes (CI gate)
check: fmt-check lint test
    @echo "✅ All quality checks passed!"

# Auto-fix: apply formatting and lint fixes (agent's first response)
fix: fmt lint-fix
    @echo "✨ Auto-fixes applied!"

# =============================================================================
# 🧪 Testing & Verification
# =============================================================================

# Unit/integration tests: supports argument passthrough
# Usage: just test (all) | just test path/to/file (specific)
test *args="":
    @echo "🧪 Running unit tests..."
    @if [ -d "{{test_dir}}" ]; then \
        uv run pytest {{args}}; \
    else \
        echo "⚠️  No tests directory found. Skipping tests."; \
    fi

# E2E tests: supports argument passthrough
e2e *args="":
    @echo "🎭 Running E2E tests..."
    @echo "⚠️  E2E tests not configured for this project."

# =============================================================================
# 🧩 Granular Tasks (Components of 'check' & 'fix')
# =============================================================================

# --- Format ---

fmt-check:
    @echo "📏 Checking formatting..."
    uv run ruff format --check {{src_dir}}

fmt:
    @echo "💅 Formatting code..."
    uv run ruff format {{src_dir}}

# --- Lint ---

lint:
    @echo "🔍 Linting..."
    uv run ruff check {{src_dir}}

lint-fix:
    @echo "🧹 Fixing lint errors..."
    uv run ruff check --fix {{src_dir}}

# --- Typecheck ---

typecheck:
    @echo "📐 Checking types..."
    @echo "⚠️  Type checking not configured (mypy not in dependencies)."

# =============================================================================
# 🛠️ Operations & Utilities
# =============================================================================

# Start dev server
dev:
    @echo "🚀 Starting dev server..."
    @echo "⚠️  No dev server configured for this project."

# Production build
build:
    @echo "🏗️ Building artifact..."
    @echo "⚠️  No build process configured for this project."

# Remove artifacts
clean:
    @echo "🗑️ Cleaning artifacts..."
    rm -rf .ruff_cache .pytest_cache .mypy_cache __pycache__ .coverage htmlcov
    find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
    find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
