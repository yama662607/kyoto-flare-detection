# =============================================================================
# ⚙️ Configuration & Variables
# =============================================================================

set dotenv-load := true
set shell := ["bash", "-c"]

# Python プロジェクト用の変数
src_dir := "src"
test_dir := "tests"

# =============================================================================
# 🤖 Standard Interface (AI Agent Protocol)
# =============================================================================

# デフォルト: 読み取り専用の全体チェックを実行
default: check

# 環境構築: 依存関係のインストール、ツールチェーンのセットアップ
setup:
    @echo "📦 Setting up environment..."
    uv sync --all-extras

# 全体品質検証: コードを変更せずに品質を検証 (CIゲート)
check: fmt-check lint test
    @echo "✅ All quality checks passed!"

# 自動修正: フォーマットとLint修正を適用 (Agentの第一手)
fix: fmt lint-fix
    @echo "✨ Auto-fixes applied!"

# =============================================================================
# 🧪 Testing & Verification
# =============================================================================

# ユニット/統合テスト: 引数パススルー対応
# Usage: just test (全実行) | just test path/to/file (特定実行)
test *args="":
    @echo "🧪 Running unit tests..."
    @if [ -d "{{test_dir}}" ]; then \
        uv run pytest {{args}}; \
    else \
        echo "⚠️  No tests directory found. Skipping tests."; \
    fi

# E2Eテスト: 引数パススルー対応
e2e *args="":
    @echo "🎭 Running E2E tests..."
    @echo "⚠️  E2E tests not configured for this project."

# =============================================================================
# 🧩 Granular Tasks (Components of 'check' & 'fix')
# =============================================================================

# --- Format (整形) ---

fmt-check:
    @echo "📏 Checking formatting..."
    uv run ruff format --check {{src_dir}}

fmt:
    @echo "💅 Formatting code..."
    uv run ruff format {{src_dir}}

# --- Lint (静的解析) ---

lint:
    @echo "🔍 Linting..."
    uv run ruff check {{src_dir}}

lint-fix:
    @echo "🧹 Fixing lint errors..."
    uv run ruff check --fix {{src_dir}}

# --- Typecheck (型検査) ---

typecheck:
    @echo "📐 Checking types..."
    @echo "⚠️  Type checking not configured (mypy not in dependencies)."

# =============================================================================
# 🛠️ Operations & Utilities
# =============================================================================

# 開発サーバー起動
dev:
    @echo "🚀 Starting dev server..."
    @echo "⚠️  No dev server configured for this project."

# 本番ビルド
build:
    @echo "🏗️ Building artifact..."
    @echo "⚠️  No build process configured for this project."

# アーティファクト削除
clean:
    @echo "🗑️ Cleaning artifacts..."
    rm -rf .ruff_cache .pytest_cache .mypy_cache __pycache__ .coverage htmlcov
    find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
    find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
