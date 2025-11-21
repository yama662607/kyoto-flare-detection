# Default task
default: check

# Run all checks (lint, format, typecheck)
check:
    ruff check src/
    ruff format --check src/

# Apply fixes
fix:
    ruff check --fix src/
    ruff format src/

# Run development server
dev:
    echo "No dev server configured"

# Run tests
test:
    echo "No tests configured"

# Build project
build:
    echo "No build configured"
