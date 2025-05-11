.DEFAULT_GOAL := help

help: ## ⁉️ Displays this help message
	@echo "Usage: make [target]"
	@echo ""
	@echo "Available targets:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-20s %s\n", $$1, $$2}'

test-unit: ## 🧪 Runs unit tests with coverage
	@echo "🧪 Running unit tests with coverage..."
	uv run pytest --cov=src/promptdrifter
	@echo "✅ Unit tests finished."

lint: ## ✨ Runs linter (ruff check)
	@echo "✨ Running linter..."
	uv run ruff check .
	@echo "👍 Linting finished."

lint-fix: ## 🪄 Runs linter with auto-fix (ruff check --fix)
	@echo "🪄 Running linter with auto-fix..."
	uv run ruff check --fix .
	@echo "🎉 Linting and fixing finished."

.PHONY: test-unit lint lint-fix help
