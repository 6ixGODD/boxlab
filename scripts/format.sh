#!/bin/sh
# ============================================================================
# Code Formatting Script
# ============================================================================
# Description: Format Python code using ruff
# Usage: scripts/format.sh [directory]
# Default directory: boxlab/
# ============================================================================

set -e

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Source common utilities
# shellcheck source=bin/tools/common.sh
. "$SCRIPT_DIR/tools/common.sh"

# Variables
TARGET_DIR="${1:-boxlab}"
POETRY_CMD="poetry"

# Check if dev dependencies are installed
check_dependencies() {
	log_info "Checking formatting dependencies..."

	if ! $POETRY_CMD run python -c "import ruff" 2>/dev/null; then
		log_warn "ruff not found, installing dev dependencies..."
		$POETRY_CMD install --extras dev
	fi

	log_success "Dependencies check complete"
}

# Run ruff check with auto-fix
run_ruff_check() {
	log_step "Running ruff check (linting and import sorting)..."

	if ! dir_exists "$TARGET_DIR"; then
		log_warn "Directory not found: $TARGET_DIR"
		return 1
	fi

	if $POETRY_CMD run ruff check --fix "$TARGET_DIR"; then
		log_success "ruff check complete"
		return 0
	else
		log_error "ruff check failed"
		return 1
	fi
}

# Run ruff format
run_ruff_format() {
	log_step "Running ruff format (code formatting)..."

	if ! dir_exists "$TARGET_DIR"; then
		log_warn "Directory not found: $TARGET_DIR"
		return 1
	fi

	if $POETRY_CMD run ruff format "$TARGET_DIR"; then
		log_success "ruff format complete"
		return 0
	else
		log_error "ruff format failed"
		return 1
	fi
}

# Main
main() {
	print_header "Code Formatting"

	log_info "Target directory: $TARGET_DIR"
	echo ""

	check_dependencies
	echo ""

	check_result=0
	format_result=0

	run_ruff_check || check_result=$?
	echo ""

	run_ruff_format || format_result=$?
	echo ""

	print_separator
	if [ $check_result -eq 0 ] && [ $format_result -eq 0 ]; then
		log_success "All formatting complete!"
	else
		log_error "Some formatting failed"
		exit 1
	fi
	print_separator
}

main
