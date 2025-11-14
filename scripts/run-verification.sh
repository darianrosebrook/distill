#!/bin/bash
# Production Readiness Verification Master Script
#
# Orchestrates all verification checks and generates comprehensive report.
# Run this script to verify production readiness across all criteria.
#
# Usage:
#   bash scripts/run-verification.sh

set -e  # Exit on any error

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
EVIDENCE_DIR="$PROJECT_ROOT/docs/internal/audits/readiness/evidence"
REPORTS_DIR="$PROJECT_ROOT/docs/internal/audits/readiness/reports"
LOG_FILE="$EVIDENCE_DIR/verification-run-$(date +%Y%m%d-%H%M%S).log"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging function
log() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') [run-verification] $*" | tee -a "$LOG_FILE"
}

# Error handling
error_exit() {
    echo -e "${RED}ERROR: $1${NC}" | tee -a "$LOG_FILE"
    echo "Check log file: $LOG_FILE" | tee -a "$LOG_FILE"
    exit 1
}

# Success/Warning messages
success() {
    echo -e "${GREEN}✅ $1${NC}" | tee -a "$LOG_FILE"
}

warning() {
    echo -e "${YELLOW}⚠️  $1${NC}" | tee -a "$LOG_FILE"
}

info() {
    echo -e "${BLUE}ℹ️  $1${NC}" | tee -a "$LOG_FILE"
}

# Check if Python virtual environment is available
check_python() {
    # Check for virtual environment first
    if [ -f "venv/bin/activate" ]; then
        log "Found virtual environment at venv/"
        PYTHON_CMD="venv/bin/python"
        PIP_CMD="venv/bin/pip"
        source venv/bin/activate
    elif [ -f ".venv/bin/activate" ]; then
        log "Found virtual environment at .venv/"
        PYTHON_CMD=".venv/bin/python"
        PIP_CMD=".venv/bin/pip"
        source .venv/bin/activate
    else
        # Fall back to system Python
        if ! command -v python3 &> /dev/null && ! command -v python &> /dev/null; then
            error_exit "Python not found. Please install Python 3.x or create a virtual environment"
        fi

        if command -v python3 &> /dev/null; then
            PYTHON_CMD="python3"
        else
            PYTHON_CMD="python"
        fi

        log "No virtual environment found, using system Python"
    fi

    # Check Python version
    PYTHON_VERSION=$($PYTHON_CMD --version 2>&1 | grep -oP '\d+\.\d+' || $PYTHON_CMD --version 2>&1 | grep -o '\d\+\.\d\+')
    PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d. -f1)
    PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d. -f2)

    if [ "$PYTHON_MAJOR" -lt 3 ] || ([ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -lt 10 ]); then
        error_exit "Python 3.10+ required, found $PYTHON_VERSION"
    fi

    # Check if required packages are available
    if ! $PYTHON_CMD -c "import pytest" 2>/dev/null; then
        error_exit "pytest not found. Please install testing dependencies: pip install pytest pytest-cov"
    fi

    log "Using Python: $PYTHON_CMD ($PYTHON_VERSION)"
}

# Run a verification script
run_verification_script() {
    local script_name="$1"
    local output_dir="$2"
    local description="$3"

    log "Running $description..."
    info "Running $description..."

    if [ ! -f "$SCRIPT_DIR/$script_name" ]; then
        error_exit "Script not found: $SCRIPT_DIR/$script_name"
    fi

    # Create output directory
    mkdir -p "$output_dir"

    # Run the script
    if $PYTHON_CMD "$SCRIPT_DIR/$script_name" --output "$output_dir"; then
        success "$description completed successfully"
        return 0
    else
        warning "$description found issues"
        return 1
    fi
}

# Main verification function
run_verification() {
    local exit_code=0

    # Create directories
    mkdir -p "$EVIDENCE_DIR" "$REPORTS_DIR"

    # Start logging
    log "Starting production readiness verification"
    log "Project root: $PROJECT_ROOT"
    log "Evidence directory: $EVIDENCE_DIR"
    log "Reports directory: $REPORTS_DIR"

    echo
    echo "🔍 PRODUCTION READINESS VERIFICATION"
    echo "====================================="
    echo

    # 1. Test Execution Verification
    info "Step 1: Test Execution Verification"
    if run_verification_script "verify-tests.py" "$EVIDENCE_DIR/test-execution" "test execution verification"; then
        success "All tests pass"
    else
        warning "Test failures found - check evidence files"
        exit_code=1
    fi
    echo

    # 2. Coverage Verification
    info "Step 2: Coverage Verification"
    if run_verification_script "verify-coverage.py" "$EVIDENCE_DIR/coverage" "code coverage verification"; then
        success "Coverage thresholds met"
    else
        warning "Coverage thresholds not met - check evidence files"
        exit_code=1
    fi
    echo

    # 3. Linting Verification
    info "Step 3: Linting Verification"
    if run_verification_script "verify-linting.py" "$EVIDENCE_DIR/linting" "linting verification"; then
        success "Zero linting errors"
    else
        warning "Linting errors found - check evidence files"
        exit_code=1
    fi
    echo

    # 4. Code Quality Verification
    info "Step 4: Code Quality Verification"
    if run_verification_script "verify-code-quality.py" "$EVIDENCE_DIR/code-quality" "code quality verification"; then
        success "Code quality gates met"
    else
        warning "Code quality issues found - check evidence files"
        exit_code=1
    fi
    echo

    # 5. Security Verification
    info "Step 5: Security Verification"
    if run_verification_script "verify-security.py" "$EVIDENCE_DIR/security" "security verification"; then
        success "Security controls verified"
    else
        warning "Security issues found - check evidence files"
        exit_code=1
    fi
    echo

    # 6. Generate Comprehensive Report
    info "Step 6: Generating Comprehensive Report"
    if run_verification_script "generate-report.py" "$REPORTS_DIR" "comprehensive report generation"; then
        success "Comprehensive report generated"
    else
        warning "Report generation failed"
        exit_code=1
    fi
    echo

    return $exit_code
}

# Hardware verification note (manual)
show_hardware_note() {
    echo
    info "Hardware Verification (Manual)"
    echo "Apple Silicon M1/M2/M3 Max hardware verification requires manual execution:"
    echo "1. Execute CoreML models on physical M1 Max hardware"
    echo "2. Measure ANE residency (>90% target)"
    echo "3. Benchmark latency/throughput against performance targets"
    echo "4. Save results to: $EVIDENCE_DIR/hardware/"
    echo
}

# Show results summary
show_results_summary() {
    local exit_code=$1

    echo
    echo "📊 VERIFICATION RESULTS SUMMARY"
    echo "==============================="

    if [ $exit_code -eq 0 ]; then
        success "All automated checks PASSED"
        echo
        echo "🎉 Production readiness criteria MET for automated checks!"
        echo
        echo "Next steps:"
        echo "1. Review evidence files in: $EVIDENCE_DIR"
        echo "2. Complete manual hardware verification"
        echo "3. Update docs/AUDIT_END_TO_END_READINESS.md with verified status"
        echo "4. Consider claiming production-ready status"
    else
        warning "Some checks FAILED - production readiness NOT met"
        echo
        echo "Failed checks require attention:"
        echo "1. Review evidence files in: $EVIDENCE_DIR"
        echo "2. Fix identified issues (tests, coverage, linting, etc.)"
        echo "3. Re-run verification: bash scripts/run-verification.sh"
        echo "4. Complete manual hardware verification"
    fi

    echo
    echo "📁 Evidence Location: $EVIDENCE_DIR"
    echo "📄 Reports Location: $REPORTS_DIR"
    echo "📋 Log File: $LOG_FILE"
    echo
}

# Main execution
main() {
    echo "🔍 Starting Production Readiness Verification..."
    echo

    # Pre-flight checks
    check_python

    # Run verification
    if run_verification; then
        VERIFICATION_EXIT_CODE=0
    else
        VERIFICATION_EXIT_CODE=1
    fi

    # Show hardware verification note
    show_hardware_note

    # Show results summary
    show_results_summary $VERIFICATION_EXIT_CODE

    # Exit with verification result
    exit $VERIFICATION_EXIT_CODE
}

# Run main function
main "$@"
