#!/bin/bash
set -e

echo "╔═══════════════════════════════════════════════════════════╗"
echo "║  Gaba-Burn Overpower Implementation Test Suite            ║"
echo "╚═══════════════════════════════════════════════════════════╝"
echo ""

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

# Test counter
TESTS_PASSED=0
TESTS_TOTAL=0

run_test() {
    TESTS_TOTAL=$((TESTS_TOTAL + 1))
    echo -e "${CYAN}[TEST $TESTS_TOTAL]${NC} $1"
    if eval "$2"; then
        echo -e "${GREEN}✓ PASSED${NC}"
        TESTS_PASSED=$((TESTS_PASSED + 1))
        echo ""
        return 0
    else
        echo -e "${YELLOW}✗ FAILED${NC}"
        echo ""
        return 1
    fi
}

# 1. Build tests
echo -e "${CYAN}═══ Build Tests ═══${NC}"
run_test "Build gaba-pqc" "cargo build -p gaba-pqc --release"
run_test "Build gaba-pqc with Metal" "cargo build -p gaba-pqc --release --features metal"
run_test "Build gaba-metal-backend" "cargo build -p gaba-metal-backend --release --features metal"
run_test "Build gaba-native-kernels" "cargo build -p gaba-native-kernels --release"
run_test "Build gaba-train-cli" "cargo build -p gaba-train-cli --release"
run_test "Build gaba-train-cli full" "cargo build -p gaba-train-cli --release --features full"

# 2. Unit tests
echo -e "${CYAN}═══ Unit Tests ═══${NC}"
run_test "Test gaba-pqc" "cargo test -p gaba-pqc --release"
run_test "Test gaba-native-kernels" "cargo test -p gaba-native-kernels --release"

# 3. Feature tests
echo -e "${CYAN}═══ Feature Tests ═══${NC}"
if [[ "$OSTYPE" == "darwin"* ]]; then
    run_test "Test Metal backend" "cargo test -p gaba-metal-backend --release --features metal"
    run_test "Test PQC with Metal" "cargo test -p gaba-pqc --release --features metal"
else
    echo "Skipping Metal tests (not on macOS)"
fi

# 4. CLI tests
echo -e "${CYAN}═══ CLI Tests ═══${NC}"
CLI_BIN="./target/release/gaba-train"

if [ -f "$CLI_BIN" ]; then
    run_test "CLI info command" "$CLI_BIN info"
    run_test "CLI help" "$CLI_BIN --help"
    
    # Create test directory
    mkdir -p /tmp/gaba-test
    
    run_test "CLI generate data" "$CLI_BIN generate --output /tmp/gaba-test --traffic-samples 100 --route-samples 10"
    
    if [ -f "/tmp/gaba-test/traffic.csv" ]; then
        run_test "CLI train traffic" "$CLI_BIN traffic --data /tmp/gaba-test/traffic.csv --output /tmp/gaba-test --epochs 5 --lr 0.01"
    fi
    
    # Cleanup
    rm -rf /tmp/gaba-test
else
    echo "CLI binary not found, skipping CLI tests"
fi

# 5. Performance benchmarks
echo -e "${CYAN}═══ Performance Benchmarks ═══${NC}"
if [ -f "$CLI_BIN" ]; then
    echo "Running GEMM benchmarks..."
    $CLI_BIN bench --size small || true
    echo ""
fi

# Summary
echo "╔═══════════════════════════════════════════════════════════╗"
echo "║  Test Results                                             ║"
echo "╚═══════════════════════════════════════════════════════════╝"
echo ""
echo -e "Tests Passed: ${GREEN}$TESTS_PASSED${NC} / $TESTS_TOTAL"
echo ""

if [ $TESTS_PASSED -eq $TESTS_TOTAL ]; then
    echo -e "${GREEN}✓ ALL TESTS PASSED${NC}"
    exit 0
else
    echo -e "${YELLOW}⚠ Some tests failed${NC}"
    exit 1
fi
