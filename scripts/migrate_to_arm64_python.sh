#!/bin/bash
# Migrate to ARM64 native Python installations
# This script updates PATH and creates symlinks to prioritize ARM64 Python

set -e

echo "🔍 Detecting Python installations..."

# Check ARM64 Homebrew Python
ARM64_PYTHON_311="/opt/homebrew/bin/python3.11"
if [ ! -f "$ARM64_PYTHON_311" ]; then
    echo "❌ ARM64 Python 3.11 not found at $ARM64_PYTHON_311"
    echo "   Installing via Homebrew..."
    arch -arm64 brew install python@3.11
fi

# Verify ARM64 architecture
ARCH=$(arch -arm64 "$ARM64_PYTHON_311" -c "import platform; print(platform.machine())" 2>/dev/null)
if [ "$ARCH" != "arm64" ]; then
    echo "❌ Python at $ARM64_PYTHON_311 is not ARM64 (got: $ARCH)"
    exit 1
fi

echo "✅ ARM64 Python 3.11 found: $ARM64_PYTHON_311"
echo "   Architecture: $ARCH"

# Check current shell config
SHELL_CONFIG="$HOME/.zshrc"
if [ ! -f "$SHELL_CONFIG" ]; then
    echo "❌ Shell config not found: $SHELL_CONFIG"
    exit 1
fi

echo ""
echo "📝 Updating shell configuration..."

# Check if ARM64 Homebrew PATH is already prioritized
if grep -q 'export PATH="/opt/homebrew/bin:$PATH"' "$SHELL_CONFIG"; then
    echo "✅ ARM64 Homebrew PATH already prioritized in $SHELL_CONFIG"
else
    echo "➕ Adding ARM64 Homebrew PATH priority to $SHELL_CONFIG"
    # Add after conda initialization but before other PATH modifications
    cat >> "$SHELL_CONFIG" << 'EOF'

# ============================================================================
# ARM64 PYTHON PRIORITY (Added by migrate_to_arm64_python.sh)
# ============================================================================
# Prioritize ARM64 Homebrew Python over x86_64 versions
export PATH="/opt/homebrew/bin:$PATH"
# Remove x86_64 Homebrew from PATH if present
export PATH=$(echo "$PATH" | tr ':' '\n' | grep -v "^/usr/local/bin$" | grep -v "^/usr/local/bin:" | tr '\n' ':' | sed 's/:$//')
EOF
    echo "✅ Updated $SHELL_CONFIG"
fi

echo ""
echo "🔗 Creating symlinks for common Python commands..."

# Create symlinks in a local bin directory that's in PATH
LOCAL_BIN="$HOME/.local/bin"
mkdir -p "$LOCAL_BIN"

# Create symlinks to ARM64 Python
ln -sf "$ARM64_PYTHON_311" "$LOCAL_BIN/python3.11" 2>/dev/null || true
ln -sf "$ARM64_PYTHON_311" "$LOCAL_BIN/python3" 2>/dev/null || true
ln -sf "$ARM64_PYTHON_311" "$LOCAL_BIN/python" 2>/dev/null || true

# Also create pip symlinks
ARM64_PIP_311="/opt/homebrew/bin/pip3.11"
if [ -f "$ARM64_PIP_311" ]; then
    ln -sf "$ARM64_PIP_311" "$LOCAL_BIN/pip3.11" 2>/dev/null || true
    ln -sf "$ARM64_PIP_311" "$LOCAL_BIN/pip3" 2>/dev/null || true
    ln -sf "$ARM64_PIP_311" "$LOCAL_BIN/pip" 2>/dev/null || true
fi

echo "✅ Created symlinks in $LOCAL_BIN"

echo ""
echo "🧪 Testing ARM64 Python and PyTorch availability..."

# Test ARM64 Python
ARM64_VERSION=$(arch -arm64 "$ARM64_PYTHON_311" --version 2>&1)
echo "   Python version: $ARM64_VERSION"

# Test PyTorch availability
PYTORCH_CHECK=$(arch -arm64 "$ARM64_PYTHON_311" -m pip install torch --dry-run 2>&1 | grep -E "torch-[0-9]" | head -1 || echo "Not found")
if [[ "$PYTORCH_CHECK" == *"torch"* ]]; then
    echo "✅ PyTorch available for ARM64 Python"
    echo "   $PYTORCH_CHECK"
else
    echo "⚠️  PyTorch check inconclusive (may need to install)"
fi

echo ""
echo "✅ Migration complete!"
echo ""
echo "📋 Next steps:"
echo "   1. Reload your shell: source ~/.zshrc"
echo "   2. Verify Python: python3.11 --version"
echo "   3. Check architecture: python3.11 -c 'import platform; print(platform.machine())'"
echo "   4. Install PyTorch: python3.11 -m pip install torch"
echo "   5. Install project: python3.11 -m pip install -e \".[dev]\""
echo ""
echo "💡 To use ARM64 Python in current session:"
echo "   export PATH=\"/opt/homebrew/bin:\$PATH\""

