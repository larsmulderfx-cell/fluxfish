#!/bin/bash
# Liquid Chess Engine - Lichess Deployment Script

set -e

# Configuration
ENGINE_NAME="liquid-chess-2.0"
VERSION="1.0.0"
BUILD_TYPE="Release"
PYTHON_VERSION="3.9"

echo "🚀 Building Liquid Chess Engine for Lichess..."

# Create build directory
mkdir -p build
cd build

# Configure with maximum optimizations
cmake .. \
    -DCMAKE_BUILD_TYPE=$BUILD_TYPE \
    -DCMAKE_CXX_FLAGS="-O3 -march=native -mtune=native -flto -fno-exceptions -fno-rtti" \
    -DPYTHON_EXECUTABLE=$(which python$PYTHON_VERSION)

# Build with all cores
make -j$(nproc) liquid_chess

echo "Engine built successfully"

# Install Python dependencies
cd ..
pip install -r requirements.txt

# Download or train LRT model
if [ ! -f "data/models/lrt_chess.jax" ]; then
    echo "📥 Downloading pre-trained LRT model..."
    wget -O data/models/lrt_chess.jax \
        https://storage.googleapis.com/liquid-chess-models/lrt_v1.jax
fi

# Create deployment package
DEPLOY_DIR="deploy/$ENGINE_NAME"
mkdir -p $DEPLOY_DIR

# Copy binaries
cp build/liquid_chess $DEPLOY_DIR/
cp -r python/liquid_chess $DEPLOY_DIR/
cp data/models/lrt_chess.jax $DEPLOY_DIR/

# Create configuration
cat > $DEPLOY_DIR/config.json << EOF
{
    "engine_name": "$ENGINE_NAME",
    "version": "$VERSION",
    "threads": 4,
    "hash_size": 256,
    "lrt_enabled": true,
    "adaptive_thinking": true,
    "lichess_token": "${LICHESS_TOKEN}"
}
EOF

# Create startup script
cat > $DEPLOY_DIR/start.sh << 'EOF'
#!/bin/bash

# Set up environment
export PYTHONPATH=$PWD:$PYTHONPATH
export OMP_NUM_THREADS=${THREADS:-4}
export MKL_NUM_THREADS=${THREADS:-4}

# Run UCI engine
if [ "$1" = "uci" ]; then
    ./liquid_chess
elif [ "$1" = "lichess" ]; then
    python -m liquid_chess.bot.lichess_bot \
        --engine ./liquid_chess \
        --token $LICHESS_TOKEN \
        --config config.json
else
    echo "Usage: $0 [uci|lichess]"
    exit 1
fi
EOF

chmod +x $DEPLOY_DIR/start.sh

echo "📦 Deployment package created in $DEPLOY_DIR"
echo ""
echo "To run on Lichess:"
echo "1. Get a bot token from https://lichess.org/account/oauth/token"
echo "2. export LICHESS_TOKEN=your_token_here"
echo "3. cd $DEPLOY_DIR && ./start.sh lichess"