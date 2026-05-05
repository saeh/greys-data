#!/bin/bash
# Daily run script for greys-data

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR/.."

# Activate virtual environment if it exists
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
fi

# Run the generation
echo "Starting data generation at $(date)"
python main.py generate

echo "Data generation completed at $(date)"
