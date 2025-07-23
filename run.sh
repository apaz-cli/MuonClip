#!/bin/bash

# Check if an optimizer argument is provided
if [ $# -eq 1 ]; then
    # Run training with the specified optimizer
    optimizer=$1
    echo "Training with $optimizer..."
    python test.py --optimizer "$optimizer"
else
    # Run training with all three optimizers
    echo "Running training with all optimizers..."

    echo "Training with Adam..."
    python test.py --optimizer adam

    echo "Training with Muon..."
    python test.py --optimizer muon

    echo "Training with MuonClip..."
    python test.py --optimizer muonclip

    echo "All training runs completed!"
fi
