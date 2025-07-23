#!/bin/bash

# Run training with all three optimizers
echo "Running training with all optimizers..."

echo "Training with Adam..."
python test.py --optimizer adam

echo "Training with Muon..."
python test.py --optimizer muon

echo "Training with MuonClip..."
python test.py --optimizer muonclip

echo "All training runs completed!"
