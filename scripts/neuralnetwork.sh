#! /bin/bash

# scaling 
# normalizing
# scaling + pca 15
# normalizing + pca 5
# scaling + normalizing 
# scaling + normalizing + pca 15

# Experiment 1
python3 neuralnetwork.py --standardize True

# Experiment 2
python3 neuralnetwork.py --normalize True

# Experiment 3
python3 neuralnetwork.py --standardize True --pca 15

# Experiment 4
python3 neuralnetwork.py --normalize True --pca 5

# Experiment 5
python3 neuralnetwork.py --standardize True --normalize True

# Experiment 6
python3 neuralnetwork.py --standardize True --normalize True --pca 15