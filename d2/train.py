import argparse
import os
import sys

# Ensure root is on path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from train_sparse_d2 import main as train_d2_main

if __name__ == "__main__":
    train_d2_main()
