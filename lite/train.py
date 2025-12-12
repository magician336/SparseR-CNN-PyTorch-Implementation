import argparse
import os
import sys

# Ensure root is on path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from train import main as train_main

if __name__ == "__main__":
    train_main()
