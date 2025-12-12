import argparse
import os
import sys

# Ensure root is on path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from infer_sparse_d2 import main as infer_d2_main

if __name__ == "__main__":
    infer_d2_main()
