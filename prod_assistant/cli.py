# E:\LLMOps\pulse-flow\prod_assistant\cli.py

import sys

def main():
    """Entry point for the pulse-flow CLI."""
    print("🚀 Pulse-Flow is running!")
    if sys.argv[1:]:
        print("Args passed:", sys.argv[1:])
