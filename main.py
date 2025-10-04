# main.py

import sys
from prod_assistant.core.bootstrap import bootstrap_app


def main():
    """Entry point for the pulse-flow CLI."""
    print("🚀 Pulse-Flow is running!")

    bootstrap_app()


if __name__ == "__main__":
    main()
