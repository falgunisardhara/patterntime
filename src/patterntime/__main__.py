"""
PatternTime module entrypoint.
Allows running CLI via: python -m patterntime <command>
"""

import sys
from patterntime.cli import main

if __name__ == "__main__":
    sys.exit(main())
