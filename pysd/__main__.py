import sys
from .cli import main

if __name__ == "__main__":
    if len(sys.argv) == 1:
        main(["--help"])
    else:
        main(sys.argv[1:])
