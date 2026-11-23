"""Allow ``python -m sinmodel`` to launch the graphical editor."""

from .gui import main

if __name__ == "__main__":
    raise SystemExit(main())
