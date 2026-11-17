"""Allow ``python -m slideseek``, which works even where the console script is unavailable."""

from slideseek.cli import app

if __name__ == "__main__":
    app()
