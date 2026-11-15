"""Allow ``python -m lecturelens``, which works even where the console script is unavailable."""

from lecturelens.cli import app

if __name__ == "__main__":
    app()
