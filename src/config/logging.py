import logging
from rich.logging import RichHandler


def create_logger(name: str) -> logging.Logger:
    FORMAT = "%(levelname)s %(asctime)s [%(funcName)s] %(message)s"
    handler = RichHandler(level=logging.INFO, rich_tracebacks=True)
    formatter = logging.Formatter(FORMAT, datefmt="[%X]")
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.addHandler(handler)
    logger.propagate = False
    return logger
