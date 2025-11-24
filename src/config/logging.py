import logging
from rich.logging import RichHandler


def setup_logging() -> None:
    FORMAT = "%(levelname)s %(asctime)s [%(funcName)s] %(message)s"
    logging.basicConfig(
        level="INFO", format=FORMAT, datefmt="[%X]", handlers=[RichHandler()]
    )
