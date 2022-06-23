import logging

logging.basicConfig(
    format="[%(levelname)s][%(asctime)s]:%(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger()
