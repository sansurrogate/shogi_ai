import logging
import sys


def init_logger(log_level=logging.WARNING):
    logger = logging.getLogger()
    for hdlr in logger.handlers:
        if isinstance(hdlr, logging.StreamHandler):
            logger.removeHandler(hdlr)

    stream_handler = logging.StreamHandler(stream=sys.stderr)
    formatter = logging.Formatter(
        "[%(asctime)s][%(name)s][%(levelname)s] - %(message)s"
    )
    stream_handler.setFormatter(formatter)
    stream_handler.setLevel(log_level)
    logger.addHandler(stream_handler)
