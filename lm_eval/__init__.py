import logging
import os

from .evaluator import evaluate, simple_evaluate


def setup_logging(verbosity=logging.INFO):
    # Configure the root logger
    log_level = os.environ.get("LOGLEVEL", verbosity) or verbosity

    level_map = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
        "CRITICAL": logging.CRITICAL,
    }

    log_level = level_map.get(str(log_level).upper(), logging.INFO)
    if not logging.root.handlers:
        logging.basicConfig(
            format="%(asctime)s,%(msecs)03d %(levelname)-8s [%(name)s:%(lineno)d] %(message)s",
            datefmt="%Y-%m-%d:%H:%M:%S",
            level=log_level,
        )
        if log_level == logging.DEBUG:
            third_party_loggers = ["urllib3", "filelock", "fsspec"]
            for logger_name in third_party_loggers:
                logging.getLogger(logger_name).setLevel(logging.INFO)
    else:
        logging.getLogger().setLevel(log_level)
    # Prevent logging from propagating to the root logger multiple times
    # logging.getLogger(__name__).propagate = False


setup_logging()
