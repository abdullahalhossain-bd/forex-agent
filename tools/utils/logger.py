import logging
def get_logger(name):
    logger = logging.getLogger(name)
    if not logger.handlers:
        logging.basicConfig(level=logging.WARNING)
    return logger
