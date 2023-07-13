import logging

def _get_logger(name=None, *, level=logging.WARNING, fmt=logging.BASIC_FORMAT):
    logger = logging.getLogger(name)
    logger.setLevel(level)
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter(fmt))
    logger.addHandler(ch)
    return logger


logger = _get_logger(
    __name__,
    # level=logging.DEBUG,
    fmt=
    '%(asctime)s [\033[1;4m%(levelname)s\033[0m %(processName)s:%(threadName)s] %(filename)s:%(lineno)s %(message)s'
)