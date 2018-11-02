import logging

def logger(module, level='info'):
    """

    :param module: module name
    :param level:  logging level , check LEVELS variable for available options
    :return: logger object
    """
    formatter = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    LEVELS = {'debug': logging.DEBUG,
              'info': logging.INFO,
              'warning': logging.WARNING,
              'error': logging.ERROR,
              'critical': logging.CRITICAL}
    level_name = LEVELS.get(level, logging.NOTSET)
    logging.basicConfig(format=formatter, level=level_name)
    log = logging.getLogger(module)
    return log