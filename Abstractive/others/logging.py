from __future__ import absolute_import

import logging
import os

logger = logging.getLogger()


def init_logger(log_file=None, log_file_level=logging.NOTSET):
    log_path = '/'.join(log_file.split('/')[:-1]) + '/'
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    log_format = logging.Formatter("[%(asctime)s %(levelname)s] %(message)s")
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_format)
    logger.handlers = [console_handler]

    if log_file and log_file != '':
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(log_file_level)
        file_handler.setFormatter(log_format)
        logger.addHandler(file_handler)

    return logger


def get_console_logger(name):
    logFormatter = logging.Formatter(
        "%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s")
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        ch = logging.StreamHandler()
        ch.setFormatter(logFormatter)
        logger.addHandler(ch)
    return logger


def get_global_console_logger():
    return get_console_logger('global')