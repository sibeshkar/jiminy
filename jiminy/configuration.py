import logging
from jiminy.gym import configuration

jiminy_logger = logging.getLogger('jiminy')
jiminy_logger.setLevel(logging.INFO)

extra_logger = logging.getLogger('jiminy.extra')
extra_logger.setLevel(logging.INFO)

if hasattr(configuration, '_extra_loggers'):
    configuration._extra_loggers.append(jiminy_logger)
    configuration._extra_loggers.append(extra_logger)
