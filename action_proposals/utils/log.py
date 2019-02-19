import time
import sys
from functools import partial
import inspect

_DEBUG = '\033[92m[DEBUG]'
_INFO = '\033[94m[ INFO]'
_WARN = '\033[93m[ WARN]'
_ERROR = '\033[91m[ERROR]'

_HEADER = '\033[95m'
_BOLD = '\033[1m'
_UNDERLINE = '\033[4m'

_ENDC = '\033[0m'

levels = [_DEBUG, _INFO, _WARN, _ERROR, _HEADER, _BOLD, _UNDERLINE]
DEBUG, INFO, WARN, ERROR, HEADER, BOLD, UNDERLINE = range(len(levels))

log_level = INFO


def log(level, content):
    if level >= log_level:
        print("{} {} \033[4m ...{}\033[0m:\033[95m {} {}".format(levels[level], time.asctime(), inspect.getfile(sys._getframe(1))[-30:], content, _ENDC))


log_debug = partial(log, DEBUG)
log_info = partial(log, INFO)
log_warn = partial(log, WARN)
log_error = partial(log, ERROR)


if __name__ == '__main__':
    log_level = DEBUG

    log(DEBUG, "hello world")
    log(INFO, "hello world")
    log(WARN, "hello world")
    log(ERROR, "hello world")

    log(HEADER, "hello world")
    log(BOLD, "hello world")
    log(UNDERLINE, "hello world")
