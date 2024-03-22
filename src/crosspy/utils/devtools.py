import logging

_mode = logging.DEBUG


def enable():
    global _mode
    _mode = logging.DEBUG


def disable():
    global _mode
    _mode = logging.NOTSET


def devtool(toolfunc):
    def noop(*_, **__):
        pass

    return noop if _mode == logging.NOTSET else toolfunc


@devtool
def get_logger(name=None, *, level=logging.WARNING, fmt=logging.BASIC_FORMAT):
    """
    logger = get_logger(
        __name__,
        level=_mode,
        fmt=
        '%(asctime)s [\033[1;4m%(levelname)s\033[0m %(processName)s:%(threadName)s] %(filename)s:%(lineno)s %(message)s'
    )
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter(fmt))
    logger.addHandler(ch)
    return logger


from threading import get_native_id


class MemSize:
    def __init__(self, size):
        self.size = size
        size = abs(size)
        self.B = size & ((1 << 10) - 1)
        size >>= 10
        self.KB = size & ((1 << 10) - 1)
        size >>= 10
        self.MB = size & ((1 << 10) - 1)
        size >>= 10
        self.GB = size


    def __repr__(self):
        return ('-' if self.size < 0 else '') + (' '.join((
            (f"{self.GB} GB",) if self.GB else ()) + (
            (f"{self.MB} MB",) if self.MB else ()) + (
            (f"{self.KB} KB",) if self.KB else ()) + (
            (f"{self.B} B",) if self.B else ())) or '0 B')
    

    def __sub__(self, other):
        return MemSize(self.size - other.size)


last_gmem = None


@devtool
def report_gpu_meminfo(message=""):
    from crosspy.device import gpu
    gmem = {g.id: MemSize(g.mem_info[1] - g.mem_info[0]) for g in gpu}
    print(message, gmem)
    global last_gmem
    if last_gmem is not None:
        print("\tDiff", {gid: gmem[gid] - last_gmem[gid] for gid in gmem})
    last_gmem = gmem
    return gmem
