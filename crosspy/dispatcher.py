class SerialDispatcher:
    def __init__(self, id, *args, **kwargs) -> None:
        self.id = id

    def __call__(self, func, *args, **kwds):
        func()
        return self.id
    
    
class DummyTaskSpace:
    def __init__(self, name):
        self.name = name
    
    def __getitem__(self, index):
        return index
    
    def __len__(self):
        return id(self)

    def wait(self):
        pass

_dispatcher = SerialDispatcher
_taskspace = DummyTaskSpace("")
_foreground_tasks = []
_background_tasks = []

def set(sched, ts=None):
    global _dispatcher
    global _taskspace
    _dispatcher = sched
    if ts is not None:
        _taskspace = ts

def barrier(non_blocking=False):
    global _background_tasks, _foreground_tasks
    if non_blocking:
        _foreground_tasks, _background_tasks = [], _foreground_tasks
    else:
        _taskspace.wait()
        _foreground_tasks, _background_tasks = [], []

def launch(func, id=None, **kwargs):
    if id is None: id = len(_taskspace)
    _foreground_tasks.append(id)
    return _dispatcher(_taskspace[id], dependencies=[_taskspace[t] for t in _background_tasks], **kwargs)(func)