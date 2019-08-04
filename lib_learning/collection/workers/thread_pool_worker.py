import yaml
import traceback
from threading import Thread

from lib_learning.collection.workers.base_worker import Worker


class ThreadPoolWorker(Worker):
    """ Basic implementation of a generic worker which is intended to be used with the lib_learning.collection
        work scheduler paradigm. The worker recieves work parameters in the form of a dictionary from an interface,
        attempts to do perform the paramterized work, and then reports the result of the work back to the interface.

        Details on the paradigm can be found in the documentation at ib_learning/collections/README.md.

    Inputs:
        interface <lib_learning.collection.interfaces.base_interface.Interface>: A lib_learning.collection Interface
            object from which the worker will recieve
        n_threads <int>
        do_fn <function>: A function mapping a work block in the form of a python dictionary to some work done.
            Thread safety of the do_fn must be managed by the implementor according to their own needs.
        loggers <list: <logging.Logger>
    """
    def __init__(self, interface, n_threads, do_fn, loggers):
        assert len(loggers) == n_threads

        self.n_threads = n_threads
        self.interface = interface
        self.do_fn = do_fn
        self.loggers = loggers
        self.workers = self.create_workers()


    def create_workers(self):
        workers = []
        for i in range(self.n_threads):
            workers.append(Worker(self.interface, self.do_fn, self.loggers[i]))
        return workers


    def start(self):
        for w in self.workers:
            w.start()
