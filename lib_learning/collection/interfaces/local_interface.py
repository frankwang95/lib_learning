import multiprocessing as mp
from queue import  Queue
from lib_learning.collection.interfaces.base_interface import Interface


class LocalInterface(Interface):
    """ An Interface implementation which interacts which supports concurrent schedulers and workers through a local
        interpreter interface.

        Realistically, this interface should only be run with a single scheduler and worker - distributed work is
        probably not benefitial given the python GIL.
    """
    def __init__(self, multiprocessing=False):
        if multiprocessing:
            self.work = mp.Queue()
        else:
            self.work = Queue()
        self.confirmations = {}


    def push_work(self, block):
        if block['_scheduler_name'] not in self.confirmations:
            self.confirmations[block['_scheduler_name']] = []

        self.work.put(block)


    def get_work(self):
        return self.work.get()


    def push_confirmation(self, block):
        self.confirmations[block['_scheduler_name']].append(block)


    def get_confirmation(self, scheduler_name):
        if scheduler_name in self.confirmations:
            confirmations = self.confirmations[scheduler_name]
            self.confirmations[scheduler_name] = []
            return confirmations

        return []
