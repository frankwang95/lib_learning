from multiprocessing import Manager
from lib_learning.collection.interfaces.local_interface import LocalInterface


class MPInterface(LocalInterface):
    """ An Interface implementation which interacts which supports concurrent schedulers and workers through a local
        interpreter interface.

        Realistically, this interface should only be run with a single scheduler and worker - distributed work is
        probably not benefitial given the python GIL.
    """
    def __init__(self):
        self.manager = Manager()
        self.work = self.manager.Queue()
        self.confirmations = self.manager.dict()
