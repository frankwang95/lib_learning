import abc

class BaseMonitor(object):
    def __init__(self, values_init, update_interval):
        self.values = values_init
        self.update_interval = update_interval
        self.network = None

    def link_to_network(self, network):
        self.network = network
        return self

    @abc.abstractmethod
    def evaluate(self):
        pass

    def check_update(self):
        if self.update_interval is not None:
            return self.network.epochs % self.update_interval == 0
