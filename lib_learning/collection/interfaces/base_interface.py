import abc


class Interface(object):
    """ A definition class for a two-way storage queue which connects work schedulers to workers. Schedulers push work
        to the interface and get confirmations that work is completed. Workers

        Implementations of this class must insure that all four instances of methods given below are thread safe as
        the intention is for a single interface to support multiple scheulders and workers simultaneously.

        Examples of Implementations Models:
        - local/remote, single/pool schedulers
        - local/remote workers, single/pool workers
        - in-interpreter/restAPI/pubsub interfaces
    """


    @abc.abstractmethod
    def push_work(self, block):
        raise NotImplementedError()


    @abc.abstractmethod
    def get_work(self):
        raise NotImplementedError()


    @abc.abstractmethod
    def push_confirmation(self, block):
        raise NotImplementedError()


    @abc.abstractmethod
    def get_confirmation(self, scheduler_name):
        raise NotImplementedError()
