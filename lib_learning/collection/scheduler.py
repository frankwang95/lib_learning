import time
import yaml
import threading
from queue import PriorityQueue


class BlockedError(Exception):
    pass


class Scheduler(object):
    """ Independent Task Scheduler
    """
    def __init__(
        self, scheduler_name, interface, block_generator, logger, revert_fn=None, blocking=False,
        task_timeout=600, confirm_interval=20
    ):
        if revert_fn is None:
            def identity(x):
                return x
            revert_fn = identity

        self.scheduler_name = scheduler_name
        self.interface = interface
        self.block_generator = block_generator
        self.revert_fn = revert_fn
        self.blocking = blocking
        self.logger = logger
        self.task_timeout = task_timeout
        self.confirm_interval = confirm_interval

        # A lock to prevent interface work pushes during confirmations
        self.global_lock = threading.Lock()
        # maps retrieval_time to iter reconstruction params
        self.pending_work = {}

        thread = threading.Thread(target=self.pending_work_checker)
        thread.setDaemon(True)
        thread.start()


    def push_next_block(self, **kwargs):
        self.logger.info('recieved new work request with parameters {}'.format(
            yaml.dump(kwargs, default_flow_style=False)
        ))
        if self.blocking and len(self.pending_work) > 0:
            raise BlockedError('previous work block not yet completed in a blocking scheduler')

        with self.global_lock:
            retrieval_datetime = time.time()
            next_block = self.block_generator.get_next(**kwargs)
            next_block = self.tag_block(next_block, retrieval_datetime)
            self.pending_work[retrieval_datetime] = next_block
            self.interface.push_work(next_block)

        return next_block


    def tag_block(self, block, retrieval_time):
        assert '_retrieval_datetime' not in block
        assert '_scheduler_name' not in block
        assert '_status' not in block
        block['_retrieval_datetime'] = retrieval_time
        block['_scheduler_name'] = self.scheduler_name
        block['_status'] = 'PENDING'
        return block


    def pending_work_checker(self):
        while True:
            time.sleep(self.confirm_interval)

            with self.global_lock:
                self.logger.info('checking for work status updates...')
                self.check_confirmations()
                self.check_timeouts()


    def check_confirmations(self):
        confirmations = self.interface.get_confirmation(self.scheduler_name)
        self.logger.info('got {} new status updates'.format(len(confirmations)))

        for block in confirmations:
            rt = block['_retrieval_datetime']

            if block['_status'] == 'SUCCESS':
                self.logger.info('block {} computation succeeded'.format(rt))
                self.block_generator.update(block)

            else:
                self.logger.exception('block {} failed with exception\n{}'.format(rt, block['_status']))
                self.revert_fn(block)
                self.block_generator.reset(block)

            del self.pending_work[rt]


    def check_timeouts(self):
        failure_cutoff = time.time() - self.task_timeout
        timeouts = [ts for ts in self.pending_work if ts < failure_cutoff]
        for ts in timeouts:
            self.logger.exception('block {} timed out'.format(ts))
            self.revert_fn(self.pending_work[ts])
            self.block_generator.reset(self.pending_work[ts])
            del self.pending_work[ts]
