

import time
import yaml
import threading
from queue import PriorityQueue


class BlockedError(Exception):
    pass


class Scheduler(object):
    """ An implementation of an independent Task Scheduler which manages batch processing tasks. Schedulers communicate
        with workers (.workers.base_worker.Worker) via an interface (.interfaces.base_interface.Interface).

    Inputs:
        scheduler_name
        interface
        block_generators
        logger
        task_timeout
        confirm_interval
    """
    def __init__(
        self, scheduler_name, interface, block_generator, logger, task_timeout=600, confirm_interval=20,
        tries=1, default_retry_delay=60
    ):
        self.scheduler_name = scheduler_name
        self.interface = interface
        self.block_generator = block_generator
        self.logger = logger
        self.tries = tries
        self.default_retry_delay = default_retry_delay
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

        with self.global_lock:
            generated_work = self.block_generator.get_next(**kwargs)

            if not isinstance(generated_work, list):
                generated_work = [generated_work]

            for block in generated_work:
                retrieval_datetime = time.time()
                block = self.tag_block(block, retrieval_datetime)
                self.pending_work[retrieval_datetime] = block
                self.interface.push_work(block)

        return generated_work


    def tag_block(self, block, retrieval_time):
        assert '_retrieval_datetime' not in block
        assert '_scheduler_name' not in block
        assert '_status' not in block
        assert '_finish_datetime' not in block
        block['_retrieval_datetime'] = retrieval_time
        block['_scheduler_name'] = self.scheduler_name
        block['_status'] = 'PENDING'
        block['finish_datetime'] = None
        if '_tries' not in block:
            block['_tries'] = self.tries
        if '_retry_delay' not in block:
            block['_retry_delay'] = self.default_retry_delay
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
            elif block['_tries'] > 1:
                if time.time() - block['_finish_datetime'] > block['_retry_delay']:
                    block['_status'] = 'PENDING'
                    block['_finish_datetime'] = None
                    block['_tries'] -= 1
                    self.interface.push_work(block)
                    self.logger.warning('block {} failed with {} tries remaining'.format(rt, block['_tries']))
            else:
                self.logger.exception('block {} failed permanantly with exception\n{}'.format(rt, block['_status']))

            if rt in self.pending_work: # handles case where job timeouts but then later is successful
                del self.pending_work[rt]


    def check_timeouts(self):
        failure_cutoff = time.time() - self.task_timeout
        timeouts = [ts for ts in self.pending_work if ts < failure_cutoff]
        for ts in timeouts:
            self.logger.exception('block {} timed out'.format(ts))
            del self.pending_work[ts]
