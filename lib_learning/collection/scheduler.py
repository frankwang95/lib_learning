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
        scheduler_name <str>: A name for the scheduler. Schedulers sharing the same interface must have unique names.
        interface <Interface>: A interface object connecting this scheduler to a worker pool.
        block_generators <BlockGenerator>: A block generator object to generate work blocks from user inputs.
        logger <Logger>: A logger object to write scheduler logs to.
        task_timeout <int>: A positive integer designating the number of seconds to wait before to timing out unfinished
            work blocks.
        confirm_interval <int>: A positive integer designating the length of the pause in seconds that the scheduler
            waits between checking pending work blocks.
        tries <int>: A postiive integer designating the default number of times a work block will be retried upon
            failure. Work blocks can include a '_tries' key to overwrite this default value.
        default_retry_delay <int>: A positive integer designating at least how long to wait before initiating a retry.
            Work blocks can include a '_retry_delay' key to overwrite this default value.
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
        # store of failed blocks for potential retry
        self.failed_blocks = {}

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
                assert '_retrieval_datetime' not in block
                assert '_scheduler_name' not in block
                assert '_status' not in block
                assert '_finish_datetime' not in block
                retrieval_datetime = time.time()
                block = self.tag_block(block, retrieval_datetime)
                self.pending_work[retrieval_datetime] = block
                self.interface.push_work(block)

        return generated_work


    def tag_block(self, block, retrieval_time):
        block['_retrieval_datetime'] = retrieval_time
        block['_scheduler_name'] = self.scheduler_name
        block['_status'] = 'PENDING'
        block['_finish_datetime'] = None
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
                self.process_failures()


    def process_failures(self):
        deletions = []
        for old_rt, block in self.failed_blocks.items():
            if block['_tries'] <= 1:
                self.logger.exception('block {} failed permanantly'.format(old_rt))
                deletions.append(old_rt)
            elif time.time() - block['_finish_datetime'] > block['_retry_delay']:
                new_rt = time.time()
                block = self.tag_block(block, new_rt)
                block['_tries'] -= 1
                self.interface.push_work(block)
                self.logger.warning('retrying block {} under ts {} with {} tries remaining'.format(
                    old_rt, new_rt, block['_tries']
                ))
                deletions.append(old_rt)
        for rt in deletions:
            del self.failed_blocks[rt]


    def check_confirmations(self):
        confirmations = self.interface.get_confirmation(self.scheduler_name)
        self.logger.info('got {} new status updates'.format(len(confirmations)))

        for block in confirmations:
            rt = block['_retrieval_datetime']

            if block['_status'] == 'SUCCESS':
                self.logger.info('block {} computation succeeded'.format(rt))
            else:
                self.failed_blocks[rt] = block
                self.logger.exception('block {} failed with exception\n{}'.format(rt, block['_status']))

            if rt in self.pending_work:
                del self.pending_work[rt]


    def check_timeouts(self):
        failure_cutoff = time.time() - self.task_timeout
        timeouts = [ts for ts in self.pending_work if ts < failure_cutoff]
        for ts in timeouts:
            self.logger.exception('block {} timed out'.format(ts))
            self.failed_blocks[ts] = self.pending_work[ts]
            del self.pending_work[ts]
