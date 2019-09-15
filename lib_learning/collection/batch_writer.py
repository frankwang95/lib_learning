import pandas as pd
import MySQLdb as sql
from sqlalchemy import create_engine
import threading
from retry import retry


SQL_RETRIES = 10
RETRY_DELAY = 5


class BatchWriter(object):
    """ Template is a dictionary which maps an attribute name to a column in the target SQL DB.
    """
    def __init__(self, logger, template, table_name, sql_parameters, batch_size=16):
        self.logger = logger
        self.template = template
        self.table_name = table_name
        self.sql_parameters = sql_parameters
        self.batch_size = batch_size
        self.primary_key = self.get_primary_key()

        self.push_lock = threading.Lock()

        self.work_queue = None
        self.num_items = None
        self.reset_work_queue()

        self.logger.info("initialized a batch SQL writer into table {} with primary key {}".format(
            self.table_name,
            self.primary_key
        ))


    def push(self, item):
        self.logger.info("batch writer recieved new item with primary key {}".format(getattr(item, self.primary_key)))

        with self.push_lock:
            new_entry = {self.template[k]: getattr(item, k) for k in self.template}
            self.work_queue = self.work_queue.append(new_entry, ignore_index=True)
            self.num_items += 1

            if self.num_items >= self.batch_size:
                self.unsafe_flush()


    def flush(self):
        with self.push_lock:
            self.unsafe_flush()


    def reset_work_queue(self):
        self.work_queue = pd.DataFrame(columns=self.template.values())
        self.num_items = 0


    def unsafe_flush(self):
        if self.work_queue.shape[0] > 0:
            self.logger.info("flushing work queue of {} items".format(self.work_queue.shape[0]))
            self.clear_existing_entries()
            self.write_new_entries()


    def get_primary_key(self):
        query = 'SHOW KEYS FROM {} WHERE Key_name="PRIMARY"'.format(self.table_name)
        self.logger.info('executing query {}'.format(query))
        db = sql.connect(**self.sql_parameters)
        cur = db.cursor()
        cur.execute(query)
        db.commit()
        result_set = cur.fetchall()
        cur.close()
        db.close()
        return result_set[0][4]


    @retry(Exception, tries=SQL_RETRIES, delay=RETRY_DELAY)
    def clear_existing_entries(self):
        if self.work_queue.shape[0] == 1:
            query = 'DELETE FROM {} WHERE {}="{}";'.format(
                self.table_name,
                self.primary_key,
                self.work_queue[self.primary_key][0]
            )
        else:
            query = 'DELETE FROM {} WHERE {} IN {};'.format(
                self.table_name,
                self.primary_key,
                tuple(self.work_queue[self.primary_key])
            )

        self.logger.info('executing query {}'.format(query))
        db = sql.connect(**self.sql_parameters)
        cur = db.cursor()
        cur.execute(query)
        db.commit()
        cur.close()
        db.close()


    @retry(Exception, tries=SQL_RETRIES, delay=RETRY_DELAY)
    def write_new_entries(self):
        engine = create_engine("mysql://{}:{}@{}:{}/{}?charset=utf8mb4".format(
            self.sql_parameters['user'],
            self.sql_parameters['passwd'],
            self.sql_parameters['host'],
            self.sql_parameters['port'],
            self.sql_parameters['db'],
        ), convert_unicode=True, encoding='utf-8')
        con = engine.connect()

        self.work_queue.drop_duplicates(subset=self.primary_key, keep='last', inplace=True)
        self.work_queue.to_sql(self.table_name, con=con, if_exists='append', index=False)
        self.reset_work_queue()
        con.close()
