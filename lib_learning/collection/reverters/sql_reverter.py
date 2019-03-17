from datetime import datetime
import MySQLdb as sql


class SQLReverter(object):
    def __init__(self, sql_params):
        self.sql_params = sql_params


    def revert_fn(self, block, tables):
        removal_cutoff = datetime.utcfromtimestamp(int(block['_retrieval_datetime']))

        db = sql.connect(**self.sql_params)
        cur = db.cursor()

        for table in tables:
            cur.execute('DELETE FROM {} WHERE retrieval_datetime>="{}"'.format(table, removal_cutoff))

        db.commit()
        cur.close()
        db.close()
