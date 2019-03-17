from datetime import datetime
import MySQLdb as sql


class SQLReverter(object):
    def __init__(self, sql_params, tables):
        self.sql_params = sql_params
        self.tables = tables


    def revert_fn(self, block):
        removal_cutoff = datetime.utcfromtimestamp(int(block['_retrieval_datetime']))

        db = sql.connect(**self.sql_params)
        cur = db.cursor()

        for table in self.tables:
            cur.execute('DELETE FROM {} WHERE retrieval_datetime>="{}"'.format(table, removal_cutoff))

        db.commit()
        cur.close()
        db.close()
