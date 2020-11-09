"""
Assumption: The only assumption I would be doing is that the present working directory would be the root
folder of the project
"""
import unittest
from sqlite_db import SQLDb


class TestDB(unittest.TestCase):
    def test_create_db(self):
        sql_db = SQLDb("test")
        sql_db.insert_entry(1, 1.0, 2.0, 3.0, "datapoint1", "cat", "dog")
        sql_db.close_connection()

    def test_print_table_names(self):
        sql_db = SQLDb("test", create_table=False)
        print(sql_db.get_all_table_names())

    def test_table_to_dataframe(self):
        sql_db = SQLDb("test", create_table=False)
        # This is a very specific test
        # TODO: This should be modified to a more generic one
        df = sql_db.get_pandas_frame("[mnist_2020-11-08-22:59:17]")
        print(df)


if __name__ == '__main__':
    unittest.main()
