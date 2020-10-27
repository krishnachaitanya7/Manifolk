"""
Assumption: The only assumption I would be doing is that the present working directory would be the root
folder of the project
"""
import unittest
from sqlite_db import SQLDb


class TestDB(unittest.TestCase):
    def test_create_db(self):
        sql_db = SQLDb("test")
        sql_db.create_table()
        sql_db.insert_entry(1, 1.0, 2.0, 3.0, "datapoint1", "cat", "dog")
        sql_db.close_connection()


if __name__ == '__main__':
    unittest.main()
