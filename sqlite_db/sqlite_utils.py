from pathlib import Path
import sqlite3
from datetime import datetime


class SQLDb:
    def __init__(self, table_name, db_path="manifolk.db"):
        if not Path(db_path).is_file():
            raise Exception(f"The Database path should be a full path of a file even if it doesnt exist. "
                            f"A new databse will be created at that path. Please pass full file path name")
        self.db_path = db_path
        datetime_now = datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
        self.table_name = f"[{table_name}_{datetime_now}]"
        self.conn = None
        self.table_cursor = None

    def create_table(self):
        self.conn = sqlite3.connect(self.db_path)
        self.table_cursor = self.conn.cursor()
        self.table_cursor.execute(f'CREATE TABLE '
                                  f'{self.table_name} '
                                  f'(epoch int, '
                                  f'X real, '
                                  f'Y real, '
                                  f'Z real, '
                                  f'DATAPOINT_NAME text, '
                                  f'ORIGINAL_LABEL text, '
                                  f'CLASSIFIED_AS_LABEL text)')
        self.conn.commit()

    def sanitize_input(self, *args):
        epoch = args[0]
        if not isinstance(epoch, int):
            raise Exception(f"The input x is not a int value: {epoch}")
        x = args[1]
        if not isinstance(x, float):
            raise Exception(f"The input x is not a float value: {x}")
        y = args[2]
        if not isinstance(y, float):
            raise Exception(f"The input x is not a float value: {y}")
        z = args[3]
        if not isinstance(z, float):
            raise Exception(f"The input x is not a float value: {z}")
        datapoint_name = args[4]
        if not isinstance(datapoint_name, str):
            raise Exception(f"The input x is not a string value: {datapoint_name}")
        original_label = args[5]
        if not isinstance(original_label, str):
            raise Exception(f"The input x is not a string value: {original_label}")
        classified_label = args[6]
        if not isinstance(classified_label, str):
            raise Exception(f"The input x is not a string value: {classified_label}")

    def insert_entry(self, *args):
        self.sanitize_input(*args)
        epoch = args[0]
        x = args[1]
        y = args[2]
        z = args[3]
        datapoint_name = args[4]
        original_label = args[5]
        classified_label = args[6]
        self.table_cursor.execute(f"""INSERT INTO {self.table_name} VALUES 
                                    ("{epoch}",
                                     "{x}", 
                                     "{y}", 
                                     "{z}", 
                                     "{datapoint_name}", 
                                     "{original_label}", 
                                     "{classified_label}")""")
        self.conn.commit()

    def close_connection(self):
        self.conn.close()
