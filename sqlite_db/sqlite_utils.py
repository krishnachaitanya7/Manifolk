import sqlite3
from datetime import datetime
import numpy as np
import pandas as pd


class SQLDb:
    def __init__(self, table_name="", db_path="manifolk.db", create_table=True):
        # if not Path(db_path).is_file():
        #     raise Exception(f"The Database path should be a full path of a file even if it doesnt exist. "
        #                     f"A new databse will be created at that path. Please pass full file path name")
        self.db_path = db_path
        datetime_now = datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
        self.table_name = f"[{table_name}_{datetime_now}]"
        self.conn = None
        self.table_cursor = None
        if create_table:
            self.create_table()

    def create_table(self):
        self.conn = sqlite3.connect(self.db_path)
        self.table_cursor = self.conn.cursor()
        self.table_cursor.execute(
            f"CREATE TABLE "
            f"{self.table_name} "
            f"(epoch int, "
            f"X real, "
            f"Y real, "
            f"Z real, "
            f"DATAPOINT_NAME text, "
            f"ORIGINAL_LABEL text, "
            f"CLASSIFIED_AS_LABEL text)"
        )
        self.conn.commit()

    def sanitize_input(self, *args):
        epoch = args[0]
        if not isinstance(epoch, int):
            raise Exception(f"The input x is not a int value: {epoch}")
        x = args[1]
        if not (isinstance(x, float) or isinstance(x, np.float32)):
            raise Exception(f"The input x is not a float value: {x}")
        y = args[2]
        if not (isinstance(y, float) or isinstance(y, np.float32)):
            raise Exception(f"The input x is not a float value: {y}")
        z = args[3]
        if not (isinstance(z, float) or isinstance(z, np.float32)):
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
        self.table_cursor.execute(
            f"""INSERT INTO {self.table_name} VALUES 
                                    ("{epoch}",
                                     "{x}", 
                                     "{y}", 
                                     "{z}", 
                                     "{datapoint_name}", 
                                     "{original_label}", 
                                     "{classified_label}")"""
        )
        self.conn.commit()

    def insert(
        self, epoch: int, tsne_array: np.array, original_labels: list, predicted_labels: list, datapoint_ids: list
    ):
        for each_point, original_label, predicted_label, datapoint_id in zip(
            tsne_array, original_labels, predicted_labels, datapoint_ids
        ):
            x, y, z = each_point
            self.insert_entry(epoch, x, y, z, datapoint_id, original_label, predicted_label)

    def get_all_table_names(self):
        conn = sqlite3.connect(self.db_path)
        res = conn.execute("SELECT name FROM sqlite_master WHERE type='table';")
        all_table_names = []
        for name in res:
            all_table_names.append(f"[{name[0]}]")
        conn.close()
        return all_table_names

    def get_pandas_frame(self, table_name):
        conn = sqlite3.connect(self.db_path)
        df = pd.read_sql_query(f"SELECT * from {table_name}", conn)
        conn.close()
        return df

    def close_connection(self):
        self.conn.close()
