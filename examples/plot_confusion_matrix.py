"""
Print accuracy values of all labels in the dataframe
Use this script, only if you have one epoch, otherwise it will screw up accuracy values
"""
import pandas as pd
from sqlite_db import SQLDb

if __name__ == "__main__":
    sql_db = SQLDb("test", create_table=False)
    df = sql_db.get_pandas_frame(sql_db.get_all_table_names()[0])
    pd.set_option('display.max_rows', None)
    print(df['ORIGINAL_LABEL'].eq(df['CLASSIFIED_AS_LABEL']).groupby(df['ORIGINAL_LABEL']).mean())
