import sqlite3
from datetime import datetime
import numpy as np
import pandas as pd


class SQLDb:
    """
    SQLite database handler for managing TSNE data visualization points.
    Provides methods for creating, storing, and retrieving 3D visualization data
    with associated metadata like original labels and classified labels.
    """

    def __init__(self, table_name="", db_path="manifolk.db", create_table=True):
        # The commented code below was likely a path validation check
        # if not Path(db_path).is_file():
        #     raise Exception(f"The Database path should be a full path of a file even if it doesnt exist. "
        #                     f"A new databse will be created at that path. Please pass full file path name")

        # Store the database file path
        self.db_path = db_path

        # Create a unique table name by appending current timestamp to avoid collisions
        datetime_now = datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
        self.table_name = f"[{table_name}_{datetime_now}]"

        # Initialize connection objects to None
        self.conn = None
        self.table_cursor = None

        # Optionally create the table immediately (default behavior)
        if create_table:
            self.create_table()

    def create_table(self):
        """
        Create a new SQLite table with the schema needed for storing TSNE visualization data.
        Schema includes:
        - epoch: training epoch/iteration number
        - X, Y, Z: 3D coordinates for visualization
        - DATAPOINT_NAME: unique identifier for each data point
        - ORIGINAL_LABEL: ground truth label
        - CLASSIFIED_AS_LABEL: model's predicted label
        """
        # Establish connection to the SQLite database
        self.conn = sqlite3.connect(self.db_path)
        self.table_cursor = self.conn.cursor()

        # Execute SQL to create the table with required columns
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
        # Commit the changes to the database
        self.conn.commit()

    def sanitize_input(self, *args):
        """
        Validate input types before inserting into the database.
        Ensures data integrity by checking that each value has the expected type.

        Args:
            *args: A sequence of values to validate in the following order:
                  epoch (int), x (float), y (float), z (float),
                  datapoint_name (str), original_label (str), classified_label (str)

        Raises:
            Exception: If any input value doesn't match its expected type
        """
        # Validate epoch is an integer
        epoch = args[0]
        if not isinstance(epoch, int):
            raise Exception(f"The input epoch is not an int value: {epoch}")

        # Validate X coordinate is a float
        x = args[1]
        if not (isinstance(x, float) or isinstance(x, np.float32)):
            raise Exception(f"The input x is not a float value: {x}")

        # Validate Y coordinate is a float
        y = args[2]
        if not (isinstance(y, float) or isinstance(y, np.float32)):
            raise Exception(f"The input y is not a float value: {y}")

        # Validate Z coordinate is a float
        z = args[3]
        if not (isinstance(z, float) or isinstance(z, np.float32)):
            raise Exception(f"The input z is not a float value: {z}")

        # Validate datapoint_name is a string
        datapoint_name = args[4]
        if not isinstance(datapoint_name, str):
            raise Exception(f"The input datapoint_name is not a string value: {datapoint_name}")

        # Validate original_label is a string
        original_label = args[5]
        if not isinstance(original_label, str):
            raise Exception(f"The input original_label is not a string value: {original_label}")

        # Validate classified_label is a string
        classified_label = args[6]
        if not isinstance(classified_label, str):
            raise Exception(f"The input classified_label is not a string value: {classified_label}")

    def insert_entry(self, *args):
        """
        Insert a single data point entry into the database.

        Args:
            *args: A sequence of values in the following order:
                  epoch (int), x (float), y (float), z (float),
                  datapoint_name (str), original_label (str), classified_label (str)
        """
        # Validate input types
        self.sanitize_input(*args)

        # Extract values from args
        epoch = args[0]
        x = args[1]
        y = args[2]
        z = args[3]
        datapoint_name = args[4]
        original_label = args[5]
        classified_label = args[6]

        # Execute SQL INSERT statement
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
        # Commit the transaction
        self.conn.commit()

    def insert(
        self, epoch: int, tsne_array: np.array, original_labels: list, predicted_labels: list, datapoint_ids: list
    ):
        """
        Insert multiple data points for a given epoch into the database.

        Args:
            epoch (int): The training epoch/iteration number
            tsne_array (np.array): Array of 3D points from TSNE dimensionality reduction
            original_labels (list): List of ground truth labels for each point
            predicted_labels (list): List of model-predicted labels for each point
            datapoint_ids (list): List of unique identifiers for each data point
        """
        # Iterate through all data points and their associated metadata
        for each_point, original_label, predicted_label, datapoint_id in zip(
            tsne_array, original_labels, predicted_labels, datapoint_ids
        ):
            # Unpack X, Y, Z coordinates from the point
            x, y, z = each_point
            # Insert each point as an individual entry
            self.insert_entry(epoch, x, y, z, datapoint_id, original_label, predicted_label)

    def get_all_table_names(self):
        """
        Retrieve all table names from the SQLite database.

        Returns:
            list: List of all table names in the database with square brackets
                 to handle special characters in table names
        """
        # Create a new connection to the database
        conn = sqlite3.connect(self.db_path)
        # Query the sqlite_master table to get all table names
        res = conn.execute("SELECT name FROM sqlite_master WHERE type='table';")
        all_table_names = []
        # Process results and format table names with square brackets
        for name in res:
            all_table_names.append(f"[{name[0]}]")
        # Close the connection
        conn.close()
        return all_table_names

    def get_pandas_frame(self, table_name):
        """
        Load a table from the database into a pandas DataFrame.

        Args:
            table_name (str): Name of the table to load

        Returns:
            pandas.DataFrame: DataFrame containing all data from the specified table
        """
        # Create a new connection to the database
        conn = sqlite3.connect(self.db_path)
        # Use pandas to read the SQL query result directly into a DataFrame
        df = pd.read_sql_query(f"SELECT * from {table_name}", conn)
        # Close the connection
        conn.close()
        return df

    def close_connection(self):
        """
        Close the database connection.
        Should be called when the database operations are complete.
        """
        self.conn.close()
