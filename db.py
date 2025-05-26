import sqlite3

class DbManagement:

    def __init__(self, db_name: str="test.db"):
        self.db_name = db_name
        self.connection = self.start_connection()
        self.cursor = self.connection.cursor()

    def start_connection(self):
        return sqlite3.connect(self.db_name)

    def execute_query(self, query: str = ""):
        if query == "":
            return None
        self.cursor.execute(query)
        self.connection.commit()


    def close_connection(self):
        self.connection.close()
