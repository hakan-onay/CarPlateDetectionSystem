import sqlite3
from datetime import datetime

class DatabaseManager:
    def __init__(self, db_name="LPR.db"):
        self.db_name = db_name
        self._create_table()

    def _connect(self):
        return sqlite3.connect(self.db_name)

    def _create_table(self):
        conn = self._connect()
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS Plates (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                plate TEXT UNIQUE,
                owner TEXT,
                date_time TEXT
                vehicle_type TEXT
            )
        """)
        conn.commit()
        conn.close()

    def insert_plate(self, plate_text, owner):
        conn = self._connect()
        cursor = conn.cursor()
        date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cursor.execute("INSERT INTO Plates (plate, owner, date_time) VALUES (?, ?, ?)", (plate_text, owner, date))
        conn.commit()
        conn.close()

    def get_owner(self, plate_text):
        conn = self._connect()
        cursor = conn.cursor()
        cursor.execute("SELECT owner FROM Plates WHERE plate = ?", (plate_text,))
        result = cursor.fetchone()
        conn.close()
        return result[0] if result else None
