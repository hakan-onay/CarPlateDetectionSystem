import sqlite3
from datetime import datetime


class DatabaseManager:
    def __init__(self, db_path="LPR.db"):
        self.db_path = db_path
        self._initialize_db()

    def _initialize_db(self):
        # Initialize the database with required tables.
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Create Plates table if not exists
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS Plates (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                plate TEXT UNIQUE NOT NULL,
                owner TEXT,
                vehicle_type TEXT,
                date_time TEXT
            )
        """)

        conn.commit()
        conn.close()

    def insert_plate(self, plate, owner="Unknown", vehicle_type="Unknown"):
        # Insert a new plate into the database.
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        try:
            cursor.execute("""
                INSERT INTO Plates (plate, owner, vehicle_type, date_time)
                VALUES (?, ?, ?, ?)
            """, (plate, owner, vehicle_type, datetime.now().strftime("%Y-%m-%d %H:%M:%S")))

            conn.commit()
            return True
        except sqlite3.IntegrityError:
            return False
        finally:
            conn.close()

    def get_owner(self, plate):
        # Get owner and vehicle type for a given plate.
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            SELECT owner, vehicle_type FROM Plates WHERE plate = ?
        """, (plate,))

        result = cursor.fetchone()
        conn.close()

        if result:
            return result[0], result[1]
        return None, None

    def get_all_plates(self):
        # Get all plates from the database.
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("SELECT * FROM Plates ORDER BY id DESC")
        plates = cursor.fetchall()

        conn.close()
        return plates