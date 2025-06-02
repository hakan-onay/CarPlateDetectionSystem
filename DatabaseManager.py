import sqlite3
from datetime import datetime


class DatabaseManager:
    def __init__(self, db_name="LPR.db"):
        self.db_name = db_name
        self._create_table()

    def _connect(self):
        return sqlite3.connect(self.db_name)

    def _create_table(self):
        # Create table if not exists
        conn = self._connect()
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS Plates (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                plate TEXT UNIQUE,
                owner TEXT,
                vehicle_type TEXT,
                date_time TEXT
            )
        """)
        conn.commit()
        conn.close()

    def insert_plate(self, plate_text, owner, vehicle_type=None):
        """Insert new plate with proper error handling"""
        try:
            conn = self._connect()
            cursor = conn.cursor()
            date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            # Check if plate already exists
            cursor.execute("SELECT 1 FROM Plates WHERE plate = ?", (plate_text,))
            if cursor.fetchone():
                # Update existing record
                cursor.execute("""
                    UPDATE Plates 
                    SET owner = ?, vehicle_type = ?, date_time = ?
                    WHERE plate = ?
                """, (owner, vehicle_type, date, plate_text))
            else:
                # Insert new record
                cursor.execute("""
                    INSERT INTO Plates (plate, owner, vehicle_type, date_time) 
                    VALUES (?, ?, ?, ?)
                """, (plate_text, owner, vehicle_type, date))

            conn.commit()
        except sqlite3.Error as e:
            print(f"Database error: {e}")
            raise
        finally:
            conn.close()

    def get_owner(self, plate_text):
        """Get owner info with better error handling"""
        try:
            conn = self._connect()
            cursor = conn.cursor()
            cursor.execute("""
                SELECT owner, vehicle_type FROM Plates WHERE plate = ?
            """, (plate_text,))
            result = cursor.fetchone()
            return result if result else (None, None)
        except sqlite3.Error as e:
            print(f"Database error: {e}")
            return (None, None)
        finally:
            conn.close()