import sqlite3
import pickle
import os

class DBManager:
    def __init__(self, db_path="../database/attendance.db"):
        # Database folder exist nahi karta toh banayega
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.conn = sqlite3.connect(db_path)
        self.cursor = self.conn.cursor()
        self.create_tables()

    def create_tables(self):
        # Face Embeddings store karne ke liye table
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT UNIQUE NOT NULL,
                embedding BLOB NOT NULL
            )
        ''')
        # Daily Attendance logs ke liye table
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS attendance_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY(user_id) REFERENCES users(id)
            )
        ''')
        self.conn.commit()

    def register_user(self, name, embedding):
        try:
            # 512-D numpy array ko binary string me convert karna
            emb_bytes = pickle.dumps(embedding)
            self.cursor.execute("INSERT INTO users (name, embedding) VALUES (?, ?)", (name, emb_bytes))
            self.conn.commit()
            print(f"Success: '{name}' ka face fingerprint database me save ho gaya.")
            return True
        except sqlite3.IntegrityError:
            print(f"Error: '{name}' already registered hai.")
            return False

    def load_all_users(self):
        self.cursor.execute("SELECT id, name, embedding FROM users")
        rows = self.cursor.fetchall()
        users_data = []
        for row in rows:
            user_id, name, emb_bytes = row
            embedding = pickle.loads(emb_bytes) # Binary back to Numpy Array
            users_data.append({"id": user_id, "name": name, "embedding": embedding})
        return users_data