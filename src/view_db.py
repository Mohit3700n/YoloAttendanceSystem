import sqlite3
import pandas as pd

# Database connect karo
conn = sqlite3.connect('../database/attendance.db')

print("--- REGISTERED USERS ---")
# Embedding BLOB column ko hide kiya hai warna terminal bhar jayega
print(pd.read_sql_query("SELECT id, name FROM users", conn))

print("\n--- ATTENDANCE LOGS ---")
print(pd.read_sql_query("SELECT * FROM attendance_logs", conn))

conn.close()