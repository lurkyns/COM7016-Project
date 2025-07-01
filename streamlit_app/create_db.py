#Just creates the table if it doesn’t exist. It doesn’t save any food data.

import sqlite3
import os

# Connect to the DB file (this will create it if it doesn't exist)
conn = sqlite3.connect("unknown_foods.db")
c = conn.cursor()

# Create the table
c.execute("""
    CREATE TABLE IF NOT EXISTS unknown_foods (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        food_name TEXT,
        first_name TEXT,
        last_name TEXT,
        email TEXT,
        date_added TEXT
    )
""")

conn.commit()
conn.close()

print("unknown_foods.db created!")

print(os.path.abspath("unknown_foods.db"))

