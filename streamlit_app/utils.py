# Utils.py alone does ot create new database file unless someone uses the app. That’s why create_db.py is useful.
import sqlite3
from datetime import datetime

def save_unknown_food(food_name, first_name, last_name, email):
    print("SAVING FOOD:", food_name, first_name, last_name, email)
    conn = sqlite3.connect(r"C:\Users\lurky\OneDrive\Escritorio\COM7016-Project\unknown_foods.db")

    c = conn.cursor()
    
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
    
    date_added = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    c.execute("""
        INSERT INTO unknown_foods (food_name, first_name, last_name, email, date_added)
        VALUES (?, ?, ?, ?, ?)
    """, (food_name, first_name, last_name, email, date_added))
    
    conn.commit()
    conn.close()
print("unknown_foods.db created!")