from pymongo import MongoClient
from datetime import datetime

# Connect to local MongoDB
client = MongoClient("mongodb://localhost:27017/")

# Create a database
db = client["diabeticAI"]

# Create a collection (this is like a table)
collection = db["unknown_foods"]

# Sample data
data = {
    "food_name": "Bubble tea",
    "first_name": "Osa",
    "last_name": "Arthur",
    "email": "osa@example.com",
    "date_added": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
}

# Insert document
result = collection.insert_one(data)
print("\n TEST COMPLETE! Inserted document ID:", result.inserted_id, "\n")
