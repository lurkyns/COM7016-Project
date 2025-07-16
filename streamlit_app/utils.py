from pymongo import MongoClient
from datetime import datetime

def save_unknown_food_mongo(food_name, first_name, last_name, email):
    client = MongoClient("mongodb://localhost:27017/")
    db = client["diabeticAI"]
    collection = db["unknown_foods"]

    data = {
        "food_name": food_name,
        "first_name": first_name,
        "last_name": last_name,
        "email": email,
        "date_added": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

    result = collection.insert_one(data)
    print("Inserted document ID:", result.inserted_id)
