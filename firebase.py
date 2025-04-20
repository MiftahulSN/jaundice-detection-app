import os
import json
import firebase_admin
from firebase_admin import credentials, db


cred = credentials.Certificate("input_your_credential_key_json_file_path_here")
firebase_admin.initialize_app(cred, {
    'databaseURL': 'input_your_db_url_here'
})

def upload_data(path, data):
    database = db.reference(path)
    # database.set(data)
    # database.push(data)
    print("Data uploaded successfully!")

def get_data(path):
    database = db.reference(path)
    data = database.get()
    print("Data fetched successfully!")
    return data


if __name__ == "__main__" :
    
    example_data = {
        "data_1": {
            "value": 100,
            "timestamp": "2025-04-16"
        },
        "data_2": {
            "value": 500,
            "timestamp": "2025-04-16"
        }
    }

    upload_data("examples", example_data)

    data = get_data("examples")
    print(data)