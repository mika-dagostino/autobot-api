import json

# Read the JSON file
with open(r'creds.json', 'r') as file:
    data = json.load(file)
    CSV_DIR = data["vehicle_csv_path"]
    UNIQUE_VALUES_DIR = data["unique_values_path"]
    VECTOR_DATABASE_DIR = data["vector_database_path"]
    PARAMS = data["normal_columns"]
    DEFAULT_PREF_DICT = data["default_pref_dict"]