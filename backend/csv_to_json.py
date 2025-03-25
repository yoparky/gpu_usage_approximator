import csv
import json

CSV_FILE_PATH = "./data/raw_data/gpu_specs_v6.csv"
JSON_FILE_PATH = "./data/processed_data/test.json"

csvfile = open(CSV_FILE_PATH, 'r')
jsonfile = open(JSON_FILE_PATH, 'w')

def csv_to_json_file(csv_file_path, json_file_path):
    try:
        with open(csv_file_path, "r", encoding='utf-8-sig') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
    except Exception as e:
        print(f"Error opening csv file: {e}")
        return

    try:
        with open(json_file_path, 'w', encoding='utf-8-sig') as f:
            json.dump(rows, f)
    except Exception as e:
        print(f"Error writing json file: {e}")
        return

def json_file_to_string(json_file_path):
    try:
        with open(json_file_path, "r", encoding='utf-8-sig') as f:
            d = json.load(f)
            # print(d) # list of dicts
            print(type(d)) # list
            # print(d[1]) # dictionary entry of gpu
            # print(type(d[0])) # dict
            return d
    except Exception as e:
        print(f"Error reading json file: {e}")
        return

csv_to_json_file(CSV_FILE_PATH, JSON_FILE_PATH)
json_file_to_string(JSON_FILE_PATH)
