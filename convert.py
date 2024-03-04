import csv
import json

# Load the JSON file
with open('train.json', 'r') as json_file:
    data = json.load(json_file)

# Prepare CSV file
with open('train.csv', 'w', newline='', encoding='utf-8') as csv_file:
    writer = csv.writer(csv_file, quoting=csv.QUOTE_NONNUMERIC)

    # Write each entry to the CSV file
    for entry in data:
        review = entry['reviews'].replace('\n', '\\n')
        sentiment = entry['sentiments']
        writer.writerow([review, sentiment])
