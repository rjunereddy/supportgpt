import csv
import json
import os
import random

CSV_FILES = [
    'dataset-tickets-multi-lang-4-20k.csv',
    'aa_dataset-tickets-multi-lang-5-2-50-version.csv'
]

OUTPUT_JSON = 'data/raw/support_tickets.json'
SAMPLE_SIZE = 500 # Using a sample so the local embedding doesn't take hours

def convert():
    tickets = []
    ticket_id = 1
    
    for file in CSV_FILES:
        if not os.path.exists(file):
            print(f"Skipping {file}, not found.")
            continue
            
        with open(file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Handle different CSV formats (one has 'version' column, one doesn't)
                product = row.get('queue', 'General Support')
                category = row.get('type', 'Inquiry')
                subject = row.get('subject', '')
                body = row.get('body', '')
                answer = row.get('answer', '')
                
                tickets.append({
                    "id": f"T-CSV-{ticket_id}",
                    "product": product,
                    "issue": f"{subject}\n\n{body}",
                    "resolution": answer,
                    "category": category
                })
                ticket_id += 1

    print(f"Total tickets found in CSVs: {len(tickets)}")
    
    os.makedirs(os.path.dirname(OUTPUT_JSON), exist_ok=True)
    with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
        json.dump(tickets, f, indent=4)
        
    print(f"Successfully wrote {len(tickets)} tickets to {OUTPUT_JSON}")

if __name__ == "__main__":
    convert()
