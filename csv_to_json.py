import pandas as pd
import json

# Load your CSV file
csv_file = "data/raw/dft-road-casualty-statistics-collision-2023.csv"
df = pd.read_csv(csv_file, low_memory=False)

# Convert to JSON
json_file = "data/raw/collisions.json"
df.to_json(json_file, orient="records", lines=True)

print(f"✅ JSON file saved to: {json_file}")
