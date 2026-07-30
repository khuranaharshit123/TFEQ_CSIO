import os
import json
from bson import decode_file_iter

def bson_to_json(bson_file_path, json_file_path=None):
    """
    Converts a BSON file to a JSON file.

    Parameters:
    bson_file_path (str): Path to the input BSON file.
    json_file_path (str): Path to save the output JSON file.
                           If None, saves in the same folder with .json extension.
    """

    # Check if file exists
    if not os.path.exists(bson_file_path):
        print(f"Error: File not found:\n{bson_file_path}")
        return

    # Generate output filename if not provided
    if json_file_path is None:
        json_file_path = os.path.splitext(bson_file_path)[0] + ".json"

    documents = []

    try:
        # Read BSON documents
        with open(bson_file_path, "rb") as bson_file:
            for document in decode_file_iter(bson_file):
                documents.append(document)

        # Save as JSON
        with open(json_file_path, "w", encoding="utf-8") as json_file:
            json.dump(documents, json_file, indent=4, default=str)

        print(f"Conversion completed successfully!")
        print(f"JSON saved at:\n{json_file_path}")
        print(f"Total documents converted: {len(documents)}")

    except Exception as e:
        print("Error while converting BSON to JSON:")
        print(e)


# ==========================
# GIVE YOUR BSON FILE PATH HERE
# ==========================
bson_path = r"D:\harshit\bson_file\accelerogram.bson"

# Optional: specify output JSON path
json_path = None    
# Example:
# json_path = r"C:\Users\Harshit\Desktop\output.json"

bson_to_json(bson_path, json_path)