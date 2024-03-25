import pandas as pd
import json

def convert_csv_to_jsonl(csv_file_path, output_directory, category_labels):
    # Read the CSV file into a DataFrame
    df = pd.read_csv(csv_file_path)

    # Define the function to generate JSONL data
    def generate_jsonl_for_category(df, category_name, options):
        jsonl_data = []

        for _, row in df.iterrows():
            # Find the index of the answer in the options list
            answer_idx = options.index(row[category_name]) if row[category_name] in options else None
            # Correspond the options with letters
            option_dict = {chr(65 + i): opt for i, opt in enumerate(options)}
            # Skip if the answer is not found in the options (to avoid errors)
            if answer_idx is None:
                continue
            json_object = {
                "report": row["Original_Report"],
                "answer": row[category_name],
                "options": option_dict,
                "answer_idx": chr(65 + answer_idx)  # Convert index to letter (A, B, C, etc.)
            }
            jsonl_data.append(json.dumps(json_object))

        return jsonl_data

    # Generate JSONL content for each category and write to files
    output_file_paths = {}
    for category, labels in category_labels.items():
        jsonl_data = generate_jsonl_for_category(df, category, labels)
        jsonl_file_name = f"{category.lower().replace(' ', '_')}_data.jsonl"
        output_file_path = f'{output_directory}/{jsonl_file_name}'
        output_file_paths[category] = output_file_path
        with open(output_file_path, 'w') as file:
            for item in jsonl_data:
                file.write(item + "\n")

    return output_file_paths

# Usage
category_labels = {
    "Modality": ['BIO-MG', 'BIO-MG-MRI', 'BIO-MG-US', 'BIO-MRI', 'BIO-US', 'MG', 'MG-MRI', 'MG-US', 'MRI', 'Other', 'US'],
    "PreviousCa": ['Negative', 'Positive', 'Suspicious'],
    "Density": ['<= 75%', 'Dense', 'Fatty', 'Heterogeneous', 'Not Stated', 'Scattered'],
    "Purpose": ['Diagnostic', 'Screening', 'Unknown'],
    "BPE": ['Marked', 'Mild', 'Minimal', 'Moderate', 'Not Stated'],
    "Menopausal Status": ['Not Stated', 'Post-Menopausal', 'Pre-Menopausal'],
    # BI-RADS is omitted as per instruction
}

# Call the function with the path to the CSV file and the output directory
csv_file_path = '/path/to/your/input.csv'  # Replace with your actual CSV file path
output_directory = '/path/to/output'  # Replace with your actual output directory path
output_file_paths = convert_csv_to_jsonl(csv_file_path, output_directory, category_labels)

print(output_file_paths
