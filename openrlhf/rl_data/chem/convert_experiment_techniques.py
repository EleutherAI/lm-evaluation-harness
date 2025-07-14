#!/usr/bin/env python3
import csv
import json
import os

def convert_csv_to_jsonl(input_file, output_file):
    """
    Convert the ExperimentTechniques.csv file to JSONL format.
    
    Args:
        input_file: Path to the input CSV file
        output_file: Path to the output JSONL file
    """
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    jsonl_data = []
    
    # Read the CSV file
    with open(input_file, 'r', encoding='utf-8') as f:
        # Read first line to get headers
        first_line = f.readline().strip()
        headers = [h.strip() for h in first_line.split(',')]
        
        # Reset file pointer
        f.seek(0)
        
        # Create reader with stripped headers
        reader = csv.DictReader(f)
        
        for row in reader:
            # Get the question key (which might have a space)
            question_key = next((key for key in row.keys() if 'Question' in key), None)
            answer_key = next((key for key in row.keys() if 'Answer' in key), None)
            
            if not question_key or not answer_key:
                print(f"Warning: Could not find Question/Answer columns. Available columns: {list(row.keys())}")
                continue
                
            # Skip empty rows
            if not row[question_key] or row[question_key].isspace():
                continue
                
            # Create a structured question
            question = f"Task: You are a materials scientist. {row[question_key].strip()}"
            
            # Create a structured answer
            answer = f"The answer is: {row[answer_key]}" if row[answer_key] else ""
            
            # Create the JSON entry with safe gets for other fields
            entry = {
                "question": question,
                "answer": answer,
                "domain": row.get('Domain', row.get('Domain ', '')),
                "difficulty": row.get('Difficulty', row.get('Difficulty ', '')),
                "type": row.get('Type', row.get('Type ', '')),
                "task": row.get('Task', row.get('Task ', 'Experimental Techniques')),
                "reference": row.get('Reference', row.get('Reference ', '')),
                "explanation": row.get('Explanation', row.get('Explanation ', ''))
            }
            
            # Only include entries that have both question and answer
            if entry["question"] and entry["answer"]:
                jsonl_data.append(entry)
    
    # Write the JSONL file
    with open(output_file, 'w', encoding='utf-8') as f:
        for entry in jsonl_data:
            f.write(json.dumps(entry) + '\n')
    
    print(f"Converted {len(jsonl_data)} entries from CSV to JSONL format")
    print(f"Output saved to {output_file}")

if __name__ == "__main__":
    input_file = "data/ExperimentTechniques.csv"
    output_file = "data/experiment_techniques.jsonl"
    
    convert_csv_to_jsonl(input_file, output_file) 