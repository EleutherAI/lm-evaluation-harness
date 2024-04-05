import os
import pandas as pd

def combine_csvs(directory_path):
    csv_files = [file for file in os.listdir(directory_path) if file.endswith('.csv')]

    dataframes = []
    
    for csv_file in csv_files:
        file_path = os.path.join(directory_path, csv_file)
        df = pd.read_csv(file_path)
        dataframes.append(df)
        
    combined_df = pd.concat(dataframes, ignore_index = True)

    combined_df.to_csv('combined_data.csv', index=False)

combine_csvs('/Users/marina.levay/Documents/GitHub/lm-evaluation-harness/lm_eval/tasks/scheming_evals')
