## Should be used to merge local input files (.csv) into one input file in the local folder ##

import os
import pandas as pd

# combining all csv files in the scheming_evals folder
def combine_csvs(directory_path):
    csv_files = [file for file in os.listdir(directory_path) if file.endswith('.csv')]

    dataframes = []
    
    for csv_file in csv_files:
        file_path = os.path.join(directory_path, csv_file)
        df = pd.read_csv(file_path)
        dataframes.append(df)
        
    combined_df = pd.concat(dataframes, ignore_index = True)
    
    # switching the answer (target) of MC task for written choice instead of the number, since the request used in the experiment is currently limited to the generate_until request instead of multiple_choice
    for i in range(len(combined_df['answer'])):
        index_for_choice = str(combined_df['answer'][i])
        choice = combined_df[index_for_choice][i]
        combined_df['answer'][i] = choice

    combined_df.to_csv('combined_data.csv', index=False)

combine_csvs('path/to/local/repo')
