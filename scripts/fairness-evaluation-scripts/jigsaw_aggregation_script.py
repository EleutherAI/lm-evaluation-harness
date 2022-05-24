import pandas as pd
import numpy as np
import argparse

def get_fairness_evaluation(file_path):
        file_pd = pd.read_csv(file_path, sep="|", header=[0])
        file_pd.columns = file_pd.columns.str.replace(' ','')
        identity_to_fpr = {}
        out = {}
        dimension_to_identity_set = {"race": ["black", "white", "asian", "latino", "other_race_or_ethnicity"],
                                         "gender_nonbinary": ["male", "female", "transgender", "other_gender"]}
        file_pd["Metric"] = file_pd["Metric"].apply(lambda x: x.strip())
        for prompt in list(file_pd["Prompt"].unique()):
            if "-" in prompt:
                continue
            out[prompt] = {}
            prompt_pd = file_pd[file_pd["Prompt"] == prompt]
            for dimension, identity_set in dimension_to_identity_set.items():
                for identity in identity_set:
                    fp = prompt_pd[prompt_pd["Metric"] == f"{identity}_fp"]["Value"].iloc[0].strip()
                    tn = prompt_pd[prompt_pd["Metric"] == f"{identity}_tn"]["Value"].iloc[0].strip()
                    if float(fp) + float(tn) == 0:
                    	identity_to_fpr[f"{identity}_fpr"] = 0
                    else:
                    	identity_to_fpr[f"{identity}_fpr"] =  float(fp) / (float(fp) + float(tn))
                out[prompt][f"{dimension}_var"] = np.var(list(identity_to_fpr.values()))
                out[prompt][f"{dimension}_std"] = np.std(list(identity_to_fpr.values()))
        out = pd.DataFrame(out)
        out.to_csv(f"results_{file_path.replace('txt', 'csv')}")
        return out


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--file-path',  type=str, required=True,
	                    help='path to raw jigsaw results.')
	args = parser.parse_args()
	yo = get_fairness_evaluation(args.file_path)
