from datasets import DatasetDict, Dataset, load_dataset, Value, ClassLabel, Features
import csv
import datasets

class BiradsDataset(datasets.GeneratorBasedBuilder):
    """Birads dataset for classifying Modality, PreviousCa, Density, Purpose, BPE, Menopausal_Status, Dx from report text"""

    VERSION = datasets.Version("1.0.0")

    def _info(self):
        # Adjust features based on the columns in your CSV file.
        return datasets.DatasetInfo(
            description="Breast radiology report along with Modality, PreviousCa, Density, Purpose, BPE, Menopausal_Status, Dx labels"
            features=datasets.Features(
                {
                    "original_report" = Value("string"),
                    "Modality": ClassLabel(names=['BIO-MG','BIO-MG-MRI', 'BIO-MG-US', 'BIO-MRI', 'BIO-US', 'MG', 'MG-MRI', 'MG-US', 'MRI', 'Other', 'US']),
                    "PreviousCa": ClassLabel(names=['Negative','Positive', 'Suspicious']),
                    "Density": ClassLabel(names=['<= 75%','Dense', 'Fatty', 'Heterogeneous', 'Not Stated', 'Scattered']),
                    "Purpose": ClassLabel(names=['Diagnostic','Screening', 'Unknown']),
                    "BPE": ClassLabel(names=['Marked','Mild', 'Minimal', 'Moderate', 'Not Stated']),
                    "Menopausal_Status": ClassLabel(names=['Not Stated','Post-Menopausal', 'Pre-Menopausal']),
                    # "Dx": ClassLabel(names=['',''])
                }
            ),
        )
    
    def _split_generators(self, dl_manager):
        """Returns SplitGenerators"""
        return [
            datasets.SplitGenerator(name=datasets.Split.TRAIN, gen_kwargs={'filepath': '~/Downloads/bogus_birads_data.csv'}),
        ]
    
    def _generate_examples(self, filepath):
        """Yields examples"""
        with open(filepath, encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for id_, row in enumerate(reader):
                # Processing each column based on format of csv
                yield id_, {
                    "original_report": row["original_report"],
                    "modality": row["Modality"],
                    "previousCa": row["PreviousCa"],
                    "density": row["Density"],
                    "purpose": row["Purpose"],
                    "bpe": row["BPE"],
                    "menopausal_status": row["Menopausal_Status"],
                    # "dx": row["Dx"],
                }