from lm_eval.api.task import PerplexityTask
from typing import Optional, Dict, Any
import datasets

class Mc4PerplexityTask(PerplexityTask):
    def download(
        self,
        data_dir: Optional[str] = None,
        cache_dir: Optional[str] = None,
        download_mode=None,
    ) -> None:
        unloaded_dataset = datasets.load_dataset(
            path=self.DATASET_PATH,
            name=self.DATASET_NAME,
            data_dir=data_dir,
            cache_dir=cache_dir,
            download_mode=download_mode,
            streaming=True
        )
        
        self.dataset = {}
        for split, sub_dataset in unloaded_dataset.items():
            self.dataset[split] == []
            for i, data_pt in enumerate(sub_dataset):
                self.dataset[split].append(data_pt)
                if i == 1000:
                    break

    def has_validation_docs(self):
        """Whether the task has a validation set"""
        return False

    def has_test_docs(self):
        """Whether the task has a test set"""
        return False