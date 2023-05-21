import datasets

from lm_eval.base import PerplexityTask
from lm_eval.utils import escaped_split


class JsonPerplexity(PerplexityTask):

    VERSION = 0
    DATASET_NAME = "json"

    def __init__(self, data_dir=None, cache_dir=None, download_mode=None):
        """
        :param data_dir: str
            Use this to specify the path to manually downloaded JSON test data.
            This also needs to include the split key and text key for the data
            in the following format:
            ```
            split:text:/absolute/path/to/data.json
            ```

            If you do not have splits inside the JSON file, it should be "train".
            Colons in the split or text key can be escaped by backslashes.
        :param cache_dir: str
            The directory to read/write the `Task` dataset. This follows the
            HuggingFace `datasets` API with the default cache directory located at:
                `~/.cache/huggingface/datasets`
            NOTE: You can change the cache location globally for a given process
            by setting the shell environment variable, `HF_DATASETS_CACHE`,
            to another directory:
                `export HF_DATASETS_CACHE="/path/to/another/directory"`
        :param download_mode: datasets.DownloadMode
            How to treat pre-existing `Task` downloads and data.
            - `datasets.DownloadMode.REUSE_DATASET_IF_EXISTS`
                Reuse download and reuse dataset.
            - `datasets.DownloadMode.REUSE_CACHE_IF_EXISTS`
                Reuse download with fresh dataset.
            - `datasets.DownloadMode.FORCE_REDOWNLOAD`
                Fresh download and fresh dataset.
        """
        self._split, self._key, data_file = escaped_split(data_dir, ":", 2)
        self.load(data_file)
        self._training_docs = None
        self._fewshot_docs = None

    def download(self, data_dir=None, cache_dir=None, download_mode=None):
        raise TypeError("cannot download an arbitrary JSON dataset")

    def load(self, data_file):
        self.dataset = datasets.load_dataset("json", data_files=data_file)

    def has_validation_docs(self):
        return False

    def has_test_docs(self):
        return True

    def test_docs(self):
        return map(self._process_doc, self.dataset[self._split])

    def _process_doc(self, doc):
        return doc[self._key]
