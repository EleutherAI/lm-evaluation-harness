import abc
import random


class LM(abc.ABC):
    @abc.abstractmethod
    def loglikelihood(self, context, continuation):
        """Compute log-likelihood of generating a continuation from a context

        :param context: str
            Context string
        :param continuation: str
            The continuation over which log likelihood will be calculated. If 
            there is a word boundary, the space should be in the continuation. 
            For example, context="hello" continuation=" world" is correct.
        :return: float
        """
        pass

    @classmethod
    def create_from_arg_string(cls, arg_string):
        """Constructor method, in case models need additional arguments
        e.g. OpenAI API engine, paths for loading, other params

        :param arg_string: str
            Left up to individual model class to handle

        """
        return cls()


class Dataset(abc.ABC):
    @abc.abstractmethod
    def __init__(self):
        self.download()
        self._traindocs = None

    def download(self):
        """Downloads the task dataset if necessary"""
        pass

    @abc.abstractmethod
    def has_training_docs(self):
        """Whether the task has a training set"""
        pass
    
    @abc.abstractmethod
    def has_validation_docs(self):
        """Whether the task has a validation set"""
        pass

    @abc.abstractmethod
    def has_test_docs(self):
        """Whether the task has a test set"""
        pass

    @abc.abstractmethod
    def training_docs(self):
        """

        :return: Iterable[obj]
            A iterable of any object, that doc_to_text can handle
        """
        pass
    
    @abc.abstractmethod
    def validation_docs(self):
        pass
    
    @abc.abstractmethod
    def test_docs(self):
        pass
    
    def fewshot_examples(self, k):
        if self._traindocs is None:
            self._traindocs = list(self.training_docs())

        return random.sample(self._traindocs, k)

    @abc.abstractmethod
    def doc_to_text(self, doc, include_target=True):
        pass
    
    @abc.abstractmethod
    def evaluate(self, docs, lm, provide_description, num_fewshot):
        """Take iterable of docs and evaluates, returning a dict with the following format:

        {
            "major": float,
            "minor": dict,
            "higher_is_better": bool,
        }

        * `major` should be a single, representative number, for programmatic comparison
        * `minor` should be a dictionary containing all relevant sub-metrics
        * `higher_is_better` determines whether a higher metric is better
        """
        pass

    def fewshot_description(self):
        return ""

    def fewshot_context(self, doc, num_fewshot, provide_description):
        raw_description = self.fewshot_description()
        description = (raw_description + "\n===\n\n") if provide_description and raw_description else ""
        labeled_examples = "\n\n".join(
            map(self.doc_to_text, self.fewshot_examples(k=num_fewshot))
        ) + "\n\n"
        example = self.doc_to_text(doc, include_target=False).strip()
        return description + labeled_examples + example