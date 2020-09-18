import abc
import random


class LM(abc.ABC):
    @abc.abstractmethod
    def generate(self, context, max_gen_length):
        """Conditional text generation with an LM

        :param context: str
            Context string for conditional generation
        :param max_gen_length: int
            Maximum number of tokens to generate
        :return: str

        """
        pass

    @abc.abstractmethod
    def loglikelihood(self, context, continuation):
        """Compute log-likelihood of a generation a continuation from a context

        Assume that the final text will simple be
            context + continuation

        :param context: str
            Context string for conditional generation
        :param continuation: str
            Maximum number of tokens to generate
        :return: float
        """
        pass

    @classmethod
    def num_tokens(cls, string):
        """Return the number of tokens in a string, based on tokenization

        :param string: str
            Input string
        :return: int
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
        traindocs = list(self.training_docs())
        random.shuffle(traindocs)
        return traindocs[:k]

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
