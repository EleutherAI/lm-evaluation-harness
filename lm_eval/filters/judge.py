import os

from lm_eval.api.filter import Filter
from lm_eval.api.registry import register_filter
from lm_eval.models.openai_completions import LocalChatCompletion


@register_filter("judge")
class JudgeFilter(Filter):
    """A filter that evaluates the correctness of a question-answering system's answers, using an LM Judge"""

    PROMPT = """You are an expert evaluator of question-answering systems. Your task is to determine if a given answer matches the ground truth answer in meaning and accuracy. You should respond with "yes", "no", or "unknown".

    Guidelines for evaluation:
    1. Focus on semantic meaning rather than exact wording
    2. Consider numerical accuracy when applicable
    3. Account for partial answers that contain the correct information plus additional details
    4. Recognize equivalent phrasings and synonyms
    5. Be lenient with minor grammatical differences
    6. For multi-part questions, all parts must be correct
    7. For questions requiring specific units, check unit correctness
    8. Respond with "unknown" when:
       - The answer is ambiguous and could be interpreted multiple ways
       - There is insufficient context to determine correctness
       - The ground truth is incomplete or unclear
       - The comparison requires external knowledge not provided

    Input format:
    Question: [The question being asked]
    Answer: [The answer given by the system]
    Ground Truth: [The known correct answer]

    Your response must be exactly "yes", "no", or "unknown", with no additional explanation.

    Example 1:
    Question: What is the capital of France?
    Answer: The capital city of France is Paris.
    Ground Truth: Paris
    Your response: yes

    Example 2:
    Question: How many planets are in our solar system?
    Answer: There are seven planets in our solar system.
    Ground Truth: Eight planets (Mercury, Venus, Earth, Mars, Jupiter, Saturn, Uranus, and Neptune)
    Your response: no

    Example 3:
    Question: What is the GDP of France in 2023?
    Answer: The economic output was substantial.
    Ground Truth: 3.05 trillion USD
    Your response: unknown

    Your response must be exactly "yes", "no", or "unknown", with no additional explanation!
    """

    def __init__(self, url, **kwargs) -> None:
        """
        pass a string `regex` to run `re.compile(r"regex")` on.
        `fallback` defines the output returned if no matches for the regex are located.
        """
        assert os.environ.get("AI_API_KEY") is not None, (
            "Please set the AI_API_KEY environment variable to use the JudgeFilter (can be empty string)"
        )
        self.model = LocalChatCompletion(base_url=url, **kwargs)

    def apply(self, resps: list[list[str]], docs: list[dict]) -> list[list[str]]:
        inputs = [
            self.PROMPT
            + "\n\n"
            + f"Question: {doc['question']}\nAnswer: {resp}\nGround Truth: {doc['answer']}"
            for resp, doc in zip(resps, docs)
        ]

        res = self.model.simple_async_generate([inputs], gen_kwargs={})
        return [[x] for x in res]
