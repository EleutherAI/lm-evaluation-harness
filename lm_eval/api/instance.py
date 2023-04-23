from dataclasses import dataclass, field
from typing import Literal

@dataclass
class Instance:
    request_type: str = Literal["loglikelihood", "loglikelihood_rolling", "greedy_until"]
    doc: dict = None
    arguments: tuple = None
    id_: int = None
    metadata: tuple = None # TODO: better typehints here
    resps: list = field(default_factory=list)
    filtered_resps: dict = field(default_factory=dict)

    task_name: str = None
    doc_id: str = None
    repeats: str = None

    def __post_init__(self):
        # unpack metadata field
        self.task_name, self.doc_id, self.repeats = self.metadata
     
    @property
    def args(self):
        """
        Returns (string,) where `string` is the string to calculate loglikelihood over
        """
        return self.arguments if isinstance(self.arguments, tuple) else (self.arguments,)
