from dataclasses import dataclass, field
from typing import Literal, Optional, Tuple


OutputType = Literal[
    "loglikelihood", "loglikelihood_rolling", "generate_until", "multiple_choice"
]


@dataclass
class Instance:
    request_type: OutputType
    doc: dict
    arguments: tuple
    idx: int
    metadata: Tuple[Optional[str], Optional[int], Optional[int]] = field(
        default_factory=lambda: (None, None, None)
    )
    resps: list = field(default_factory=list)
    filtered_resps: dict = field(default_factory=dict)

    # initialized after init
    task_name: Optional[str] = None
    doc_id: Optional[int] = None
    repeats: Optional[int] = None

    def __post_init__(self) -> None:
        # unpack metadata field
        self.task_name, self.doc_id, self.repeats = self.metadata

    @property
    def args(self):
        """
        Returns (string,) where `string` is the string to calculate loglikelihood over
        """
        return (
            self.arguments if isinstance(self.arguments, tuple) else (self.arguments,)
        )

    @args.setter
    def args(self, new_arguments: tuple) -> None:
        """
        Update the arguments of this instance with a new one
        """
        if isinstance(new_arguments, tuple):
            assert (
                len(new_arguments) == len(self.args)
            ), "Must set new Instance arguments to have same size + types as old arguments"
            self.arguments = new_arguments
        else:
            raise ValueError("Must set new Instance args to a tuple!")
