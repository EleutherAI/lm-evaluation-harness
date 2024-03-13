from dataclasses import dataclass, field
from typing import Callable, Literal, Optional, Tuple


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


class ContextInstance(Instance):
    def __init__(
        self,
        requests_updater: Optional[Callable] = None,
        storage_updater: Optional[Callable] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._update_request = requests_updater
        self._update_storage = storage_updater

    @property
    def update_request(self):
        if getattr(self, "_update_request") is not None:
            return self._update_request
        raise NotImplementedError("Method for updating request is not defined.")

    @property
    def update_storage(self):
        if getattr(self, "_update_storage") is not None:
            return self._update_storage
        raise NotImplementedError("Method for updating storage is not defined.")
