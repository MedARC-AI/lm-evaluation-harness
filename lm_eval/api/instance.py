from dataclasses import dataclass, field
from typing import Literal, Tuple


@dataclass
class Instance:
    request_type: Literal["loglikelihood", "loglikelihood_rolling", "generate_until"]
    doc: dict
    arguments: tuple
    idx: int
    metadata: Tuple[str, int, int, bool] = field(
        default_factory=lambda: (None, None, None, False)
    )  # TODO: better typehints here
    resps: list = field(default_factory=list)
    filtered_resps: dict = field(default_factory=dict)

    # initialized after init
    task_name: str = None
    doc_id: str = None
    repeats: str = None

    def __post_init__(self) -> None:
        # unpack metadata field
        self.task_name, self.doc_id, self.repeats, self.shuffle_choices = self.metadata

    @property
    def args(self):
        """
        Returns (string,) where `string` is the string to calculate loglikelihood over
        """
        return (
            self.arguments if isinstance(self.arguments, tuple) else (self.arguments,)
        )
