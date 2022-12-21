from dataclasses import dataclass, field


@dataclass()
class LogisticRegressionParams:
    random_state: int = field(default=0)
    penalty: str = field(default="l2")
    C: float = field(default=0.9)
