from abc import ABC, abstractmethod
from typing import NamedTuple, Optional

class Verdict(NamedTuple):
    is_violation: bool
    is_refusal: bool
    score: float
    category: str
    rationale: str

class BaseJudge(ABC):
    @abstractmethod
    def evaluate(self, prompt: str, response: str, language: str = "en") -> Verdict:
        """
        Evaluate a target model response given the prompt and language.
        Returns a structured Verdict.
        """
        pass

