from abc import ABC, abstractmethod
from typing import Any

from ai_healthcare.schemas.agent import AgentName, AgentTask


class BaseAgent(ABC):
    """Abstract interface implemented by every specialized agent."""

    @property
    @abstractmethod
    def name(self) -> AgentName:
        """Return the agent's registered name."""
        ...

    @abstractmethod
    def execute(
        self,
        task: AgentTask,
        context: dict[str, Any],
    ) -> Any:
        """Execute a task and return its raw output."""
        ...
