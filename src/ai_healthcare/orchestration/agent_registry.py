from pydantic import TypeAdapter, ValidationError

from ai_healthcare.agents.base import BaseAgent
from ai_healthcare.schemas.agent import AgentName


_AGENT_NAME_ADAPTER = TypeAdapter(AgentName)


class AgentRegistry:
    """Register and retrieve specialized agents by their names."""

    def __init__(self) -> None:
        self._agents: dict[AgentName, BaseAgent] = {}

    def register(self, agent: BaseAgent) -> None:
        """Register an agent instance."""

        if not isinstance(agent, BaseAgent):
            raise TypeError("Only BaseAgent instances can be registered.")

        try:
            name = _AGENT_NAME_ADAPTER.validate_python(agent.name)
        except ValidationError as exc:
            raise ValueError(
                f"Invalid agent name: {agent.name!r}"
            ) from exc

        if name in self._agents:
            raise ValueError(f"Agent '{name}' is already registered.")

        self._agents[name] = agent

    def get(self, name: AgentName) -> BaseAgent:
        """Retrieve a registered agent by name."""

        try:
            return self._agents[name]
        except KeyError as exc:
            raise KeyError(
                f"Agent '{name}' is not registered."
            ) from exc

    def has(self, name: AgentName) -> bool:
        """Check whether an agent is registered."""

        return name in self._agents

    @property
    def registered_names(self) -> tuple[AgentName, ...]:
        """Return the names of all registered agents."""

        return tuple(self._agents.keys())
