
# Phase 2C — Orchestrator Execution Contract

**Version:** 0.1  
**Status:** Implemented and tested  
**Scope:** Python orchestration layer for the AI Healthcare multi-agent system

---

## 1. Objective

Phase 2C implements the execution infrastructure that connects the
Commander with specialized agents.

The architecture separates responsibilities into:

1. Commander: generates a structured execution plan.
2. Orchestrator: validates the plan, resolves dependencies and dispatches tasks.
3. AgentRegistry: manages registered agent instances.
4. BaseAgent: defines a common execution interface.
5. ResponseAssembler: combines individual execution results.

This phase establishes the execution contract. It does not implement
the medical capabilities of specialized agents.

## 2. Architecture

```text
User Request
     |
     v
Commander
     |
     | CommanderPlan
     v
Orchestrator
     |
     | Validate plan
     | Resolve dependencies
     v
AgentRegistry
     |
     | Retrieve registered agent
     v
BaseAgent.execute()
     |
     | AgentExecutionResult
     v
ResponseAssembler
     |
     | AssembledResponse
     v
Application / Response Layer
```

### 2.1. Module responsibilities

| Module | Responsibility |
|---|---|
| `agents/commander.py` | Generates and validates CommanderPlan using a structured-output provider. |
| `agents/base.py` | Defines the interface required by specialized agents. |
| `orchestration/agent_registry.py` | Registers and retrieves agent instances. |
| `orchestration/orchestrator.py` | Validates plans, resolves dependencies, executes tasks and tracks results. |
| `orchestration/response_assembler.py` | Validates and aggregates execution results. |
| `schemas/agent.py` | Defines CommanderPlan, AgentTask and agent identifiers. |
| `schemas/execution.py` | Defines the result contract for agent execution. |

## 3. Core Contracts

### 3.1. CommanderPlan

CommanderPlan represents a bounded execution proposal.

Main fields:

- `intent`: recognized user intent.
- `status`: ready, needs_clarification or unsupported.
- `tasks`: ordered task proposals with dependencies.
- `clarification_question`: optional question when clarification is required.

Each AgentTask contains:

- `task_id`: unique task identifier.
- `agent`: one whitelisted agent name.
- `objective`: task description.
- `depends_on`: identifiers of prerequisite tasks.

Validation rules:

- Task IDs must be unique.
- An agent may appear at most once in a plan.
- All dependencies must exist.
- Dependency cycles are rejected.
- Ready plans must contain at least one task.
- Clarification and unsupported plans must not dispatch tasks.
- Agent identifiers must belong to the whitelist.

### 3.2. AgentName allowlist

The currently supported identifiers are:

- `medical_rag`
- `symptom`
- `medication`
- `psychology_support`

Being whitelisted does not mean an agent has already been implemented.

### 3.3. BaseAgent

Every specialized agent must implement:

```python
@property
def name(self) -> AgentName:
    ...

def execute(
    self,
    task: AgentTask,
    context: dict[str, Any],
) -> Any:
    ...
```

The BaseAgent contract does not implement medical logic, load models,
or communicate with Ollama.

Agent implementations are responsible for their own task-specific processing.

### 3.4. AgentRegistry

AgentRegistry manages explicitly registered agent instances.

Supported operations:

- `register(agent)`: register an agent.
- `get(name)`: retrieve a registered agent.
- `has(name)`: check whether an agent exists.
- `registered_names`: inspect registered agent names.

The registry rejects:
- Objects that do not inherit BaseAgent.
- Invalid agent names.
- Duplicate registrations.

The registry does not create agents automatically.

## 4. Orchestrator Execution Contract

### 4.1. Initialization

```python
registry = AgentRegistry()

orchestrator = Orchestrator(
    registry=registry,
)
```

The orchestrator maintains an enabled-agent allowlist separately
from the registry.

These concepts are different:

- Enabled: the agent is permitted by the orchestrator.
- Registered: an executable agent instance is available.

A whitelisted and enabled agent that is not registered produces
`not_implemented`.

A disabled agent causes plan validation to fail with
`AgentNotEnabledError`.

### 4.2. Execution

```python
plan = commander.plan(user_message)

execution_results = orchestrator.execute_plan(
    plan,
    context={"request_id": "example-001"},
)

response = ResponseAssembler().assemble(execution_results)
```

The Commander only generates a plan. Calling `execute_plan()` is
a separate orchestration operation.

The execution sequence is:

1. Validate the CommanderPlan.
2. Derive a stable topological execution order.
3. Check prerequisite task results.
4. Retrieve the required agent from AgentRegistry.
5. Pass task information and dependency results to the agent.
6. Capture the agent output or failure.
7. Store an AgentExecutionResult.
8. Continue with the next eligible task.

Execution is sequential in version 0.1.

### 4.3. Dependency context

The orchestrator supplies:

```python
context["dependency_results"]
```

This is a mapping from prerequisite task IDs to their
AgentExecutionResult objects.

Example:

```python
dependency_result = context[
    "dependency_results"
]["task_1"]

previous_output = dependency_result.output
```

Only declared dependencies are included in this mapping.

The `dependency_results` key is reserved and cannot be supplied
directly by the caller.

### 4.4. Failure handling

A task whose dependency did not succeed is skipped.

Independent tasks continue to execute even when another task fails.

The orchestrator catches ordinary `Exception` instances raised
during agent execution and records a sanitized error message.

Internal exception details are not copied into the execution result.

## 5. AgentExecutionResult

Each executed task produces an AgentExecutionResult.

Fields:

- `task_id`
- `agent`
- `status`
- `output`
- `error`
- `skipped_reason`

Supported execution statuses:

| Status | Meaning |
|---|---|
| `succeeded` | Agent execution completed successfully. |
| `failed` | A registered agent raised an exception. |
| `not_implemented` | The requested agent is not registered. |
| `skipped` | A prerequisite task did not succeed. |

The result schema enforces consistency between status and
associated output/error fields.

The `output` field is currently flexible (`Any`).
Specialized output schemas are required before production use.

## 6. ResponseAssembler

ResponseAssembler aggregates a sequence of AgentExecutionResult objects
into an immutable AssembledResponse.

Supported overall statuses:

| Status | Rule |
|---|---|
| `completed` | All tasks succeeded. |
| `partial` | At least one task succeeded, but not all tasks succeeded. |
| `failed` | Results exist, but no task succeeded. |
| `no_results` | No execution results were provided. |

The response includes:

- `results`
- `total_tasks`
- `succeeded_count`
- `failed_count`
- `not_implemented_count`
- `skipped_count`

Duplicate task IDs are rejected.

ResponseAssembler does not generate natural-language medical answers
and does not independently verify the clinical correctness of agent outputs.

## 7. Error and Safety Boundaries

### 7.1. Agent dispatch

Only whitelisted agent identifiers are accepted by the plan schema.

Only explicitly registered agent instances can execute tasks.

The Commander cannot create Python classes or arbitrary executable
instructions through its plan output.

### 7.2. Patient data

Phase 2C does not implement patient-record access.

The execution context is currently supplied by the caller.
No EHR/EMR storage, authentication, patient ownership verification,
or per-agent patient-data permissions are implemented here.

Future patient-record integration must enforce:
- Authenticated user identity.
- Patient-record ownership and access authorization.
- Minimum necessary data access for each agent.
- Audit logging for patient-data access.
- Clear separation between patient-provided facts and external medical evidence.

### 7.3. Medical evidence

The current execution framework does not establish clinical validity.

Medical RAG retrieval, evidence provenance, medication verification,
safety escalation and output validation must be implemented and tested
inside the relevant specialized agents and application services.

## 8. Testing

The Phase 2C implementation is verified using unit and integration tests.

### 8.1. Test coverage

| Test module | Tests |
|---|---:|
| `test_chat_service.py` | 3 |
| `test_commander.py` | 35 |
| `test_commander_evaluation.py` | 7 |
| `test_orchestrator.py` | 12 |
| `test_orchestrator_execution.py` | 5 |
| `test_orchestration_execution.py` | 2 |
| **Total** | **64** |

### 8.2. Verified scenarios

- Commander plan validation.
- Agent allowlist enforcement.
- Dependency validation and cycle rejection.
- Stable topological execution order.
- Registered agent execution.
- Dependency result propagation.
- Missing agent handling.
- Agent execution failure handling.
- Dependency-based task skipping.
- Independent task continuation.
- Response status aggregation.
- Execution result counting.
- Complete multi-agent workflow with simulated components.

### 8.3. Run tests

```powershell
$env:PYTHONPATH = "src"
.\.venv\Scripts\python.exe -m pytest tests -v
```

Latest verified result:

```text
64 passed
```

Integration tests use fake providers and fake agents.
They do not call Ollama or test actual medical-agent quality.

## 9. Explicit Non-Goals

The following capabilities are not implemented in Phase 2C:

- Actual Medical RAG Agent.
- Actual Symptom Agent.
- Actual Medication Agent.
- Actual Psychology Support Agent.
- Automatic agent creation.
- Ollama model loading or inference through specialized agents.
- Patient registration and EHR/EMR persistence.
- Patient-specific authorization.
- Parallel agent execution.
- LLM-based response synthesis.
- Fine-tuning.
- Clinical validation or deployment.

These remain subsequent implementation tasks.

## 10. Phase 2C Acceptance

Phase 2C establishes:

1. A validated Commander task contract.
2. A common BaseAgent interface.
3. An explicit AgentRegistry.
4. Sequential dependency-aware execution.
5. Structured execution results and failure handling.
6. Response aggregation.
7. Unit and integration test coverage.

**Acceptance status:** Implemented and tested locally, with 64 passing tests.

This acceptance confirms the orchestration infrastructure,
not the clinical capabilities or production readiness of the system.
