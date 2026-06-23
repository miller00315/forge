"""Forge: LangGraph workflow for iterative technical support with tool calling.

The graph gathers context, invokes a chat model, executes repository tools,
verifies completion, and routes between retry, strategy change, and finish nodes.
"""

import time
from enum import Enum
import operator
from pathlib import Path
from typing import Annotated

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.types import Command
from typing_extensions import NotRequired, TypedDict

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

REAL_MODEL = False
DONE_KEYWORD = "done"

TRIES_PER_STRATEGY = 3
INITIAL_MAX_TRIES = 20
BUDGET_EXTENSION = TRIES_PER_STRATEGY
MAX_STRATEGY_CHANGES = 1
RECURSION_LIMIT = (
    1  # gather
    + TRIES_PER_STRATEGY * (1 + MAX_STRATEGY_CHANGES) * 3
    + MAX_STRATEGY_CHANGES
    + 2  # finish + safety margin
)

GRAPH_IMAGE_PATH = Path(__file__).parent / "forge_graph.png"
PASS_THRESHOLD = 3

# ---------------------------------------------------------------------------
# Simulated repository
# ---------------------------------------------------------------------------

REPO = {"auth.py": "def valida_email(e): \n    return '@' in e # bug: case-sensitive\n"}

_applied_correction = {"done": False}

FORGE_INTRODUCTION = SystemMessage(
    content=(
        "You are a technical support assistant that fixes system issues. "
        f"When the fix covers the issue, end with the word {DONE_KEYWORD.upper()}."
    )
)

if REAL_MODEL:
    from langchain.chat_models import init_chat_model

    model = init_chat_model("claude-haiku-4-5", model_provider="anthropic")
else:
    from langchain_core.language_models.fake_chat_models import GenericFakeChatModel

    model = GenericFakeChatModel(
        messages=iter(
            [
                AIMessage(content="I will investigate email validation."),
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "search_in_repo",
                            "args": {"query": "valida_email"},
                            "id": "c1",
                            "type": "tool_call",
                        }
                    ],
                ),
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "read_file",
                            "args": {"path": "auth.py"},
                            "id": "c2",
                            "type": "tool_call",
                        }
                    ],
                ),
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "apply_correction",
                            "args": {
                                "path": "auth.py",
                                "correction": (
                                    "def valida_email(e): \n    "
                                    "return '@' in e # bug: case-sensitive\n"
                                ),
                            },
                            "id": "c3",
                            "type": "tool_call",
                        }
                    ],
                ),
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "execute_test",
                            "args": {"suite": "test_email_validation"},
                            "id": "c4",
                            "type": "tool_call",
                        }
                    ],
                ),
                AIMessage(
                    content=(
                        "Fix applied: the issue case is complete. "
                        f"{DONE_KEYWORD.upper()}."
                    )
                ),
            ]
        )
    )

# ---------------------------------------------------------------------------
# Input / output schemas
# ---------------------------------------------------------------------------


class SCENARIOS(Enum):
    SUCCESS = "success"
    STUCK = "stuck"
    BUDGET_EXCEEDED = "budget_exceeded"


class STOP_REASONS(Enum):
    BUDGET_EXCEEDED = "budget_exceeded"
    STUCK_DETECTED = "stuck_detected"
    NO_OUTCOME = "no_outcome"
    OSCILATION_DETECTED = "oscillation_detected"
    SUCCESS = "success"


class ACTION_PATTERNS(Enum):
    UNIFORM = "uniform"
    ALTERNATING = "alternating"


class InputForge(TypedDict):
    """Public graph input."""

    issue: str
    scenario: NotRequired[SCENARIOS]
    max_tries: NotRequired[int]
    start: NotRequired[float]
    max_seconds: NotRequired[float]
    patience: NotRequired[int]


class OutputForge(TypedDict):
    """Public graph output."""

    report: str
    history: list
    total_cost: float
    messages: list


class StateForge(MessagesState):
    """Internal graph state."""

    issue: str
    scenario: SCENARIOS
    tries: int
    start: float
    max_tries: int
    max_seconds: float
    patience: int
    scores: Annotated[list[float], operator.add]
    best_score: float
    test_ok: bool
    report: str
    actions: Annotated[list, operator.add]
    stop_reason: STOP_REASONS
    strategy_changes: int
    messages: Annotated[list, operator.add]
    total_cost: Annotated[float, operator.add]
    history: Annotated[list, operator.add]


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


@tool("execute_test", description="Execute a test suite and return the result.")
def execute_test(suite: str) -> str:
    """Run a test suite against the simulated repository."""
    if _applied_correction["done"]:
        return "Result: 2 passed, 0 failed."
    return "Result: 0 passed, 2 failed."


@tool("read_file", description="Read a file and return the content.")
def read_file(path: str) -> str:
    """Return file contents from the in-memory repository."""
    if path not in REPO:
        return f"ERROR: '{path}' not found in repo. Files: {list(REPO)}"
    return REPO[path]


@tool("apply_correction", description="Apply a correction to a file.")
def apply_correction(path: str, correction: str) -> str:
    """Overwrite a repository file and mark the correction as applied."""
    if path not in REPO:
        return f"ERROR: '{path}' not found in repo. Files: {list(REPO)}"
    REPO[path] = correction
    _applied_correction["done"] = True
    return f"OK: path '{path}' updated ({len(correction)} chars)."


@tool(
    "search_in_repo",
    description="Search for a string in repo file names and file contents.",
)
def search_in_repo(query: str) -> str:
    """Find files whose path or content contains the query string."""
    if not query:
        return "ERROR: query must not be empty."

    matches = []
    for path, content in REPO.items():
        in_path = query in path
        in_content = query in content
        if in_path or in_content:
            locations = []
            if in_path:
                locations.append("path")
            if in_content:
                locations.append("content")
            matches.append(f"{path} ({', '.join(locations)})")

    if not matches:
        return f"No matches for '{query}'. Files: {list(REPO)}"

    return f"Found {len(matches)} file(s): {matches}"


TOOLS = [read_file, apply_correction, execute_test, search_in_repo]

# ---------------------------------------------------------------------------
# Workflow nodes
# ---------------------------------------------------------------------------


def _resolve_scenario(state: StateForge) -> SCENARIOS:
    """Return the active scenario, defaulting to SUCCESS."""
    scenario = state.get("scenario", SCENARIOS.SUCCESS)
    if isinstance(scenario, str):
        return SCENARIOS(scenario)
    return scenario


def _has_repeating_period(sequence: list[str], period: int) -> bool:
    """Return True if sequence equals its prefix repeated (e.g. ABAB has period 2)."""
    if period <= 0 or period > len(sequence):
        return False
    return all(sequence[i] == sequence[i - period] for i in range(period, len(sequence)))


def detect_action_pattern(window: list[str]) -> ACTION_PATTERNS | None:
    """Classify a window as uniform (period 1) or alternating (period 2)."""
    if not window:
        return None
    if _has_repeating_period(window, 1):
        return ACTION_PATTERNS.UNIFORM
    if len(window) >= 2 and _has_repeating_period(window, 2) and len(set(window)) == 2:
        return ACTION_PATTERNS.ALTERNATING
    return None


def detect_repetitive_pattern(
    actions: list[str],
    window_size: int = 4,
) -> ACTION_PATTERNS | None:
    """Classify the last N actions as uniform, alternating, or no pattern."""
    if len(actions) < window_size:
        return None
    return detect_action_pattern(actions[-window_size:])


def _action_label(scenario: SCENARIOS, tries: int, score: float) -> str:
    """Map an attempt to action label A or B for pattern detection."""
    match scenario:
        case SCENARIOS.STUCK:
            return "A"
        case SCENARIOS.BUDGET_EXCEEDED:
            return "A" if score <= 1 else "B"
        case SCENARIOS.SUCCESS:
            return "B" if score >= PASS_THRESHOLD else "A"


def try_and_verify(state: StateForge) -> dict:
    """Simulate one attempt and score it according to the active scenario."""
    tries = state.get("tries", 0) + 1
    scenario = _resolve_scenario(state)

    match scenario:
        case SCENARIOS.SUCCESS:
            score = min(tries, PASS_THRESHOLD)
        case SCENARIOS.STUCK:
            score = 1
        case SCENARIOS.BUDGET_EXCEEDED:
            score = 1 if tries % 2 else 2

    return {
        "tries": tries,
        "scores": [score],
        "actions": [_action_label(scenario, tries, score)],
        "history": [f"try {tries}: {score}/{PASS_THRESHOLD} tests passed"],
    }


def gather(state: StateForge) -> dict:
    """Seed the conversation with the issue or reuse existing context."""
    if not state.get("messages"):
        return {
            "messages": [HumanMessage(content=f"Issue: {state['issue']}")],
            "history": ["gather: initial context"],
        }

    return {"history": ["gather: reused context"]}


def verify(state: StateForge) -> dict:
    """Check whether the latest attempt score meets the pass threshold."""
    scores = state.get("scores", [])
    last_score = scores[-1] if scores else 0
    passed = last_score >= PASS_THRESHOLD

    return {
        "test_ok": passed,
        "history": [f"attempt {'passed' if passed else 'failed'}"],
    }


def change_strategy(state: StateForge) -> dict:
    """Reset failure window and extend the attempt budget."""
    current_budget = state.get("max_tries", INITIAL_MAX_TRIES)
    return {
        "history": ["decision: change strategy"],
        "messages": [HumanMessage(content="Let's try a different strategy.")],
        "strategies_results": [],
        "max_tries": current_budget + BUDGET_EXTENSION,
    }


def decide(state: StateForge) -> Command:
    """Route to finish, strategy change, or another act attempt."""
    scores = state.get("scores", [])
    last = scores[-1] if scores else 0
    best = max(scores)

    if last >= PASS_THRESHOLD:
        return Command(
            update={
                "stop_reason": STOP_REASONS.SUCCESS,
                "best_score": best,
            },
            goto="finish",
        )

    if state.get("tries", 0) >= state.get("max_tries", INITIAL_MAX_TRIES):
        return Command(
            update={
                "stop_reason": STOP_REASONS.BUDGET_EXCEEDED,
                "best_score": best,
            },
            goto="finish",
        )

    if time.time() - state.get("start", 0) >= state.get("max_seconds", 0):
        return Command(
            update={
                "stop_reason": STOP_REASONS.NO_OUTCOME,
                "best_score": best,
            },
            goto="finish",
        )

    patience = state.get("patience", 3)
    if (
        patience > 0
        and len(scores) > patience
        and max(scores[-patience:]) <= max(scores[:-patience])
    ):
        return Command(
            update={
                "stop_reason": STOP_REASONS.STUCK_DETECTED,
                "best_score": best,
            },
            goto="finish",
        )

    actions = state.get("actions", [])
    pattern = detect_repetitive_pattern(actions, window_size=4)
    if pattern is not None:
        return Command(
            update={
                "stop_reason": STOP_REASONS.OSCILATION_DETECTED,
                "best_score": best,
                "history": [
                    f"decision: {pattern.value} pattern detected in last 4 actions"
                ],
            },
            goto="finish",
        )

    return Command(goto="try_and_verify")


def finish(state: StateForge) -> dict:
    """Build the final report from the stop reason."""

    match state["stop_reason"]:
        case STOP_REASONS.SUCCESS:
            report = (
                f"The issue has been corrected in {state['tries']} attempts. "
                f"Reason: {state['stop_reason']}."
            )
            return {"report": report}
        case STOP_REASONS.BUDGET_EXCEEDED:
            report = (
                f"The issue has not been corrected in {state['tries']} attempts. "
                f"Reason: {state['stop_reason']}."
            )
            return {"report": report}
        case STOP_REASONS.STUCK_DETECTED:
            report = (
                f"The issue has not been corrected in {state['tries']} attempts. "
                f"Reason: {state['stop_reason']}."
            )
            return {"report": report}
        case STOP_REASONS.NO_OUTCOME:
            report = (
                f"The issue has not been corrected in {state['tries']} attempts. "
                f"Reason: {state['stop_reason']}."
            )
            return {"report": report}
        case STOP_REASONS.OSCILATION_DETECTED:
            report = (
                f"The issue has not been corrected in {state['tries']} attempts. "
                f"Reason: {state['stop_reason']}."
            )
            return {"report": report}

    return {
        "report": (
            "Process finished without a known outcome after "
            f"{state.get('tries', 0)} attempts."
        ),
    }


# ---------------------------------------------------------------------------
# Graph assembly
# ---------------------------------------------------------------------------


build_graph = StateGraph(
    StateForge,
    input_schema=InputForge,
    output_schema=OutputForge,
)

build_graph.add_node("gather", gather)
build_graph.add_node("try_and_verify", try_and_verify)
build_graph.add_node("tools", ToolNode(TOOLS))
build_graph.add_node("verify", verify)
build_graph.add_node("change_strategy", change_strategy)
build_graph.add_node(
    "decide",
    decide,
    destinations=("try_and_verify", "finish", "change_strategy"),
)
build_graph.add_node("finish", finish)

build_graph.add_edge(START, "gather")
build_graph.add_edge("gather", "try_and_verify")
build_graph.add_conditional_edges(
    "try_and_verify",
    tools_condition,
    {"tools": "tools", "__end__": "verify"},
)
build_graph.add_edge("change_strategy", "try_and_verify")
build_graph.add_edge("tools", "try_and_verify")
build_graph.add_edge("verify", "decide")
build_graph.add_edge("finish", END)

forge_graph = build_graph.compile()


def save_graph_png(path: Path = GRAPH_IMAGE_PATH) -> Path:
    """Render the compiled graph to a PNG file."""
    forge_graph.get_graph().draw_mermaid_png(output_file_path=str(path))
    return path


def print_messages(messages: list) -> None:
    """Print the conversation transcript in a readable format."""
    for message in messages:
        if getattr(message, "tool_calls", None):
            tool_call = message.tool_calls[0]
            print(f"[AI -> tool] {tool_call['name']}({tool_call['args']})")
        elif message.type == "tool":
            print(f"[tool -> AI] {message.content}")
        elif message.type == "ai":
            print(f"[AI -> Human] {message.content}")
        elif message.type == "human":
            print(f"[Human -> AI] {message.content}")
        else:
            print(f"[{message.type}] {message.content}")


if __name__ == "__main__":
    save_graph_png()
    print(f"Graph saved at: {GRAPH_IMAGE_PATH}")

    for scenario in SCENARIOS:
        result = forge_graph.invoke(
            {
                "issue": "The system is not responding to the customer email.",
                "scenario": scenario,
                "max_tries": INITIAL_MAX_TRIES,
                "start": time.time(),
                "max_seconds": 60.0,
                "patience": 3,
            },
            config={"recursion_limit": RECURSION_LIMIT},
        )
        print(result["report"])
        print("\nTimeline:")
        for event in result.get("history", []):
            print(f"- {event}")
        messages = result.get("messages", [])
        if messages:
            print("\nMessages:")
            print_messages(messages)

