"""Forge: LangGraph workflow for iterative technical support with tool calling.

Edit ``ATTEMPT_MESSAGES`` to drive the fake model. Each model response is evaluated
by parsing tags in the output ([Score:], [Action:], [Discovery:], etc.).
Adjust ``max_tries``, ``max_seconds`` and ``patience`` in ``run_forge()`` to
exercise different stop conditions.
"""

import operator
import re
import time
from enum import Enum
from pathlib import Path
from typing import Annotated

from langchain_core.messages import (
    AIMessage, HumanMessage, SystemMessage, RemoveMessage
)
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
PATTERN_WINDOW = 4
PRUNING_KEEP_LAST = 4
SUMMARIZE_MESSAGE_THRESHOLD = 8
PASS_THRESHOLD = 3
TRIES_PER_STRATEGY = 3
INITIAL_MAX_TRIES = 20
BUDGET_EXTENSION = TRIES_PER_STRATEGY
MAX_STRATEGY_CHANGES = 1

GRAPH_IMAGE_PATH = Path(__file__).parent / "forge_graph.png"

FORGE_INTRODUCTION = SystemMessage(
    content=(
        "You are a technical support assistant that fixes system issues. "
        f"When the fix covers the issue, end with the word {DONE_KEYWORD.upper()}."
    )
)

runtime_model = None

# ---------------------------------------------------------------------------
# Simulated repository
# ---------------------------------------------------------------------------

REPO = {
    "auth.py": "def valida_email(e): \n    return '@' in e # bug: case-sensitive\n"
}

_applied_correction = {"done": False}

# ---------------------------------------------------------------------------
# Schemas and enums
# ---------------------------------------------------------------------------


class STOP_REASONS(Enum):
    SUCCESS = "success"
    BUDGET_EXCEEDED = "budget_exceeded"
    STUCK_DETECTED = "stuck_detected"
    NO_OUTCOME = "no_outcome"
    OSCILATION_DETECTED = "oscillation_detected"


class ACTION_PATTERNS(Enum):
    UNIFORM = "uniform"
    ALTERNATING = "alternating"


class NODES(str, Enum):
    """Graph node identifiers."""

    GATHER = "gather"
    PRUNING = "pruning"
    ACT = "act"
    TOOLS = "tools"
    VERIFY = "verify"
    CHANGE_STRATEGY = "change_strategy"
    DECIDE = "decide"
    FINISH = "finish"
    SUMMARIZE = "summarize"


class InputForge(TypedDict):
    issue: str
    max_tries: NotRequired[int]
    start: NotRequired[float]
    max_seconds: NotRequired[float]
    patience: NotRequired[int]


class OutputForge(TypedDict):
    report: str
    history: list
    stop_reason: str
    total_cost: float
    messages: list
    discoveries: list
    dead_ends: list


class StateForge(MessagesState):
    issue: str
    tries: int
    start: float
    max_tries: int
    max_seconds: float
    patience: int
    scores: Annotated[list[float], operator.add]
    best_score: float
    test_ok: bool
    report: str
    dead_ends: Annotated[list[str], operator.add]
    discoveries: Annotated[list[str], operator.add]
    actions: list[str]
    stop_reason: STOP_REASONS
    strategy_changes: int
    messages: Annotated[list, operator.add]
    total_cost: Annotated[float, operator.add]
    history: Annotated[list[str], operator.add]

# ---------------------------------------------------------------------------
# Model helpers
# ---------------------------------------------------------------------------


def build_model(attempt_messages: list[AIMessage]):
    """Build a chat model whose responses come from ``attempt_messages``."""
    if REAL_MODEL:
        from langchain.chat_models import init_chat_model

        return init_chat_model("claude-haiku-4-5", model_provider="anthropic")

    from langchain_core.language_models.fake_chat_models import GenericFakeChatModel

    return GenericFakeChatModel(messages=iter(attempt_messages))


def ai(content: str, **kwargs) -> AIMessage:
    """Build one fake model turn. Embed [Score:] and [Action:] for evaluation."""
    return AIMessage(content=content, **kwargs)


def reset_repo_state() -> None:
    global _applied_correction

    _applied_correction = {"done": False}
    REPO["auth.py"] = (
        "def valida_email(e): \n    return '@' in e # bug: case-sensitive\n"
    )

# ---------------------------------------------------------------------------
# Fake model turns — edit this list to test different flows
# ---------------------------------------------------------------------------


ATTEMPT_MESSAGES: list[AIMessage] = [
    ai(
        "Investigating email validation. "
        "[Score: 1] [Action: A] "
        "[Discovery: Customer reports valid emails are rejected silently]"
    ),
    ai(
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
    ai(
        "Search hit auth.py — reading the validator next. "
        "[Score: 1] [Action: A] "
        "[Discovery: valida_email is defined in auth.py]"
    ),
    ai(
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
    ai(
        "The check is too strict for mixed-case addresses. "
        "[Score: 1] [Action: A] "
        "[Discovery: validation uses a naive '@' substring match] "
        "[Dead end: restarting the auth service did not change behavior]"
    ),
    ai(
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
    ai(
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
    ai(
        f"Fix applied. Tests green. {DONE_KEYWORD.upper()}. "
        "[Score: 3] [Action: B] "
        "[Discovery: test_email_validation passes after auth.py patch]"
    ),
]

# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


@tool("execute_test", description="Execute a test suite and return the result.")
def execute_test(suite: str) -> str:
    if _applied_correction["done"]:
        return "Result: 2 passed, 0 failed."
    return "Result: 0 passed, 2 failed."


@tool("read_file", description="Read a file and return the content.")
def read_file(path: str) -> str:
    if path not in REPO:
        return f"ERROR: '{path}' not found in repo. Files: {list(REPO)}"
    return REPO[path]


@tool("apply_correction", description="Apply a correction to a file.")
def apply_correction(path: str, correction: str) -> str:
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
# Response evaluation (parses model output, not a parallel spec)
# ---------------------------------------------------------------------------

TAG_PATTERN = re.compile(r"\[(Score|Action|Discovery|Dead end):([^\]]+)\]")


def _parse_tag(content: str, tag: str) -> str | None:
    for name, value in TAG_PATTERN.findall(content):
        if name == tag:
            return value.strip()
    return None


def evaluate_model_response(message: AIMessage) -> dict:
    """Derive score, action and journal entries from the model message."""
    content = message.content or ""

    score_raw = _parse_tag(content, "Score")
    if score_raw is not None:
        score = float(score_raw)
    elif DONE_KEYWORD in content.lower():
        score = float(PASS_THRESHOLD)
    elif message.tool_calls:
        score = 0.0
    else:
        score = 1.0

    action = _parse_tag(content, "Action") or "A"
    discoveries = [
        value.strip()
        for name, value in TAG_PATTERN.findall(content)
        if name == "Discovery"
    ]
    dead_ends = [
        value.strip()
        for name, value in TAG_PATTERN.findall(content)
        if name == "Dead end"
    ]
    passed = score >= PASS_THRESHOLD or DONE_KEYWORD in content.lower()

    return {
        "score": score,
        "action": action,
        "discoveries": discoveries,
        "dead_ends": dead_ends,
        "test_ok": passed,
    }


def _has_repeating_period(sequence: list[str], period: int) -> bool:
    if period <= 0 or period > len(sequence):
        return False
    return all(
        sequence[i] == sequence[i - period] for i in range(period, len(sequence))
    )


def detect_action_pattern(window: list[str]) -> ACTION_PATTERNS | None:
    if not window:
        return None
    if _has_repeating_period(window, 1):
        return ACTION_PATTERNS.UNIFORM
    if len(window) >= 2 and _has_repeating_period(window, 2) and len(set(window)) == 2:
        return ACTION_PATTERNS.ALTERNATING
    return None


def detect_repetitive_pattern(
    actions: list[str],
    window_size: int = PATTERN_WINDOW,
) -> ACTION_PATTERNS | None:
    if len(actions) < window_size:
        return None
    return detect_action_pattern(actions[-window_size:])


def is_score_stuck(scores: list[float], patience: int) -> bool:
    if patience <= 0 or len(scores) <= patience:
        return False
    return max(scores[-patience:]) <= max(scores[:-patience])


def recursion_limit_for(message_count: int) -> int:
    return max(30, 1 + message_count * 6)

# ---------------------------------------------------------------------------
# Workflow nodes
# ---------------------------------------------------------------------------


def gather(state: StateForge) -> dict:
    if state.get("messages"):
        return {"history": [f"{NODES.GATHER.value}: reused context"]}

    return {
        "messages": [HumanMessage(content=f"Issue: {state['issue']}")],
        "actions": [],
        "history": [f"{NODES.GATHER.value}: initial context"],
    }


def act(state: StateForge) -> dict:
    """Invoke the model and evaluate its response."""
    if runtime_model is None:
        raise RuntimeError("runtime_model is not set — call run_forge() first.")

    tries = state.get("tries", 0) + 1

    journal_parts = []
    if state.get("discoveries"):
        journal_parts.append("Discoveries: " + "; ".join(state["discoveries"]))
    if state.get("dead_ends"):
        journal_parts.append("Dead ends: " + "; ".join(state["dead_ends"]))

    intro_content = FORGE_INTRODUCTION.content
    if journal_parts:
        intro_content += "\n" + "\n".join(journal_parts)

    result = runtime_model.invoke(
        [SystemMessage(content=intro_content)] + state["messages"]
    )

    evaluation = evaluate_model_response(result)
    actions = list(state.get("actions", []))
    actions.append(evaluation["action"])

    preview = (result.content or "[tool call]")[:50]
    history = [
        f"{NODES.ACT.value} (try {tries}): "
        f"{evaluation['score']}/{PASS_THRESHOLD} — {preview}"
    ]
    for discovery in evaluation["discoveries"]:
        history.append(f"notebook +discovery: {discovery}")
    for dead_end in evaluation["dead_ends"]:
        history.append(f"notebook +dead end: {dead_end}")

    update: dict = {
        "messages": [result],
        "tries": tries,
        "actions": actions,
        "discoveries": evaluation["discoveries"],
        "dead_ends": evaluation["dead_ends"],
        "test_ok": evaluation["test_ok"],
        "history": history,
    }
    if not result.tool_calls or _parse_tag(result.content or "", "Score") is not None:
        update["scores"] = [evaluation["score"]]

    return update


def pruning(state: StateForge) -> dict:
    """Keep the first message and the last ``PRUNING_KEEP_LAST`` messages."""
    messages = state.get("messages", [])

    if len(messages) <= 1 + PRUNING_KEEP_LAST:
        return {"history": [f"{NODES.PRUNING.value}: window within limit"]}

    to_remove = [
        message
        for message in messages[1:-PRUNING_KEEP_LAST]
        if message.id is not None
    ]
    if not to_remove:
        return {"history": [f"{NODES.PRUNING.value}: nothing to remove"]}

    return {
        "messages": [RemoveMessage(id=message.id) for message in to_remove],
        "history": [
            f"{NODES.PRUNING.value}: removed {len(to_remove)} message(s), "
            f"kept first + last {PRUNING_KEEP_LAST}"
        ],
    }


def summarize(state: StateForge) -> dict:
    """Keep the first message and the last two; summarize the middle into one SystemMessage."""
    keep_last = 2
    messages = state.get("messages", [])

    if len(messages) <= 1 + keep_last:
        return {"history": [f"{NODES.SUMMARIZE.value}: window within limit"]}

    middle = messages[1:-keep_last]
    if not middle:
        return {"history": [f"{NODES.SUMMARIZE.value}: nothing to summarize"]}

    def _format_message(message) -> str:
        role = type(message).__name__.replace("Message", "")
        content = message.content or ""
        if getattr(message, "tool_calls", None):
            calls = ", ".join(tc["name"] for tc in message.tool_calls)
            return f"{role}[tool_calls: {calls}]"
        preview = content.replace("\n", " ").strip()
        if len(preview) > 300:
            preview = preview[:300] + "..."
        return f"{role}: {preview}" if preview else f"{role}: (empty)"

    middle_text = "\n".join(f"- {_format_message(message)}" for message in middle)

    if REAL_MODEL:
        if runtime_model is None:
            raise RuntimeError("runtime_model is not set — call run_forge() first.")
        result = runtime_model.invoke(
            [
                SystemMessage(
                    content=(
                        "Summarize the conversation excerpt below. "
                        "Keep key facts, discoveries, tool outcomes, and dead ends."
                    )
                ),
                HumanMessage(content=middle_text),
            ]
        )
        summary_content = result.content or middle_text
    else:
        summary_content = (
            "Summarized pruned context (keep these points in the next steps):\n"
            f"{middle_text}"
        )

    to_remove = [message for message in middle if message.id is not None]
    if not to_remove:
        return {"history": [f"{NODES.SUMMARIZE.value}: no removable middle messages"]}

    return {
        "messages": [
            *[RemoveMessage(id=message.id) for message in to_remove],
            SystemMessage(content=summary_content),
        ],
        "history": [
            f"{NODES.SUMMARIZE.value}: summarized {len(middle)} message(s), "
            f"kept first + last {keep_last}"
        ],
    }


def verify(state: StateForge) -> dict:
    scores = state.get("scores", [])
    last_score = scores[-1] if scores else 0
    passed = last_score >= PASS_THRESHOLD or state.get("test_ok", False)

    return {
        "test_ok": passed,
        "history": [
            f"{NODES.VERIFY.value}: attempt {'passed' if passed else 'failed'}"
        ],
    }


def change_strategy(state: StateForge) -> dict:
    current_budget = state.get("max_tries", INITIAL_MAX_TRIES)
    return {
        "history": [f"{NODES.CHANGE_STRATEGY.value}: switching approach"],
        "messages": [HumanMessage(content="Let's try a different strategy.")],
        "actions": [],
        "max_tries": current_budget + BUDGET_EXTENSION,
    }


def decide(state: StateForge) -> Command:
    scores = state.get("scores", [])
    actions = state.get("actions", [])
    last_score = scores[-1] if scores else 0
    best_score = max(scores) if scores else 0
    patience = state.get("patience", 3)

    if state.get("test_ok") or last_score >= PASS_THRESHOLD:
        return Command(
            update={"stop_reason": STOP_REASONS.SUCCESS, "best_score": best_score},
            goto=NODES.FINISH,
        )

    if state.get("tries", 0) >= state.get("max_tries", INITIAL_MAX_TRIES):
        return Command(
            update={
                "stop_reason": STOP_REASONS.BUDGET_EXCEEDED,
                "best_score": best_score,
            },
            goto=NODES.FINISH,
        )

    if time.time() - state.get("start", 0) >= state.get("max_seconds", 0):
        return Command(
            update={"stop_reason": STOP_REASONS.NO_OUTCOME, "best_score": best_score},
            goto=NODES.FINISH,
        )

    pattern = detect_repetitive_pattern(actions)
    if pattern == ACTION_PATTERNS.ALTERNATING:
        return Command(
            update={
                "stop_reason": STOP_REASONS.OSCILATION_DETECTED,
                "best_score": best_score,
                "history": [
                    f"{NODES.DECIDE.value}: {pattern.value} pattern in last "
                    f"{PATTERN_WINDOW} actions"
                ],
            },
            goto=NODES.FINISH,
        )

    if is_score_stuck(scores, patience):
        return Command(
            update={
                "stop_reason": STOP_REASONS.STUCK_DETECTED,
                "best_score": best_score,
                "history": [f"{NODES.DECIDE.value}: score plateau detected"],
            },
            goto=NODES.FINISH,
        )

    if pattern == ACTION_PATTERNS.UNIFORM:
        if state.get("strategy_changes", 0) < MAX_STRATEGY_CHANGES:
            return Command(
                update={
                    "strategy_changes": state.get("strategy_changes", 0) + 1,
                    "history": [
                        f"{NODES.DECIDE.value}: {pattern.value} "
                        "pattern — changing strategy"
                    ],
                },
                goto=NODES.CHANGE_STRATEGY,
            )
        return Command(
            update={
                "stop_reason": STOP_REASONS.STUCK_DETECTED,
                "best_score": best_score,
                "history": [
                    f"{NODES.DECIDE.value}: uniform pattern after strategy exhausted"
                ],
            },
            goto=NODES.FINISH,
        )

    messages = state.get("messages", [])
    if len(messages) > SUMMARIZE_MESSAGE_THRESHOLD:
        return Command(
            update={
                "history": [
                    f"{NODES.DECIDE.value}: {len(messages)} messages "
                    f"(>{SUMMARIZE_MESSAGE_THRESHOLD}) — summarizing"
                ],
            },
            goto=NODES.SUMMARIZE,
        )

    return Command(goto=NODES.PRUNING)


def finish(state: StateForge) -> dict:
    reason = state["stop_reason"]
    tries = state.get("tries", 0)

    if reason == STOP_REASONS.SUCCESS:
        report = f"Issue corrected in {tries} attempts. Reason: {reason.value}."
    else:
        report = f"Issue not corrected in {tries} attempts. Reason: {reason.value}."

    return {"report": report, "stop_reason": reason.value}

# ---------------------------------------------------------------------------
# Graph assembly
# ---------------------------------------------------------------------------


build_graph = StateGraph(
    StateForge,
    input_schema=InputForge,
    output_schema=OutputForge,
)

build_graph.add_node(NODES.GATHER, gather)
build_graph.add_node(NODES.ACT, act)
build_graph.add_node(NODES.PRUNING, pruning)
build_graph.add_node(NODES.SUMMARIZE, summarize)
build_graph.add_node(NODES.TOOLS, ToolNode(TOOLS))
build_graph.add_node(NODES.VERIFY, verify)
build_graph.add_node(NODES.CHANGE_STRATEGY, change_strategy)
build_graph.add_node(
    NODES.DECIDE,
    decide,
    destinations=(
        NODES.PRUNING,
        NODES.SUMMARIZE,
        NODES.FINISH,
        NODES.CHANGE_STRATEGY,
    ),
)
build_graph.add_node(NODES.FINISH, finish)

build_graph.add_edge(START, NODES.GATHER)
build_graph.add_edge(NODES.GATHER, NODES.ACT)
build_graph.add_conditional_edges(
    NODES.ACT,
    tools_condition,
    {"tools": NODES.TOOLS, "__end__": NODES.VERIFY},
)
build_graph.add_edge(NODES.TOOLS, NODES.ACT)
build_graph.add_edge(NODES.VERIFY, NODES.DECIDE)
build_graph.add_edge(NODES.PRUNING, NODES.ACT)
build_graph.add_edge(NODES.SUMMARIZE, NODES.ACT)
build_graph.add_edge(NODES.CHANGE_STRATEGY, NODES.ACT)
build_graph.add_edge(NODES.FINISH, END)

forge_graph = build_graph.compile()


def save_graph_png(path: Path = GRAPH_IMAGE_PATH) -> Path:
    graph = forge_graph.get_graph()
    graph.draw_mermaid_png(
        output_file_path=str(path),
        max_retries=5,
        retry_delay=2.0,
    )
    return path


def run_forge(
    issue: str = "The system is not responding to the customer email.",
    attempt_messages: list[AIMessage] | None = None,
    *,
    max_tries: int = INITIAL_MAX_TRIES,
    max_seconds: float = 60.0,
    patience: int = 3,
) -> dict:
    """Configure the model from ``attempt_messages`` and run the graph once."""
    global runtime_model

    reset_repo_state()
    messages = attempt_messages if attempt_messages is not None else ATTEMPT_MESSAGES
    runtime_model = build_model(messages)

    return forge_graph.invoke(
        {
            "issue": issue,
            "max_tries": max_tries,
            "start": time.time(),
            "max_seconds": max_seconds,
            "patience": patience,
        },
        config={"recursion_limit": recursion_limit_for(len(messages))},
    )


def print_messages(messages: list) -> None:
    for message in messages:
        role = type(message).__name__.replace("Message", "").lower()
        content = message.content or ""
        if hasattr(message, "tool_calls") and message.tool_calls:
            calls = ", ".join(tc["name"] for tc in message.tool_calls)
            content = f"[tool_calls: {calls}]"
        print(f"  [{role}] {content[:120]}")


def print_notebook(discoveries: list[str], dead_ends: list[str]) -> None:
    print("\nNotebook:")
    print("  Discoveries:")
    if discoveries:
        for index, entry in enumerate(discoveries, start=1):
            print(f"    {index}. {entry}")
    else:
        print("    (none)")
    print("  Dead ends:")
    if dead_ends:
        for index, entry in enumerate(dead_ends, start=1):
            print(f"    {index}. {entry}")
    else:
        print("    (none)")


if __name__ == "__main__":
    save_graph_png()
    print(f"Graph saved at: {GRAPH_IMAGE_PATH}\n")

    result = run_forge()
    print(f"Stop reason: {result['stop_reason']}")
    print(f"Report: {result['report']}")
    print("\nTimeline:")
    for entry in result["history"]:
        print(f"  {entry}")
    print("\nMessages:")
    print_messages(result["messages"])
    print_notebook(result.get("discoveries", []), result.get("dead_ends", []))
