import operator
from pathlib import Path
from typing import Annotated
from typing_extensions import TypedDict, NotRequired
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.types import Command
from langchain_core.messages import AIMessage, SystemMessage, HumanMessage
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.tools import tool


REAL_MODEL = False


# -- Repo  simulado--

REPO = {"auth.py": "def valida_email(e): \n    return '@' in e # bug: case-sensitive\n"}

_applied_correction = {"done": False}


FORGE_INTRODUCTION = SystemMessage(
    content="Você é um assistente de suporte técnico que resolve problemas de sistemas."
    "Quando a correção cobrir o problema, termine com a palavra PRONTO."
)


if REAL_MODEL:
    from langchain.chat_models import init_chat_model
    model = init_chat_model("claude-haiku-4-5", model_provider="anthropic")
else:
    from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
    model = GenericFakeChatModel(
        messages=iter(
            [
                AIMessage(content="Vou investigar a validação de email."),
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "search_in_repo",
                            "args": {
                                "query": "valida_email"
                            },
                            "id": "c1",
                            "type": "tool_call"
                        }
                    ]
                ),
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "read_file",
                            "args": {
                                "path": "auth.py"
                            },
                            "id": "c2",
                            "type": "tool_call"
                        }
                    ]
                ),
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "apply_correction",
                            "args": {
                                "path": "auth.py",
                                "correction": "def valida_email(e): \n    "
                                    "return '@' in e # bug: case-sensitive\n"
                            },
                            "id": "c3",
                            "type": "tool_call"
                        }
                    ]
                ),
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "execute_test",
                            "args": {
                                "suite": "test_email_validation"
                            },
                            "id": "c4",
                            "type": "tool_call"
                        }
                    ]
                ),
                AIMessage(
                    content="Correção aplicada: o caso da issue está completo. PRONTO."
                )
            ]
        )
    )


# -- facade --

class InputForge(TypedDict):
    issue: str
    max_tries: NotRequired[int]


class OutputForge(TypedDict):
    report: str
    history: list
    total_cost: float
    messages: list

# -- State --


TRIES_PER_STRATEGY = 3
INITIAL_MAX_TRIES = 7
BUDGET_EXTENSION = TRIES_PER_STRATEGY
MAX_STRATEGY_CHANGES = 1
RECURSION_LIMIT = (
    1  # gather
    + TRIES_PER_STRATEGY * (1 + MAX_STRATEGY_CHANGES) * 3
    + MAX_STRATEGY_CHANGES
    + 2  # finish + margem
)


class StateForge(MessagesState):
    issue: str
    tries: int
    max_tries: int
    test_ok: bool
    report: str
    stop_reason: str
    strategy_changes: int
    messages: Annotated[list, operator.add]
    total_cost: Annotated[float, operator.add]
    history: Annotated[list, operator.add]
    strategies_results: list

# -- Tools --


@tool("execute_test", description="Execute a test suite and return the result.")
def execute_test(suite: str) -> str:
    if _applied_correction["done"]:
        return "Resultado: 2 passaram, 0 falharam."
    return "Resultado: 0 passaram, 2 falharam."


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

    return f"Ok: path '{path}' updated ({len(correction)} chars)."


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

# -- Nodes of workflow --


def gather(state: StateForge) -> dict:
    if not state.get("messages"):
        return {
            "messages": [HumanMessage(content=f"Issue: {state['issue']}")],
            "history": ["gather: contexto inicial"],
        }

    return {"history": ["gather: contexto reaproveitado"]}


def act(state: StateForge) -> dict:
    tries = state.get("tries", 0) + 1

    result = model.invoke([FORGE_INTRODUCTION] + state['messages'])

    return {
        "messages": [result],
        "tries": tries,
        "history": [f"act (try {tries}): {result.content[:60]}"],
    }


def verify(state: StateForge) -> dict:
    """
    Simulate a verification of the correction.
    """
    passed = "pronto" in state['messages'][-1].content.lower()

    status = "success" if passed else "failed"
    strategies = list(state.get("strategies_results", []))
    strategies.append(status)

    return {
        "test_ok": passed,
        "history": [f"tentativa {'verdes' if passed else 'vermelhos'}"],
        "strategies_results": strategies,
    }


def change_strategy(state: StateForge) -> dict:
    current_budget = state.get("max_tries", INITIAL_MAX_TRIES)
    return {
        "history": ["decision: mudar estratégia"],
        "messages": [HumanMessage(content="Vamos tentar outra estratégia.")],
        "strategies_results": [],
        "max_tries": current_budget + BUDGET_EXTENSION,
    }


def decide(state: StateForge) -> Command:
    if state["test_ok"]:
        return Command(
            update={
                "stop_reason": "success",
                "history": ["decision: encerrar -testes verdes"]
            },
            goto="finish"
        )

    if state["tries"] >= state.get("max_tries", INITIAL_MAX_TRIES):
        return Command(
            update={
                "stop_reason": "budget_exceeded",
                "history": ["decision: encerrar - orçamento excedido"]
            },
            goto="finish"
        )

    strategies = state.get("strategies_results", [])
    last_three_failed = (
        len(strategies) >= TRIES_PER_STRATEGY
        and all(result == "failed" for result in strategies[-TRIES_PER_STRATEGY:])
    )
    if last_three_failed:
        if state.get("strategy_changes", 0) < MAX_STRATEGY_CHANGES:
            return Command(
                update={
                    "strategy_changes": state.get("strategy_changes", 0) + 1,
                },
                goto="change_strategy",
            )
        return Command(
            update={
                "stop_reason": "budget_exceeded",
                "history": ["decision: encerrar - orçamento excedido"],
            },
            goto="finish",
        )

    return Command(
        update={
            "history": ["decision: iterate - ainda há orçamento"]
        },
        goto="act"
    )


def finish(state: StateForge) -> dict:
    """
    Finish the process.
    """
    if state["stop_reason"] == "success":
        report = f"The issue has been corrected in {state['tries']} attempts."
        report += f"Reason: {state['stop_reason']}."

        return {
            "report": report,
        }

    elif state["stop_reason"] == "budget_exceeded":
        report = f"The issue has not been corrected in {state['tries']} attempts."
        report += f"Reason: {state['stop_reason']}."

        return {
            "report": report,
        }

    return {
        "report": (
            "Process finished without a known outcome after "
            f"{state.get('tries', 0)} attempts."
        ),
    }

# -- Graph --


build_graph = StateGraph(
    StateForge,
    input_schema=InputForge,
    output_schema=OutputForge
)

build_graph.add_node("gather", gather)

build_graph.add_node("act", act)

build_graph.add_node("tools", ToolNode(TOOLS))

build_graph.add_node("verify", verify)

build_graph.add_node("change_strategy", change_strategy)

build_graph.add_node(
    "decide",
    decide,
    destinations=("act", "finish", "change_strategy")
)

build_graph.add_node("finish", finish)

# -- Edges of workflow --

build_graph.add_edge(START, "gather")

build_graph.add_edge("gather", "act")

build_graph.add_conditional_edges(
    "act",
    tools_condition,
    {"tools": "tools", "__end__": "verify"},
)

build_graph.add_edge("change_strategy", "act")

build_graph.add_edge("tools", "act")

build_graph.add_edge("verify", "decide")

build_graph.add_edge("finish", END)

forge_graph = build_graph.compile()

GRAPH_IMAGE_PATH = Path(__file__).parent / "forge_graph.png"


def save_graph_png(path: Path = GRAPH_IMAGE_PATH) -> Path:
    forge_graph.get_graph().draw_mermaid_png(output_file_path=str(path))
    return path

# -- Run --


if __name__ == "__main__":

    save_graph_png()
    print(f"Grafo salvo em: {GRAPH_IMAGE_PATH}")

    result = forge_graph.invoke(
        {
            "issue": "O sistema não está respondendo ao email do cliente.", 
            "max_tries": INITIAL_MAX_TRIES
        },
        config={"recursion_limit": RECURSION_LIMIT},
    )

    print(result['report'])
    print("\nLinha do tempo:")
    for event in result.get("history", []):
        print(f"- {event}")

    messages = result.get("messages", [])
    if messages:
        print("\nMessages:")
    for message in messages:
        if getattr(message, "tool_calls", None):
            tool_calls = message.tool_calls[0]

            print(f"[AI -> tool]- {tool_calls['name']}({tool_calls['args']})")
        elif message.type == "tool":
            print(f"[tool -> AI]- {message.content}")

        elif message.type == "ai":
            print(f"[AI -> Human]- {message.content}")

        elif message.type == "human":
            print(f"[Human -> AI]- {message.content}")

        else:
            print(f"[{message.type}]- {message.content}")
