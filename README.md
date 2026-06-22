# Forge

Protótipo de workflow em [LangGraph](https://langchain-ai.github.io/langgraph/) que simula um ciclo de correção de problemas: propor correção, verificar, decidir se continua ou encerra.

O objetivo é experimentar padrões de grafos com estado, histórico acumulado, roteamento condicional e controle de orçamento de tentativas.

## Fluxo do grafo

```
START → try_correct → verify → decide ──(iterar)──→ try_correct
                              └──(encerrar)──→ finish → END
```

| Nó | Responsabilidade |
|---|---|
| `try_correct` | Incrementa tentativas, simula custo e registra a correção proposta |
| `verify` | Simula verificação (30% de chance de sucesso) e registra o resultado |
| `decide` | Decide continuar ou encerrar com base no teste e no limite de tentativas |
| `finish` | Gera o relatório final |

## Estado

O grafo usa `StateForge`, com schemas de entrada e saída explícitos:

- **Entrada (`InputForge`)**: `issue` — descrição do problema
- **Saída (`OutputForge`)**: `report`, `history`, `total_cost`

Campos principais do estado interno:

| Campo | Descrição |
|---|---|
| `tries` | Número de tentativas de correção |
| `max_tries` | Limite de tentativas (padrão: 5) |
| `test_ok` | Resultado da última verificação |
| `stop_reason` | Motivo do encerramento (`success` ou `budget_exceeded`) |
| `history` | Linha do tempo de eventos (com reducer `operator.add`) |
| `total_cost` | Custo acumulado das tentativas (com reducer `operator.add`) |

### Histórico

Cada ciclo registra eventos legíveis, por exemplo:

```
tentativa 1: correção proposta
tentativa 1: falhou
decision: iterate - ainda há orçamento
tentativa 2: correção proposta
tentativa 2: sucesso
decision: encerrar -testes verdes
```

### Critérios de parada

O nó `decide` encerra o fluxo quando:

1. **Sucesso** — a verificação passou (`stop_reason: success`)
2. **Orçamento excedido** — `tries >= max_tries` (`stop_reason: budget_exceeded`)

Caso contrário, retorna para `try_correct` via `Command(goto=...)`.

## Como executar

```bash
python -m venv .venv
source .venv/bin/activate
pip install langgraph typing_extensions

python main.py
```

Saída esperada: relatório final, linha do tempo (`history`) e custo total.

## Estrutura do projeto

```
forge/
├── main.py       # Grafo, nós e execução
├── .gitignore
└── README.md
```

## Evolução (changelog)

> Atualize esta seção a cada commit relevante.

| Commit | Descrição |
|---|---|
| _inicial_ | Grafo LangGraph com ciclo try/verify/decide, histórico acumulado, custo simulado, roteamento via `Command` e schemas de entrada/saída |
