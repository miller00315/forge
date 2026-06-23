# Forge

Protótipo de workflow em [LangGraph](https://langchain-ai.github.io/langgraph/) que simula um agente de suporte técnico: investiga uma issue, usa ferramentas para buscar, ler e corrigir código, executa testes e decide se continua, muda de estratégia ou encerra.

O objetivo é experimentar grafos com estado, mensagens, tool calling, histórico acumulado, roteamento condicional e controle de orçamento de tentativas.

## Fluxo do grafo

```
START → gather → act ──(tool_calls?)──→ tools ──→ act
                      └──(sem tools)──→ verify → decide ──(iterar)──→ act
                                                    ├──(mudar estratégia)──→ change_strategy → act
                                                    └──(encerrar)──→ finish → END
```

| Nó | Responsabilidade |
|---|---|
| `gather` | Monta o contexto inicial (`HumanMessage` com a issue) |
| `act` | Invoca o modelo de chat e registra a tentativa |
| `tools` | Executa tool calls do agente |
| `verify` | Verifica se a última resposta contém `PRONTO` |
| `decide` | Roteia via `Command`: iterar, mudar estratégia ou encerrar |
| `change_strategy` | Reseta janela de falhas e estende o orçamento |
| `finish` | Gera o relatório final |

O roteamento após `act` usa `tools_condition`: se a última mensagem tiver `tool_calls`, vai para `tools`; caso contrário, vai para `verify`.

## Ferramentas

Repositório simulado em memória (`REPO`):

| Tool | Função |
|---|---|
| `search_in_repo(query)` | Busca string no **nome** e no **conteúdo** dos arquivos |
| `read_file(path)` | Lê o conteúdo de um arquivo do repo |
| `apply_correction(path, correction)` | Aplica patch no arquivo e marca correção como feita |
| `execute_test(suite)` | Roda suite de testes (passa só após correção aplicada) |

Exemplo de `search_in_repo`:

```python
search_in_repo("valida_email")  # → auth.py (content)
search_in_repo("auth.py")       # → auth.py (path)
```

## Modelo

Por padrão (`REAL_MODEL = False`), usa `GenericFakeChatModel` com respostas e `tool_calls` pré-definidos — útil para desenvolver sem API key.

Fluxo simulado do fake model:

1. Resposta inicial em texto
2. `search_in_repo("valida_email")`
3. `read_file("auth.py")`
4. `apply_correction(...)`
5. `execute_test("test_email_validation")`
6. Mensagem final com `PRONTO`

Com `REAL_MODEL = True`, usa `init_chat_model` (requer pacote `langchain` e credenciais no `.env`).

## Estado

Schemas explícitos de entrada e saída:

- **Entrada (`InputForge`)**: `issue` (obrigatório), `max_tries` (opcional, padrão 7)
- **Saída (`OutputForge`)**: `report`, `history`, `total_cost`, `messages`

Campos principais do estado interno (`StateForge`):

| Campo | Descrição |
|---|---|
| `messages` | Histórico de conversa (reducer `operator.add`) |
| `tries` | Número de invocações do nó `act` |
| `max_tries` | Orçamento de tentativas (pode ser estendido ao mudar estratégia) |
| `test_ok` | Resultado da última verificação |
| `stop_reason` | `success` ou `budget_exceeded` |
| `strategy_changes` | Quantas vezes a estratégia foi alterada |
| `strategies_results` | Resultados da janela atual (`success` / `failed`) |
| `history` | Linha do tempo legível (reducer `operator.add`) |
| `total_cost` | Custo acumulado (reducer `operator.add`) |

### Orçamento e estratégia

Constantes em `main.py`:

| Constante | Valor | Significado |
|---|---|---|
| `TRIES_PER_STRATEGY` | 3 | Falhas consecutivas antes de mudar estratégia |
| `INITIAL_MAX_TRIES` | 7 | Orçamento inicial |
| `BUDGET_EXTENSION` | 3 | Tentativas extras ao mudar estratégia |
| `MAX_STRATEGY_CHANGES` | 1 | Máximo de mudanças de estratégia |

Comportamento no pior caso (sempre falha na verificação):

1. Três ciclos `act` falham → `change_strategy` (orçamento sobe para 10)
2. Mais três ciclos `act` falham → encerra com `budget_exceeded` (6 invocações de `act` no total)

### Critérios de parada (`decide`)

1. **Sucesso** — última mensagem contém `PRONTO`
2. **Orçamento excedido** — `tries >= max_tries`
3. **Estratégia esgotada** — 3 falhas na janela atual e limite de mudanças atingido

## Como executar

```bash
python -m venv .venv
source .venv/bin/activate
pip install langgraph langchain-core typing_extensions

python main.py
```

Saída esperada:

- `forge_graph.png` — diagrama do grafo
- Relatório final (`report`)
- Linha do tempo (`history`)
- Transcript de mensagens (`messages`)

## Estrutura do projeto

```
forge/
├── main.py           # Grafo, tools, modelo e execução
├── forge_graph.png   # Diagrama gerado automaticamente
├── .gitignore
├── .env              # Credenciais (quando REAL_MODEL = True)
└── README.md
```

## Evolução (changelog)

> Atualize esta seção a cada commit relevante.

| Commit | Descrição |
|---|---|
| _inicial_ | Grafo LangGraph com ciclo try/verify/decide, histórico, roteamento via `Command` |
| _v2_ | Nós `gather`/`act`, tool calling (`ToolNode`), repo simulado, verificação via `PRONTO`, mudança de estratégia, saída com `messages`, export PNG |
| _atual_ | Tool `search_in_repo` (busca em path e conteúdo), fluxo fake com search→read→correct→test, `INITIAL_MAX_TRIES` ajustado para 7 |
