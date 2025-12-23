<p align="center">
  <img src="https://img.shields.io/badge/python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python 3.10+"/>
  <img src="https://img.shields.io/badge/SQLite-WAL%20+%20FTS5-003B57?style=for-the-badge&logo=sqlite&logoColor=white" alt="SQLite"/>
  <img src="https://img.shields.io/badge/LLM-llama.cpp-FF6B35?style=for-the-badge" alt="llama.cpp"/>
</p>

<h1 align="center">⚡ CogOS</h1>
<h3 align="center">A Production-Grade Cognitive Operating System for AI Agents</h3>

<p align="center">
  <em>Where every claim is evidence-grounded, every action is verified, and hallucination is not an option.</em>
</p>

---

## The Problem

Modern AI agents confidently produce plausible-sounding nonsense. They hallucinate facts, invent citations, and present speculation as truth. In production environments—where decisions have consequences—this is unacceptable.

**CogOS takes a different approach: if a claim can't be traced back to verifiable evidence, it doesn't get said.**

## Architecture

CogOS implements a rigorous **Plan → Execute → Reason → Verify → Render** pipeline that treats every response as a court case—claims require evidence, evidence requires provenance, and the system abstains rather than fabricates.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              User Query                                     │
└─────────────────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                            PLANNER                                          │
│  • Rule-based (deterministic arithmetic, string ops)                        │
│  • LLM-guided (complex multi-step plans)                                    │
│  Outputs: StepMemorySearch | StepToolCall | StepWriteNote | StepCreateTask  │
└─────────────────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                          TOOL EXECUTION                                     │
│  • calc, count_chars, now                                                   │
│  • memory_search (hybrid lexical + vector)                                  │
│  • read_text_file, write_text_file                                          │
│  • web_search (optional, domain-allowlisted)                                │
│  Every execution → Evidence record with trust_score                         │
└─────────────────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                            REASONER                                         │
│  • Conservative: tool-derived claims only, zero hallucination               │
│  • LLM: evidence-grounded claims with support_spans                         │
│  • Search: best-of-N sampling for verifiable claim sets                     │
│  Outputs: ProposedAnswer with Claim[] + evidence_ids + support_spans        │
└─────────────────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           VERIFIER                                          │
│  • Evidence IDs must exist in memory                                        │
│  • Support spans must appear VERBATIM in evidence                           │
│  • Numeric claims must have numbers grounded in evidence                    │
│  • Trust score thresholds enforced                                          │
│  Rejected claims are DROPPED, not paraphrased                               │
└─────────────────────────────────────────────────────────────────────────────┘
                                     │
                            ┌────────┴────────┐
                            │   verified?     │
                            └────────┬────────┘
                         yes │               │ no
                             ▼               ▼
                    ┌────────────┐   ┌────────────────────┐
                    │  RENDERER  │   │      NOTARY        │
                    │  Response  │   │  Escalate to human │
                    └────────────┘   │  review (task)     │
                                     └────────────────────┘
```

## Core Principles

### 🔒 Evidence-First

Every claim links to `evidence_ids` and `support_spans`. The Verifier rejects claims where:
- Evidence doesn't exist in memory
- Support spans aren't verbatim substrings of evidence
- Numbers in claims aren't grounded in evidence text

### 🧠 Hybrid Memory

SQLite-backed with:
- **FTS5** for blazing-fast lexical search
- **Vector embeddings** for semantic similarity
- **RRF fusion** combining both for superior retrieval
- **WAL mode** for concurrent reads during background daemon activity

### 🤖 Pluggable Intelligence

| Component | Options |
|-----------|---------|
| **Embedder** | `hash` (zero-dep, 384d) or `st` (sentence-transformers) |
| **LLM Backend** | `stub` (offline) or `llama_cpp` (local GGUF models) |
| **Planner** | `rule` (deterministic regex) or `llm` (model-guided) |
| **Reasoner** | `conservative` (zero hallucination), `llm`, or `search` (best-of-N) |

### 🏃 Background Daemons

CogOS runs autonomous maintenance via an event-driven daemon system:

| Daemon | Purpose |
|--------|---------|
| **ReflectionDaemon** | Archives conversation fragments as searchable notes |
| **PruningDaemon** | Summarizes old episodes into digests, bounds memory growth |
| **ConnectionMiner** | Discovers and links related notes automatically |
| **ConsistencyAuditor** | Flags contradictory notes for human review |
| **TaskSolverDaemon** | Background worker for queued tasks |

### 🚨 Notary Escalation

When CogOS cannot produce verified claims even after auto-research, the **Notary** hard-cuts the response and escalates to human review. No confident fabrications—just honest abstention.

```python
"I can't verify any claims from trusted evidence.
I've flagged this for human review and will not proceed further on this thread."
```

### 💡 Proactive Initiative

The `InitiativeManager` surfaces background insights using a utility-based scoring model:

```
impulse = (expected_utility × confidence × actionability) - (interruption_cost + risk)
```

Rate-limited and cooldown-protected to avoid spam.

## Quick Start

### Installation

```bash
git clone https://github.com/theapemachine/architecture.git
cd architecture
pip install -r requirements.txt

# Optional: local LLM support
pip install -r requirements-llama.txt
```

### Run Interactive Chat

```bash
# Minimal (offline, deterministic)
python cogos.py chat

# With local LLM
python cogos.py chat \
  --llm-backend llama_cpp \
  --llama-auto-download \
  --planner llm \
  --reasoner search

# With web search (domain-allowlisted)
python cogos.py chat \
  --allow-web-search \
  --auto-research \
  --notary
```

### Self-Test

```bash
python cogos.py selftest
# Selftest OK ✅
```

### Download Model

```bash
python cogos.py download-model
# Downloads TinyLlama-1.1B-Chat GGUF to ./models/
```

## CLI Commands

Once in the chat REPL:

| Command | Description |
|---------|-------------|
| `/tools` | List available tools |
| `/notes` | Show recent notes |
| `/skills` | Show learned skills |
| `/episodes` | View conversation history |
| `/tasks` | List background tasks |
| `/task add <title>::<desc>` | Create a task |
| `/poll` | Retrieve proactive messages |
| `/quit` | Exit |

## Configuration

Key CLI flags (see `--help` for full list):

```bash
--db cogos.db              # SQLite database path
--session-id default       # Session isolation
--embedder hash|st         # Embedding model
--llm-backend stub|llama_cpp
--planner rule|llm
--reasoner conservative|llm|search
--allow-side-effects       # Enable write_text_file
--allow-web-search         # Enable web_search tool
--auto-research            # Auto-hydrate from web if memory empty
--notary                   # Enable human escalation
--prune-episodes           # Auto-summarize old episodes
--min-evidence-trust 0.5   # Reject low-trust evidence
```

Environment variables:
- `COGOS_LLAMA_MODEL` - Path to local GGUF file
- `COGOS_LLAMA_MODEL_DIR` - Directory for downloaded models
- `COGOS_LLAMA_GPU_LAYERS` - GPU offloading layers

## Memory Schema

```sql
episodes    -- Conversation turns (user/assistant)
notes       -- Extracted knowledge with tags, links, confidence
evidence    -- Tool outputs with SHA256 deduplication
skills      -- Learned procedures (preconditions, steps, tests)
tasks       -- Background work queue with priority/status
proactive   -- Pending initiative messages
```

All tables support hybrid retrieval via FTS5 + vector embeddings.

## Design Decisions

### Why SQLite?

- Single-file deployment
- WAL mode handles concurrent daemon writes
- FTS5 is faster than external search for this scale
- Zero operational overhead

### Why Verbatim Span Verification?

LLMs paraphrase freely. Requiring exact substring matches prevents:
- Citation drift ("The paper says X" when it says Y)
- Numeric hallucination (inventing statistics)
- Confidence laundering (restating low-confidence info as fact)

### Why Abstention Over Hedging?

"I'm not sure, but..." still plants false information. CogOS either:
1. Produces verified claims with evidence
2. Abstains and escalates to humans

There is no middle ground where unverified speculation reaches the user.

## Extending CogOS

### Adding Tools

```python
from cogos.tools import ToolSpec, ToolBus

class MyInput(BaseModel):
    query: str

class MyOutput(BaseModel):
    result: str

def my_handler(inp: MyInput) -> MyOutput:
    return MyOutput(result=f"Processed: {inp.query}")

tools.register(ToolSpec(
    name="my_tool",
    description="Does something useful",
    input_model=MyInput,
    output_model=MyOutput,
    handler=my_handler,
    side_effects=False,  # True requires --allow-side-effects
))
```

### Adding Daemons

```python
from cogos.daemons import Daemon, DaemonContext

class MyDaemon(Daemon):
    name = "my_daemon"
    tick_every_s = 10.0

    def on_event(self, evt: Event, ctx: DaemonContext) -> None:
        if evt.type == "note_added":
            # React to new notes
            pass

    def tick(self, ctx: DaemonContext) -> None:
        # Periodic maintenance
        pass
```

## Roadmap

- [ ] Hierarchical task decomposition with parent/child dependencies
- [ ] Skill synthesis from successful task completions
- [ ] Multi-agent coordination via shared memory
- [ ] Web UI for task review and note exploration
- [ ] Fine-tuned verifier models

## Philosophy

CogOS embodies a simple belief: **AI systems should know what they don't know.**

Every architectural decision flows from this principle:
- Evidence linking prevents hallucination
- Span verification prevents paraphrasing drift
- Trust scoring prevents evidence poisoning
- Notary escalation prevents confident fabrication
- Abstention prevents speculation

The goal is not a system that always answers—it's a system you can actually trust.

---

<p align="center">
  <strong>CogOS: Trust Through Transparency</strong>
</p>

