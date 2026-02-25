# 🕸️ Agentic Graph-RAG Over Social-Network Knowledge Graphs

> **Author:** Solomon Adegoke — Data Scientist · AI/ML Engineer · Software Engineer · Engineering Physics  
> **Dataset:** [MUSAE Facebook Page–Page Network](https://snap.stanford.edu/data/facebook-large-page-page-network.html) (Stanford SNAP)  
> **Stack:** Python · NetworkX · PyTorch · PyTorch Geometric · LangGraph · OpenAI API

---

## 📌 Overview

Traditional **Retrieval-Augmented Generation (RAG)** pipelines retrieve *documents* — text chunks matched by semantic similarity. This works well for knowledge bases and PDFs, but falls short when the knowledge itself is *relational*.

This project introduces an **Agentic Graph-RAG** system that retrieves **subgraphs** instead of documents — extracting structurally relevant neighborhoods from a real-world social network and using them as context for downstream AI reasoning.

Built on a **Facebook Page–Page network** with **22,000+ nodes** and **300,000+ directed edges**, the system combines:

- 🔬 **Graph feature engineering** — PageRank, degree, k-core, clustering, topic similarity
- 🧠 **Graph Convolutional Network (GCN)** — a learned influence ranking model
- 🤖 **LangGraph agent** — a four-step agentic pipeline (Plan → Retrieve → Score → Synthesize)
- 💬 **LLM synthesis** — GPT-4o-mini generates actionable, evidence-grounded outreach reports

---

## 🗺️ Architecture

```
User Query (natural language)
        │
        ▼
   ┌─────────┐
   │  PLAN   │  ← Infer topics from query (politician, company, news, brand...)
   └────┬────┘
        │
        ▼
   ┌──────────┐
   │ RETRIEVE │  ← Seed topic-matched nodes → expand k-hop neighborhoods → extract subgraph
   └────┬─────┘
        │
        ▼
   ┌─────────┐
   │  SCORE  │  ← Run trained GCN on subgraph → produce per-node influence scores
   └────┬────┘
        │
        ▼
   ┌───────────┐
   │ SYNTHESIZE│  ← LLM generates ranked influencer list + rationale + outreach plan
   └─────┬─────┘
         │
         ▼
   Final Answer (human-readable, graph-grounded)
```

**Key Design Principle:**
- The **graph** handles topology and structure
- The **GCN** handles learned influence estimation
- The **LLM** handles explanation and synthesis

No component is asked to do a job it wasn't built for.

---

## 📂 Project Structure

```
├── data/
│   └── curated/
│       ├── users.csv              # Node attributes: topics, followers, posts, bio
│       └── edges_follow.csv       # Directed follow edges (src → dst)
│
├── Agentic_Graph-RAG_Over_Social-Network_Knowledge_Graphs.ipynb
│
└── README.md
```

---

## 🔧 Installation

### Prerequisites
- Python 3.9+
- pip

### Install Dependencies

```bash
pip install langgraph pydantic pandas networkx matplotlib tqdm scipy openai

# PyTorch (CPU)
pip install torch==2.8.0+cpu torchvision==0.23.0+cpu torchaudio==2.8.0+cpu \
    --index-url https://download.pytorch.org/whl/cpu

# PyTorch Geometric
pip install torch_geometric==2.6.1
```

### Environment Variables

```bash
export OPENAI_API_KEY="your-openai-api-key"
export OPENAI_MODEL="gpt-4o-mini"   # optional, defaults to gpt-4o-mini
```

---

## 📊 Dataset

| Property | Value |
|---|---|
| Source | MUSAE Facebook Page–Page Network |
| Nodes | 22,470 Facebook pages |
| Edges | 342,004+ directed follow edges |
| Node attributes | topics, followers, following, posts_30d, bio |
| Edge type | FOLLOW (directed) |

The dataset is automatically downloaded from hosted CSVs when the notebook is first run. No manual download required.

---

## ⚙️ Pipeline Components

### 1. Graph Construction

Converts `users.csv` and `edges_follow.csv` into a directed `nx.DiGraph` using NetworkX. Each node stores structured attributes; edges carry direction and type metadata.

```python
G, users_df = build_graph(df_users, df_edges)
# |V| = 22,470 | |E| = 342,004
```

---

### 2. Feature Engineering

For every node, seven structural and semantic features are computed:

| Feature | Description |
|---|---|
| `pagerank` | Global influence via random walk convergence |
| `deg_in` | Number of incoming follow edges |
| `deg_out` | Number of outgoing follow edges |
| `kcore` | Structural cohesion / network embeddedness |
| `clust` | Clustering coefficient of local neighborhood |
| `posts_30d` | Recent posting activity (activity signal) |
| `topic_sim` | Cosine similarity between node topics and query |

All features are Z-score normalized before being passed to the GCN.

---

### 3. GCN Influence Ranker

A two-layer **Graph Convolutional Network** built with PyTorch Geometric:

```
Input features (dim=7)
      │
  GCNConv(7 → 64) + ReLU + Dropout
      │
  GCNConv(64 → 64) + ReLU + Dropout
      │
  Linear(64 → 1)
      │
  Influence score (scalar per node)
```

**Training objective:** Pairwise ranking loss — the model learns to correctly order pairs of nodes by relative influence, not predict absolute scores. This is a fundamentally better framing for influencer identification tasks.

```python
loss = -log(sigmoid(score_i - score_j))   # for all pairs where i > j
```

**Training details:**

| Hyperparameter | Value |
|---|---|
| Epochs | 100 |
| Learning rate | 2e-3 |
| Weight decay | 1e-4 |
| Optimizer | Adam |
| Pairs per epoch | 1,024 |
| Dropout | 0.1 |
| Seed | 42 (fully deterministic) |

---

### 4. Graph Retrieval (Graph-RAG)

Given a natural-language query, the retrieval module:

1. **Parses** the query into topic labels (`politician`, `company`, `news`, `brand`, `sports`, etc.)
2. **Seeds** the graph with nodes whose topics match the query
3. **Expands** k=2 hops outward from each seed (both successors and predecessors)
4. **Returns** a subgraph capped at 800 nodes for efficient GCN inference

```python
subgraph = graph_retrieve(G, entities, k=2, top_n=800)
```

This replaces document retrieval with **structural retrieval** — the context window is now a slice of the knowledge graph.

---

### 5. LangGraph Agent

The agent is built using `langgraph.graph.StateGraph` with typed state passing between four nodes:

```python
class AgentState(TypedDict, total=False):
    query:    str
    plan:     str
    entities: Dict[str, Any]
    subgraph: nx.DiGraph
    scores:   Dict[str, float]
    answer:   str
```

**Flow:**

```
START → plan → retrieve → score → synthesize → END
```

Each node is a pure function that reads from and writes to `AgentState`. The agent is compiled and invoked with a single natural-language query string.

---

### 6. LLM Synthesis

The top-10 scored nodes are formatted as evidence lines (node ID, score, degree, topics, activity) and passed to GPT-4o-mini with a structured prompt requesting:

1. A ranked list of 3–5 key influencers
2. A one-sentence rationale per influencer grounded in graph evidence
3. A short, actionable outreach plan

```python
res = app.invoke({"query": "Find influencers for politician and company pages"})
print(res["answer"])
```

---

## 🚀 Usage

### Run the Full Pipeline

Open and run the notebook end-to-end:

```bash
jupyter notebook Agentic_Graph-RAG_Over_Social-Network_Knowledge_Graphs.ipynb
```

### Example Query

```python
query = "Find influencers for politician and company pages"
res = app.invoke({"query": query})
print(res["answer"])
```

### Example Output (structure)

```
=== Agent Answer ===

Top Influencers:
1. node_XXXX | score: 0.87 | Rationale: High PageRank + dense cross-community connections...
2. node_XXXX | score: 0.81 | Rationale: Active posting + strong in-degree from news pages...
3. node_XXXX | score: 0.76 | Rationale: High k-core value indicates deep network embeddedness...

Outreach Plan:
- Prioritize node_XXXX for initial contact given its bridging role between...
```

### Visualize Top Influencer Neighborhoods

```python
plot_top3(G, res["scores"], hops=1)
```

This renders ego-graphs for the top 3 ranked nodes, with node sizes scaled by in-degree.

---

## 🧪 Reproducibility

All random seeds are set globally for full determinism:

```python
SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.use_deterministic_algorithms(True)
```

GCN pair sampling also uses a seeded `torch.Generator` — results are identical across runs.

---

## 📈 Results Summary

| Metric | Value |
|---|---|
| Graph size | 22,470 nodes · 342,004+ edges |
| Subgraph retrieved per query | ~800 nodes |
| GCN training time (CPU) | ~2–4 minutes (100 epochs) |
| Inference time per query | < 5 seconds |
| LLM synthesis | GPT-4o-mini · temp=0 · max_tokens=500 |

---

## 🔭 Extensions & Future Work

- **Temporal Graph RAG** — incorporate edge timestamps to model information decay and trend detection
- **Heterogeneous GNN** — model different node types (pages, users, topics) with RGCN or HGT
- **Multi-hop reasoning** — enable the agent to chain multiple retrieval steps before synthesizing
- **Community detection** — use Louvain or spectral clustering to retrieve community-aware subgraphs
- **Fine-tuned LLM ranker** — replace the heuristic target with human-annotated influence labels
- **Streaming agent** — expose the pipeline as a real-time API with FastAPI + WebSocket updates

---

## 📚 References

- Kipf, T. N., & Welling, M. (2017). *Semi-Supervised Classification with Graph Convolutional Networks.* ICLR. [arXiv:1609.02907](https://arxiv.org/abs/1609.02907)
- Rozemberczki, B., et al. (2021). *Multi-Scale Attributed Node Embedding.* Journal of Complex Networks. [MUSAE Dataset](https://snap.stanford.edu/data/facebook-large-page-page-network.html)
- LangGraph Documentation — [https://langchain-ai.github.io/langgraph/](https://langchain-ai.github.io/langgraph/)
- PyTorch Geometric Documentation — [https://pytorch-geometric.readthedocs.io/](https://pytorch-geometric.readthedocs.io/)

---

## 🪪 License

This project is for educational and research purposes. The MUSAE dataset is publicly available under its original Stanford SNAP license.

---

<p align="center">
  Built with 🔗 graphs · 🧠 neural networks · 🤖 agents · ✍️ language models
</p>