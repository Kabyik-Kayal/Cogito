# Cogito - Self Correcting Graph RAG

### **1. The Project Definition**

**Goal:** Build a RAG system that *refuses* to answer if it cannot verify the facts, rather than hallucinating.
**The "Hard" Dataset:** Do not use Wikipedia or generic news. Use **NVIDIA CUDA Documentation** or **Kubernetes Docs**.

* *Why?* They are highly technical, version-dependent, and heavily cross-referenced. Standard RAG fails here because it mixes up API versions (e.g., v1.2 vs v1.22).

---

### **2. The Architecture (The "State Machine")**

We are dropping the linear chain for a **Cyclic Graph**.

#### **Nodes (The Agents):**

1. **`RetrieveNode`**: Fetches raw chunks from the Vector DB.
2. **`GraphAugmentNode`**: Uses NetworkX to find "neighboring" chunks (e.g., if you retrieve "Function A", this node forces retrieval of "Function A's Parameters" even if vector similarity missed it).
3. **`GenerateNode`**: Drafts the initial answer.
4. **`AuditNode` (The Differentiator)**:
* Extracts claims from the draft.
* Checks: *Is this claim supported by the retrieved text?*
* Output: `Pass`, `Fail`, or `Needs_Correction`.


5. **`RewriteNode`**: If Audit fails, this node strips the unsupported claims or rewrites the query to find better proof.

#### **Edges (The Logic):**

* `Start` → `Retrieve` → `GraphAugment` → `Generate` → `Audit`
* **IF** `Audit` == `Pass` → `End`
* **IF** `Audit` == `Fail` → `Rewrite` → `Retrieve` (Loop back)

---

### **3. The Tech Stack**

* **Orchestration:** **LangGraph**. (Mandatory. It is the Python framework for state machines).
* **Vector DB:** **ChromaDB**. (Local, fast, widely used).
* **Graph "DB":** **NetworkX**.
* *Clarification:* You will not use this as a persistent database (that’s Neo4j). You will use NetworkX to **build the graph in memory** during ingestion to map relationships (Parent <-> Child documents), then save this structure as a Python pickle or JSON. This proves you understand Graph Theory without needing a heavy Java server like Neo4j.


* **Model Serving:** **vLLM** or **Llama.cpp Server**.
* *The Move:* Don't just use Ollama. Run a **quantized (GGUF)** version of `Mistral-7B-Instruct` or `Llama-3-8B`. This aligns with your interest in quantization.


* **Evaluation:** **DeepEval**. (It is currently sharper than Ragas for specific "Hallucination" metrics).

---

### **4. The Roadmap (4 Weeks to MVP)**

#### **Phase 1: The "Smart" Ingestion (Data Engineering)**

* **Task:** Scrape the documentation.
* **The Twist:** Don't just chunk text.
* Create **Nodes** for every Section Header.
* Create **Edges** for every Hyperlink in the HTML.
* *Outcome:* You have a NetworkX graph where `Node A` (API Overview) connects to `Node B` (Code Example).


* **Store:** Embed the text into ChromaDB, but store the `Graph_Node_ID` as metadata.

#### **Phase 2: The Graph-Augmented Retrieval**

* **Task:** Write the retrieval logic.
* **Logic:**
1. Vector Search: Get top 3 chunks.
2. Graph Expansion: Look up those 3 chunks in your NetworkX graph. Grab their "Children" or "Linked" nodes.
3. Context Window: Feed *both* the vector results and the graph neighbors to the LLM.


* **Why this wins:** You solve the "Missing Context" problem. Vector search finds the *name* of the function; Graph search finds the *parameters* listed in the next paragraph.

#### **Phase 3: The Hallucination Auditor (The "Grader")**

* **Task:** Build the `AuditNode`.
* **Implementation:**
* **Prompt Engineering:** "You are a QA Auditor. Here is a generated sentence. Here is the source text. Does the source text explicitly support the sentence? Answer YES/NO."
* **Optimization:** Use a *smaller*, faster model for this (e.g., `Llama-3-8B-Quantized`) to keep latency low.
* **Fail-safe:** If the answer is "NO", the loop triggers.



#### **Phase 4: The Interface & Eval**

* **Frontend:** **Streamlit**.
* *Design:* Brutalist. Black background (`#000000`), neon green terminal text (`#00FF00`), sharp borders (`1px solid white`), no rounded corners.
* *Feature:* Show the "Trace". Display: "Draft 1 (Failed Audit) -> Searching Again... -> Draft 2 (Passed)".


* **Evaluation:** Run a dataset of 50 questions. Compare "Standard RAG" vs "Cogito".
* *Metric:* "Faithfulness" (from DeepEval).



---

### **5. Resources to Learn**

* **LangGraph Official Tutorial:**
* *Resource:* [LangGraph: Adaptive RAG](https://langchain-ai.github.io/langgraph/tutorials/rag/langgraph_adaptive_rag/)
* *Focus:* This is the bible for this project. Copy the patterns, but change the data.


* **NetworkX for RAG:**
* *Concept:* Search for "GraphRAG with NetworkX".
* *Code Pattern:* Learn `nx.node_link_graph` to save/load your graph structure.


* **Evaluation:**
* *Resource:* [DeepEval Github](https://github.com/confident-ai/deepeval)
* *Metric:* Look specifically at the `HallucinationMetric`.
---

### **6. Project Structure**

```
Cogito/
├── .env                    # Environment variables
├── .gitignore              # Git ignore rules
├── readme.md               # Project documentation
├── requirements.txt        # Python dependencies
│
├── config/                 # Configuration files
│   ├── __init__.py
│   └── paths.py            # Path configurations
│
├── data/                   # Raw and processed data
│
├── logs/                   # Application logs
│
├── models/                 # Downloaded GGUF models
│
├── results/                # Evaluation results
│
├── scripts/                # Executable scripts
│   ├── evaluate.py         # Run evaluation pipeline
│   └── ingest.py           # Run ingestion pipeline
│
├── src/                    # Main source code
│   ├── __init__.py
│   ├── graph.py            # LangGraph state machine
│   ├── state.py            # State definitions
│   │
│   ├── db/                 # Database layer
│   │   ├── __init__.py
│   │   ├── graph_store.py  # NetworkX graph storage
│   │   └── vector_store.py # ChromaDB vector storage
│   │
│   ├── evaluation/         # Evaluation metrics
│   │   ├── __init__.py
│   │   ├── evaluator.py    # DeepEval integration
│   │   └── metrics.py      # Custom metrics
│   │
│   ├── frontend/           # Streamlit UI
│   │   ├── __init__.py
│   │   └── app.py          # Main Streamlit app
│   │
│   ├── ingestion/          # Data ingestion pipeline
│   │   ├── __init__.py
│   │   ├── parser.py       # Document parsing
│   │   ├── pipeline.py     # Ingestion orchestration
│   │   └── scraper.py      # Documentation scraper
│   │
│   ├── model/              # LLM model management
│   │   ├── __init__.py
│   │   └── download_models.py  # GGUF model downloader
│   │
│   └── nodes/              # LangGraph nodes (agents)
│       ├── __init__.py
│       ├── audit.py        # AuditNode - claim verification
│       ├── generate.py     # GenerateNode - answer drafting
│       ├── graph_augment.py # GraphAugmentNode - graph expansion
│       ├── retrieve.py     # RetrieveNode - vector search
│       └── rewrite.py      # RewriteNode - query rewriting
│
└── utils/                  # Utility modules
    ├── custom_exception.py # Custom exception handling
    └── logger.py           # Logging configuration
```
