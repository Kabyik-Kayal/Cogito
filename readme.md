# COGITO - Self Correcting RAG

> *"The first principle is that you must not fool yourself — and you are the easiest person to fool."* — Richard Feynman


![Cogito Interface](assets/demo.png)

## The Problem with Parrots
You see, standard language models are a bit like very well-read parrots. You ask them, "What is the capital of France?" and they say "Paris!" because they've heard it a million times. It's wonderful.

But if you ask them something trickier, something technical like "How do I configure the memory pool in CUDA 12.1?"... well, sometimes they just *guess*. They put words together that sound right, like a parrot mimicking a scientist. But the parrot doesn't understand physics! It just knows the rhythm of the words. In our business, we call this a "hallucination," but really, it's just confident guessing.

And that is a catastrophe if you are trying to build software. You don't want a guess; you want the truth.

## Trust but Verify
So, how do we fix this? We don't just make the brain bigger. We change how it thinks. We build a machine that acts less like a parrot and more like a scientist.

A scientist doesn't just blurt out the first thing that comes to mind. A scientist does research. They look things up. They check their sources. And—this is the most important part—if the facts don't match their theory, they throw the theory out and start again.

**Cogito** is a "Self-Correcting Graph RAG." That's a fancy name for a simple three-step process:
1.  **Draft** an answer based on what we read.
2.  **Audit** that answer to see if it's actually supported by the text.
3.  **Rewrite** it if we were lying.

## How It Works (The Machinery)
We don't use a straight line for this. A straight line is "Retrieval -> Generation -> Done." That's the old way. We use a **Loop**.

Imagine a room with four people in it:

### 1. The Researcher (`RetrieveNode`)
This fellow runs to the library (our Vector Database) and grabs a handful of books that look relevant. "Here!" he says, "This book mentions CUDA memory!"

### 2. The Connector (`GraphAugmentNode`)
This one is clever. She knows that knowledge is connected. If the Researcher brings a page about "Function A," she says, "Wait a minute, you can't understand Function A without knowing about its Parameters, which are on *this other page*." She follows the links (the Graph) and grabs the extra context the Researcher missed.

### 3. The Writer (`GenerateNode`)
He takes all these pages and writes a draft answer. "Based on this text, you configure the memory pool like this..."

### 4. The Auditor (`AuditNode`) - The Most Important Person
This is our strict professor. He looks at the Writer's draft, and he looks at the source text, and he asks one question: *"Can you prove it?"*

If the Writer says "X is true," but the text doesn't say that, the Auditor slams his hand on the table. **"FAIL!"** he says. "Go back and do it again!"

And so, the Writer has to rewrite the query, the Researcher goes back to the library, and they try again. They keep trying until they get it right, or until they admit they just don't know.

## The Parts List (Tech Stack)
To build this machine, we used some very specific tools:

*   **LangGraph**: This is the conductor. It manages the state—who speaks when, and who passes papers to whom. It turns our flowchart into code.
*   **ChromaDB**: The filing cabinet. It stores the text chunks so we can find them by meaning (vector search).
*   **NetworkX**: The map. It remembers how documents link to each other (hyperlinks, sections). This is how we find the "hidden" connections.
*   **Llama.cpp**: The brain. We're using local, quantized models (like Mistral or Llama 3) because you don't need a supercomputer to check facts—you just need a sharp one.
*   **DeepEval**: The scorecard. It measures how often we tell the truth.

## The Blueprints (Technical details)
For those who want to see the engine block, here is how we wired it up.

### 1. The Map Legend (Graph Schema)
When we say "Graph," we don't mean a pretty picture. We mean a specific data structure in `NetworkX`.
*   **Nodes**: represent distinct *chunks of knowledge*.
    *   `ID`: The file path or header (e.g., `docs/cuda_api.html#memory-pool`).
    *   `Content`: The actual text.
    *   `Type`: Is it a Code Block? A Concept? A Warning?
*   **Edges**: represent *relationships*.
    *   `PARENT_OF`: A section header owns its paragraphs.
    *   `LINKS_TO`: An HTML hyperlink found in the text.
    *   `MENTIONS`: If chunk A talks about "Texture Memory," and chunk B defines it.

### 2. The Professor's Rubric (Audit Logic)
The **Auditor** (`AuditNode`) isn't magic; it's a prompt with a very low temperature (`0.0`). We don't want creativity here; we want cold, hard logic.
The logic flows like this:
1.  **Extract Claims**: "The draft says: *'Use cudaMallocManaged for unified memory.'*"
2.  **Verify**: Search the retrieved context for that exact rule.
3.  **Verdict**:
    *   **PASS**: The text explicitly supports the claim.
    *   **FAIL**: The text contradicts it, or is silent.

### 3. The Lab Layout (Project Structure)
Here is where we keep everything. I've labeled the important bits.

```text
Cogito/
├── config/                 # Configuration files
│   └── paths.py            # Path constants (Model paths, DB paths)
├── data/                   # The Raw Materials (PDFs, HTML)
├── models/                 # The Brains (GGUF files go here)
├── scripts/
│   ├── evaluate.py         # Run the DeepEval metrics suite
│   └── ingest.py           # The Librarian (Scrapes & Indexes data)
├── src/
│   ├── db/                 # Database Interfaces
│   │   ├── graph_store.py  # NetworkX wrapper (saves/loads pickle)
│   │   └── vector_store.py # ChromaDB wrapper
│   ├── evaluation/
│   │   ├── evaluator.py    # DeepEval integration logic
│   │   └── metrics.py      # Custom faithfulness/hallucination metrics
│   ├── frontend/
│   │   └── app.py          # The Streamlit Dashboard
│   ├── ingestion/
│   │   ├── parser.py       # Chunking & Node creation logic
│   │   ├── pipeline.py     # Orchestrates scraping -> graph -> db
│   │   └── scraper.py      # Fetches raw documentation
│   ├── nodes/              # The Workers (LangGraph Nodes)
│   │   ├── audit.py        # The Professor (Hallucination Checker)
│   │   ├── generate.py     # The Writer (Drafts answers)
│   │   ├── graph_augment.py# The Connector (Expands context)
│   │   ├── retrieve.py     # The Researcher (Vector Search)
│   │   └── rewrite.py      # The Editor (Fixes bad queries)
│   ├── graph.py            # The Conductor (The State Machine definition)
│   └── state.py            # The Data Schema (GraphState TypedDict)
└── utils/                  # The Wrenches
    ├── custom_exception.py
    └── logger.py           # Centralized logging
```

## How to Play with It

First, you need to make sure you have atleast a 16GB GPU (can be lower if you change the 8B parameter model used currently with any 3B model), then set up your lab.

### 0. Get the Resources
```bash
git clone "https://github.com/Kabyik-Kayal/Cogito.git"
cd Cogito
```

### 1. Install the Equipment
```bash
conda create -n cogito python=3.11 -y
conda activate cogito
pip install uv
uv pip install -r requirements.txt
```
### 2. Download the LLM
```bash
python -m src.model.download_models
```

### 2. Run the Machine
Now, start the "State Machine." This spins up the little society of agents and gives you a web interface to watch them work.
```bash
uvicorn src.frontend.app:app --reload
```
FOR the ***FIRST TIME***, after opening the webapp, wait for few minutes to let the tokenizer download properly.

You can upload documents into the database or provide links to scrape data.

![Cogito Ingestion Interface](assets/Ingestion.png)

When you ask a question, watch the logs. You will see the **Audit** happen. You will see it **Fail**. And you will see it **Correct Itself**. Or answer explicitly that it **Lacks** the information to answer, instead of **Hallucinating**.

![Cogito Ingestion Interface](assets/Generation.png)

It is a beautiful thing to watch a machine admit it was wrong.

You can also find the info of the available Collections, initialize them before hand to reduce the first query time, or delete the collections completely.

![Cogito Ingestion Interface](assets/Information.png)

---
*"Nature cannot be fooled."*
