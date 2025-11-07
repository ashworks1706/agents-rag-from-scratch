# agents-rag-from-scratch

A compact, hands-on notebook and notes for learning AI agents, agentic workflows, and Retrieval-Augmented Generation (RAG) from first principles through practical implementation.

This repository contains a single interactive notebook—`tutorial.ipynb`—that documents the theory, math, and code experiments used while learning about agent architectures, retrieval strategies, and building simple RAG systems from scratch.

## What this is

- A learning notebook with background theory, mathematical notes, and runnable code examples.
- Focused on understanding how agentic systems and RAG pipelines work at a low level (not a production framework).
- Intended for students, researchers, and developers who want a minimal, transparent walkthrough rather than a packaged library.

## Contents

- `tutorial.ipynb` — the main notebook. It includes explanations, equations, and code cells demonstrating retrieval, vector stores, and simple agent loops.
- `LICENSE` — repository license.

## Quick start

1. Install a Python environment (preferably 3.10+).
2. Create and activate a virtual environment and install any dependencies you need to run the notebook (examples below).
3. Open `tutorial.ipynb` in Jupyter, run cells, and follow along.

Example (using pip and a virtualenv):

```bash
# create a venv and activate it (POSIX shell)
python -m venv .venv
source .venv/bin/activate

# install lightweight tools you may need (adjust as required)
pip install jupyterlab numpy scipy scikit-learn faiss-cpu

# start the notebook
jupyter lab
```

Note: the notebook may use other libraries depending on the examples inside. If you see import errors, install the missing packages with pip.

## Usage

- Open `tutorial.ipynb` and run cells sequentially. The notebook is organized to introduce concepts, show math/intuition, and then demonstrate code.
- Use the code cells as a basis to experiment with different retrieval models, embeddings, or agent policies.

## Goals and scope

- Teach core concepts behind retrieval-augmented generation and agentic workflows.
- Provide a minimal, educational implementation rather than a production-ready system.
- Keep the code and math transparent so you can extend or re-implement pieces for experiments.

## Contributing / Notes

- This is primarily a personal learning notebook. Pull requests that improve explanations, fix typos, or add small, well-explained experiments are welcome.
- If you add non-trivial dependencies, consider updating this README with a short requirements list or adding a `requirements.txt`.

## License

This project includes a `LICENSE` file—see it for licence terms.

---

If you'd like, I can also:

- extract a `requirements.txt` from the notebook imports,
- add a short example script that runs one of the notebook demos from the command line, or
- add a quick development section with recommended Python versions and testing instructions.

Tell me which of those you'd like next.

---

Concise codebase summary

- What this repo contains

  - `tutorial.ipynb`: the single, runnable notebook with all examples and notes. It contains both explanatory Markdown and Python code cells used throughout the tutorial.
- What the code/demo implements (high level)

  - Prompt engineering examples and reusable prompt templates
  - LLM hyperparameter experiments (temperature / top-p effects)
  - Tool examples: built-in provider tools, small custom @tool functions, and a simulated MCP server + tools
  - Agent examples that call tools and chains built on a shared LLM instance
  - Memory systems and comparisons (buffer, summary, window, token-limited, entity memory, combined memory)
  - A minimal skill registry and example skills (tiny, didactic implementations)
  - Workflow examples: prompt chaining, routing/specialist patterns, and simple orchestrators
  - Full RAG pipeline notes and demos: document loading, splitting/chunking, embeddings, vector storage and retrieval (dense / sparse / hybrid), and a minimal orchestrator that ties retrieval into generation
  - Evaluation and demo utilities (confidence scoring, voting/ensemble patterns)
- How to use

  - Open `tutorial.ipynb` in Jupyter or Colab and run cells sequentially. The notebook is designed as a guided tutorial; code cells are runnable but may require API keys or optional libraries for some provider integrations.
- Quick notes

  - The notebook is educational and didactic: examples are intentionally small and explicit rather than production-ready. Use it as a learning resource or a starting point for experiments.
