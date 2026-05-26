# MIND

## 1. Overview

MIND is a memory-enhanced prompt detoxification framework for black-box large language models. Given a prompt that may elicit a toxic response from a target model, MIND first queries the target model and evaluates the generated response with the Perspective API. If the initial response is already below the toxicity threshold, MIND returns it directly. Otherwise, MIND retrieves relevant successful optimization experiences from memory and uses an optimizer model to rewrite the prompt, aiming to reduce toxicity while preserving the original task intent.

This repository provides the core implementation of MIND, including:

- single-prompt detoxification with MIND;
- configurable target, optimizer, evaluator, and embedding models;
- Memory Retrieval (MR) and Memory Maintenance (MM);
- toxicity evaluation with the Perspective API;
- utility evaluation with an LLM judge;
- a shared memory server for accumulating and maintaining successful optimization experiences.

## 2. Environment, Datasets, and API Keys

### 2.1 Installation

Python 3.9 or later is recommended.

```powershell
pip install -r requirements.txt
```

### 2.2 Datasets

The paper evaluates MIND on three publicly available toxicity-oriented datasets. Users can download these datasets and use prompts from them as inputs to MIND.

| Dataset | Description | Link |
|---|---|---|
| RealToxicityPrompts (RTP) | Web-sourced prompts for evaluating toxic degeneration in language models. | [allenai/real-toxicity-prompts](https://huggingface.co/datasets/allenai/real-toxicity-prompts) |
| ToxicChat | Toxicity samples from real user-AI conversations. | [lmsys/toxic-chat](https://huggingface.co/datasets/lmsys/toxic-chat) |
| AttaQ | Adversarial safety-seeking / red-teaming prompts. | [ibm-research/AttaQ](https://huggingface.co/datasets/ibm-research/AttaQ) |

For example, the datasets can be loaded with Hugging Face `datasets`:

```python
from datasets import load_dataset

rtp = load_dataset("allenai/real-toxicity-prompts")
toxic_chat = load_dataset("lmsys/toxic-chat")
attaq = load_dataset("ibm-research/AttaQ")
```

The default entry point in this repository accepts a single prompt through `--prompt`. To run dataset-level experiments, sample prompts from the datasets above and call `main.py` in a loop or build a batch runner on top of this entry point.

### 2.3 API Keys

Copy `.env.example` to `.env`:

```powershell
copy .env.example .env
```

Then fill in the required API keys:

```text
PERSPECTIVE_API_KEY=your_perspective_api_key
SF_KEY=your_siliconflow_api_key
OPENROUTER_API_KEY=your_openrouter_api_key
OPENAI_API_KEY=your_openai_api_key
```

| Variable | Purpose |
|---|---|
| `PERSPECTIVE_API_KEY` | Perspective API key for toxicity evaluation. |
| `SF_KEY` | SiliconFlow API key for optimizer / embedding models. |
| `OPENROUTER_API_KEY` | OpenRouter API key for target / judge models. |
| `OPENAI_API_KEY` | Optional; only needed when using the direct OpenAI API. |

## 3. Running MIND

### 3.1 Start the Memory Server

MIND uses a shared memory server by default to store and maintain successful optimization experiences. Start the memory server before running `main.py`:

```powershell
python -m utils.memory.memory_server
```

You can also explicitly set the memory directory and MM parameters:

```powershell
python -m utils.memory.memory_server `
  --memory-dir artifacts/memory/shared `
  --M 300 `
  --N 100 `
  --embedding-model siliconflow/BAAI/bge-m3 `
  --mm-alpha 1.0 `
  --mm-beta 1.0 `
  --lambda-val 0.5
```

By default, memory files are stored under:

```text
artifacts/memory/shared/jsonl/active_memory.jsonl
artifacts/memory/shared/jsonl/full_history.jsonl
artifacts/memory/shared/jsonl/pending_new_examples.jsonl
```

- `full_history.jsonl` stores all successful experiences that reduce toxicity.
- `pending_new_examples.jsonl` stores newly collected successful experiences before MM is triggered.
- `active_memory.jsonl` stores the active memory maintained by MM and used by MR.

Memory server parameters:

| Argument | Default | Description |
|---|---:|---|
| `--host` | `127.0.0.1` | Host address for the memory server. |
| `--port` | `6001` | Port for the memory server. |
| `--memory-dir` | `artifacts/memory/shared` | Directory for memory files. |
| `--M` | `300` | Maximum active memory size. |
| `--N` | `100` | Number of new successful experiences that trigger one MM update. |
| `--embedding-model` | `siliconflow/BAAI/bge-m3` | Embedding model used by MM. |
| `--mm-alpha` | `1.0` | Quality weight in MM. |
| `--mm-beta` | `1.0` | Diversity weight in MM. |
| `--lambda-val` | `0.5` | Weight used in the example quality score `Q_d`. |

### 3.2 Run MIND

After starting the memory server, run MIND on a single prompt:

```powershell
python main.py --prompt "toxic prompt from dataset"
```

Full example:

```powershell
python main.py `
  --prompt "toxic prompt from dataset" `
  --target-model openrouter/meta-llama/llama-3.1-8b-instruct `
  --optimizer-model siliconflow/Qwen/Qwen3.6-27B `
  --judge-model openrouter/deepseek/deepseek-v4-flash `
  --embedding-model siliconflow/BAAI/bge-m3 `
  --T 3 `
  --tau 0.1 `
  --lambda-val 0.5 `
  --mr-alpha 1.0 `
  --mr-beta 1.0 `
  --mr-gamma 1.0 `
  --k 3 `
  --memory-mode server `
  --memory-server-url http://127.0.0.1:6001
```

Local memory mode is also supported:

```powershell
python main.py `
  --prompt "toxic prompt from dataset" `
  --memory-mode local `
  --memory-dir artifacts/memory/local
```

`main.py` parameters:

| Argument | Default | Description |
|---|---:|---|
| `--prompt` | Required | Input prompt for the target model. |
| `--target-model` | `openrouter/meta-llama/llama-3.1-8b-instruct` | Target model to detoxify. |
| `--optimizer-model` | `siliconflow/Qwen/Qwen3.6-27B` | Optimizer model for prompt rewriting. |
| `--judge-model` | `openrouter/deepseek/deepseek-v4-flash` | LLM judge for utility evaluation. |
| `--embedding-model` | `siliconflow/BAAI/bge-m3` | Embedding model used by MR. |
| `--T` | `3` | Maximum number of MIND optimization iterations. |
| `--tau` | `0.1` | Toxicity threshold below which no optimization is needed. |
| `--lambda-val` | `0.5` | Weight between toxicity reduction and utility in `Q_d`. |
| `--mr-alpha` | `1.0` | Similarity weight in MR. |
| `--mr-beta` | `1.0` | Quality weight in MR. |
| `--mr-gamma` | `1.0` | Diversity weight in MR. |
| `--mm-alpha` | `1.0` | Quality weight in MM; mainly used in local memory mode. |
| `--mm-beta` | `1.0` | Diversity weight in MM; mainly used in local memory mode. |
| `--M` | `300` | Maximum active memory size; mainly used in local memory mode. |
| `--N` | `100` | Number of new successful experiences that trigger one MM update; mainly used in local memory mode. |
| `--k` | `3` | Number of memory examples retrieved by MR. |
| `--memory-mode` | `server` | Use shared memory server or local memory. |
| `--memory-server-url` | `http://127.0.0.1:6001` | URL of the memory server. |
| `--memory-dir` | `artifacts/memory/shared` | Memory directory used in local memory mode. |
| `--verbose` | Disabled | Print the full JSON result. |

In `server` memory mode, MM-related parameters should be set when starting `utils.memory.memory_server`. The `--M`, `--N`, `--mm-alpha`, and `--mm-beta` arguments in `main.py` are mainly used for `local` memory mode.
