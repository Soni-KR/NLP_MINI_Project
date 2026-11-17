FinRAG answers questions in French or English using BIAT's 2021, 2023 and 2025 management reports. It retrieves French source passages locally and generates cited answers through OpenRouter. [GitHub repository](https://github.com/Soni-KR/NLP_MINI_Project)

## Run locally

1. Clone the repository and open its root folder, where `Dockerfile` and `finrag.py` are located.
2. Put the three PDFs from the handoff archive in that folder, using the exact filenames below. These copies define the evaluated corpus and its PDF page numbers. The PDFs are excluded from Git, and the Docker build requires all three. [Private Kaggle dataset, if you have access](https://www.kaggle.com/datasets/mouradkraiem/finrag-biat-reports)
3. Start Docker. For answer generation, copy `.env.example` to `.env` and set `OPENROUTER_API_KEY` to your own key without quotes. Keep `.env` local. The first build and model download need internet access.

| File | PDF pages |
|---|---:|
| `rapport_biat_2021.pdf` | 296 |
| `rapport_biat_2023.pdf` | 126 |
| `rapport_biat_2025.pdf` | 131 |

Run these commands from the repository root.

```powershell
docker build -t finrag:local .

# Ask a question and print the cited answer
docker run --rm --env-file .env -v finrag-hf:/root/.cache/huggingface finrag:local "Quel était le résultat net de la BIAT en 2024 ?"

# Inspect retrieved passages. This also works without an API key.
docker run --rm -v finrag-hf:/root/.cache/huggingface finrag:local --top-k 5 --show-passages "Quel était le résultat net de la BIAT en 2024 ?"

# List numbered development questions and expected answers for a demonstration
docker run --rm finrag:local --questions
```

The `finrag-hf` volume stores the embedding model and passage embeddings under `/root/.cache/huggingface`. Citations such as `[2025 p. 27]` identify the report year and the one-based position in that PDF. A passage can describe an earlier fiscal year.

## Web interface

The browser interface uses the same retrieval and generation pipeline as the CLI.

```powershell
docker run --rm --name finrag-web --env-file .env -p 127.0.0.1:5000:5000 -e FINRAG_HOST=0.0.0.0 `
  -v finrag-hf:/root/.cache/huggingface --entrypoint python finrag:local /app/app.py
```

Open `http://127.0.0.1:5000`. The first question loads the model and builds the search indexes. Later requests reuse them. After changing `.env`, stop the container and run the command again so Docker imports the new key.

For a local Python installation, use an environment with compatible Python dependencies, set the key in `.env`, and run the following commands.

```powershell
python -m pip install --index-url https://download.pytorch.org/whl/cpu torch==2.10.0+cpu
python -m pip install -r requirements.txt
python app.py
```

The Python app loads the project `.env` automatically. If an existing Anaconda environment reports a NumPy compatibility error, reinstall the pinned requirements with `python -m pip install --force-reinstall -r requirements.txt`. FinRAG uses PyTorch and disables Transformers' TensorFlow backend.

The language control switches the interface and generated answers between French and English. Source excerpts retain the original French, and clicking a citation opens its passage. Language and theme choices are saved in the browser. The title and question shortcuts use the locally bundled [Dockbar](https://github.com/CatsJuice/dockbar) component.

## Pipeline

PyMuPDF extracts text blocks and qualifying ruled tables with their report and page labels. Sentence and line boundaries guide chunks with a 384-token target and roughly 48 tokens of overlap. Table chunks repeat their headers and nearby context.

A pinned revision of `intfloat/multilingual-e5-small` produces normalized 384-dimensional vectors on CPU. FAISS exact inner-product search provides semantic rankings. TF-IDF adds normalized French word and bigram matches. Hybrid retrieval combines up to 60 candidates from each ranking with reciprocal rank fusion and a 1.5× lexical weight.

Each requested year gets a separate search. A year with an available report searches that report, while another year, such as 2024, searches all three. Within each selected page, a query phrase match chooses the passage. The default k is 5 distinct pages per year search, so a multi-year question can return more than five sources. A small glossary normalizes common English financial terms for search.

OpenRouter receives the question and selected passages. The configured model is `nvidia/nemotron-3.5-super-vl:free`, with `nvidia/nemotron-3.5-lightning:free` as fallback. The prompt requests explicit units, fiscal years, bank-only or consolidated scope, and page citations. An eligible historical refusal gets one retry using a focused subset of the retrieved passages. The app distinguishes a model refusal, an empty or invalidly cited reply, and an API error.

## Evaluation

`test_cases.json` is the development suite of 26 French questions, with 21 answerable questions and 5 expected refusals. These cases influenced the implementation.

`test_unseen.json` is a separate set of 10 French questions, with 8 answerable questions and 2 expected refusals. Three answerable facts had already been inspected during development. The other five answerable questions and both refusal questions were added independently after the runtime changes. All questions use the same three PDFs, so these results measure additional questions on the existing corpus.

Expected answers and accepted source pages support evaluation and question-list display. Retrieval and generation use the question and PDF excerpts. Gold answers are not inserted into the model context.

Compare all three retrieval methods at k=3, 5 and 10, including corpus statistics.

```powershell
docker run --rm -v finrag-hf:/root/.cache/huggingface --entrypoint python finrag:local /app/evaluate.py `
  --cases /app/test_cases.json --methods dense lexical hybrid --top-k 3 5 10 --corpus-stats

docker run --rm -v finrag-hf:/root/.cache/huggingface --entrypoint python finrag:local /app/evaluate.py `
  --cases /app/test_unseen.json --methods dense lexical hybrid --top-k 3 5 10
```

Retrieval passes a question when every expected value occurs in a retrieved passage from an accepted source page. Refusal cases are assessed during generation. The generation scorer links the expected value, unit and citation within a claim and checks the fiscal year when specified. Manual review still checks the indicator, scope and meaning of the cited source.

Run generation with the selected retrieval method and k. This calls OpenRouter.

```powershell
docker run --rm --env-file .env -v finrag-hf:/root/.cache/huggingface --entrypoint python finrag:local /app/evaluate.py `
  --cases /app/test_cases.json --generate --generation-method hybrid --generation-k 5

docker run --rm --env-file .env -v finrag-hf:/root/.cache/huggingface --entrypoint python finrag:local /app/evaluate.py `
  --cases /app/test_unseen.json --generate --generation-method hybrid --generation-k 5
```

## Recorded results

Recorded on 2026-11-17 against runtime commit `bafb62e`. The tables report one run per French suite on the same three PDFs. The separate set includes three revisited facts, five fresh answerable questions and two fresh refusal questions. Its small size and shared corpus limit conclusions about broader performance.

Complete retrieval evidence coverage includes answerable questions only.

| Suite and method | k=3 | k=5 | k=10 |
|---|---:|---:|---:|
| Development, FAISS dense | 9/21 | 9/21 | 12/21 |
| Development, TF-IDF lexical | 16/21 | 18/21 | 21/21 |
| Development, hybrid | 15/21 | 21/21 | 21/21 |
| Separate set, FAISS dense | 6/8 | 6/8 | 6/8 |
| Separate set, TF-IDF lexical | 8/8 | 8/8 | 8/8 |
| Separate set, hybrid | 8/8 | 8/8 | 8/8 |

Live generation used hybrid retrieval at k=5. The evaluator links expected values or text to accepted citations, with unit and fiscal-year checks where specified.

| Suite | Cited answers passed | Expected refusals passed |
|---|---:|---:|
| Development | 21/21 | 5/5 |
| Separate set | 8/8 | 2/2 |

Both suites recorded zero incorrect refusals, missed refusals, empty or citation-rejected replies, and API errors. All 36 calls returned `nvidia/nemotron-3.5-super-vl:free` with one attempt per case. An assistant review of all answers against their selected passages agreed with the automatic outcomes. Repeated runs can vary with model and provider availability. The review covers this run and corpus.

## Components and limitations

| Component | Current behavior | Limitation | Follow-up |
|---|---|---|---|
| PDF corpus | Three reports contain 553 PDF pages. Each passage retains its source year and PDF page. | A newer report can state an earlier fiscal year's figure. Editions can differ in pagination. | Preserve the exact tested PDFs and verify the fiscal year attached to each figure. |
| Extraction and tables | PyMuPDF reads text blocks and qualifying ruled tables, preserving table headers and nearby context. | Borderless tables, merged headers and column reading order can lose associations. | Inspect the relevant PDF page and its extracted passage when a financial claim is unclear. |
| Chunking and embeddings | Token-aware chunks feed normalized multilingual E5-small vectors. | A label, unit or value can fall outside a selected chunk. | Inspect the selected and neighboring chunks before changing chunking or the embedding model. |
| Retrieval and years | FAISS, TF-IDF and hybrid rankings share the same year routing and page selection. | Wording and the choice of chunk on a correct page can affect evidence coverage. | Compare methods and k on both suites, then add independent questions for any new change. |
| English search | A glossary maps common English financial terms to French search terms. | Its vocabulary covers a limited set of metrics and qualifiers. | Evaluate additional English wording and scope requests. |
| Generation and citations | OpenRouter generates from excerpts, and recognized citations must belong to the supplied pages. | A permitted citation can still accompany an incorrect metric, value, unit or scope. Free endpoints have availability and quota limits. | Review financial claims against the cited passage and report API failures separately from answer errors. |
| Evaluation | Development and separate question sets score evidence coverage, cited answers and refusals. | Both sets share one corpus, and numeric checks require semantic review. | Keep development and separate-set results distinct and inspect each failure. |
| Runtime and cache | Docker packages the CLI and Flask app. A validated cache stores passages and vectors using PDF hashes, model revision and chunking settings. | First-run model loading and indexing add latency. The Kaggle notebook represents an earlier pipeline. | Reuse the named volume, bump `CACHE_VERSION` when extraction changes, and measure cold and warm startup separately. |
