FinRAG answers questions in French using BIAT's 2021, 2023 and 2025 management reports. Start from commit `7957c99` on `main`. The first issues to investigate are retrieval sensitivity to wording and an intermittent refusal on a question about 2024. [GitHub repository](https://github.com/Soni-KR/NLP_MINI_Project)

## Run locally

1. Clone the GitHub repository and open its root folder, where `Dockerfile` and `finrag.py` are located.
2. Put the three PDFs from the handoff archive in that folder. Keep the filenames below. The PDFs are excluded from Git, and the Docker build requires all three. The archive contains the exact copies used for evaluation. [Private Kaggle dataset, if you have access](https://www.kaggle.com/datasets/mouradkraiem/finrag-biat-reports)
3. Start Docker. For answer generation, copy `.env.example` to `.env` and set `OPENROUTER_API_KEY` to your own key without quotes. Keep `.env` local. The Python web app loads this `.env` file automatically. The first build and model download need internet access.

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

# Inspect the passages for the same wording at k=5. This works without an API key.
docker run --rm -v finrag-hf:/root/.cache/huggingface finrag:local --top-k 5 --show-passages "Quel était le résultat net de la BIAT en 2024 ?"

# Score retrieval at k=3, 5 and 10
docker run --rm -v finrag-hf:/root/.cache/huggingface --entrypoint python finrag:local /app/evaluate.py

# Also check generated values, citations and refusals at k=5
docker run --rm --env-file .env -v finrag-hf:/root/.cache/huggingface --entrypoint python finrag:local /app/evaluate.py --generate
```

## Interface web

The browser interface uses the same retrieval and generation pipeline as the CLI. Install the
requirements, set `OPENROUTER_API_KEY`, then start the server from the repository root:

```powershell
python -m pip install -r requirements.txt
python app.py
```

Open `http://127.0.0.1:5000`. The index is built lazily on the first question and reused for
subsequent requests. With the Docker image, use the existing image and replace its CLI entrypoint:

```powershell
docker run --rm --env-file .env -p 127.0.0.1:5000:5000 -e FINRAG_HOST=0.0.0.0 `
  -v finrag-hf:/root/.cache/huggingface --entrypoint python finrag:local /app/app.py
```

The language control switches the interface and generated answers between French and English.
Source excerpts retain the original French, and clicking a citation opens its passage.
Language and theme choices are saved in the browser. The title and question shortcuts use the
locally bundled [Dockbar](https://github.com/CatsJuice/dockbar) component.

If Anaconda reports that a module compiled with NumPy 1.x cannot run with NumPy 2.x, repair
the environment with:

```powershell
python -m pip install --force-reinstall -r requirements.txt
```

FinRAG forces Transformers to use PyTorch, so TensorFlow/Keras is not required.

The `finrag-hf` volume saves the embedding model and passage embeddings for later runs. Citations such as `[2025 p. 27]` refer to the report year and the one-based position in that PDF. The year discussed in the passage can be earlier.

## Recorded results

| Check | Result |
|---|---|
| Complete retrieval evidence on the 8 answerable benchmark questions | 7/8 at k=3, 8/8 at k=5 and k=10 |
| Live OpenRouter run at k=5 | 8/8 contained the expected numbers and reference citations, 2/2 expected refusals, 0 API errors |
| Five additional French rewordings | 3/5 had complete evidence at k=5 and k=10 |

The benchmark questions are in `test_cases.json` and were used to tune the ranking. At k=3, the benchmark miss was Q05, the 2023 employee count. Retrieval passes when the expected number appears in a retrieved chunk from an accepted page for every requested year. The generation scorer checks figures and citations independently. Review the answer itself to confirm the indicator, unit, year, bank-only or consolidated scope, and which citation supports each figure.

## Issues to reproduce

These outcomes were reported from manual testing. Use the exact prompts and compare the retrieved passages before changing the code.

| Prompt | Recorded behavior | Expected result |
|---|---|---|
| Quel était le résultat net de la BIAT en 2024 ? | Refused once and answered on a separate run with passage display. The cause is open. | 357,8 MD for the bank alone, supported by the 2025 report [2025 p. 27]. |
| Compare les résultats nets de la banque seule en 2021, 2023 et 2025. | Refused. The inspected context was dominated by covers, contents and generic headings and lacked the required figures. | 266,8 MD [2021 p. 26], 331,4 MD [2023 p. 24], 385,3 MD [2025 p. 24]. |
| Compare le résultat net en 2021, 2023 et 2025. | Returned the three expected bank-only figures with citations. | Same figures as the preceding prompt. |
| Quel résultat net la BIAT a-t-elle réalisé en 2027 ? | Correctly refused. | Refuse because the corpus has no actual result for 2027. |

## Where to continue

1. **Add regression cases.** Add the two net-income phrasings, the 2024 historical question, the 2027 refusal, and a few new rewordings. Record the expected values and supporting PDF pages. Measure complete evidence at k=5 before and after each change.
2. **Fix ranking on the failed comparison.** Use `retrieve()` in `finrag.py` to find where the supporting chunks rank for each wording and year. Check whether table-of-contents and generic chunks displace evidence. Try one change at a time and compare it on the original questions and the new cases.
3. **Trace the 2024 refusal.** Repeat the exact question at k=5 and capture the passages. If the passages contain 357,8 MD, compare the raw OpenRouter reply with the final displayed answer. The citation check can turn a reply with missing or out-of-context citations into the same refusal message the model can produce directly. The `--show-passages` flag displays evidence and uses the same retrieval settings.

The runtime is in `finrag.py`. PyMuPDF extracts text blocks and qualifying ruled tables. Chunks prefer sentence and paragraph boundaries, use a 384-token target cap and overlap by roughly 48 tokens. E5-small runs locally, FAISS and TF-IDF provide the two search rankings, and reciprocal rank fusion combines them with a 1.5× keyword weight. An explicitly named indexed year restricts search to that report for that part of the query. A question that names only an unindexed year such as 2024 searches across all three reports. OpenRouter generates the answer from the retrieved chunks.

## Components, limitations and possible fixes

The fixes below are proposals for the next iteration. Use the recorded failures and new questions to measure their effect.

| Component | What we built and why | Result or limitation | Possible fix |
|---|---|---|---|
| **PDF corpus** | Three BIAT management reports from 2021, 2023 and 2025. Each passage carries its report year and PDF page so readers can trace the answer. | **553 PDF pages.** A newer report can also contain an earlier year’s figures. The exact PDF editions matter for reproducing citations. | Keep the tested files and checksums. Track the **fiscal year of the figure** separately from the **year of its source report** where the text or table makes that clear. |
| **Text extraction** | PyMuPDF reads text blocks, cleans spacing and soft hyphens, and keeps page boundaries. | **495 pages** contributed passages. Scanned text and complex columns can still be missed or read out of order. | Inspect the source page whenever something is missing from the extracted text. Apply OCR to useful image-only pages and adjust reading order where columns get mixed. |
| **Financial tables** | PyMuPDF extracts qualifying ruled tables with multiple year columns. Each table chunk repeats the headers and nearby context to keep values with their labels. | Verified that the 2025 report’s p. 29 keeps the *Bénéfice de l’année* row with its year columns. Borderless tables and merged headers can still lose that association. | Audit the tables behind failed questions. Preserve each row’s indicator, year, unit and scope together. |
| **Chunking** | Split at paragraph, sentence or line boundaries where possible, with a **384-token cap** and about **48 tokens of overlap**. This keeps nearby context together for embedding. | **1,229 passages.** The longest input including the embedding prefix was **394 of 512 tokens**. A figure can still land in a different chunk from its label. | Inspect the chunks on the right page when retrieval misses. If the answer sits in a neighboring chunk, test adding that neighbor within a token budget. |
| **Embeddings** | Local multilingual E5-small with Sentence Transformers and CPU PyTorch. It turns questions and passages into normalized **384-dimensional vectors** for semantic search. | The model can locate related text across languages, but similarity can favor a related paragraph about the wrong year or figure. | Measure evidence rank across French and English rewordings before trying a larger model. |
| **FAISS** | Exact vector search over the whole corpus or an individual report. It gives FinRAG a semantic ranking for the passages. | FAISS alone covered **7/14 at k=5** in the current benchmark. | Keep exact search for this corpus size. If evidence appears low in the ranking, inspect the competing hits before changing the index. |
| **TF-IDF and rank fusion** | TF-IDF adds exact word and phrase matches. Reciprocal rank fusion combines the two lists, looking at up to **60 candidates each** and giving the keyword list **1.5× weight**. | The hybrid reached **13/14 at k=5**. The longer net-income comparison still misses, while its shorter rewording succeeds. | Compare where the right passages rank in each list. Test French term normalization and filtering cover or contents pages against held-out prompts. |
| **Year handling and top-k** | Explicit indexed years route search to their reports. A question that only mentions 2024 searches all three reports. The default selects up to **five distinct PDF pages per search**, with one chunk per page. | Mixed requests such as 2023 and 2024 can narrow search to the 2023 report. The page rule can also select the wrong chunk from the right page. | Add a mixed-year regression and check a cross-report fallback. Test a second same-page chunk only where it recovers evidence. |
| **English query handling** | A small glossary maps common terms such as *net income* to *résultat net* for search. Generation still receives the original question. | At k=5, **0/3 English presets** had complete evidence before normalization and **3/3** did afterward. That result covers the three presets. | Add unseen English phrasings and scope qualifiers to the benchmark before expanding the glossary. |
| **Augmentation and generation** | FinRAG sends the question plus retrieved passages and page labels to NVIDIA Nemotron through OpenRouter. The prompt checks year, unit and scope. An eligible historical refusal gets one retry with passages from newer reports that mention the requested year. | The 2024 question sometimes refused even with the figure in context. After the prompt and retry changes, **3 English runs and 1 French run** returned **357.8 million dinars**. The unsupported 2027 question still refused. | Log the model used and the outcome of each attempt. Test evidence ranking within the prompt when several retrieved figures compete. |
| **Citations and refusals** | An answer must contain a recognized citation, and every recognized citation must come from the passages sent to the model. The app distinguishes an explicit refusal, a failed citation check and an API error. | A valid page label can still accompany the wrong number, unit or scope. | Verify each financial claim together with its year, amount, unit, scope and cited passage. |
| **Evaluation** | `test_cases.json` holds **17 questions**, including **14 answerable and 3 expected refusals**. `evaluate.py` measures retrieval at k=3, 5 and 10 and can run generation checks. | Current complete evidence coverage is **11/14 at k=3**, **13/14 at k=5**, and **13/14 at k=10**. The evaluator checks answer values and citations separately, so swapped years can slip through. The README’s **8/8 answers and 2/2 refusals** came from the earlier 10-question suite. | Score whole claims and add separate counts for retrieval misses, false refusals, citation failures and API errors. Rerun the full suite in both languages through the web app’s search path. |
| **Runtime and cache** | Docker packages the CLI and Flask app. The `finrag-hf` volume stores the model and passage embeddings. The web app builds its search indexes on the first request and reuses them. | A fresh Docker build and live web checks passed. Loading E5 and building the indexes still costs time on the first request. The Kaggle notebook represents an earlier pipeline. | Measure cold and warm latency before optimizing. Use the current Docker version for experiments so results stay comparable. |nload require internet access. | Follow the setup above and verify a fresh build and retrieval run when changing dependencies. |
| Hugging Face model and passage cache | Running without the volume repeats downloads or indexing. The passage cache version must change when extraction or chunking changes. | Mount `finrag-hf` and bump `CACHE_VERSION` after changes to the indexed text. Compare cold and warm runs when investigating startup. |
