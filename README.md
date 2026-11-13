FinRAG answers questions in French using BIAT's 2021, 2023 and 2025 management reports. Start from commit `7957c99` on `main`. The first issues to investigate are retrieval sensitivity to wording and an intermittent refusal on a question about 2024. [GitHub repository](https://github.com/Soni-KR/NLP_MINI_Project)

## Run locally

1. Clone the GitHub repository and open its root folder, where `Dockerfile` and `finrag.py` are located.
2. Put the three PDFs from the handoff archive in that folder. Keep the filenames below. The PDFs are excluded from Git, and the Docker build requires all three. The archive contains the exact copies used for evaluation. [Private Kaggle dataset, if you have access](https://www.kaggle.com/datasets/mouradkraiem/finrag-biat-reports)
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

# Inspect the passages for the same wording at k=5. This works without an API key.
docker run --rm -v finrag-hf:/root/.cache/huggingface finrag:local --top-k 5 --show-passages "Quel était le résultat net de la BIAT en 2024 ?"

# Score retrieval at k=3, 5 and 10
docker run --rm -v finrag-hf:/root/.cache/huggingface --entrypoint python finrag:local /app/evaluate.py

# Also check generated values, citations and refusals at k=5
docker run --rm --env-file .env -v finrag-hf:/root/.cache/huggingface --entrypoint python finrag:local /app/evaluate.py --generate
```

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

| Component | Possible problem or limitation | Possible fix |
|---|---|---|
| BIAT PDF corpus | The local report editions differ from the public downloads. Adding pages or swapping versions can change rankings and page references. | Keep the tested PDFs as the baseline. Record checksums and recheck reference pages whenever the corpus changes. |
| PyMuPDF text extraction and cleanup | Image-only text is missed and pages with no extracted letters are skipped. Text blocks can still mix columns or break labels across lines. | Inspect sparse pages for useful content and compare problem passages with the PDF. Apply OCR to affected pages when needed. |
| PyMuPDF ruled table extraction | Tables with missing rules, merged headers or undetected year columns can fall back to plain text. A figure can lose its unit or column label. | Inspect the tables behind failed questions and extract their rows with the original labels, year headers and units. |
| Token chunking and overlap | A long sentence or financial statement can still straddle two chunks. Overlap can repeat evidence in the retrieved context. | Check the chunks around the failed figure. Adjust boundaries or overlap and measure evidence coverage and duplication. |
| Year filters and page metadata | Search filters by report year. A question that names both an indexed year and an unindexed year can miss a historical figure in a different report. PDF positions can differ from printed page numbers. | Track the fiscal year separately from the report year for dated rows and passages. Add a test that needs a historical figure from a newer report. |
| E5-small embeddings through Sentence Transformers and CPU PyTorch | Equivalent French phrases can receive different similarity scores. A high score can also favor a related passage with the wrong figure. | Use held-out paraphrases to compare query normalization or another embedding model. Track both evidence rank and indexing time. |
| FAISS and NumPy vectors | Exact search compares against every vector in the selected index, and the indexes are rebuilt when the app starts. | Measure startup, memory and search time before trying a persistent or approximate index. |
| TF-IDF keyword search in scikit-learn | Singular and plural forms, synonyms and generic headings can change the ranking. | Test French term normalization and filtering obvious cover or contents chunks. Check both comparison phrasings. |
| Reciprocal rank fusion | The 60-candidate windows and 1.5× keyword weight can favor generic matches. They were tuned on the small benchmark. | Compare FAISS, TF-IDF and fused ranks on the same held-out questions before changing weights or candidate counts. |
| Top-k and page deduplication | The default keeps up to five distinct pages per report and one chunk per page. The selected chunk can omit a figure elsewhere on the same page. Larger k can add irrelevant text. | Measure coverage at k=3, 5 and 10. Try additional same-page context only where it recovers evidence within a set token budget. |
| Prompt and retrieved context | The model can mix years, units or bank-only and consolidated figures when passages contain competing values. | Check the exact passages sent. Preserve table headers and units, and validate each comparison row against its source. |
| OpenRouter, OpenAI Python client and NVIDIA Nemotron models | The Super VL model and Lightning fallback can produce different answers from the same passages. Provider errors or changes can interrupt generation. | Record the returned model and any API error code. Repeat selected prompts with fixed evidence to measure variation. |
| Citation validation and refusals | The citation check confirms format and retrieved page membership. A correct reply with a malformed citation can become a refusal, and an accepted citation can point to a passage that does not support the claim. | Record whether the model refused or citation validation rejected the reply. Verify figure, year, unit and scope against the cited passage. |
| `evaluate.py` and `test_cases.json` | The small benchmark checks numbers and citations separately. A result can pass while attaching a figure to the wrong year. | Add the manual failures and held-out paraphrases. Score each figure together with its year, unit, scope and reference page. |
| CLI and `--show-passages` | The default view shows only the final answer. Passage display reveals the selected chunks, while raw replies and rejection reasons need separate inspection. | Use passage display to reproduce failures. Add optional diagnostics if the same failures are hard to trace. |
| Docker, Python 3.11 and pinned dependencies | A fresh clone needs the exact three PDFs. A cold build and first model download require internet access. | Follow the setup above and verify a fresh build and retrieval run when changing dependencies. |
| Hugging Face model and passage cache | Running without the volume repeats downloads or indexing. The passage cache version must change when extraction or chunking changes. | Mount `finrag-hf` and bump `CACHE_VERSION` after changes to the indexed text. Compare cold and warm runs when investigating startup. |
