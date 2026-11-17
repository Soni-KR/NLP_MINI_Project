import argparse
import json
import re
import sys
from decimal import Decimal
from pathlib import Path

from finrag import (
    CITATION, GENERATION_FAILURE, REFUSAL, TEST_CASES_PATH,
    answer, build_index, localized_search_query, make_client, retrieve,
)


PROJECT_DIR = Path(__file__).resolve().parent
NUMBER = re.compile(r"[+−-]?(?:\d{1,3}(?:[^\S\r\n]+\d{3})+|\d+)(?:[^\S\r\n]*[.,]\d+)*")
CITATIONS = re.compile(rf"{CITATION.pattern}(?:(?:\s|,|\bet\b|\band\b)*{CITATION.pattern})*")
DINARS = r"(?:TND|DT|dinars?(?:\s+tunisiens?)?|Tunisian\s+dinars?)"
OTHER_CURRENCY = r"(?:euros?|dollars?|EUR|USD)"
UNIT = re.compile(
    rf"(?P<million_tnd>\b(?:millions?\s+(?:(?:de|of)\s+)?{DINARS}|(?-i:MDT?)|M\s*TND)\b)"
    rf"|(?P<thousand_tnd>\b(?:milliers?\s+(?:de\s+)?|thousands?(?:\s+of)?\s+){DINARS}\b)"
    r"|(?P<branches>\b(?:agences?|branches|branch|points?\s+de\s+vente)\b)"
    r"|(?P<people>\b(?:collaborateurs?|employés?|salariés?|personnes?|employees?|staff|people)\b)"
    rf"|(?P<other>\b(?:(?:milliards?|billions?)\s+(?:(?:de|of)\s+)?{DINARS}"
    rf"|(?:millions?\s+(?:(?:de|of)\s+)?)?{OTHER_CURRENCY}|{DINARS})\b|[€$])"
    r"|(?P<percent>%|\b(?:pour cent|percent)\b)",
    re.IGNORECASE,
)


def amount(token):
    token = re.sub(r"\s+", "", token).replace("−", "-")
    parts = re.split(r"[.,]", token)
    grouped = (not ("." in token and "," in token)
               and len(parts[0].lstrip("+-")) <= 3
               and all(len(part) == 3 for part in parts[1:]))
    number = "".join(parts) if len(parts) == 1 or grouped else "".join(parts[:-1]) + "." + parts[-1]
    return Decimal(number)


def amounts(text):
    return {amount(token) for token in NUMBER.findall(text)}


def contains_expected(text, item):
    if "text" in item:
        return bool(re.search(rf"\b{re.escape(item['text'])}\b", text, re.IGNORECASE))
    return Decimal(item["value"]) in amounts(text)


def missing_evidence(evidence, hits):
    missing = []
    for item in evidence:
        pages = {tuple(page) for page in item["pages"]}
        if not any((hit["year"], hit["page"]) in pages and contains_expected(hit["text"], item) for hit in hits):
            period = item.get("fiscal_year", item["pages"][0][0])
            value = item.get("text", item.get("value"))
            references = "/".join(f"{year}p{page}" for year, page in item["pages"])
            missing.append(f"{period} {value} [{references}]")
    return missing


def print_corpus_stats(data_dir, index_data):
    import pymupdf

    passages, model, indexes, _ = index_data
    print("Corpus — pages PDF, pages indexées, pages sans passage, passages, tokens min–max, passages avec |")
    for year in indexes:
        if year is None:
            continue
        selected = [passage for passage in passages if passage["year"] == year]
        sizes = [len(tokens) for tokens in model.tokenizer([p["text"] for p in selected],
                                                          add_special_tokens=False,
                                                          verbose=False)["input_ids"]]
        with pymupdf.open(data_dir / f"rapport_biat_{year}.pdf") as report:
            page_count = len(report)
        indexed = len({passage["page"] for passage in selected})
        pipe_passages = sum("|" in passage["text"] for passage in selected)
        print(f"{year}  {page_count}, {indexed}, {page_count - indexed}, "
              f"{len(selected)}, {min(sizes)}–{max(sizes)}, {pipe_passages}")


def evaluate_retrieval(cases, top_ks, index_data, method="hybrid"):
    answerable = [(number, case) for number, case in enumerate(cases, 1) if case["evidence"]]
    totals = dict.fromkeys(top_ks, 0)
    print(f"\nRecherche {method} — couverture complète des preuves")
    print("Chaque élément attendu doit apparaître dans un passage d’une page de référence.")
    print("Question\t" + "\t".join(f"k={k}" for k in top_ks))
    for number, case in answerable:
        row = [f"Q{number:02}"]
        query = localized_search_query(case["question"], case.get("language", "fr"))
        for k in top_ks:
            hits = retrieve(query, k, *index_data, method=method)
            missing = missing_evidence(case["evidence"], hits)
            totals[k] += not missing
            row.append("non (" + ", ".join(missing) + ")" if missing else "oui")
        print("\t".join(row))
    for k, count in totals.items():
        rate = count / len(answerable) if answerable else 0
        print(f"k={k}  {count}/{len(answerable)} ({rate:.1%})")


def matching_claims(reply, evidence, unit):
    # A citation group closes a prose claim. Markdown table rows arrive one per line.
    claims = []
    captions = []
    for line in reply.splitlines():
        groups = list(CITATIONS.finditer(line))
        if not groups:
            captions.append(line)
            continue
        prefix = NUMBER.search(line[:groups[0].start()])
        if prefix:
            captions.append(line[:prefix.start()])
        suffix = line[groups[-1].end():]
        if re.match(r"^\s*[,;.]?\s*(?:en|in)\b", suffix, re.IGNORECASE):
            captions.append(suffix)
        start = 0
        for group in groups[:-1]:
            claims.append(line[start:group.end()])
            start = group.end()
        claims.append(line[start:])

    shared_units = {match.lastgroup for match in UNIT.finditer(" ".join(captions))
                    if unit == "percent" or match.lastgroup != "percent"}
    for item in evidence:
        allowed = {f"[{year} p. {page}]" for year, page in item["pages"]}
        fiscal_year = item.get("fiscal_year")
        for claim in claims:
            if allowed.isdisjoint(CITATION.findall(claim)):
                continue
            text = CITATION.sub(" ", claim).replace("*", "")
            if fiscal_year is not None:
                years = set(re.findall(r"\b20\d{2}\b", text))
                if (len(evidence) > 1 and years != {str(fiscal_year)}
                        or len(evidence) == 1 and years and str(fiscal_year) not in years):
                    continue
            if "text" in item:
                if contains_expected(text, item):
                    break
                continue
            expected = Decimal(item["value"])
            claim_units = {match.lastgroup for match in UNIT.finditer(text)
                           if unit == "percent" or match.lastgroup != "percent"}
            for match in NUMBER.finditer(text):
                if amount(match[0]) != expected:
                    continue
                suffix = re.sub(r"^\s*(?:\|\s*)?(?:(?:en|in)\s+)?", "", text[match.end():], flags=re.IGNORECASE)
                local_unit = UNIT.match(suffix)
                if local_unit:
                    matches_unit = local_unit.lastgroup == unit
                else:
                    matches_unit = (claim_units or shared_units) == {unit}
                if matches_unit:
                    break
            else:
                continue
            break
        else:
            return False
    return True


def evaluate_generation(cases, index_data, client, top_k, method="hybrid"):
    from openai import APIError, AuthenticationError

    answers_ok = refusals_ok = false_refusals = missed_refusals = generation_failures = api_errors = 0
    answerable = sum(bool(case["evidence"]) for case in cases)
    print(f"\nGénération avec recherche {method}, k={top_k}")
    for number, case in enumerate(cases, 1):
        language = case.get("language", "fr")
        query = localized_search_query(case["question"], language)
        hits = retrieve(query, top_k, *index_data, method=method)
        try:
            reply = answer(case["question"], hits, client, language=language)
        except AuthenticationError:
            print("OpenRouter refuse OPENROUTER_API_KEY (401).", file=sys.stderr)
            return 1
        except APIError as error:
            print(f"Q{number:02}  erreur OpenRouter ({type(error).__name__})")
            api_errors += 1
            continue
        if reply == GENERATION_FAILURE:
            print(f"Q{number:02}  réponse vide ou citations rejetées")
            generation_failures += 1
            continue

        evidence = case["evidence"]
        if not evidence:
            refused = reply == REFUSAL
            refusals_ok += refused
            missed_refusals += not refused
            print(f"Q{number:02}  refus {'correct' if refused else 'manqué'}")
            continue
        if reply == REFUSAL:
            print(f"Q{number:02}  refus incorrect")
            false_refusals += 1
            continue

        correct = matching_claims(reply, evidence, case.get("unit"))
        answers_ok += correct
        print(f"Q{number:02}  réponse et sources liées {'oui' if correct else 'non'}")

    print(f"Réponses et sources liées  {answers_ok}/{answerable}")
    print(f"Refus attendus  {refusals_ok}/{len(cases) - answerable}")
    print(f"Refus incorrects  {false_refusals}")
    print(f"Refus manqués  {missed_refusals}")
    print(f"Réponses vides ou citations rejetées  {generation_failures}")
    print(f"Erreurs OpenRouter  {api_errors}")
    print("Le contrôle automatique rapproche la réponse et ses sources dans une même ligne ou proposition. "
          "Vérifiez aussi l’indicateur, le périmètre et le sens de la source.")
    return int(answers_ok != answerable or refusals_ok != len(cases) - answerable)


def main():
    parser = argparse.ArgumentParser(description="Évalue la recherche FinRAG et, sur demande, les réponses.")
    parser.add_argument("--data-dir", type=Path, default=PROJECT_DIR, help="Dossier des trois rapports PDF.")
    parser.add_argument("--cases", type=Path, default=TEST_CASES_PATH, help="Fichier JSON des questions de test.")
    parser.add_argument("--corpus-stats", action="store_true", help="Affiche aussi les statistiques du corpus indexé.")
    parser.add_argument("--top-k", type=int, nargs="+", default=[3, 5, 10], help="Valeurs de k (défaut : 3 5 10).")
    parser.add_argument("--methods", nargs="+", choices=("dense", "lexical", "hybrid"), default=["hybrid"],
                        help="Méthodes à comparer en recherche (défaut : hybrid).")
    parser.add_argument("--generate", action="store_true", help="Teste aussi les réponses OpenRouter.")
    parser.add_argument("--generation-k", type=int, default=5,
                        help="Nombre de passages par rapport pour la génération (défaut : %(default)s).")
    parser.add_argument("--generation-method", choices=("dense", "lexical", "hybrid"), default="hybrid",
                        help="Méthode utilisée pour générer les réponses (défaut : hybrid).")
    args = parser.parse_args()
    if min(args.top_k) < 1:
        parser.error("--top-k doit contenir des valeurs supérieures ou égales à 1.")
    if args.generation_k < 1:
        parser.error("--generation-k doit être supérieur ou égal à 1.")

    try:
        cases = json.loads(args.cases.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        parser.error(f"Impossible de lire les questions : {error}")
    if not isinstance(cases, list) or not cases:
        parser.error("Le fichier de questions doit contenir une liste JSON non vide.")

    if args.generate:
        client = make_client()
        if client is None:
            parser.error("Définissez OPENROUTER_API_KEY pour utiliser --generate.")

    try:
        index_data = build_index(args.data_dir)
    except (OSError, ValueError) as error:
        print(error, file=sys.stderr)
        return 1

    if args.corpus_stats:
        print_corpus_stats(args.data_dir, index_data)
    for method in dict.fromkeys(args.methods):
        evaluate_retrieval(cases, sorted(set(args.top_k)), index_data, method=method)
    return evaluate_generation(cases, index_data, client, args.generation_k,
                               method=args.generation_method) if args.generate else 0


if __name__ == "__main__":
    raise SystemExit(main())
