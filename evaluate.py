import argparse
import json
import re
import sys
from decimal import Decimal
from pathlib import Path

from finrag import CITATION, REFUSAL, TEST_CASES_PATH, answer, build_index, make_client, retrieve


PROJECT_DIR = Path(__file__).resolve().parent
NUMBER = re.compile(r"[+−-]?(?:\d{1,3}(?:[^\S\r\n]+\d{3})+|\d+)(?:[^\S\r\n]*[.,]\d+)*")


def amounts(text):
    values = set()
    for token in NUMBER.findall(text):
        token = re.sub(r"\s+", "", token).replace("−", "-")
        parts = re.split(r"[.,]", token)
        grouped = (not ("." in token and "," in token)
                   and len(parts[0].lstrip("+-")) <= 3
                   and all(len(part) == 3 for part in parts[1:]))
        number = "".join(parts) if len(parts) == 1 or grouped else "".join(parts[:-1]) + "." + parts[-1]
        values.add(Decimal(number))
    return values


def missing_evidence(evidence, hits):
    missing = []
    for item in evidence:
        pages = {tuple(page) for page in item["pages"]}
        expected = Decimal(item["value"])
        if not any((hit["year"], hit["page"]) in pages and expected in amounts(hit["text"]) for hit in hits):
            missing.append(str(item["pages"][0][0]))
    return missing


def evaluate_retrieval(cases, top_ks, index_data):
    answerable = [(number, case) for number, case in enumerate(cases, 1) if case["evidence"]]
    totals = dict.fromkeys(top_ks, 0)
    print("Couverture complète des preuves pour les questions avec une réponse")
    print("Chaque valeur attendue doit apparaître dans un passage d’une page de référence.")
    print("Question\t" + "\t".join(f"k={k}" for k in top_ks))
    for number, case in answerable:
        row = [f"Q{number:02}"]
        for k in top_ks:
            hits = retrieve(case["question"], k, *index_data)
            missing = missing_evidence(case["evidence"], hits)
            totals[k] += not missing
            row.append("non (" + ", ".join(missing) + ")" if missing else "oui")
        print("\t".join(row))
    for k, count in totals.items():
        print(f"k={k}  {count}/{len(answerable)} ({count / len(answerable):.1%})")


def evaluate_generation(cases, index_data, client, top_k):
    from openai import APIError, AuthenticationError

    values_ok = citations_ok = answers_ok = refusals_ok = errors = 0
    answerable = sum(bool(case["evidence"]) for case in cases)
    print(f"\nGénération à k={top_k}")
    for number, case in enumerate(cases, 1):
        hits = retrieve(case["question"], top_k, *index_data)
        try:
            reply = answer(case["question"], hits, client)
        except AuthenticationError:
            print("OpenRouter refuse OPENROUTER_API_KEY (401).", file=sys.stderr)
            return 1
        except APIError as error:
            print(f"Q{number:02}  erreur OpenRouter ({type(error).__name__})")
            errors += 1
            continue

        evidence = case["evidence"]
        if not evidence:
            refused = reply == REFUSAL
            refusals_ok += refused
            print(f"Q{number:02}  refus {'correct' if refused else 'manqué'}")
            continue

        values = amounts(CITATION.sub(" ", reply))
        citations = set(CITATION.findall(reply))
        expected_values = all(Decimal(item["value"]) in values for item in evidence)
        expected_pages = all(any(f"[{year} p. {page}]" in citations for year, page in item["pages"])
                             for item in evidence)
        values_ok += expected_values
        citations_ok += expected_pages
        answers_ok += expected_values and expected_pages
        print(f"Q{number:02}  valeurs {'oui' if expected_values else 'non'}, "
              f"citations {'oui' if expected_pages else 'non'}")

    print(f"Valeurs attendues  {values_ok}/{answerable}")
    print(f"Pages de référence citées  {citations_ok}/{answerable}")
    print(f"Valeurs et citations  {answers_ok}/{answerable}")
    print(f"Refus attendus  {refusals_ok}/{len(cases) - answerable}")
    print(f"Erreurs OpenRouter  {errors}")
    print("Ces contrôles repèrent les chiffres et les pages séparément. Vérifiez les unités, "
          "le sens des citations et l’association de chaque chiffre à son année.")
    return int(errors > 0)


def main():
    parser = argparse.ArgumentParser(description="Évalue la recherche FinRAG et, sur demande, les réponses.")
    parser.add_argument("--data-dir", type=Path, default=PROJECT_DIR, help="Dossier des trois rapports PDF.")
    parser.add_argument("--top-k", type=int, nargs="+", default=[3, 5, 10], help="Valeurs de k (défaut : 3 5 10).")
    parser.add_argument("--generate", action="store_true", help="Teste aussi les réponses OpenRouter.")
    parser.add_argument("--generation-k", type=int, default=5,
                        help="Nombre de passages par rapport pour la génération (défaut : %(default)s).")
    args = parser.parse_args()
    if min(args.top_k) < 1:
        parser.error("--top-k doit contenir des valeurs supérieures ou égales à 1.")
    if args.generation_k < 1:
        parser.error("--generation-k doit être supérieur ou égal à 1.")

    if args.generate:
        client = make_client()
        if client is None:
            parser.error("Définissez OPENROUTER_API_KEY pour utiliser --generate.")

    cases = json.loads(TEST_CASES_PATH.read_text(encoding="utf-8"))
    try:
        index_data = build_index(args.data_dir)
    except (OSError, ValueError) as error:
        print(error, file=sys.stderr)
        return 1

    evaluate_retrieval(cases, sorted(set(args.top_k)), index_data)
    return evaluate_generation(cases, index_data, client, args.generation_k) if args.generate else 0


if __name__ == "__main__":
    raise SystemExit(main())
