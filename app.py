import os
import threading
from pathlib import Path

from flask import Flask, jsonify, render_template, request

from finrag import (
    ANSWER_LANGUAGES,
    ENGLISH_REFUSAL,
    GENERATION_FAILURE,
    REFUSAL,
    answer,
    build_index,
    citation,
    localized_search_query,
    make_client,
    retrieve,
)


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_TOP_K = 5
MAX_TOP_K = 10

app = Flask(__name__)
index_lock = threading.Lock()
index_data = None


def get_index():
    global index_data
    if index_data is None:
        with index_lock:
            if index_data is None:
                index_data = build_index(PROJECT_DIR)
    return index_data


@app.get("/")
def home():
    return render_template("index.html")


@app.get("/health")
def health():
    return jsonify({"status": "ok", "indexed": index_data is not None})


@app.post("/api/ask")
def ask():
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return jsonify({"error": "La requête doit contenir un objet JSON."}), 400

    language = payload.get("language", "fr")
    if not isinstance(language, str) or language not in ANSWER_LANGUAGES:
        return jsonify({"error": "La langue doit être fr ou en."}), 400

    def error_response(status, french, english):
        return jsonify({"error": english if language == "en" else french}), status

    question = payload.get("question", "")
    if not isinstance(question, str) or not question.strip():
        return error_response(400, "La question doit contenir du texte.", "Enter a question.")

    top_k = payload.get("top_k", DEFAULT_TOP_K)
    if type(top_k) is not int:
        return error_response(400, "Le nombre de passages doit être un entier.",
                              "The number of passages must be an integer.")
    if not 1 <= top_k <= MAX_TOP_K:
        return error_response(400, f"Le nombre de passages doit être compris entre 1 et {MAX_TOP_K}.",
                              f"The number of passages must be between 1 and {MAX_TOP_K}.")

    try:
        from openai import APIError, AuthenticationError

        client = make_client()
        if client is None:
            return error_response(503, "OPENROUTER_API_KEY n'est pas configurée.",
                                  "OPENROUTER_API_KEY is not configured.")

        try:
            passages, model, indexes, keyword_search = get_index()
        except ValueError:
            app.logger.exception("FinRAG could not index the reports")
            return error_response(503, "FinRAG n’a pas pu indexer les rapports.",
                                  "FinRAG could not index the reports.")
        query = localized_search_query(question.strip(), language)
        hits = retrieve(query, top_k, passages, model, indexes, keyword_search)
        reply = answer(question.strip(), hits, client, language=language)
    except ModuleNotFoundError:
        return error_response(503, "Une dépendance Python est absente. "
                              "Exécutez python -m pip install -r requirements.txt",
                              "A Python dependency is missing. Run python -m pip install -r requirements.txt")
    except FileNotFoundError:
        return error_response(503, "Les trois rapports PDF sont nécessaires pour démarrer FinRAG.",
                              "FinRAG needs all three PDF reports.")
    except ImportError:
        return error_response(503, "Les dépendances scientifiques sont incompatibles. "
                              "Exécutez python -m pip install --force-reinstall -r requirements.txt",
                              "The scientific dependencies are incompatible. "
                              "Run python -m pip install --force-reinstall -r requirements.txt")
    except ValueError as error:
        return error_response(400, str(error), "FinRAG could not process this question.")
    except AuthenticationError:
        return error_response(502, "OpenRouter refuse la clé API. Vérifiez OPENROUTER_API_KEY dans le fichier .env.",
                              "OpenRouter rejected the API key. Check OPENROUTER_API_KEY in the .env file.")
    except APIError as error:
        status_code = getattr(error, "status_code", None)
        if status_code == 429:
            details = error.body if isinstance(error.body, dict) else {}
            if isinstance(details.get("error"), dict):
                details = details["error"]
            metadata = details.get("metadata")
            daily_limit = (isinstance(metadata, dict)
                           and metadata.get("limit_source") == "openrouter_free_tier_daily")
            if daily_limit or "free-models-per-day" in str(details.get("message", "")):
                return error_response(
                    502, "Le quota quotidien gratuit de votre compte OpenRouter est épuisé. "
                    "Attendez sa réinitialisation ou augmentez votre quota dans OpenRouter.",
                    "Your OpenRouter account's daily free-model quota is exhausted. "
                    "Wait for its daily reset or increase your quota in OpenRouter.",
                )
            return error_response(502, "OpenRouter applique une limite temporaire de requêtes. "
                                  "Réessayez dans quelques instants.",
                                  "OpenRouter is temporarily limiting requests. Try again shortly.")
        if status_code and 400 <= status_code < 500:
            return error_response(502, f"OpenRouter a refusé la requête (HTTP {status_code}). "
                                  "Vérifiez le modèle et la clé API.",
                                  f"OpenRouter rejected the request (HTTP {status_code}). "
                                  "Check the model and API key.")
        return error_response(502, "OpenRouter est momentanément indisponible. Réessayez dans quelques instants.",
                              "OpenRouter is temporarily unavailable. Try again in a moment.")

    status = "failure" if reply == GENERATION_FAILURE else "refusal" if reply == REFUSAL else "answer"
    if language == "en":
        if status == "refusal":
            reply = ENGLISH_REFUSAL
        elif status == "failure":
            reply = "The reply was empty or its citations could not be verified. Try again."
    return jsonify({
        "answer": reply,
        "status": status,
        "sources": [{"citation": citation(hit), "text": hit["text"]} for hit in hits],
    })


if __name__ == "__main__":
    app.run(
        host=os.getenv("FINRAG_HOST", "127.0.0.1"),
        port=int(os.getenv("FINRAG_PORT", "5000")),
        debug=False,
    )
