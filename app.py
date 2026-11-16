import os
import threading
from pathlib import Path

from flask import Flask, jsonify, render_template, request

from finrag import (
    GENERATION_FAILURE,
    REFUSAL,
    answer,
    build_index,
    citation,
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
    payload = request.get_json(silent=True) or {}
    question = payload.get("question", "")
    if not isinstance(question, str) or not question.strip():
        return jsonify({"error": "La question doit contenir du texte."}), 400

    try:
        top_k = int(payload.get("top_k", DEFAULT_TOP_K))
    except (TypeError, ValueError):
        return jsonify({"error": "Le nombre de passages doit être un entier."}), 400
    if not 1 <= top_k <= MAX_TOP_K:
        return jsonify({"error": f"Le nombre de passages doit être compris entre 1 et {MAX_TOP_K}."}), 400

    try:
        client = make_client()
        if client is None:
            return jsonify({"error": "OPENROUTER_API_KEY n'est pas configurée."}), 503

        from openai import APIError, AuthenticationError

        passages, model, indexes, keyword_search = get_index()
        hits = retrieve(question.strip(), top_k, passages, model, indexes, keyword_search)
        reply = answer(question.strip(), hits, client)
    except ModuleNotFoundError as error:
        if error.name not in {"openai", "dotenv"}:
            raise
        return jsonify({
            "error": "Une dépendance Python est absente. Exécutez : python -m pip install -r requirements.txt"
        }), 503
    except FileNotFoundError:
        return jsonify({"error": "Les trois rapports PDF sont nécessaires pour démarrer FinRAG."}), 503
    except ImportError:
        return jsonify({
            "error": "Les dépendances scientifiques sont incompatibles. "
                     "Exécutez : python -m pip install --force-reinstall -r requirements.txt"
        }), 503
    except ValueError as error:
        return jsonify({"error": str(error)}), 400
    except AuthenticationError:
        return jsonify({
            "error": "OpenRouter refuse la clé API. Vérifiez OPENROUTER_API_KEY dans le fichier .env."
        }), 502
    except APIError as error:
        status_code = getattr(error, "status_code", None)
        if status_code == 429:
            message = "La limite OpenRouter est atteinte. Réessayez plus tard ou utilisez une autre clé."
        elif status_code and 400 <= status_code < 500:
            message = f"OpenRouter a refusé la requête (HTTP {status_code}). Vérifiez le modèle et la clé API."
        else:
            message = "OpenRouter est momentanément indisponible. Réessayez dans quelques instants."
        return jsonify({"error": message}), 502

    status = "failure" if reply == GENERATION_FAILURE else "refusal" if reply == REFUSAL else "answer"
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
