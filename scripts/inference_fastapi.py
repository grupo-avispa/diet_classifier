import os
import torch
import argparse
from fastapi import FastAPI
import uvicorn
from diet_classifier.inference.server import DIETServer
import langs


# One DIETServer per language, filled in main()
servers: dict = {}
default_lang: str = None

# Instantiates the FastAPI app.
app = FastAPI(title="DIET Inference API", version="1.0")

@app.get("/")
async def root():
    """
    Root Endpoint: Provides basic information about the API.
    """
    return {"message": "DIET inference API is running",
            "endpoints": ["/health", "/predict"],
            "model": "DIETClassifier",
            "languages": sorted(servers),
            "default_language": default_lang}


@app.get("/health")
async def health():
    """
    Health Check Endpoint: Verifies that the API is running and the model is loaded.
    """
    return {"status": "healthy", "model_loaded": bool(servers), "languages": sorted(servers)}


@app.post("/predict")
async def predict(request: dict):
    """
    Prediction Endpoint: Accepts {"text": "...", "language": "es"} and returns model predictions.
    "language" is optional and defaults to the server's default language.
    """
    if not servers:
        return {"error": "Model server is not initialized", "status": "error"}
    if "text" not in request:
        return {"error": "No text provided", "status": "error"}
    language = request.get("language") or default_lang
    if language not in servers:
        return {"error": f"Language '{language}' not available. Available: {sorted(servers)}",
                "status": "error"}
    try:
        results = servers[language].predict([request["text"]])
        return {"status": "success", "language": language, "result": results[0]}
    except Exception as e:
        return {"error": str(e), "status": "error"}

def main():
    global default_lang
    # Get command line arguments
    parser = argparse.ArgumentParser(description="Start DIET FastAPI Inference Server")
    parser.add_argument("-p", "--port", type=int, default=8000, help="Port to run the FastAPI server on")
    parser.add_argument("--langs", nargs="+", default=None,
                        help="Languages to load (default: every language with a trained model in models/)")
    parser.add_argument("--default-lang", default=None,
                        help="Language used when a request has no 'language' (default: first loaded)")
    parser.add_argument("--device", default="cpu", choices=["cpu", "auto", "cuda"],
                        help="Device to use (default: cpu; auto: CUDA only if it works, else CPU)")
    args = parser.parse_args()
    selected = args.langs or langs.trained_languages()
    if not selected:
        raise SystemExit("No trained models found in models/<lang>/. Run scripts/train.py first.")
    device = langs.pick_device(args.device)
    print(f"Using device: {device}")
    # Initialize one DIETServer per language
    for lang in selected:
        files = langs.model_files(lang)
        print(f"Loading language '{lang}'...")
        servers[lang] = DIETServer(
            device=device,
            model_path=files["model"],
            word_dict_path=files["word_dict"],
            ngram_dict_path=files["ngram_dict"],
            entity_labels_path=files["entity_labels"],
            intent_labels_path=files["intent_labels"],
            model_config_path=files["model_config"]
        )
    default_lang = args.default_lang or selected[0]
    if default_lang not in servers:
        raise SystemExit(f"Default language '{default_lang}' is not among the loaded ones: {sorted(servers)}")
    print(f"Languages loaded: {sorted(servers)} (default: {default_lang})")
    # Start the FastAPI server
    uvicorn.run(app, host="0.0.0.0", port=args.port)


if __name__ == "__main__":
    main()