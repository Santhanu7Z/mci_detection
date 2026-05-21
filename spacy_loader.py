# spacy_loader.py

import spacy

def load_spacy_model():
    """
    Offline-safe spaCy loader.
    Gracefully falls back to a blank tokenizer if the model is unavailable.
    """
    candidate_models = [
        "en_core_web_sm",
        "en_core_web_md",
        "en_core_web_lg"
    ]

    for model_name in candidate_models:
        try:
            nlp = spacy.load(model_name)
            print(f"✅ Loaded spaCy model: {model_name}")
            return nlp
        except Exception:
            continue

    print("⚠️ No spaCy language model found.")
    print("⚠️ Falling back to blank English tokenizer.")
    return spacy.blank("en")
