"""Train the DIET model of one or more languages.

    python scripts/generate_data.py            # once, or whenever names/templates change
    python scripts/train.py --lang en
    python scripts/train.py --lang all

Writes models/<lang>/{diet_model.pt, word_dict.json, ngram_dict.json, intent_labels.json,
entity_labels.json, model_config.json}.
"""

import argparse
import json
import math
import os
import shutil
import time

import torch

from diet_classifier.training import DataLoader, Trainer
from diet_classifier.model import DIETModel, SparseFeatureExtractor
import langs
from langs import load_json


def make_extractor(word_dict_size, ngram_dict_size, ngram_overflow_size=100, ngram_min=2, ngram_max=5):
    return SparseFeatureExtractor(
        word_dict_size=word_dict_size,
        ngram_dict_size=ngram_dict_size,
        ngram_overflow_size=ngram_overflow_size,
        ngram_min=ngram_min,
        ngram_max=ngram_max,
        pad_token="[PAD]",
        cls_token="[CLS]",
        unk_token="[UNK]"
    )


def round_up(value: float, step: int) -> int:
    return int(math.ceil(value / step) * step)


def train_language(lang: str, args, device: str):
    print("=" * 60)
    print(f"Training language: {lang}")
    print("=" * 60)
    data_path = os.path.join(langs.data_dir(lang), "data.yml")
    if not os.path.isfile(data_path):
        raise FileNotFoundError(f"{data_path} not found. Run scripts/generate_data.py first.")
    entity_labels = load_json(os.path.join(langs.data_dir(lang), "entity_labels.json"))
    intent_labels = load_json(os.path.join(langs.data_dir(lang), "intent_labels.json"))
    out = langs.model_dir(lang)
    files = langs.model_files(lang)
    os.makedirs(out, exist_ok=True)

    data_loader = DataLoader(data_path=data_path,
                             batch_size=args.batch_size,
                             intent_labels=intent_labels,
                             entity_labels=entity_labels,
                             cls_token="[CLS]",
                             pad_token="[PAD]",
                             pad_entity_tag="PAD")
    all_texts = [sample["text"] for batch in data_loader.data for sample in batch]

    # Size the dictionaries to the corpus so that no word (e.g. a device name) is truncated to [UNK]
    probe = make_extractor(10**6, 10**6)
    probe.build_word_dict(all_texts)
    probe.build_ngram_dict(all_texts)
    config = dict(langs.DEFAULT_MODEL_CONFIG)
    config["word_dict_size"] = max(config["word_dict_size"], round_up(len(probe.word_dict) * 1.15, 50))
    config["ngram_dict_size"] = max(config["ngram_dict_size"], round_up(len(probe.ngram_dict) * 1.1, 100))
    print(f"Vocabulary: {len(probe.word_dict)} words, {len(probe.ngram_dict)} n-grams "
          f"-> word_dict_size={config['word_dict_size']}, ngram_dict_size={config['ngram_dict_size']}")

    sparse_extractor = make_extractor(**config)
    sparse_extractor.build_word_dict(all_texts)
    sparse_extractor.build_ngram_dict(all_texts)
    sparse_extractor.save_dicts(files["word_dict"], files["ngram_dict"])
    with open(files["model_config"], "w", encoding="utf8") as f:
        json.dump(config, f, indent=2)
    shutil.copy(os.path.join(langs.data_dir(lang), "entity_labels.json"), files["entity_labels"])
    shutil.copy(os.path.join(langs.data_dir(lang), "intent_labels.json"), files["intent_labels"])

    model = DIETModel(
        device=device,
        sparse_extractor=sparse_extractor,
        num_entity_tags=len(entity_labels),
        num_intent_tags=len(intent_labels),
        pad_entity_tag_idx=entity_labels.index("PAD"),
        eos_entity_tag_idx=entity_labels.index("EOS"),
        bos_entity_tag_idx=entity_labels.index("BOS")
    )

    trainer = Trainer(
        model=model,
        device=device,
        lr=args.lr,
        epochs=args.epochs,
        data_loader=data_loader,
        checkpoint_path=out,
        checkpoint_name="diet_model.pt"
    )
    trainer.train()
    if args.plot:
        trainer.plot_loss_history()

    # Sanity check with the best checkpoint
    model.load_state_dict(torch.load(files["model"], map_location=device))
    model.eval()
    input_batch = langs.SAMPLE_SENTENCES.get(lang, [])
    if not input_batch:
        return
    # The model expects equally long inputs (+1 for [CLS]), so pad with [PAD]
    max_len = max(len(t.split()) for t in input_batch)
    input_batch = [t + " [PAD]" * (max_len - len(t.split())) for t in input_batch]
    with torch.no_grad():
        tensor_entities, tensor_intent = model(input_batch)
    print("--" * 30)
    for b, sentence in enumerate(input_batch):
        init_time = time.perf_counter()
        print(f"Sentence: '{sentence}'")
        print("Predicted entity tags:", [entity_labels[idx] for idx in tensor_entities[b].tolist()])
        print("Predicted intent:", intent_labels[torch.argmax(tensor_intent[b]).item()])
        print(f"Inference time: {(time.perf_counter() - init_time) * 1000:.2f} ms")
        print("--" * 30)
    print(f"Model saved to {files['model']}")


def main():
    parser = argparse.ArgumentParser(description="Train the DIET model of one or more languages")
    parser.add_argument("--lang", nargs="+", required=True,
                        help="Language codes (en es fr de it ...) or 'all' for every language in data/")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--plot", action="store_true", help="Show the loss curve at the end")
    parser.add_argument("--device", default="cpu", choices=["cpu", "auto", "cuda"],
                        help="Device to use (default: cpu; auto: CUDA only if it works, else CPU)")
    args = parser.parse_args()

    device = langs.pick_device(args.device)
    print(f"Using device: {device}")
    selected = langs.data_languages() if args.lang == ["all"] else args.lang
    for lang in selected:
        train_language(lang, args, device)


if __name__ == "__main__":
    main()
