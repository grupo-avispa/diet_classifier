"""Generate the per-language training data from templates and the names catalog.

    data/templates/<lang>.yml + data/names.json  ->  data/<lang>/{data,validation}.yml
                                                      data/<lang>/{intent,entity}_labels.json
                                                      <export>/valid_entities/<lang>.json (optional)

Slots such as ``[{sala}](sala)`` in a template are replaced by the names (in that language)
of the catalog entries of that type. Every name is used at least ``--min-per-name`` times
(``--ha-weight`` times more for the entries that exist in Home Assistant).

Usage:
    python scripts/generate_data.py                  # all languages found in data/templates
    python scripts/generate_data.py --lang en fr
    python scripts/generate_data.py --export ../homelai-rpi-shs/config
"""

import argparse
import json
import math
import os
import random
import re
import shutil

import yaml

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "..", "data")
SLOT_RE = re.compile(r"\[\{(\w+)\}\]\((\w+)\)")
KINDS = ("sala", "dispositivo", "modo")


def load_json(path):
    with open(path, "r", encoding="utf8") as f:
        return json.load(f)


def save_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
        f.write("\n")


def available_languages():
    tpl = os.path.join(DATA_DIR, "templates")
    return sorted(f[:-4] for f in os.listdir(tpl) if f.endswith(".yml"))


def name_pool(names, lang, ha_weight):
    """Names per kind for a language; Home Assistant entries are repeated ``ha_weight`` times."""
    pool = {}
    for kind in KINDS:
        items = []
        for entry in names.get(kind, {}).values():
            for n in entry["names"].get(lang, []):
                items.extend([n] * (ha_weight if entry.get("ha") else 1))
        pool[kind] = items
    return pool


def fill(templates, pool, min_per_name, rng):
    """Expand templates; slotted templates are repeated until every name is used enough times."""
    out = []
    slotted = [t for t in templates if SLOT_RE.search(t)]
    for t in templates:
        m = SLOT_RE.search(t)
        if not m:
            out.append(t)
            continue
        kind = m.group(2)
        items = pool[kind]
        if not items:
            raise ValueError(f"No names for slot '{kind}' (template: {t})")
        # distinct names are drawn round-robin from a shuffled list, so coverage is even
        reps = max(1, math.ceil(min_per_name * len(items) / len(slotted)))
        order = list(items)
        rng.shuffle(order)
        for i in range(reps):
            name = order[i % len(order)]
            out.append(SLOT_RE.sub(lambda mm: f"[{name}]({mm.group(2)})", t))
    return list(dict.fromkeys(out))  # dedupe, keep order


def balance(sections, ratio):
    """Repeat examples of small intents until they reach ``ratio`` x the median intent size."""
    sizes = sorted(len(v) for v in sections.values())
    target = int(ratio * sizes[len(sizes) // 2])
    for intent, examples in sections.items():
        base = list(examples)
        while len(examples) < target:
            examples.append(base[len(examples) % len(base)])
    return sections


def write_yml(path, sections):
    with open(path, "w", encoding="utf8") as f:
        f.write('version: "3.1"\n\nnlu:\n\n')
        for intent, examples in sections.items():
            f.write(f"- intent: {intent}\n  examples: |\n")
            for e in examples:
                f.write(f"    - {e}\n")
            f.write("\n")


def generate(lang, names, intents, args):
    with open(os.path.join(DATA_DIR, "templates", f"{lang}.yml"), "r", encoding="utf8") as f:
        tpl = yaml.safe_load(f)
    pool = name_pool(names, lang, args.ha_weight)
    valid_intents = [i for kind in intents.values() for i in kind]
    out_dir = os.path.join(DATA_DIR, lang)
    os.makedirs(out_dir, exist_ok=True)

    for section, fname, per_name, seed in (("nlu", "data.yml", args.min_per_name, args.seed),
                                           ("validation", "validation.yml", args.min_per_name_val, args.seed + 1)):
        rng = random.Random(f"{lang}-{seed}")
        sections = {}
        for item in tpl[section]:
            if item["intent"] not in valid_intents:
                raise ValueError(f"{lang}: intent '{item['intent']}' is not in data/intents.json")
            lines = [l.strip().lstrip("-").strip() for l in item["examples"].strip().split("\n")]
            sections[item["intent"]] = fill(lines, pool, per_name, rng)
        missing = set(valid_intents) - set(sections)
        if missing:
            raise ValueError(f"{lang}: no '{section}' templates for intents {sorted(missing)}")
        if section == "nlu" and args.balance > 0:
            sections = balance(sections, args.balance)
        write_yml(os.path.join(out_dir, fname), sections)
        print(f"  {lang}/{fname}: {sum(len(v) for v in sections.values())} examples")

    save_json(os.path.join(out_dir, "intent_labels.json"), valid_intents)
    entity_labels = ["EOS", "BOS", "PAD", "O"] + [f"{p}-{k}" for k in KINDS for p in ("B", "I")]
    save_json(os.path.join(out_dir, "entity_labels.json"), entity_labels)

    if args.export:
        # Runtime lists: only the entities that exist in Home Assistant
        valid = {k: [n for e in names.get(k, {}).values() if e.get("ha") for n in e["names"].get(lang, [])]
                 for k in KINDS}
        save_json(os.path.join(args.export, "valid_entities", f"{lang}.json"), valid)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--lang", nargs="+", help="Languages to generate (default: all in data/templates)")
    parser.add_argument("--names", default=os.path.join(DATA_DIR, "names.json"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-per-name", type=int, default=3, help="Min. training examples per name")
    parser.add_argument("--min-per-name-val", type=int, default=1, help="Min. validation examples per name")
    parser.add_argument("--ha-weight", type=int, default=2, help="Weight of names that exist in Home Assistant")
    parser.add_argument("--balance", type=float, default=0.5,
                        help="Repeat examples of small training intents up to this fraction of the median size (0 = off)")
    parser.add_argument("--export", help="Dir (e.g. homelai-rpi-shs/config) to copy names, intents and valid_entities to")
    args = parser.parse_args()

    names = load_json(args.names)
    intents = load_json(os.path.join(DATA_DIR, "intents.json"))
    langs = args.lang or available_languages()
    for lang in langs:
        print(f"[{lang}]")
        generate(lang, names, intents, args)
    if args.export:
        os.makedirs(args.export, exist_ok=True)
        shutil.copy(args.names, os.path.join(args.export, "names.json"))
        shutil.copy(os.path.join(DATA_DIR, "intents.json"), os.path.join(args.export, "valid_intents.json"))
        print(f"Exported names.json, valid_intents.json and valid_entities/ to {args.export}")


if __name__ == "__main__":
    main()
