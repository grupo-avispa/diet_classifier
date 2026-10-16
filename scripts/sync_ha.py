"""Sync data/names.json with the zones, devices and modes defined in Home Assistant.

Home Assistant entities are expected to follow these conventions:

    switch.light_<id>      light of zone <id>       -> kind "sala"
    cover.<id>             blinds of zone <id>      -> kind "sala"
    switch.plug_<id>       plug of device <id>      -> kind "dispositivo"
    automation.mode_<id>   mode <id>                -> kind "modo"

For every entry in names.json, "ha" is set to true if it exists in Home Assistant (those names are
what the runtime accepts as valid entities and what training favours). Entities found in Home
Assistant that are not in the catalog are added with their friendly name for every language and
reported, so you can translate them by editing names.json. Existing translations are never changed.

Usage:
    HA_URL=http://homelai-pi.local:8123 HA_TOKEN=... python scripts/sync_ha.py
    python scripts/sync_ha.py --dry-run
    python scripts/sync_ha.py --offline-report          # only list what is not translated
"""

import argparse
import json
import os
import re
import unicodedata

import requests

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
NAMES_PATH = os.path.join(SCRIPT_DIR, "..", "data", "names.json")
LANGS = ["es", "en", "fr", "de", "it"]

# (domain, id prefix, kind)
RULES = [
    ("switch", "light_", "sala"),
    ("cover", "", "sala"),
    ("switch", "plug_", "dispositivo"),
    ("automation", "mode_", "modo"),
]


def discover(states):
    """Return {kind: {id: friendly_name}} for the entities of a Home Assistant states list."""
    found = {"sala": {}, "dispositivo": {}, "modo": {}}
    for st in states:
        entity_id = st.get("entity_id", "")
        domain, _, obj = entity_id.partition(".")
        for rule_domain, prefix, kind in RULES:
            if domain == rule_domain and obj.startswith(prefix) and len(obj) > len(prefix):
                key = obj[len(prefix):]
                friendly = (st.get("attributes", {}) or {}).get("friendly_name") or key.replace("_", " ")
                # a zone can have both a light and blinds: keep the first friendly name
                found[kind].setdefault(key, friendly)
    return found


def normalize(name: str) -> str:
    return unicodedata.normalize("NFC", name.strip().lower())


def fetch_states(url, token, timeout):
    response = requests.get(f"{url}/api/states",
                            headers={"Authorization": f"Bearer {token}"}, timeout=timeout)
    response.raise_for_status()
    return response.json()


def merge(names, found):
    """Update 'ha' flags in place; return the new entries as a list of (kind, id)."""
    added = []
    for kind, entries in found.items():
        catalog = names.setdefault(kind, {})
        for key, friendly in entries.items():
            if key not in catalog:
                catalog[key] = {"ha": True, "names": {l: [normalize(friendly)] for l in LANGS}}
                added.append((kind, key))
        for key, entry in catalog.items():
            exists = key in entries
            entry["ha"] = exists
    return added


def untranslated(names):
    """Entries that have the same names in every language (probably not translated yet)."""
    out = []
    for kind, entries in names.items():
        for key, entry in entries.items():
            if entry.get("ha") and len({json.dumps(v, ensure_ascii=False) for v in entry["names"].values()}) == 1:
                out.append((kind, key, entry["names"][LANGS[0]]))
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--names", default=NAMES_PATH)
    parser.add_argument("--url", default=os.getenv("HA_URL", "http://homelai-pi.local:8123"))
    parser.add_argument("--token", default=os.getenv("HA_TOKEN"))
    parser.add_argument("--timeout", type=int, default=int(os.getenv("HA_TIMEOUT", "10")))
    parser.add_argument("--dry-run", action="store_true", help="Show the changes without writing names.json")
    parser.add_argument("--offline-report", action="store_true", help="Do not contact HA; list untranslated entries")
    args = parser.parse_args()

    with open(args.names, "r", encoding="utf8") as f:
        names = json.load(f)

    if not args.offline_report:
        if not args.token:
            raise SystemExit("Set HA_TOKEN (or pass --token) to read the entities from Home Assistant.")
        found = discover(fetch_states(args.url, args.token, args.timeout))
        print("Found in Home Assistant: " + ", ".join(f"{len(v)} {k}" for k, v in found.items()))
        added = merge(names, found)
        for kind, key in added:
            print(f"  + new {kind} '{key}' (named '{names[kind][key]['names'][LANGS[0]][0]}' in every language)")
        flagged = {k: sum(1 for e in v.values() if e.get("ha")) for k, v in names.items()}
        print("Entries with ha=true: " + ", ".join(f"{n} {k}" for k, n in flagged.items()))
        if not args.dry_run:
            with open(args.names, "w", encoding="utf8") as f:
                json.dump(names, f, ensure_ascii=False, indent=2)
                f.write("\n")
            print(f"Updated {args.names}")

    todo = untranslated(names)
    if todo:
        print("\nTranslate these names in names.json (same name in every language):")
        for kind, key, n in todo:
            print(f"  {kind}/{key}: {n}")


if __name__ == "__main__":
    main()
