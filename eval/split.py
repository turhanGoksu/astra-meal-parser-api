"""Grouped dev/test split of the labeled eval names (Design Y).

Leakage rule (Project 2B): anything we learn from dev errors (thresholds,
new aliases) must not touch test items about the same food. So:
- in-table names are grouped by connected gold foods ("Eggs" -> egg and
  "scrambled eggs" -> egg|egg_fried share a group);
- "none" names are grouped by their first folded word ("mercimek corbasi"
  and "mercimek corbisi" stay together).
Groups (not names) are shuffled with a fixed seed and assigned until each
side reaches its share, separately for in-table and none groups.

Usage (from the project root):
    python -m eval.split
"""

import csv
import random
from collections import defaultdict
from pathlib import Path

from astra_nutrition.text import fold

LABELS_PATH = Path("data/eval/labels.csv")
PARSED_PATH = Path("data/eval/parsed_items.csv")
SPLIT_PATH = Path("data/eval/split.csv")
DEV_SHARE = 0.6
SEED = 42


def group_keys(labels: list[dict[str, str]]) -> dict[str, str]:
    """Map each folded name to a group key (union-find over gold food ids)."""
    parent: dict[str, str] = {}

    def find(x: str) -> str:
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for label in labels:
        foods = [g for g in label["gold"].split("|") if g != "none"]
        for food in foods[1:]:
            parent[find(food)] = find(foods[0])

    keys = {}
    for label in labels:
        name = fold(label["name"])
        if label["gold"] == "none":
            keys[name] = "none:" + name.split()[0]
        else:
            keys[name] = "food:" + find(label["gold"].split("|")[0])
    return keys


def assign(groups: dict[str, list[str]], rng: random.Random) -> dict[str, str]:
    """Assign whole groups to dev until dev holds DEV_SHARE of the names."""
    order = sorted(groups)
    rng.shuffle(order)
    total = sum(len(names) for names in groups.values())
    dev_count, split = 0, {}
    for key in order:
        side = "dev" if dev_count < DEV_SHARE * total else "test"
        dev_count += len(groups[key]) if side == "dev" else 0
        split[key] = side
    return split


def main() -> None:
    with open(LABELS_PATH, encoding="utf-8") as f:
        labels = list(csv.DictReader(f))
    with open(PARSED_PATH, encoding="utf-8") as f:
        sets: dict[str, set[str]] = defaultdict(set)
        for row in csv.DictReader(f):
            sets[fold(row["name"])].add(row["set"])

    keys = group_keys(labels)
    rng = random.Random(SEED)
    split: dict[str, str] = {}
    for kind in ("food:", "none:"):
        groups: dict[str, list[str]] = defaultdict(list)
        for name, key in keys.items():
            if key.startswith(kind):
                groups[key].append(name)
        split.update(assign(groups, rng))

    rows = [
        {
            "name": label["name"],
            "gold": label["gold"],
            "tags": label["tags"],
            "sets": ";".join(sorted(sets[fold(label["name"])])),
            "group": keys[fold(label["name"])],
            "split": split[keys[fold(label["name"])]],
        }
        for label in labels
    ]
    with open(SPLIT_PATH, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    for side in ("dev", "test"):
        part = [r for r in rows if r["split"] == side]
        in_table = sum(r["gold"] != "none" for r in part)
        hard = sum("hard_negative" in r["tags"] for r in part)
        groups_n = len({r["group"] for r in part})
        print(
            f"{side:4}: {len(part):3} names in {groups_n:3} groups | "
            f"in-table {in_table:3} | none {len(part) - in_table:3} | "
            f"hard negatives {hard}"
        )


if __name__ == "__main__":
    main()
