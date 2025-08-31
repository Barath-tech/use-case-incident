# memory.py
import json
import os
import numpy as np
from typing import List, Union

def load_memory(path: str):
    print(f"[load_memory] Loading memory from: {path}")
    if not os.path.exists(path):
        print("[load_memory] Not found. Initializing empty memory.")
        return {"categories": {}}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            if not isinstance(data, dict) or "categories" not in data:
                print("[load_memory] Schema mismatch. Reinitializing memory.")
                return {"categories": {}}
            return data
    except Exception:
        print("[load_memory] Corrupt JSON. Reset to empty.")
        return {"categories": {}}


def save_memory(path: str, memory: dict):
    """
    Save memory to file. Ensure embeddings are JSON-serializable lists.
    """
    print(f"[save_memory] Saving memory to: {path}")
    # Convert any numpy arrays to lists
    for k, v in memory.get("categories", {}).items():
        emb = v.get("embedding")
        if hasattr(emb, "tolist"):
            memory["categories"][k]["embedding"] = emb.tolist()
    with open(path, "w", encoding="utf-8") as f:
        json.dump(memory, f, indent=2, ensure_ascii=False)
    print("[save_memory] Memory saved.")


def add_category(memory: dict, name: str, example: Union[str, List[str]], embedding):
    """
    Add or update a category.
    - name: MUST be the GPT-returned string (sanitized).
    - example: a text string or list of texts (examples/incident ids).
    - embedding: list or numpy array (vector) representing the example(s).
    Returns memory (modified).
    """
    # sanitize name
    if not name or not isinstance(name, str) or name.strip() == "":
        name = "Uncategorized"
    name = name.strip()

    if isinstance(example, str):
        examples = [example]
    else:
        examples = list(example)

    # ensure embedding is a list (or numpy array)
    if hasattr(embedding, "tolist"):
        emb_list = embedding.tolist()
    else:
        emb_list = embedding

    cats = memory.setdefault("categories", {})

    if name not in cats:
        cats[name] = {
            "examples": examples,
            "embedding": emb_list
        }
        print(f"[add_category] Created category '{name}' with {len(examples)} example(s).")
    else:
        # append new examples without duplicates
        existing = cats[name]["examples"]
        for ex in examples:
            if ex not in existing:
                existing.append(ex)
        # Recompute embedding as mean of stored embedding and new embedding
        try:
            cur_emb = np.array(cats[name]["embedding"], dtype=float)
            new_emb = np.array(emb_list, dtype=float)
            # average the two embeddings (simple running avg)
            merged = ((cur_emb * len(existing)) + new_emb) / (len(existing) + 1)
            cats[name]["embedding"] = merged.tolist()
        except Exception:
            # fallback: overwrite if numeric ops fail
            cats[name]["embedding"] = emb_list
        print(f"[add_category] Updated category '{name}'; total examples: {len(existing)}")

    return memory
