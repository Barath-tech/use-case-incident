# categorization.py
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from llm import gpt_category_name  # your narrowed prompt function
from memory import add_category
import time

def embed_texts(texts, model):
    """Return embeddings array (numpy) or list of vectors."""
    print("🧠 Generating embeddings...")
    embs = model.encode(texts, batch_size=32, show_progress_bar=True)
    return np.array(embs)

def cluster_and_create_categories(df, embeddings, st_model, mem, n_clusters=8):
    """
    Cluster incidents, obtain a GPT short name for each cluster, and save into memory.
    RETURNS: (df, mem) where mem is the updated memory dict.
    """
    print(f"📊 Clustering into {n_clusters} categories...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(embeddings)
    df["_cluster"] = labels

    for cl in sorted(set(labels)):
        members = df[df["_cluster"] == cl]
        sample_text = members.iloc[0]["_text"]
        print(f"🔸 Creating initial category for cluster {cl}")

        time.sleep(2)  # avoid rate-limit

        # 1) Get GPT-proposed name and sanitize it
        raw_name = gpt_category_name(sample_text)
        if not raw_name:
            name = f"Uncategorized_{cl}"
        else:
            # strip extra tokens, remove "Category:" if present, and take first line
            name = extract_final_category_name(raw_name)
            if name == "":
                name = f"Uncategorized_{cl}"

        print(f"🔹 GPT name (sanitized): {name}")

        # 2) Compute embedding for representative text (as list)
        sample_emb = st_model.encode(sample_text)
        sample_emb_list = sample_emb.tolist() if hasattr(sample_emb, "tolist") else list(sample_emb)

        # 3) Add to memory (pass embedding list)
        mem = add_category(mem, name, sample_text, sample_emb_list)

    return df

def extract_final_category_name(raw_name):
    """
    Extract the final category name from the GPT response.
    Assumes the final category name is the last word or phrase in the response.
    """
    # Split the response into lines and find the last valid line
    lines = raw_name.splitlines()
    for line in reversed(lines):  # Process lines from bottom to top
        line = line.strip()
        if line:  # Ignore empty lines
            # Check if the line ends with "assistantfinal<Category>"
            if "assistantfinal" in line:
                return line.split("assistantfinal")[-1].strip()
            return line  # Return the last valid line if no "assistantfinal"
    return None  # Return None if no valid category name is found

def match_to_categories(emb, mem):
    emb_arr = np.array(emb, dtype=float)  # ensure 1D float array

    cats = mem.get("categories", {})
    if not cats:
        return None, None

    # collect embeddings + names
    mem_names, mem_embs = [], []
    for cname, cdata in cats.items():
        if "embedding" in cdata:
            cemb = np.array(cdata["embedding"], dtype=float)
            if len(cemb) == len(emb_arr):
                mem_names.append(cname)
                mem_embs.append(cemb)

    if not mem_embs:
        return None, None

    sims = cosine_similarity([emb_arr], mem_embs)[0]
    best_idx = np.argmax(sims)

    print("Embedding shape:", emb_arr.shape)
    print("Memory shapes:", [len(e) for e in mem_embs])

    return mem_names[best_idx], sims[best_idx]

