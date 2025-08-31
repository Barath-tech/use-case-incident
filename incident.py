import pandas as pd
from sentence_transformers import SentenceTransformer
from llm import gpt_summarize, gpt_category_name
from memory import load_memory, save_memory, add_category
from categorization import embed_texts, cluster_and_create_categories, match_to_categories

# ================================
# CONFIG
# ================================
EXCEL_PATH = "Incident_Data.xlsx"
MEMORY_PATH = "incident_agent_categories.json"
OUTPUT_PATH = "incidents_categorized.xlsx"

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
SIMILARITY_THRESHOLD = 0.72
N_INIT_CLUSTERS = 8


def load_excel(path):
    print(f"📥 Loading Excel data from {path}")
    df = pd.read_excel(path,header=1)
    text_col = None
    for c in df.columns:
        if c.lower() in ["summary*","notes", "description", "details", "title"]:
            text_col = c
            break
    if not text_col:
        text_col = df.select_dtypes(include="object").columns[0]
    print(f"✅ Using '{text_col}' column as text source")
    return df, text_col


def main():
    print("🚀 Incident Categorization Pipeline Started")

    # Load
    df, text_col = load_excel(EXCEL_PATH)
    df["_text"] = df[text_col].astype(str)

    df=df.head(20)

    # Embeddings
    print("🔄 Loading embedding model...")
    st_model = SentenceTransformer(EMBED_MODEL)
    inc_embs = embed_texts(df["_text"].tolist(), st_model)

    # Memory
    mem = load_memory(MEMORY_PATH)

    # Bootstrap categories if empty
    if not mem["categories"]:
        print("📊 Bootstrapping categories with clustering...")
        df = cluster_and_create_categories(df, inc_embs, st_model, mem, N_INIT_CLUSTERS)
        save_memory( MEMORY_PATH, mem)

    # Classification
    results = []
    print("🔄 Classifying incidents...")
    for i, row in df.iterrows():
        emb = inc_embs[i]
        cat_name, score = match_to_categories(emb, mem)

        if cat_name and score >= SIMILARITY_THRESHOLD:
            assigned = cat_name
            print(f"✅ Assigned existing category: {assigned}")
        else:
            cname = gpt_category_name(row["_text"])
            add_category(mem, st_model, cname, row["_text"])
            save_memory(MEMORY_PATH,mem)
            assigned, score = cname, 1.0
            print(f"🆕 Created new category: {assigned}")

        summary = gpt_summarize(row["_text"], max_tokens=40)

        results.append({
            **row.to_dict(),
            "auto_category": assigned,
            "auto_summary": summary
            # "similarity_score": round(float(score), 3)
        })

    out_df = pd.DataFrame(results)
    out_df.to_excel(OUTPUT_PATH, index=False)
    print(f"🎉 Done! Categorized incidents saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
