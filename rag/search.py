# run_all_retrievals.py
import os, ast, json
import numpy as np
import pandas as pd
from pathlib import Path

# query embedding backends
from sentence_transformers import SentenceTransformer
from openai import OpenAI

# -----------------------------
# Utils
# -----------------------------
def parse_vec(x):
    """CSV stores a Python list as a string; parse to np.array"""
    return np.array(ast.literal_eval(x), dtype=np.float32)

def l2norm(v):
    n = np.linalg.norm(v, axis=-1, keepdims=True) + 1e-12
    return v / n

def cos_sim_matrix(q, M):
    # q: (d,), M: (N, d) -> (N,)
    return M @ q

def load_df(path, expect_cluster=False):
    df = pd.read_csv(path)
    if "embedding" not in df.columns:
        raise FileNotFoundError(f"'embedding' column missing in {path}")

    # parse embeddings and normalize
    df["embedding"] = df["embedding"].apply(parse_vec)
    E = np.stack(df["embedding"].to_list())
    E = l2norm(E)
    df["embedding_norm"] = list(E)

    if expect_cluster:
        # clusters store a list of 3 rep paragraphs in 'text' as a string
        def parse_text_list(t):
            try:
                v = ast.literal_eval(str(t))
                if isinstance(v, list):
                    return v
            except Exception:
                pass
            return [str(t)]
        df["text_list"] = df["text"].apply(parse_text_list) if "text" in df.columns else [[]]*len(df)
    return df

# -----------------------------
# Query embedding functions
# -----------------------------
_free_model = None
def embed_query_free(q: str) -> np.ndarray:
    """SentenceTransformer embedding (normalized)"""
    global _free_model
    if _free_model is None:
        _free_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    v = _free_model.encode([q], normalize_embeddings=True)[0]
    return v.astype(np.float32)

def embed_query_openai(q: str) -> np.ndarray:
    """OpenAI embedding (normalized so cosine is dot)"""
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set (env var or config.json).")
    client = OpenAI(api_key=api_key)
    r = client.embeddings.create(model="text-embedding-3-small", input=q)
    v = np.array(r.data[0].embedding, dtype=np.float32)
    return l2norm(v)

def embed_query(q: str, kind: str) -> np.ndarray:
    return embed_query_free(q) if kind == "free" else embed_query_openai(q)

def retrieve_full(df_full: pd.DataFrame, qvec: np.ndarray, topk=3) -> pd.DataFrame:
    M = np.stack(df_full["embedding_norm"].to_list())
    sims = cos_sim_matrix(qvec, M)
    idx = np.argsort(-sims)[:topk]
    out = df_full.iloc[idx].copy()
    out["score"] = sims[idx]
    # nice short snippet if available
    if "text" in out.columns:
        out["snippet"] = out["text"].astype(str).str.replace(r"\s+", " ", regex=True)
    else:
        out["snippet"] = ""
    return out

def retrieve_paragraph(df_para: pd.DataFrame, qvec: np.ndarray, topk_talks=3) -> pd.DataFrame:
    M = np.stack(df_para["embedding_norm"].to_list())
    sims = cos_sim_matrix(qvec, M)
    order = np.argsort(-sims)
    seen_urls, rows = set(), []
    for i in order:
        url = df_para.iloc[i].get("url", "")
        if url in seen_urls:
            continue
        r = df_para.iloc[i]
        rows.append({
            **r.to_dict(),
            "score": float(sims[i]),
            "snippet": str(r.get("text", ""))
        })
        seen_urls.add(url)
        if len(rows) == topk_talks:
            break
    return pd.DataFrame(rows)

def retrieve_cluster(df_cluster: pd.DataFrame, qvec: np.ndarray, topk=3) -> pd.DataFrame:
    M = np.stack(df_cluster["embedding_norm"].to_list())
    sims = cos_sim_matrix(qvec, M)
    order = np.argsort(-sims)
    seen, rows = set(), []
    for i in order:
        url = df_cluster.iloc[i].get("url", "")
        if url in seen:
            continue
        r = df_cluster.iloc[i].to_dict()
        r["score"] = float(sims[i])
        reps = r.get("text_list") or []
        r["snippet"] = " | ".join(str(p) for p in reps[:3])
        rows.append(r)
        seen.add(url)
        if len(rows) == topk:
            break
    return pd.DataFrame(rows)

def main():
    questions = [
        "How can I gain a testimony of Jesus Christ?",
        "What are some ways to deal with challenges in life and find a purpose?",
        "How can I fix my car if it won't start?",
        "What is the significance of the teachings of the Book of Mormon?",
        "How can I better balance my faith and my work life?"
    ]
        
    paths = [
        "free/free_talks.csv",
        "free/free_paragraphs.csv",
        "free/free_3_clusters.csv",
        "openai/openai_talks.csv",
        "openai/openai_paragraphs.csv",
        "openai/openai_3_clusters.csv",
    ]

    base_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(base_dir, "config.json")
    
    if os.path.exists(config_path):
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                k = json.load(f).get("openaiKey")
                if k:
                    os.environ["OPENAI_API_KEY"] = k
        except Exception:
            return 1
    
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)
    full_results_path = os.path.join(base_dir, results_dir)

    combined_rows = []

    combos = [
        ("free", "full", paths[0], False),
        ("free", "paragraph", paths[1], False),
        ("free", "cluster",  paths[2], True),
        ("openai", "full", paths[3], False),
        ("openai", "paragraph", paths[4], False),
        ("openai", "cluster", paths[5], True),
    ]

    dataframes = {}
    for kind, mode, path, is_cluster in combos:
        full_path = os.path.join(base_dir, path)

        if not os.path.exists(full_path):
            print(f"[WARN] Missing file for {kind} {mode}: {full_path}")
            dataframes[(kind, mode)] = None
            continue
        
        try:
            df = load_df(full_path, expect_cluster=is_cluster)
            dataframes[(kind, mode)] = df
        except Exception as e:
            print(f"[ERROR] Failed to load {kind} {mode} from {full_path}: {e}")
            dataframes[(kind, mode)] = None

    for question in questions:
        for kind in ("free", "openai"):
            try:
                qvec = embed_query(question, kind)
            except Exception as e:
                print(f"[ERROR] Embedding query with {kind}: {e}")
                continue

            for mode in ("full", "paragraph", "cluster"):
                df = dataframes.get((kind, mode))
                if df is None or df.empty:
                    continue
                if mode == "full":
                    hits = retrieve_full(df, qvec, topk=3)
                elif mode == "paragraph":
                    hits = retrieve_paragraph(df, qvec, topk_talks=3)
                else:
                    hits = retrieve_cluster(df, qvec, topk=3)

                # standardize columns
                for _, r in hits.iterrows():
                    combined_rows.append({
                        "question": question,
                        "model": kind,
                        "mode": mode,
                        "title": r.get("title",""),
                        "speaker": r.get("speaker",""),
                        "calling": r.get("calling",""),
                        "season": r.get("season",""),
                        "year": r.get("year",""),
                        "url": r.get("url",""),
                        "score": float(r.get("score", 0.0)),
                        "snippet": r.get("snippet",""),
                    })

                # also write per-combo file
                out_csv = os.path.join(full_results_path, f"{kind}_{mode}_top3.csv")
                small = pd.DataFrame([row for row in combined_rows if row["model"]==kind and row["mode"]==mode and row["question"]==question])
                if not small.empty:
                    write_header = not os.path.exists(out_csv)
                    file_mode = "w" if write_header else "a"
                    small.to_csv(out_csv, index=False, header=write_header, mode=file_mode)

        print(f"\n=== {question} ===")
        for kind in ("free","openai"):
            for mode in ("full","paragraph","cluster"):
                subset = [r for r in combined_rows if r["question"]==question and r["model"]==kind and r["mode"]==mode]
                if subset:
                    titles = "; ".join([f"{r['title']} ({r['season']} {r['year']})" for r in subset])
                    print(f"{kind}/{mode}: {titles}")

    if combined_rows:
        all_df = pd.DataFrame(combined_rows)
        all_path = os.path.join(full_results_path, "all_results.csv")
        all_df.to_csv(all_path, index=False)
        print(f"\nWrote combined results to {all_path}")
    else:
        print("\nNo results written (check that your input CSVs exist).")
        
    return 0

if __name__ == "__main__":
    main()
