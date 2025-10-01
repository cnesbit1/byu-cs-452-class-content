# rag_gen.py
import json
import os, pandas as pd
from openai import OpenAI

MODEL = "gpt-4o"

def load_content(csv_path, question):
    df = pd.read_csv(csv_path)
    hits = df[df["question"] == question].sort_values("score", ascending=False).head(3)
    context = []
    for _, row in hits.iterrows():
        context.append(f"Title: {row['title']} ({row['season']} {row['year']}) — {row['speaker']}\nURL: {row['url']}\nExcerpt: {row['snippet']}")
    return "\n\n---\n\n".join(context)

def ask_gpt(question, context):
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    system = (
        "You are a helpful assistant. Answer ONLY using the provided excerpts from Latter-day Saint General "
        "Conference talks. If the excerpts are insufficient, say you don't know. Do not use outside sources."
    )
    user = f"Question: {question}\n\nUse these excerpts:\n\n{context}"
    r = client.chat.completions.create(
        model=MODEL,
        messages=[{"role":"system","content":system},{"role":"user","content":user}],
        temperature=0.2,
    )
    return r.choices[0].message.content.strip()

def main():
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

    
    csv_path = "results/openai_full_top3.csv"
    questions = [
        "How can I gain a testimony of Jesus Christ?",
        "What are some ways to deal with challenges in life and find a purpose?",
        "How can I fix my car if it won't start?",
        "What is the significance of the teachings of the Book of Mormon?",
        "How can I better balance my faith and my work life?"
    ]

    for question in questions:
        print(f"\n=== {question} ===")
        ctx = load_content(csv_path, question)
        response = ask_gpt(question, ctx)
        print(response)

if __name__ == "__main__":
    main()
