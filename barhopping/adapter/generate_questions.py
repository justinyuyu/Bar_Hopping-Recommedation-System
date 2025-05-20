import json
import os
import sqlite3
from tqdm import tqdm
from openai import OpenAI
from barhopping.config import BARS_DB, QUERIES_DB
from barhopping.logger import logger

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def generate_bar_questions(summary: str, num_questions: int = 20) -> list[str]:
    """Call the OpenAI chat API to create *num_questions* user queries that
    could lead to this bar summary.
    Returns a list of strings. Raises on failure.
    """
    prompt = f"""
    You are an AI assistant tasked with generating a set of realistic, natural‑sounding user queries that could lead to a review‑based summary of a specific type or category of bar in a bar‑hopping recommendation system.

    Given Review Summary:
    """
    {summary}
    """

    Instructions:
    1. Focus on the user’s intent to discover a **vibe** or **specialties** of this bar (e.g., rare whisky collection, signature smoked cocktails, vinyl‑only jazz sets).
    2. Generate **exactly {num_questions}** realistic user search queries that ask for bars sharing its vibe and specialties.
    3. Use casual, conversational language; you may include typos or slang to reflect real search behavior.
    4. All queries must be semantically equivalent (same core intent) without copying phrases verbatim.

    Output Format:
    Return **only** a JSON array of strings (no keys, no extra text):
    """

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful assistant creating bar recommendation questions."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7,
        max_tokens=num_questions * 25,
    )
    return json.loads(response.choices[0].message.content)

def _ensure_table(cur):
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS bar_questions (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            bar_id        INTEGER NOT NULL,
            question_no   INTEGER NOT NULL,
            question_text TEXT    NOT NULL,
            embedding     TEXT,
            FOREIGN KEY(bar_id) REFERENCES bars(id)
        );
        """
    )
    cur.execute("CREATE INDEX IF NOT EXISTS idx_bq_bar_id ON bar_questions(bar_id);")

def save_questions_for_bar(bar_id: int, questions: list[str]):
    """Insert/replace the generated questions for *bar_id* into the DB."""
    with sqlite3.connect(QUERIES_DB) as conn:
        cur = conn.cursor()
        _ensure_table(cur)
        cur.execute("DELETE FROM bar_questions WHERE bar_id = ?", (bar_id,))
        cur.executemany(
            "INSERT INTO bar_questions (bar_id, question_no, question_text) VALUES (?, ?, ?)",
            [(bar_id, i + 1, q) for i, q in enumerate(questions)]
        )
        conn.commit()

def process_first_n(n: int = 100, num_questions: int = 10):
    with sqlite3.connect(BARS_DB) as conn:
        bars = conn.execute("SELECT id, summary FROM bars ORDER BY id LIMIT ?", (n,)).fetchall()

    failures = []
    for bar_id, summary in tqdm(bars, desc="Generating queries"):
        try:
            qs = generate_bar_questions(summary, num_questions=num_questions)
            if not qs:
                raise ValueError("Empty response from OpenAI")
            save_questions_for_bar(bar_id, qs)
        except Exception as exc:
            logger.error("Bar %s failed: %s", bar_id, exc)
            failures.append(bar_id)

    logger.info("Processed %d bars | failures: %s", n, failures if failures else "none")

if __name__ == "__main__":
    process_first_n(108, num_questions=10)