import sqlite3
from barhopping.config import BARS_DB

def init_bars():
    query = """
        CREATE TABLE IF NOT EXISTS bars (
            id INTEGER PRIMARY KEY,
            name TEXT,
            url TEXT,
            city TEXT,
            address TEXT,
            rating TEXT,
            photo TEXT,
            summary TEXT,
            embedding TEXT
        );
    """
    with sqlite3.connect(BARS_DB) as conn:
        conn.execute(query)
        conn.commit()

def insert_bar(bar: dict):
    columns = ", ".join(bar.keys())
    placeholders = ", ".join("?" for _ in bar)
    values = list(bar.values())

    query = f"INSERT INTO bars ({columns}) VALUES ({placeholders})"

    with sqlite3.connect(BARS_DB) as conn:
        conn.execute(query, values)
        conn.commit()