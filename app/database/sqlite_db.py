from __future__ import annotations
import numpy as np
import sqlite3
from pathlib import Path


class SQLiteDatabase:
    """Handles all database interactions for ARGUS."""

    def __init__(self, db_path: str = "data/faces.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        self.connection = sqlite3.connect(self.db_path)
        self.connection.execute("PRAGMA foreign_keys = ON")

        self.create_tables()

    def get_or_create_person(self, name: str) -> int:
    cursor = self.connection.cursor()

    cursor.execute(
        "SELECT id FROM people WHERE name = ?",
        (name,),
    )

    row = cursor.fetchone()

    if row:
        return row[0]

    cursor.execute(
        "INSERT INTO people (name) VALUES (?)",
        (name,),
    )

    self.connection.commit()

    return cursor.lastrowid

    def add_embedding(
    self,
    person_id: int,
    image_name: str,
    embedding: np.ndarray,
) -> None:

    embedding = np.asarray(
        embedding,
        dtype=np.float32,
    )

    blob = embedding.tobytes()

    self.connection.execute(
        """
        INSERT INTO embeddings
        (person_id, image_name, embedding)
        VALUES (?, ?, ?)
        """,
        (
            person_id,
            image_name,
            blob,
        ),
    )

    self.connection.commit()

    def get_all_embeddings(self):

    cursor = self.connection.cursor()

    cursor.execute(
        """
        SELECT
            embeddings.id,
            people.name,
            embeddings.image_name,
            embeddings.embedding
        FROM embeddings
        JOIN people
        ON embeddings.person_id = people.id
        """
    )

    rows = cursor.fetchall()

    results = []

    for embedding_id, name, image_name, blob in rows:

        embedding = np.frombuffer(
            blob,
            dtype=np.float32,
        )

        results.append(
            {
                "id": embedding_id,
                "name": name,
                "image_name": image_name,
                "embedding": embedding,
            }
        )

    return results

    def create_tables(self):
        cursor = self.connection.cursor()

        cursor.execute("""
        CREATE TABLE IF NOT EXISTS people(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT UNIQUE NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """)

        cursor.execute("""
        CREATE TABLE IF NOT EXISTS embeddings(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            person_id INTEGER NOT NULL,
            image_name TEXT,
            embedding BLOB NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

            FOREIGN KEY(person_id)
            REFERENCES people(id)
            ON DELETE CASCADE
        )
        """)

        self.connection.commit()