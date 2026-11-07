"""Cosine similarity of two paraphrases vs an unrelated sentence: python -m evaluation.embedding_diagnostic"""
import itertools
import json
import math
from pathlib import Path

from src.retrieval.embeddings import embedding_info, get_embeddings

S = ["My purifier keeps beeping three times.", "The unit gives three short audible alarms repeatedly.", "Delivery to Karachi takes one to two working days."]


def cos(a, b):
    return sum(x * y for x, y in zip(a, b)) / (math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b)))


if __name__ == "__main__":
    v = get_embeddings().embed_documents(S)
    out = {"embedding": embedding_info(), "pairs": {f"{i}-{j}": round(cos(v[i], v[j]), 3) for i, j in itertools.combinations(range(3), 2)}, "sentences": S}
    Path(__file__).parent.joinpath("results/embedding_diagnostic.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
