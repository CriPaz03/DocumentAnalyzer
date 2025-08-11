from embedding import VectorIndex


def ask(query: str, top_k: int = 5):
    v = VectorIndex()
    v.load("index.faiss", "index_meta.json")
    hits = v.search(query, k=top_k)
    return hits

if __name__ == '__main__':
    print(ask("difese dei castelli in puglia nel XII secolo", 3))