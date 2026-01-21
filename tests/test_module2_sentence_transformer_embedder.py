import numpy as np

from beie.module2.embedding import SentenceTransformerEmbedder


def test_sentence_transformer_embedder_basic():
    embedder = SentenceTransformerEmbedder()

    texts = [
        "I love programming.",
        "Programming is great.",
        "The stock market fell today.",
    ]

    vectors = embedder.embed(texts)

    assert len(vectors) == 3
    assert all(isinstance(v, np.ndarray) for v in vectors)

    dim = vectors[0].shape[0]
    assert dim == 384

    # Similar texts should be closer than unrelated ones
    sim_01 = np.dot(vectors[0], vectors[1])
    sim_02 = np.dot(vectors[0], vectors[2])

    assert sim_01 > sim_02
