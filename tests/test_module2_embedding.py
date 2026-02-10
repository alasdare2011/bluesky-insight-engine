import numpy as np
from beie.module2.embedding import SimpleHashEmbedder

def test_embed_returns_same_count_as_texts():
    e = SimpleHashEmbedder(dim=64)
    texts = ["hello", "world", "hello world"]
    vecs = e.embed(texts)
    assert len(vecs) == len(texts)

def test_embed_vectors_have_correct_dim():
    dim = 128
    e = SimpleHashEmbedder(dim=dim)
    vecs = e.embed(["a", "b"])
    assert all(isinstance(v, np.ndarray) for v in vecs)
    assert all(v.shape == (dim,) for v in vecs)

def test_embed_is_deterministic_for_same_text():
    e = SimpleHashEmbedder(dim=64)
    v1 = e.embed(["same text"])[0]
    v2 = e.embed(["same text"])[0]
    assert np.array_equal(v1, v2)

def test_embed_different_texts_produce_different_vectors():
    e = SimpleHashEmbedder(dim=64)
    v1 = e.embed(["text one"])[0]
    v2 = e.embed(["text two"])[0]
    assert not np.array_equal(v1, v2)

def test_embed_empty_input_returns_empty_list():
    e = SimpleHashEmbedder(dim=64)
    assert e.embed([]) == []

def test_embed_empty_string_returns_vector():
    e = SimpleHashEmbedder(dim=64)
    v = e.embed([""])[0]
    assert v.shape == (64,)
    assert isinstance(v, np.ndarray)
    assert v.dtype == np.float32
