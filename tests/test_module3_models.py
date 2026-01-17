import numpy as np
from beie.module3.models import ClusteredPost, Cluster

def test_clustered_post_fields():
    cp = ClusteredPost(
        post_id="p1",
        cluster_id=2,
        embedding=np.array([1.0, 2.0]),
        clean_text="hello",
        metadata={"k": "v"},
    )
    assert cp.post_id == "p1"
    assert cp.cluster_id == 2
    assert cp.clean_text == "hello"
    assert np.allclose(cp.embedding, np.array([1.0, 2.0]))
    assert cp.metadata["k"] == "v"

def test_cluster_fields():
    c = Cluster(
        cluster_id=7,
        size=3,
        centroid=np.array([0.0, 1.0]),
        member_post_ids=["p1", "p2", "p3"],
    )
    assert c.cluster_id == 7
    assert c.size == 3
    assert c.member_post_ids == ["p1", "p2", "p3"]
    assert c.centroid.shape == (2,)
