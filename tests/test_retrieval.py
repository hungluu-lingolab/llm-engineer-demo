"""Test Buổi 4 — Embeddings (OpenAI) & Vector Store (Qdrant).

Mock OpenAI client + Qdrant client để test chạy nhanh, không gọi API / không cần server.
Verify wiring: embed gọi đúng model, search map payload → SearchHit đúng.
"""

from __future__ import annotations

from app.retrieval import embeddings, vectorstore


def test_embed_query_calls_openai(monkeypatch):
    """embed_query gọi OpenAI embeddings API với đúng model, trả vector phẳng."""
    captured = {}

    class FakeEmbeddings:
        def create(self, model, input):
            captured["model"] = model
            captured["input"] = input

            class R:
                data = [type("E", (), {"embedding": [0.1, 0.2, 0.3]})()]

            return R()

    class FakeClient:
        embeddings = FakeEmbeddings()

    monkeypatch.setattr(embeddings, "_get_client", lambda: FakeClient())
    vec = embeddings.embed_query("Công ty TNHH là gì?")

    assert captured["input"] == ["Công ty TNHH là gì?"]   # query bọc thành list
    assert captured["model"] == "text-embedding-3-small"
    assert vec == [0.1, 0.2, 0.3]                          # trả vector phẳng, không lồng


def test_embed_passages_batches(monkeypatch):
    """embed_passages gửi cả batch trong một lời gọi."""
    captured = {}

    class FakeEmbeddings:
        def create(self, model, input):
            captured["input"] = input

            class R:
                data = [
                    type("E", (), {"embedding": [0.1]})(),
                    type("E", (), {"embedding": [0.2]})(),
                ]

            return R()

    class FakeClient:
        embeddings = FakeEmbeddings()

    monkeypatch.setattr(embeddings, "_get_client", lambda: FakeClient())
    out = embeddings.embed_passages(["đoạn 1", "đoạn 2"])

    assert captured["input"] == ["đoạn 1", "đoạn 2"]
    assert out == [[0.1], [0.2]]


def test_search_maps_qdrant_hits(monkeypatch):
    """search map Qdrant hit (payload + score) → SearchHit, tách text khỏi metadata."""

    class Hit:
        def __init__(self, text, dieu, score):
            self.payload = {"text": text, "dieu": dieu, "source": "luat-dn-2020"}
            self.score = score

    class FakeResponse:
        def __init__(self, points):
            self.points = points

    class FakeClient:
        def query_points(self, collection_name, query, query_filter, limit):
            return FakeResponse(
                [Hit("Điều 46 ...", "46", 0.92), Hit("Điều 111 ...", "111", 0.75)]
            )

    monkeypatch.setattr(vectorstore, "_get_client", lambda: FakeClient())
    hits = vectorstore.search([0.1, 0.2, 0.3], top_k=2)

    assert len(hits) == 2
    assert hits[0].score == 0.92                     # Qdrant cosine = similarity trực tiếp
    assert hits[0].text == "Điều 46 ..."
    assert hits[0].metadata["dieu"] == "46"
    assert "text" not in hits[0].metadata            # text đã tách khỏi metadata
