
from sentence_transformers import CrossEncoder
from gliner import GLiNER


cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

ner_model = GLiNER.from_pretrained("urchade/gliner_medium-v2.1")

# ---------- NER extraction ----------
def extract_query_entities(query: str):
    labels = [
        "company",
        "organization",
        "year",
        "financial_metric",
        "country",
        "region"
    ]

    entities = ner_model.predict_entities(query, labels)

    result = {}
    for e in entities:
        label = e["label"].lower()
        text = e["text"]

        if label in ["company", "organization"]:
            result["company_name"] = text

        if label == "year":
            # Convert to string to match the stringified metadata in the DB
            result["fiscal_year"] = str(text)

    return result


# ---------- Cross-encoder reranking ----------
def rerank_with_cross_encoder(query: str, documents: list, top_k=20):
    pairs = [(query, doc) for doc in documents]
    scores = cross_encoder.predict(pairs)

    ranked = sorted(zip(scores, documents), reverse=True)
    return [doc for _, doc in ranked[:top_k]]