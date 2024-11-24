from sentence_transformers import SentenceTransformer

_model = None

def get_model():
    global _model
    if _model is None:
        # _model = SentenceTransformer(r"C:\Jeff_Documents\Programming\CodeJam14\Final_Vector_Embedding\fine_tuned_large_model")  # Load the model only once
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model

