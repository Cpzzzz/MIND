import logging
import numpy as np
from utils.llm.llm_calling import get_embedding


def get_sentence_embedding(sentence, model: str):
    if not sentence:
        return []
    if not model:
        raise ValueError("An embedding model must be provided.")

    try:
        emb = get_embedding(sentence, model=model)
        if not emb:
            return []

        vector = np.array(emb)
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm
        return vector.tolist()
    except Exception as error:
        logging.error("Embedding error: %s", error)
        return []
