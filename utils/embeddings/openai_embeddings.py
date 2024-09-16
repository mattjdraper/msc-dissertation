from typing import List
from tqdm import tqdm
import numpy as np

def get_embedding(client, text: str, model: str, **kwargs) -> List[float]:
    response = client.embeddings.create(input=[text], model=model, **kwargs)
    embedding = np.array(response.data[0].embedding)
    return embedding


def embed_dataset(client, data, model):
    embeddings = []
    
    print("Generating OpenAI embeddings... this may take a while.")
    for d in tqdm(data):
        embedding = get_embedding(client, d, model=model)
        embeddings.append(embedding)

    # Ensure the final list of embeddings is a 2D array
    return np.array(embeddings)