from openai.embeddings_utils import get_embedding, cosine_similarity
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import numpy as np
import openai
import os
from dotenv import load_dotenv

load_dotenv()


openai.api_key = os.getenv('API_KEY')
# models.Base.metadata.create_all(bind=engine)

app = FastAPI()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

df = pd.read_csv('_babbage.csv')
df['babbage_search'] = df.babbage_search.apply(eval).apply(np.array)


async def search(df, search_query, n=3, pprint=True):
    embedding = get_embedding(search_query, engine='babbage-search-query')
    df['similarity'] = df.babbage_search.apply(
        lambda x: cosine_similarity(x, embedding))

    res = df.sort_values('similarity', ascending=False).head(
        n).combined.astype(str)
    if pprint:
        for r in res:
            print(r[:200])
            print()
    return res


@app.get("/{string}/{id}")
async def root(id: int, string: str):
    res = await search(df, string, n=id)
    print(res)

    return {"message": res}
