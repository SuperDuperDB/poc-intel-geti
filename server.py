import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from superduperdb import Document, superduper
from superduperdb.backends.mongodb import Collection
from superduperdb import logging

mongodb_uri = os.getenv("SUPERDUPERDB_DATA_BACKEND", "mongomock://test")
db = superduper(mongodb_uri)

def qa(query, vector_search_top_k=5):
    collection = Collection("_outputs.url.chunk")
    output, sources = db.predict(
        model_name="gpt-3.5-turbo",
        input=query,
        context_select=collection.like(
            Document({"_outputs.url.chunk.0": query}),
            vector_index="vector_index",
            n=vector_search_top_k,
        ).find({}),
        context_key="_outputs.url.chunk.0.text",
    )
    if sources:
        sources = sorted(sources, key=lambda x: x["score"], reverse=True)

    return output.unpack(), sources


def process_sources(sources):
    results = []
    for source in sources:
        source_data = source["_source"]
        source_url = Collection("url").find_one({"_id": source_data}).execute(db)["url"]
        data = source.outputs("url", "chunk")
        if data["href"].startswith("http"):
            url = data["href"]
        else:
            url = source_url + data["href"]
        results.append(
            {
                "score": source["score"],
                "url": url,
            }
        )
    return results


# Create a FastAPI app instance with version, description, and lifespan manager
app = FastAPI(
    title="SuperDuperDB Question The Docs",
)

# Configure Cross-Origin Resource Sharing (CORS) settings
origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class Query(BaseModel):
    query: str
    doc_nums: int = 5


# Endpoint for showing components in the database
@app.post("/chat")
async def chat(query: Query):
    # Define the search parameters
    #
    answer, sources = qa(query.query, query.doc_nums)

    return {"answer": answer, "source_urls": process_sources(sources)}


@app.post("/add_url")
async def add_url(url: str):
    # Define the search parameters
    data = Document(**{"url": url})
    db.execute(Collection("url").insert_one(data), refresh=False)
    return {"url": url}
