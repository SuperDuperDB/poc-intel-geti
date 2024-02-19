
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from superduperdb import Document, superduper
from superduperdb.backends.mongodb import Collection
from superduperdb import logging

# It just super dupers your database
db = superduper("mongodb://127.0.0.1:27017/intel-geti")


def qa(query, vector_search_top_k=5):
    logging.info(f"QA query: {query}")
    collection = Collection("_outputs.elements.chunk")
    output, sources = db.predict(
        model_name="gpt-3.5-turbo",
        input=query,
        context_select=collection.like(
            Document({"_outputs.elements.chunk": query}),
            vector_index="vector_index",
            n=vector_search_top_k,
        ).find({}),
        context_key="_outputs.elements.chunk.0.text",
    )
    if sources:
        sources = sorted(sources, key=lambda x: x.content["score"], reverse=True)
    return output.content, sources


def process_sources(sources):
    results = []
    for source in sources:
        source_data = source.content["_source"]
        source_url = (
            Collection("pages").find_one({"_id": source_data}).execute(db)["url"]
        )
        data = source.outputs("elements", "chunk")
        url = source_url + data["url"]
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
