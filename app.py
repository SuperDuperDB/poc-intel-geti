import os
import time
from datetime import datetime


import streamlit as st
from superduperdb import Document, logging, superduper
from superduperdb.backends.mongodb import Collection

# It just super dupers your database
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
    for source in sources:
        source_data = source["_source"]
        source_url = Collection("url").find_one({"_id": source_data}).execute(db)["url"]
        data = source.outputs("url", "chunk")
        if data["href"].startswith("http"):
            url = data["href"]
        else:
            url = source_url + data["href"]
        yield {
            "score": source["score"],
            "url": url,
        }

st.set_page_config(page_title="SuperDuperDB - Documents Chatbot")

st.header("SuperDuperDB - Documents Chatbot")



def add_new_data(url):
    urls = [url]
    yield f"Added {len(urls)} new URLs to the database"
    data = Document(**{"url": url})
    db.execute(Collection("url").insert_one(data), refresh=False)
    
    yield "Data will be updated later"

[tab_qa, tab_add_data] = st.tabs(["AI Chat (Q&A)", "Add New Data"])

with tab_qa:
    st.markdown("### AI Chat (Q&A)")
    query = st.text_input("Enter your question", placeholder="Type here...", key="query")

    submit_button = st.button("Search", key="qa")
    if submit_button:
        st.markdown("#### Input/Query")
        st.markdown(query)
        answer, sources = qa(query)
        st.markdown("#### Answer:")
        st.markdown(answer)

        st.markdown("#### Related Documents:")
        for source in process_sources(sources):
            st.markdown(source["url"])

with tab_add_data:
    # 添加新数据选项卡
    st.markdown("### Add New Data")
    url_input = st.text_input("Enter URL", key="new_url")
    add_data_button = st.button("Add Data", key="add_data_button")

    if add_data_button:
        messages = add_new_data(url_input)
        for message in messages:
            st.markdown(message)
