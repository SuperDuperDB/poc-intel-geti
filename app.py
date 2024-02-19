import streamlit as st
from superduperdb import Document, logging, superduper
from superduperdb.backends.mongodb import Collection


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
    for source in sources:
        source_data = source.content["_source"]
        source_url = (
            Collection("pages").find_one({"_id": source_data}).execute(db)["url"]
        )
        data = source.outputs("elements", "chunk")
        url = source_url + data["url"]
        yield {
            "score": source["score"],
            "url": url,
        }


st.set_page_config(page_title="SuperDuperDB - Documents Chatbot")

st.header("SuperDuperDB - Documents Chatbot")

query = st.text_input("Enter your question", placeholder="Type here...", key="query")

submit_button = st.button("Search", key="qa")
if submit_button:
    st.markdown("#### Input/Query")
    st.markdown(query)
    answer, sources = qa(query, vector_search_top_k=5)
    st.markdown("#### Answer:")
    st.markdown(answer)

    st.markdown("#### Related Documents:")
    for source in process_sources(sources):
        st.markdown(source["url"])
