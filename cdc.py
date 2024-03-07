import os
import re
from collections import defaultdict
from io import StringIO
from urllib.parse import urljoin

import pandas as pd
from pydantic import main
import requests
from bs4 import BeautifulSoup
from superduperdb import Listener, Model, VectorIndex, logging, superduper
from superduperdb.backends.mongodb import Collection
from superduperdb.base.datalayer import Datalayer
from superduperdb.base.document import Document
from superduperdb.components.model import SequentialModel
from superduperdb.ext.openai import OpenAIEmbedding
from superduperdb.misc.retry import Retry
from unstructured.documents.elements import ElementType
from unstructured.partition.html import partition_html

mongodb_uri = os.getenv("SUPERDUPERDB_DATA_BACKEND", "mongomock://test")
db = superduper(mongodb_uri)


def process_code_snippets(text):
    soup = BeautifulSoup(text, "html.parser")
    pre_tags = soup.find_all("pre")

    for pre in pre_tags:
        processed_text = str(pre.text)
        new_content = "CODE::" + soup.new_string(processed_text)
        pre.clear()
        pre.append(new_content)
    return str(soup)


def process_py_class(source_html):
    soup = BeautifulSoup(source_html, "html.parser")
    dl_tags = soup.find_all("dl", class_="py class")

    for dl in dl_tags:
        dt_tag = dl.find("dt", class_="sig sig-object py")
        if not dt_tag:
            continue
        last_headerlink = dt_tag.find_all("a", class_="headerlink")[-1]
        href = last_headerlink["href"] if last_headerlink else ""
        id = dt_tag.attrs["id"]
        new_h3 = soup.new_tag("h3")
        new_a_inside_h3 = soup.new_tag("a", href=href)
        new_a_inside_h3.string = f"Class: {id}"
        new_h3.append(new_a_inside_h3)

        new_code = soup.new_tag("a")
        new_code.string = dt_tag.text
        dt_tag.insert_before(new_h3)
        dt_tag.insert_before(new_code)
        dt_tag.decompose()

    return str(soup)


def parse_url(seed_url):
    retry = Retry(exception_types=(Exception))

    @retry
    def get_response(url):
        response = requests.get(seed_url)
        return response

    print(f"parse {seed_url}")
    response = get_response(seed_url)
    # Parse the HTML content
    source_html = response.text
    source_html = process_code_snippets(source_html)
    source_html = process_py_class(source_html)

    return source_html


def url2html(url):
    try:
        html = parse_url(url)
    except Exception as e:
        logging.error(e)
        html = ""
    return html


def page2elements(page):
    elements = partition_html(text=page, html_assemble_articles=True)
    return elements


def get_title_data(element):
    data = {}
    if element.category != ElementType.TITLE:
        return data
    if "link_urls" not in element.metadata.to_dict():
        return data

    if "category_depth" not in element.metadata.to_dict():
        return data

    [link_text, *_] = element.metadata.link_texts

    if not link_text:
        return data

    link_urls = element.metadata.link_urls
    if not link_urls:
        return data
    category_depth = element.metadata.category_depth
    return {"link": link_urls[0], "category_depth": category_depth}


def element2text(element):
    title_message = get_title_data(element)
    text = element.text
    if title_message:
        title_tags = "#" * (title_message["category_depth"] + 1)
        text = title_tags + " " + text
        text = text.rstrip("#")

    elif element.category == ElementType.LIST_ITEM:
        text = "- " + text

    elif element.category == ElementType.TABLE:
        html = element.metadata.text_as_html
        html = html.replace("|", "")
        df = pd.read_html(StringIO(html))[0]
        text = df.to_markdown(index=False)
        text = text + "  \n"

    if text.startswith("CODE::"):
        text = f"```\n{text[6:]}\n```"

    return text


def get_chunk_texts(text, chunk_size=1000, overlap_size=300):
    chunks = []
    start = 0

    while start < len(text):
        if chunks:
            start -= overlap_size
        end = start + chunk_size
        end = min(end, len(text))
        chunks.append(text[start:end])
        start = end
        if start >= len(text):
            break

    return chunks


def get_chunks(elements):
    chunk_tree = defaultdict(list)
    now_depth = -1
    now_path = "root"
    for element in elements:
        title_data = get_title_data(element)
        if not title_data:
            chunk_tree[now_path].append(element)
        else:
            link = title_data["link"]
            depth = title_data["category_depth"]
            if depth > now_depth:
                now_path = now_path + "::" + link
            else:
                now_path = "::".join(now_path.split("::")[: depth + 1] + [link])
            now_depth = depth
            chunk_tree[now_path].append(element)

    chunks = []
    for node_path, node_elements in chunk_tree.items():
        new_elements = []
        nodes = node_path.split("::")
        parent_elements = []
        for i in range(1, len(nodes) - 1):
            [parent_element, *_] = chunk_tree["::".join(nodes[: i + 1])] or [None]
            if parent_element:
                parent_elements.append(parent_element)
        node_elements = [*parent_elements, *node_elements]
        content = "\n\n".join(map(lambda x: element2text(x), node_elements))
        for chunk_text in get_chunk_texts(content):
            # The url field is used to save the jump link
            # The text field is used for vector search
            # The content field is used to submit to LLM for answer
            chunk = {"href": nodes[-1], "text": chunk_text, "content": content}
            chunks.append(chunk)
    return chunks


def page2chunks(page):
    elements = page2elements(page)
    chunks = get_chunks(elements)
    return chunks


def add_model_url2html(db):
    url_model = Model(
        identifier="url2html",
        object=url2html,
        model_update_kwargs={"document_embedded": False},
    )
    url_listener = Listener(
        model=url_model,
        select=Collection("url").find(),
        key="url",
    )
    db.add(url_listener)
    print(url_listener.identifier, url_listener.outputs)
    return url_listener


def add_model_chunk(db, url_listener):
    chunk_model = Model(
        identifier="chunk",
        object=page2chunks,
        flatten=True,
        model_update_kwargs={"document_embedded": False},
    )

    chunk_listener = Listener(
        model=chunk_model,
        select=Collection("_outputs.url.url2html").find(),
        key=f"_outputs.url.url2html.{url_listener.model.version}",
    )

    db.add(chunk_listener)

    print(chunk_listener.identifier, chunk_listener.outputs)
    return chunk_listener


def add_model_embedding(db, chunk_listener):
    opeai_emb_model = OpenAIEmbedding(
        identifier="text-embedding-ada-002",
        model="text-embedding-ada-002",
    )
    preprocess_model = Model(
        identifier="preprocess",
        object=lambda x: x["text"] if isinstance(x, dict) else x,
    )

    embed_model = SequentialModel(
        identifier="embedding", predictors=[preprocess_model, opeai_emb_model]
    )
    embed_listener = Listener(
        select=Collection("_outputs.url.chunk").find(),
        key=f"_outputs.url.chunk.{chunk_listener.model.version}",  # Key for the documents
        model=embed_model,  # Specify the model for processing
        predict_kwargs={"max_chunk_size": 64},
    )
    print(embed_listener.identifier, embed_listener.outputs)
    db.add(embed_listener)
    return embed_listener


def add_vector_index(db, embed_listener):
    vector_index = VectorIndex(
        identifier="vector_index",
        indexing_listener=embed_listener,
    )
    db.add(vector_index)
    print(vector_index.identifier)


def add_model_llm(db):
    from superduperdb.ext.openai import OpenAIChatCompletion

    prompt = """
    As an Intel GETI assistant, based on the provided documents and the question, answer the question.
    If the document does not provide an answer, offer a safe response without fabricating an answer.

    Documents:
    {context}

    Question: """

    llm = OpenAIChatCompletion(identifier="gpt-3.5-turbo", prompt=prompt)

    db.add(llm)

    print(db.show("model"))


def setup():
    db.drop(force=True)
    url_listener = add_model_url2html(db)
    chunk_listener = add_model_chunk(db, url_listener)
    embed_listener = add_model_embedding(db, chunk_listener)
    add_vector_index(db, embed_listener)
    add_model_llm(db)


def vector_search(query):
    outs = db.execute(
        Collection("_outputs.url.chunk")
        .like(
            Document({"_outputs.url.chunk.0": query}), vector_index="vector_index", n=3
        )
        .find()
    )
    if outs:
        outs = sorted(outs, key=lambda x: x["score"], reverse=True)
    for out in outs:
        print("-" * 20, "\n")
        data = out.outputs("url", "chunk")
        url = data["href"]
        print(url, out["score"])
        print(data["text"])


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
    print(output.unpack())
    for out in sources:
        print("-" * 20, "\n")
        data = out.outputs("url", "chunk")
        url = data["href"]
        print(url, out["score"])
        print(data["text"])
    # return output, sources


def insert_url(url):
    datas = Document(**{"url": url})
    db.execute(Collection("url").insert_one(datas), refresh=False)


if __name__ == "__main__":
    setup()
