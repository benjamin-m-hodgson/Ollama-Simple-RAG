"""
Simple local RAG pipeline over a PDF, using Ollama for both embeddings and chat.

Run `pip install -r rag-requirements.txt` to install the required packages.
Requires a running Ollama server (`ollama serve`).

Pipeline:

    1. Ingest a PDF file
    2. Extract text from the PDF file and partition into small chunks
    3. Send the chunks to the embedding model
    4. Save the embeddings to a vector database
    5. Perform similarity search (via multi-query expansion) on the vector database to find similar documents
    6. Retrieve the similar chunks and feed them to the LLM to answer the question

Usage:
    python rag-pdf.py
    python rag-pdf.py --question "How do I report BOI?"
    python rag-pdf.py --rebuild          # discard cached embeddings first
        
"""

import argparse
import logging
import shutil
import sys
from pathlib import Path
from textwrap import dedent

from langchain_chroma import Chroma
from langchain_classic.retrievers.multi_query import MultiQueryRetriever
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.runnables import Runnable, RunnablePassthrough
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_unstructured import UnstructuredLoader
from langchain_core.output_parsers import BaseOutputParser

import ollama

# ===== LOGGING ===== #
 
# Configure our own logger rather than the root logger, so third-party
# libraries don't inherit INFO level and flood stdout.
# A logger decides whether a message passes; a handler decides where it goes.
log = logging.getLogger("simple-rag")
log.setLevel(logging.INFO)
log.addHandler(logging.StreamHandler(sys.stdout))

# Python's loggers form a hierarchy, and by default a record travels up it 
# after your handler runs: simple-rag → root. 
# Setting propagate = False stops the record at the simple-rag logger.
log.propagate = False
 
# pdfminer logs "Cannot set non-stroke color" for every malformed color
# operator in the PDF. Harmless for text extraction, but it floods stdout.
for noisy in ("pdfminer", "pikepdf", "httpx", "chromadb", "unstructured", "urllib3"):
    logging.getLogger(noisy).setLevel(logging.ERROR)


# ===== CONFIG ===== #

# Resolve relative to this file, not the current working directory, so the
# script works regardless of where you run it from.
BASE_DIR = Path(__file__).resolve().parent
DOC_PATH = BASE_DIR / "data" / "BOI.pdf"
PERSIST_DIR = BASE_DIR / "chroma_db"

CHAT_MODEL = "llama3.2:3b"
EMBEDDING_MODEL = "nomic-embed-text"
COLLECTION_NAME = "simple-rag"

DEFAULT_QUESTION = "What are the main points as a business owner I should be aware of?"

# Strategies the local unstructured library supports. "by_page" and
# "by_similarity" exist only in Unstructured's hosted Platform/API.
LOCAL_CHUNKING_STRATEGIES = {"by_title", "basic", None}


# ===== Steps 1 & 2: Load + Chunk ===== #

def load_chunks(path: Path, 
                chunking_strategy: str | None = "by_title", 
                max_characters: int = 1000,
                new_after_n_chars: int | None = None,
                combine_text_under_n_chars: int = 300,
                multipage_sections: bool = True,
                overlap: int = 200,
                overlap_all: bool = False
) -> list[Document]:
    """
    Partition a PDF into chunks suitable for embedding.

    Partitioning splits the document into semantic elements (Title,
    NarrativeText, ListItem, Table). Chunking is a separate pass that
    recombines those elements into larger units.

    Parameters
    ----------
    path : Path
        Path to the PDF to load.
    chunking_strategy : {"by_title", "basic", None}, optional
        How elements are recombined. Default "by_title".

        - ``None``| no chunking. Returns raw elements, one per Title,
          NarrativeText, ListItem, or Table. Typically many very short
          documents.
        - ``"basic"``| packs consecutive elements to maximally fill each
          chunk up to `max_characters`, ignoring document structure. A
          heading can end up separated from the text it introduces.
        - ``"by_title"``| as "basic", but a Title element closes the
          current chunk even if it would have fit. A chunk therefore never
          spans two sections.

        Unstructured's hosted Platform/API also offers "by_page" and
        "by_similarity". These are NOT available in the local library, which
        is what UnstructuredLoader uses unless partition_via_api=True. For
        page-boundary behaviour locally, use "by_title" with
        multipage_sections=False.
    max_characters : int, optional
        Hard ceiling on chunk size. An individual element longer than this is
        text-split on a word boundary; otherwise elements are never split.
        Default 1000. Match this to your embedding model's context window.
    new_after_n_chars : int | None, optional
        Soft target. Once a chunk reaches this length, close it at the 
        next element boundary if possible, but unlike `max_characters` this 
        is not enforced. When None, unstructured defaults it to `max_characters`. 
        Default None.
    combine_text_under_n_chars : int, optional
        Merge consecutive sections until the combined length reaches this
        many characters. Guards against the flood of tiny chunks produced
        when partitioning mislabels short paragraphs or list items as Title.
        Note this works against section isolation -- too high and it reunites
        sections that "by_title" just separated. Default 300.
        Only operative for chunking_strategy="by_title".
    multipage_sections : bool, optional
        When True, a section may span a page break. When False, a page break
        also closes the chunk -- the local approximation of by-page chunking.
        Default True.
        Only operative for chunking_strategy="by_title".
    overlap : int, optional
        Characters of trailing context prepended to the next chunk. Applies
        ONLY to chunks created by text-splitting an oversized element, unless
        overlap_all is True. On a document whose elements all fit within
        max_characters, this has no effect. Default 200.
    overlap_all : bool, optional
        Apply `overlap` between all chunks, including those formed from whole
        elements. This deliberately blurs otherwise clean semantic
        boundaries, so treat it as a tunable to validate against retrieval
        quality rather than a default. Default False.

    Returns
    -------
    list[Document]
        LangChain Documents. Each carries metadata including `category`
        (after chunking this is "CompositeElement", not the original element
        type), `page_number`, `source`, and `orig_elements`.

    Raises
    ------
    FileNotFoundError
        If `path` does not exist.
    ValueError
        If no content could be extracted from the PDF.
    """
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {path}")

    if chunking_strategy not in LOCAL_CHUNKING_STRATEGIES:
        raise ValueError(
            f"chunking_strategy={chunking_strategy!r} is not available in the "
            f"local unstructured library. Choose from {LOCAL_CHUNKING_STRATEGIES}."
        )

    loader_kwargs = {"file_path" : str(path)}

    if chunking_strategy is not None:
        loader_kwargs.update(
            chunking_strategy = chunking_strategy,
            max_characters = max_characters,
            overlap = overlap
        )
        if new_after_n_chars is not None:
            loader_kwargs.update(new_after_n_chars = new_after_n_chars)
        if chunking_strategy == "by_title":
            loader_kwargs.update(
                combine_text_under_n_chars = combine_text_under_n_chars,
                multipage_sections = multipage_sections
            )

    loader = UnstructuredLoader(**loader_kwargs)
    chunks = loader.load()

    if not chunks:
        raise ValueError(f"No content extracted from {path}")

    log.info("Loaded %d chunks from %s", len(chunks), path.name)
    return chunks


# ===== Steps 3 & 4: Embed + Store ===== #

def _collection_count(store: Chroma) -> int:
    """
    Number of documents in the store.
 
    Chroma exposes no public count on the LangChain wrapper, so this reaches
    into the underlying collection. Isolated here so there is one place to
    fix if the private attribute changes.
    """
    return store._collection.count()

def build_vector_db(path: Path, rebuild: bool = False) -> Chroma:
    """
    Return a Chroma store for `path`, reusing persisted embeddings when present.
 
    The PDF is only partitioned when the store is empty. Partitioning is
    typically slower than embedding, so loading it unconditionally would
    defeat the point of persisting.
    """
    if rebuild and PERSIST_DIR.exists():
        log.info("Removing cached embeddings at %s", PERSIST_DIR)
        shutil.rmtree(PERSIST_DIR)

    ollama.pull(EMBEDDING_MODEL)
    embeddings = OllamaEmbeddings(model = EMBEDDING_MODEL)

    store = Chroma(
        collection_name = COLLECTION_NAME,
        embedding_function = embeddings,
        persist_directory = str(PERSIST_DIR)
    )

    existing = _collection_count(store)
    if existing == 0:
        chunks = load_chunks(path)
        log.info("Embedding %d chunks (first run)...", len(chunks))
        store.add_documents(chunks)
        log.info("Embeddings stored in %s", PERSIST_DIR)
    else:
        log.info("Reusing %d existing embeddings", existing)
 
    return store

# ===== Steps 5 & 6: Retrieve + Answer ===== #

# LangChain's built-in parser splits on newlines but doesn't filter empty lines.
class LineListOutputParser(BaseOutputParser[list[str]]):
    """Split LLM output into non-empty, stripped lines."""

    def parse(self, text: str) -> list[str]:
        result = []
        for line in text.splitlines():
            # Keep only lines that have content after whitespace removal
            if line.strip():        
                result.append(line.strip())
        return result


# dedent() strips the shared leading whitespace so the model receives clean
# prompt text rather than an indented block.
QUERY_PROMPT = PromptTemplate(
    input_variables=["question"],
    template=dedent("""\
        You are an AI language model assistant. Your task is to generate five
        different versions of the given user question to retrieve relevant
        documents from a vector database. By generating multiple perspectives
        on the user question, your goal is to help the user overcome some of
        the limitations of distance-based similarity search. Provide these
        alternative questions separated by newlines. Do not include any
        additional text in the response - only the five different versions.
        Original question: {question}
    """),
)

RAG_TEMPLATE = dedent("""\
    Answer the question based ONLY on the following context.
    If the context does not contain the answer, say so rather than guessing.
 
    Context: {context}
 
    Question: {question}
""")

def build_chain(store: Chroma, k: int = 3) -> Runnable:
    """
    Assemble the retrieval chain.
 
    MultiQueryRetriever asks the LLM to rephrase the question several ways and
    unions the results, which helps when the user's wording does not match the
    document's. It costs one extra LLM call per query.
 
    Parameters
    ----------
    k : int, optional 
    k is the number of chunks fetched per generated query but MultiQueryRetriever 
    issues multiple queries, so the context passed to the LLM can be several 
    times k after deduplication. Default 3.
    """

    llm = ChatOllama(model = CHAT_MODEL)

    # MultiQueryRetriever.from_llm() uses a parser that keeps blank lines, so small 
    # models that separate their questions with blank lines produce empty queries.
    query_chain = QUERY_PROMPT | llm | LineListOutputParser()

    retriever = MultiQueryRetriever(
        retriever = store.as_retriever(search_kwargs = {"k": k}),
        llm_chain = query_chain,
    )

    return (
        {"context": retriever, "question": RunnablePassthrough()}
        | ChatPromptTemplate.from_template(RAG_TEMPLATE)
        | llm
        | StrOutputParser()
    )

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description = "Query a PDF with a local RAG pipeline.")
    parser.add_argument("--question", default = DEFAULT_QUESTION, help = "Question to ask.")
    parser.add_argument("--pdf", type = Path, default = DOC_PATH, help = "PDF to query.")
    parser.add_argument("--k", type = int, default = 3, help = "Chunks retrieved per query.")
    parser.add_argument(
        "--rebuild", action = "store_true", help = "Discard cached embeddings and re-embed."
    )
    return parser.parse_args()

def main():
    args = parse_args()
 
    try:
        ollama.pull(CHAT_MODEL)
    except Exception as exc:
        # ollama raises connection errors that don't make the cause obvious.
        log.error("Could not reach Ollama (%s). Is `ollama serve` running?", exc)
        sys.exit(1)
 
    store = build_vector_db(args.pdf, rebuild = args.rebuild)
    chain = build_chain(store, k = args.k)
 
    log.info("\nQuestion: %s\n", args.question)

    # Pass the question as a plain string. RunnablePassthrough forwards
    # whatever it receives, so wrapping it in a tuple would send a tuple
    # through to the prompt.
    print(chain.invoke(args.question))


if __name__ == "__main__":
    main()