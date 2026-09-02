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

    # defaults
    python simple-rag-pdf.py
 
    # ask a different question of the default document
    python simple-rag-pdf.py --question "How do I report BOI?"
 
    # every argument, with sample values
    python simple-rag-pdf.py \
        --chat-model "llama3.2:3b" \
        --question "What are the filing deadlines?" \
        --pdf data/BOI.pdf \
        --embedding-model "nomic-embed-text" \
        --k 3 \
        --chunking-strategy by_title \
        --max-characters 1000 \
        --new-after-n-chars 800 \
        --combine-text-under-n-chars 300 \
        --no-multipage-sections \
        --overlap 200 \
        --overlap-all \
        --rebuild \
        --verbose
 
    # raw elements, no chunking (one document per Title/NarrativeText/ListItem/Table)
    python simple-rag-pdf.py --chunking-strategy none --rebuild
 
    # structure-blind packing, for documents without reliable headings
    python simple-rag-pdf.py --chunking-strategy basic --max-characters 1500 --rebuild
 
IMPORTANT: embeddings are cached in chroma_db/ and the only staleness check
is whether the collection is empty. Changing the document, embedding model,
or any chunking argument will silently reuse the previous embeddings.
Pass --rebuild whenever any is changed.
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
from langchain_core.output_parsers import BaseOutputParser, StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.runnables import Runnable, RunnablePassthrough
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_unstructured import UnstructuredLoader

import ollama

# ===== LOGGING ===== #
 
# Logger for this script's own messages. Configured by configure_logging(),
# which main() calls before any other work.
log = logging.getLogger("simple-rag-pdf")

# Logs the queries MultiQueryRetriever generates. Enabled by --verbose.
MULTI_QUERY_LOGGER = "langchain_classic.retrievers.multi_query"

# Third-party loggers that inherit root's level and flood stdout. pdfminer in
# particular logs "Cannot set non-stroke color" for every malformed color
# operator in the PDF -- harmless for text extraction, but noisy.
NOISY_LOGGERS = (
    "pdfminer",
    "pikepdf",
    "httpx",
    "chromadb",
    "unstructured",
    "urllib3",
    MULTI_QUERY_LOGGER
)

def configure_logging(verbose: bool = False) -> None:
    """
    Set up logging for the script.

    Called from main() rather than at import time. That ordering matters: a
    dependency in the unstructured tree calls logging.basicConfig() when it is
    imported, which installs a root handler and sets root to INFO. Configuring
    from main() means those imports have already run, so nothing set here is
    immediately overwritten.

    Safe to call more than once -- handlers are only added if absent.

    Parameters
    ----------
    verbose : bool, optional
        Log the queries MultiQueryRetriever generates. Default False.
        CLI: --verbose
    """
    # A logger decides whether a message passes; a handler decides where it
    # goes. The guard keeps a second call from attaching a duplicate handler.
    log.setLevel(logging.INFO)
    if not log.handlers:
        log.addHandler(logging.StreamHandler(sys.stdout))

    # Python's loggers form a hierarchy, and a record travels up it after our
    # handler runs: simple-rag-pdf -> root. propagate = False stops it here, so a
    # root handler installed by a dependency cannot print it a second time.
    log.propagate = False

    # Backstop only. basicConfig() may run later and reset root's level, so
    # this is not sufficient on its own -- the explicit per-logger levels
    # below are what actually holds, because a level set directly on a logger
    # takes precedence over inheritance from its ancestors.
    logging.getLogger().setLevel(logging.WARNING)

    for name in NOISY_LOGGERS:
        logging.getLogger(name).setLevel(logging.ERROR)

    if verbose:
        # Re-enable the one logger silenced above, with its own handler and
        # propagate = False for the same reason as `log`.
        multi_query = logging.getLogger(MULTI_QUERY_LOGGER)
        multi_query.setLevel(logging.INFO)
        if not multi_query.handlers:
            multi_query.addHandler(logging.StreamHandler(sys.stdout))
        multi_query.propagate = False


# ===== CONFIG ===== #

# Resolve relative to this file, not the current working directory, so the
# script works regardless of where you run it from.
BASE_DIR = Path(__file__).resolve().parent
DOC_PATH = BASE_DIR / "data" / "BOI.pdf"
PERSIST_DIR = BASE_DIR / "chroma_db"

COLLECTION_NAME = "simple-rag"

# Defaults live here so the function signatures and the argparse defaults
# cannot drift apart.
DEFAULT_CHAT_MODEL = "llama3.2:3b"
DEFAULT_EMBEDDING_MODEL = "nomic-embed-text"
DEFAULT_QUESTION = "What are the main points as a business owner I should be aware of?"
DEFAULT_K = 3
DEFAULT_CHUNKING_STRATEGY = "by_title"
DEFAULT_MAX_CHARACTERS = 1000
DEFAULT_NEW_AFTER_N_CHARS = None
DEFAULT_COMBINE_TEXT_UNDER_N_CHARS = 300
DEFAULT_MULTIPAGE_SECTIONS = True
DEFAULT_OVERLAP = 200
DEFAULT_OVERLAP_ALL = False

# Strategies the local unstructured library supports. "by_page" and
# "by_similarity" exist only in Unstructured's hosted Platform/API.
LOCAL_CHUNKING_STRATEGIES = {"by_title", "basic", None}


# ===== Steps 1 & 2: Load + Chunk ===== #

def load_chunks(path: Path, 
                chunking_strategy: str | None = DEFAULT_CHUNKING_STRATEGY, 
                max_characters: int = DEFAULT_MAX_CHARACTERS,
                new_after_n_chars: int | None = DEFAULT_NEW_AFTER_N_CHARS,
                combine_text_under_n_chars: int = DEFAULT_COMBINE_TEXT_UNDER_N_CHARS,
                multipage_sections: bool = DEFAULT_MULTIPAGE_SECTIONS,
                overlap: int = DEFAULT_OVERLAP,
                overlap_all: bool = DEFAULT_OVERLAP_ALL
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
        CLI: --chunking-strategy

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
        CLI: --max-characters
    new_after_n_chars : int | None, optional
        Soft target. Once a chunk reaches this length, close it at the 
        next element boundary if possible, but unlike `max_characters` this 
        is not enforced. When None, unstructured defaults it to `max_characters`. 
        Default None.
        CLI: --new-after-n-chars
    combine_text_under_n_chars : int, optional
        Merge consecutive sections until the combined length reaches this
        many characters. Guards against the flood of tiny chunks produced
        when partitioning mislabels short paragraphs or list items as Title.
        Note this works against section isolation -- too high and it reunites
        sections that "by_title" just separated. Default 300.
        Only operative for chunking_strategy="by_title".
        CLI: --combine-text-under-n-chars
    multipage_sections : bool, optional
        When True, a section may span a page break. When False, a page break
        also closes the chunk -- the local approximation of by-page chunking.
        Default True.
        Only operative for chunking_strategy="by_title".
        CLI: --no-multipage-sections
    overlap : int, optional
        Characters of trailing context prepended to the next chunk. Applies
        ONLY to chunks created by text-splitting an oversized element, unless
        overlap_all is True. On a document whose elements all fit within
        max_characters, this has no effect. Default 200.
        CLI: --overlap
    overlap_all : bool, optional
        Apply `overlap` between all chunks, including those formed from whole
        elements. This deliberately blurs otherwise clean semantic
        boundaries, so treat it as a tunable to validate against retrieval
        quality rather than a default. Default False.
        CLI: --overlap-all

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
        If `chunking_strategy` is not available locally, or if no content
        could be extracted from the PDF.
    """
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {path}")

    if chunking_strategy not in LOCAL_CHUNKING_STRATEGIES:
        raise ValueError(
            f"chunking_strategy = {chunking_strategy!r} is not available in the "
            f"local unstructured library. Choose from {LOCAL_CHUNKING_STRATEGIES}."
        )

    loader_kwargs = {"file_path" : str(path)}

    if chunking_strategy is not None:
        loader_kwargs.update(
            chunking_strategy = chunking_strategy,
            max_characters = max_characters,
            overlap = overlap,
            overlap_all = overlap_all
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

# Chroma metadata values must be scalars, None, or lists of scalars.
# unstructured attaches nested dicts -- `coordinates` holds a points tuple
# plus layout dimensions -- which fail validation. by_title chunking happens
# to discard these when it merges elements into CompositeElements, so this
# only bites under "basic" and "none".
SCALAR_TYPES = (str, int, float, bool)

def _clean_metadata(docs: list[Document]) -> list[Document]:
    """
    Return copies of `docs` with metadata Chroma cannot store removed.

    Keeps scalars, None, and lists of scalars. Drops anything else -- dicts,
    tuples, and objects.

    Copies rather than mutates, so load_chunks' callers keep the full
    metadata and only what reaches Chroma is stripped.
    """
    cleaned = []
    for doc in docs:
        metadata = {}
        for key, value in doc.metadata.items():
            if value is None or isinstance(value, SCALAR_TYPES):
                metadata[key] = value
            elif isinstance(value, list) and all(
                isinstance(item, SCALAR_TYPES) for item in value
            ):
                metadata[key] = value
            # anything else (dicts, tuples, objects) is dropped
        cleaned.append(Document(page_content = doc.page_content, metadata = metadata))
    return cleaned

def _collection_count(store: Chroma) -> int:
    """
    Number of documents in the store.
 
    Chroma exposes no public count on the LangChain wrapper, so this reaches
    into the underlying collection. Isolated here so there is one place to
    fix if the private attribute changes.
    """
    return store._collection.count()

def build_vector_db(path: Path, 
                    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
                    rebuild: bool = False, 
                    **chunk_params
) -> Chroma:
    """
    Return a Chroma store for `path`, reusing persisted embeddings when present.
 
    The PDF is only partitioned when the store is empty. Partitioning is
    typically slower than embedding, so loading it unconditionally would
    defeat the point of persisting.

    Parameters
    ----------
    path : Path
        PDF to embed.
    embedding_model : str, optional
        Embedding model used to generate embeddings. Default "nomic-embed-text".
    rebuild : bool, optional
        Delete PERSIST_DIR before building, forcing a re-partition and
        re-embed. Default False.
    **chunk_params
        Forwarded to load_chunks. See its docstring for each parameter.
    """
    if rebuild and PERSIST_DIR.exists():
        log.info("Removing cached embeddings at %s", PERSIST_DIR)
        shutil.rmtree(PERSIST_DIR)

    ollama.pull(embedding_model)
    embeddings = OllamaEmbeddings(model = embedding_model)

    store = Chroma(
        collection_name = COLLECTION_NAME,
        embedding_function = embeddings,
        persist_directory = str(PERSIST_DIR)
    )

    existing = _collection_count(store)
    if existing == 0:
        chunks = load_chunks(path, **chunk_params)
        log.info("Embedding %d chunks (first run)...", len(chunks))
        store.add_documents(_clean_metadata(chunks))
        log.info("Embeddings stored in %s", PERSIST_DIR)
    else:
        log.info("Reusing %d existing embeddings", existing)
        log.info("Pass --rebuild if the document, embedding model, or chunking arguments changed.")
 
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

def format_docs(docs: list[Document]) -> str:
    """
    Join retrieved documents into the plain text passed to the LLM.

    Without this, the Document objects are stringified into the prompt --
    ids, metadata, and the orig_elements base64 blob included -- which
    floods the context.
    """
    contents = []
    for doc in docs:
        page = doc.metadata.get("page_number")
        header = f"[Page {page}]" if page else ""
        contents.append(f"{header}\n{doc.page_content}".strip())
    return "\n\n".join(contents)

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

def build_chain(store: Chroma, 
                k: int = DEFAULT_K,
                chat_model: str = DEFAULT_CHAT_MODEL
) -> Runnable:
    """
    Assemble the retrieval chain.
 
    MultiQueryRetriever asks the LLM to rephrase the question several ways and
    unions the results, which helps when the user's wording does not match the
    document's. It costs one extra LLM call per query.
 
    Parameters
    ----------
    store : Chroma
        Vector store to retrieve from.
    k : int, optional 
        k is the number of chunks fetched per generated query but MultiQueryRetriever 
        issues multiple queries, so the context passed to the LLM can be several 
        times k after deduplication. Default 3.
        CLI: --k
    chat_model : str, optional
        Model passed to ChatOllama to generate rephrased question context and answer.
        Default "llama3.2:3b"
    """

    llm = ChatOllama(model = chat_model)

    # MultiQueryRetriever.from_llm() uses a parser that keeps blank lines, so small 
    # models that separate their questions with blank lines produce empty queries.
    # Custom class LineListOutputParser() was implemented above to address this.
    query_chain = QUERY_PROMPT | llm | LineListOutputParser()

    retriever = MultiQueryRetriever(
        retriever = store.as_retriever(search_kwargs = {"k": k}),
        llm_chain = query_chain,
    )

    return (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | ChatPromptTemplate.from_template(RAG_TEMPLATE)
        | llm
        | StrOutputParser()
    )


# ===== CLI ===== #

def parse_args() -> argparse.Namespace:
    """
    Command-line interface.
 
    Chunking flags mirror load_chunks' parameters one-for-one; see that
    docstring for what each does. Defaults come from the module constants so
    the CLI and the function signatures cannot disagree.
    """
    parser = argparse.ArgumentParser(
        description = "Query a PDF with a local RAG pipeline.",
        formatter_class = argparse.ArgumentDefaultsHelpFormatter
    )
 
    # --- query ---
    query_group = parser.add_argument_group(
        "query",
        "Pass --rebuild if the embedding model is changed."
    )
    query_group.add_argument("--chat-model", type = str, default = DEFAULT_CHAT_MODEL,
                             help = "Chat model used to answer the question.")
    query_group.add_argument("--question", type = str, default = DEFAULT_QUESTION,
                             help = "Question to ask.")
    query_group.add_argument("--k", type = int, default = DEFAULT_K,
                             help = "Chunks retrieved per generated query.")
 
    # --- chunking (see load_chunks docstring) ---
    chunk_group = parser.add_argument_group(
        "chunking",
        "Pass --rebuild if any of these change, or the stale embeddings are reused."
    )
    chunk_group.add_argument("--pdf", type = Path, default = DOC_PATH,
                             help = "PDF to query.")
    chunk_group.add_argument("--embedding-model", type = str, default = DEFAULT_EMBEDDING_MODEL,
                                 help = "Embedding model used to generate embeddings.")
    # "none" maps to Python None in chunk_params_from_args(); argparse
    # choices cannot express None directly.
    chunk_group.add_argument("--chunking-strategy",
                             choices = ["by_title", "basic", "none"],
                             default = DEFAULT_CHUNKING_STRATEGY,
                             help = "How elements are recombined into chunks.")
    chunk_group.add_argument("--max-characters", type = int,
                             default = DEFAULT_MAX_CHARACTERS,
                             help = "Hard ceiling on chunk size.")
    chunk_group.add_argument("--new-after-n-chars", type = int,
                             default = DEFAULT_NEW_AFTER_N_CHARS,
                             help = "Soft chunk-size target.")
    chunk_group.add_argument("--combine-text-under-n-chars", type = int,
                             default = DEFAULT_COMBINE_TEXT_UNDER_N_CHARS,
                             help = "Merge sections below this size (by_title only).")
    chunk_group.add_argument("--no-multipage-sections", action = "store_false",
                             dest = "multipage_sections",
                             default = DEFAULT_MULTIPAGE_SECTIONS,
                             help = "Prevent a section from spanning a page break (by_title only).")
    chunk_group.add_argument("--overlap", type = int, default = DEFAULT_OVERLAP,
                             help = "Characters of context carried into the next chunk.")
    chunk_group.add_argument("--overlap-all", action = "store_true",
                             default = DEFAULT_OVERLAP_ALL,
                             help = "Apply --overlap between all chunks, not just split ones.")
 
    # --- run behaviour ---
    run_group = parser.add_argument_group("run")
    run_group.add_argument("--rebuild", action = "store_true",
                           help = "Discard cached embeddings and re-embed.")
    run_group.add_argument("--verbose", action = "store_true",
                           help = "Log the queries MultiQueryRetriever generates.")
 
    return parser.parse_args()

def chunk_params_from_args(args: argparse.Namespace) -> dict:
    """
    Collect the load_chunks keyword arguments off the argparse namespace.
 
    argparse turns --max-characters into args.max_characters.
    """
    strategy = args.chunking_strategy
    if strategy == "none":
        # argparse choices cannot express None, so "none" is the sentinel.
        strategy = None

    return {
        "chunking_strategy": strategy,
        "max_characters": args.max_characters,
        "new_after_n_chars": args.new_after_n_chars,
        "combine_text_under_n_chars": args.combine_text_under_n_chars,
        "multipage_sections": args.multipage_sections,
        "overlap": args.overlap,
        "overlap_all": args.overlap_all,
    }

def main() -> None:
    args = parse_args()

    configure_logging(verbose = args.verbose)
 
    try:
        ollama.pull(args.chat_model)
    except Exception as exc:
        # ollama raises connection errors that don't make the cause obvious.
        log.error("Could not reach Ollama (%s). Is `ollama serve` running?", exc)
        sys.exit(1)
 
    store = build_vector_db(
        path = args.pdf,
        embedding_model = args.embedding_model, 
        rebuild = args.rebuild,
        **chunk_params_from_args(args)
    )
    chain = build_chain(
        store, 
        k = args.k,
        chat_model = args.chat_model
    )
 
    log.info("\nQuestion: %s\n", args.question)

    # Pass the question as a plain string. RunnablePassthrough forwards
    # whatever it receives, so wrapping it in a tuple would send a tuple
    # through to the prompt.
    res = chain.invoke(args.question)
    if args.verbose:
        print("\n----------\n")
    print(res)


if __name__ == "__main__":
    main()