"""
Simple local RAG pipeline over a PDF, using Ollama for both embeddings and chat.

Run `pip install -r Llama/rag-requirements.txt` to install the required packages.

Pipeline:

    1. Ingest a PDF file
    2. Extract text from PDF files and partition into small chunks
    3. Send the chunks to the embedding model
    4. Save the embeddings to a vector database
    5. Perform similarity search (via multi-query expansion) on the vector database to find similar documents
    6. Retrieve the similar chunks and feed them to the LLM to answer the question
"""

import logging
from pathlib import Path

from langchain_chroma import Chroma
from langchain_classic.retrievers.multi_query import MultiQueryRetriever
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_unstructured import UnstructuredLoader

import ollama

# pdfminer logs a "Cannot set non-stroke color" warning for every malformed
# color operator in the PDF. Harmless for text extraction, but it floods stdout.
logging.getLogger("pdfminer").setLevel(logging.ERROR)
logging.basicConfig(level = logging.INFO, format = "%(message)s")
log = logging.getLogger("simple-rag")


# ===== CONFIG ===== #

# Resolve relative to this file, not the current working directory, so the
# script works regardless of where you run it from.
BASE_DIR = Path(__file__).resolve().parent
DOC_PATH = BASE_DIR / "data" / "BOI.pdf"
PERSIST_DIR = BASE_DIR / "chroma_db"
print(BASE_DIR)
print(DOC_PATH)

CHAT_MODEL = "llama3.2:3b"
EMBEDDING_MODEL = "nomic-embed-text"
COLLECTION_NAME = "simple-rag"

QUESTION = "What are the main points as a business owner I should be aware of?"


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
        (the element type), `page_number`, and `source`.

    Raises
    ------
    FileNotFoundError
        If `path` does not exist.
    ValueError
        If no content could be extracted from the PDF.
    """
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {path}")

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


doc_path = "./Llama/data/BOI.pdf"
model = "llama3.2:3b"

# Local PDF file uploads
if doc_path:
    # loader = UnstructuredLoader(file_path = doc_path)
    loader = UnstructuredPDFLoader(file_path = doc_path)
    data = loader.load()
    print(f"File loading complete. {len(data)} elements extracted.")
    if data:
        # Preview first page
        content = data[0].page_content
        print(content[:100])
        # print(content)
else:
    print("Upload a PDF file.")

# ===== Extract text from PDF files and split into small chunks ===== #

from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma

# Split and chunk
text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1200, chunk_overlap = 300)
chunks = text_splitter.split_documents(data)
print("Document splitting complete.")

# print(f"Number of chunks: {len(chunks)}")
# print(f"Example chunk: {chunks[0]}")

# ===== Add to vector database ===== #

import ollama
embedding_model = "nomic-embed-text"

ollama.pull(embedding_model)
vector_db = Chroma.from_documents(
    documents = chunks,
    embedding = OllamaEmbeddings(model = embedding_model),
    collection_name = "simple-rag"
)
print("Embedded chunks added to vector databse.")

# ===== Retrieval ===== #

from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama
from langchain_core.runnables import RunnablePassthrough
from langchain_classic.retrievers.multi_query import MultiQueryRetriever

llm = ChatOllama(model = model)

# A simple technique to generate multiple questions from a single question and then retrieve documents
# based on those questions, getting the best of both worlds.
QUERY_PROMPT = PromptTemplate(
    input_variables = ["question"],
    template = """You are an AI language model assistant. Your task is to generate five
    different versions of the given user question to retrieve relevant documents from
    a vector database. By generating multiple perspectives on the user question, your
    goal is to help the user overcome some of the limitations of the distance-based
    similarity search. Provide these alternative questions separated by newlines.
    Original question: {question}""",
)

retriever = MultiQueryRetriever.from_llm(
    vector_db.as_retriever(), llm, prompt = QUERY_PROMPT
)

# RAG prompt
rag_template = """Answer the question based ONLY on the following context: {context}
Question: {question}
"""

chat_prompt = ChatPromptTemplate.from_template(rag_template)

chain = (
    {"context" : retriever, "question" : RunnablePassthrough()}
    | chat_prompt
    | llm
    | StrOutputParser()
)

# res = chain.invoke(input=("what is the document about?",))
res = chain.invoke(
    input = ("what are the main points as a business owner I should be aware of?",)
)
# res = chain.invoke(input=("how to report BOI?",))

print(res)