import os
import base64
import requests
from pathlib import Path

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document

CHROMA_PATH = "./chroma_db"
RESUME_PATH = "./data/zidan_resume.pdf"

GITHUB_REPOS = [
    "zidan18Ahd/research-rag",
    "zidan18Ahd/ML-Models-from-scratch",
    "zidan18Ahd/Sentiment-analysis",
]

EMBED_MODEL = "all-MiniLM-L6-v2"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

def _gh_headers() -> dict:
    tok = os.getenv("GITHUB_TOKEN", "")
    return {"Authorization": f"token {tok}"} if tok else {}

def fetch_readme(repo: str) -> str:
    url = f"https://api.github.com/repos/{repo}/readme"
    resp = requests.get(url, headers=_gh_headers(), timeout=10)
    if resp.status_code == 200:
        raw = resp.json().get("content", "")
        return base64.b64decode(raw).decode("utf-8", errors="ignore")
    print(f"Could not fetch README for {repo}: HTTP {resp.status_code}")
    return ""

def fetch_commits(repo: str, n: int = 15) -> str:
    url = f"https://api.github.com/repos/{repo}/commits?per_page={n}"
    resp = requests.get(url, headers=_gh_headers(), timeout=10)
    if resp.status_code == 200:
        lines = [c["commit"]["message"].split("\n")[0] for c in resp.json()]
        return "\n".join(f"- {l}" for l in lines)
    return ""

def fetch_tree(repo: str) -> str:
    url = f"https://api.github.com/repos/{repo}/contents"
    resp = requests.get(url, headers=_gh_headers(), timeout=10)
    if resp.status_code == 200:
        items = [f["name"] for f in resp.json()]
        return ", ".join(items)
    return ""

def load_all_documents() -> list[Document]:
    docs: list[Document] = []

    # Load resume PDF
    if Path(RESUME_PATH).exists():
        loader = PyPDFLoader(RESUME_PATH)
        pdf_docs = loader.load()
        for d in pdf_docs:
            d.metadata["source"] = "resume_pdf"
            d.metadata["type"] = "resume"
        docs.extend(pdf_docs)
        print(f"Loaded resume: {len(pdf_docs)} pages")
    else:
        print(f"Resume not found at {RESUME_PATH}. Put your PDF there.")

    # Load GitHub repos
    for repo in GITHUB_REPOS:
        repo_name = repo.split("/")[-1]
        print(f"Fetching {repo}...")

        readme = fetch_readme(repo)
        if readme:
            docs.append(Document(
                page_content=f"# README: {repo_name}\n\n{readme}",
                metadata={"source": f"github:{repo}", "type": "readme", "repo": repo_name}
            ))

        commits = fetch_commits(repo)
        if commits:
            docs.append(Document(
                page_content=f"# Recent commits in {repo_name}:\n{commits}",
                metadata={"source": f"github:{repo}", "type": "commits", "repo": repo_name}
            ))

        tree = fetch_tree(repo)
        if tree:
            docs.append(Document(
                page_content=f"# File structure of {repo_name}: {tree}",
                metadata={"source": f"github:{repo}", "type": "tree", "repo": repo_name}
            ))

    return docs

def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

def build_vectorstore(force_rebuild: bool = False) -> None:
    if Path(CHROMA_PATH).exists() and not force_rebuild:
        print("Chroma DB already exists. Use REBUILD=1 env var to force rebuild.")
        return

    print("Building vector store from scratch...")
    docs = load_all_documents()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(docs)
    print(f"Split into {len(chunks)} chunks")

    embeddings = get_embeddings()
    Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_PATH,
    )
    print(f"Chroma DB saved to {CHROMA_PATH}")

def get_vectorstore() -> Chroma:
    return Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=get_embeddings(),
    )

if __name__ == "__main__":
    import sys
    force = "--rebuild" in sys.argv
    build_vectorstore(force_rebuild=force)