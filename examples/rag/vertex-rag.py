import asyncio

import google.auth
import vertexai
from googleapiclient.discovery import build
from vertexai import rag

from fast_agent import FastAgent
from fast_agent.config import get_settings

# RAG quickstart: Required roles, Prepare your Google Cloud console, Run Vertex AI RAG Engine
# https://docs.cloud.google.com/vertex-ai/generative-ai/docs/rag-engine/rag-quickstart
#
# Vertex AI RAG Engine overview: Overview, Supported regions, ...
# https://docs.cloud.google.com/vertex-ai/generative-ai/docs/rag-engine/rag-overview
#
# [Install the Vertex AI SDK for Python
# https://docs.cloud.google.com/vertex-ai/docs/start/install-sdk
#
# Admin console
# https://console.cloud.google.com/vertex-ai/rag
# Create a RAG Corpus, Import Files, and Generate a response
# uv pip install google-api-python-client

# TODO(developer): Update PROJECT_ID, LOCATION fastagent.config.yaml
CONFIG_PATH = "fastagent.secrets.yaml"

# google:
#   vertex_ai:
#     enabled: true
#     project_id: strato-space-ai   # Your project
#     location: europe-west4        # Netherlands, use Vertex RAG supported regions

_settings = get_settings(CONFIG_PATH)
_vertex_ai = getattr(_settings.google, "vertex_ai", {}) if _settings.google else {}
PROJECT_ID = _vertex_ai.get("project_id")
LOCATION = _vertex_ai.get("location")

# Configure embedding model, for example "text-embedding-005".
EMBEDDING_MODEL = "text-embedding-005"
SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]

SAMPLE_DRIVE = "1J3ubtdkmFuWDjfW3_qT2Fhsdn2pbtv-8"

if not PROJECT_ID or not LOCATION:
    raise ValueError(
        "Missing google.vertex_ai.project_id/location in fastagent.secrets.yaml"
    )


def _drive_folder_name(folder_id: str) -> str:
    credentials, _ = google.auth.default(scopes=SCOPES)
    drive_service = build("drive", "v3", credentials=credentials)
    payload = (
        drive_service.files()
        .get(
            fileId=folder_id,
            fields="id,name,mimeType",
            supportsAllDrives=True,
        )
        .execute()
    )
    return payload["name"]


# Initialize Vertex AI API once per session
# us-central1/us-east4 require allowlist; default to a GA region.

_vertex_initialized = False


def _ensure_vertexai_init() -> None:
    global _vertex_initialized
    if not _vertex_initialized:
        vertexai.init(project=PROJECT_ID, location=LOCATION)
        _vertex_initialized = True


def _create_and_import_corpus(
    display_name: str,
    paths: list[str],
) -> rag.RagCorpus:

    embedding_model_config = rag.RagEmbeddingModelConfig(
        vertex_prediction_endpoint=rag.VertexPredictionEndpoint(
            publisher_model=f"publishers/google/models/{EMBEDDING_MODEL}"
        )
    )
    rag_corpus = rag.create_corpus(
        display_name=display_name,
        backend_config=rag.RagVectorDbConfig(
            rag_embedding_model_config=embedding_model_config
        ),
    )
    rag.import_files(
        rag_corpus.name,
        paths,
        # Optional
        transformation_config=rag.TransformationConfig(
            chunking_config=rag.ChunkingConfig(
                chunk_size=512,
                chunk_overlap=100,
            ),
        ),
        max_embedding_requests_per_min=1000,  # Optional
    )
    return rag_corpus


def mini_rag(query: str, drive_id: str, top_k: int) -> object:
    _ensure_vertexai_init()
    if not drive_id:
        raise ValueError("drive_id must be a non-empty Google Drive ID.")

    paths = [f"https://drive.google.com/drive/folders/{drive_id}"]
    folder_name = _drive_folder_name(drive_id)
    key = drive_id
    display_name = f"{folder_name} | {key}"

    existing_corpus = None
    for corpus in rag.list_corpora():
        if corpus.display_name and key in corpus.display_name:
            existing_corpus = corpus
            break
    if existing_corpus:
        rag_corpus = existing_corpus
    else:
        rag_corpus = _create_and_import_corpus(
            display_name,
            paths,
        )

    rag_retrieval_config = rag.RagRetrievalConfig(
        top_k=top_k,  # Optional
        filter=rag.Filter(vector_distance_threshold=0.5),  # Optional
    )
    return rag.retrieval_query(
        rag_resources=[
            rag.RagResource(
                rag_corpus=rag_corpus.name,
                # Optional: supply IDs from `rag.list_files()`.
                # rag_file_ids=["rag-file-1", "rag-file-2", ...],
            )
        ],
        text=query,
        rag_retrieval_config=rag_retrieval_config,
    )


fast = FastAgent("Google Vertex RAG - Index google drive id to RAG")


@fast.agent(
    name="vertex rag",
    function_tools=[mini_rag],
)
async def main():
    async with fast.run() as agent:
        result = await agent(
            f"Produce a short top 5 prioritized list about customer pain points. From RAG, select 50 relevant chunks about customer pain points. Deduplicate. Links: [name](<link>). Compact output. Drive ID: {SAMPLE_DRIVE}."
        )
        print(result)
        # await agent.interactive()


if __name__ == "__main__":
    asyncio.run(main())
