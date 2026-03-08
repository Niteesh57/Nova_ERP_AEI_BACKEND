"""
chromadb_service.py ─ Store and retrieve employee image embeddings via AWS Bedrock
                       and ChromaDB PersistentClient.
"""
import os
import base64
import logging
import boto3
from typing import Optional

logger = logging.getLogger(__name__)

# ChromaDB persistence path (project_root/chroma_db/)
_CHROMA_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "chroma_db"
)
os.makedirs(_CHROMA_DIR, exist_ok=True)

_client = None
_collection = None


def _get_collection():
    """Lazily initialize ChromaDB collection."""
    global _client, _collection
    if _collection is None:
        try:
            import chromadb
            _client = chromadb.PersistentClient(path=_CHROMA_DIR)
            _collection = _client.get_or_create_collection(
                name="employee_images",
                metadata={"hnsw:space": "cosine"},
            )
            print(f"[ChromaDB] ✅ Collection ready at {_CHROMA_DIR}")
        except Exception as e:
            print(f"[ChromaDB] ❌ Failed to init: {e}")
            logger.error(f"[ChromaDB] Init error: {e}", exc_info=True)
            _collection = None
    return _collection


def _get_embedding(image_bytes: bytes) -> Optional[list]:
    """
    Call AWS Bedrock amazon.titan-embed-image-v1 to get image embedding.
    Returns a list[float] or None on failure.
    """
    from app.core.config import AWS_REGION
    try:
        client = boto3.client("bedrock-runtime", region_name=AWS_REGION)
        b64_image = base64.b64encode(image_bytes).decode("utf-8")

        import json
        body = json.dumps({
            "inputImage": b64_image,
            "embeddingConfig": {"outputEmbeddingLength": 1024},
        })

        response = client.invoke_model(
            modelId="amazon.titan-embed-image-v1",
            contentType="application/json",
            accept="application/json",
            body=body,
        )
        result = json.loads(response["body"].read())
        embedding = result.get("embedding")
        print(f"[ChromaDB] Got embedding of length {len(embedding) if embedding else 0}")
        return embedding
    except Exception as e:
        print(f"[ChromaDB] ❌ Embedding error: {e}")
        logger.error(f"[ChromaDB] Embedding error: {e}", exc_info=True)
        return None


def store_employee_embedding(
    employee_id: int,
    name: str,
    email: str,
    image_bytes: bytes,
) -> bool:
    """
    Generate image embedding and upsert into ChromaDB with name/email metadata.
    Returns True on success, False on failure.
    """
    collection = _get_collection()
    if collection is None:
        return False

    embedding = _get_embedding(image_bytes)
    if embedding is None:
        return False

    try:
        collection.upsert(
            ids=[str(employee_id)],
            embeddings=[embedding],
            metadatas=[{"name": name, "email": email}],
        )
        print(f"[ChromaDB] ✅ Stored embedding for employee_id={employee_id}")
        return True
    except Exception as e:
        print(f"[ChromaDB] ❌ Upsert failed: {e}")
        logger.error(f"[ChromaDB] Upsert error: {e}", exc_info=True)
        return False


def delete_employee_embedding(employee_id: int) -> bool:
    """Remove employee embedding from ChromaDB."""
    collection = _get_collection()
    if collection is None:
        return False
    try:
        collection.delete(ids=[str(employee_id)])
        print(f"[ChromaDB] ✅ Deleted embedding for employee_id={employee_id}")
        return True
    except Exception as e:
        print(f"[ChromaDB] ❌ Delete failed: {e}")
        logger.error(f"[ChromaDB] Delete error: {e}", exc_info=True)
        return False


def identify_face(face_image_bytes: bytes, threshold: float = 0.35) -> Optional[dict]:
    """
    Embed a face image and query ChromaDB to find the closest employee match.

    Args:
        face_image_bytes: JPEG bytes of the face crop.
        threshold: Maximum cosine distance to consider a match (0-1, lower = stricter).

    Returns:
        {"name": str, "email": str} if a match is found within the threshold,
        otherwise None (caller should treat this as "Unknown Person").
    """
    collection = _get_collection()
    if collection is None:
        return None

    embedding = _get_embedding(face_image_bytes)
    if embedding is None:
        return None

    try:
        results = collection.query(
            query_embeddings=[embedding],
            n_results=1,
            include=["metadatas", "distances"],
        )
        distances = results.get("distances", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]

        if not distances or not metadatas:
            print("[ChromaDB] ℹ️ No employees in DB to match against")
            return None

        best_distance = distances[0]
        best_meta = metadatas[0]

        print(f"[ChromaDB] Face match distance={best_distance:.4f} (threshold={threshold})")

        if best_distance <= threshold:
            print(f"[ChromaDB] ✅ Identified: {best_meta.get('name')} ({best_meta.get('email')})")
            return {"name": best_meta.get("name", "Unknown"), "email": best_meta.get("email")}
        else:
            print(f"[ChromaDB] ℹ️ No match — closest distance {best_distance:.4f} exceeds threshold {threshold}")
            return None

    except Exception as e:
        print(f"[ChromaDB] ❌ Query error: {e}")
        logger.error(f"[ChromaDB] Query error: {e}", exc_info=True)
        return None

