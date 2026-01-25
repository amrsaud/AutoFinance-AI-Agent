from langchain_core.tools import tool
import os
import requests
import logging

logger = logging.getLogger(__name__)


@tool
def retrieve_eligible_policies(
    min_income: float, employment_category: str
) -> list[dict]:
    """
    Retrieves eligible credit policies from DataRobot Vector Database based on user profile.

    Args:
        min_income: The user's monthly income in EGP.
        employment_category: The user's employment type (e.g., 'Freelancer (Tech)').

    Returns:
        List of matching policy documents with metadata (rate, tenure, dbr).
    """
    from config import Config

    config = Config()

    deployment_id = config.datarobot_vector_db_id
    if not deployment_id:
        return [{"error": "Vector DB ID not configured."}]

    # Clean URL logic
    base_url = os.getenv("DATAROBOT_ENDPOINT", "https://app.datarobot.com")
    if base_url.endswith("/api/v2"):
        endpoint = f"{base_url}/deployments/{deployment_id}/predictions"
    else:
        endpoint = f"{base_url}/api/v2/deployments/{deployment_id}/predictions"

    api_token = os.getenv("DATAROBOT_API_TOKEN")

    # Construct Query
    query = f"Credit policy for {employment_category} with income {min_income} EGP"

    # Payload with strict filtering
    payload = [
        {
            "promptText": query,
            "top_k": 5,  # Fetch top 5 to find best match
            "filters": {
                "min_income_egp": {"$lte": min_income},
                # We could filter by employment_category exact match if the metadata is strict
                # "employment_category": employment_category
            },
        }
    ]

    headers = {
        "Authorization": f"Bearer {api_token}",
        "Content-Type": "application/json",
    }

    try:
        logging.info(
            f"Querying Vector DB: {endpoint} with filters income<={min_income}"
        )
        response = requests.post(endpoint, json=payload, headers=headers)
        response.raise_for_status()

        result_row = response.json()["data"][0]
        extra = result_row.get("extraModelOutput", {})

        chunks = []

        # Robustly extract content from metadata or use default fields
        metadata = extra.get("metadata", {})

        if (
            isinstance(metadata, dict)
            and "content" in metadata
            and isinstance(metadata["content"], list)
        ):
            # Columnar format
            chunks = metadata["content"]
        elif isinstance(metadata, list):
            # Row format
            chunks = [item.get("content", str(item)) for item in metadata]
        else:
            # Fallback: maybe just return string representation of keys?
            # Or try top level citations?
            # Let's hope for content.
            pass

        if not chunks:
            # Try citations
            logging.warning(
                f"No content found in metadata. Full keys: {result_row.keys()}"
            )
            return []

        # Return list of strings (Raw Data)
        return chunks

    except Exception as e:
        logger.error(f"Vector DB Retrieval Failed: {e}")
        return []
