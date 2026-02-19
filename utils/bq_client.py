"""BigQuery client factory.

Reads GCP_PROJECT_ID from Colab Secrets (google.colab.userdata).
Falls back to a hardcoded placeholder when running outside Colab.
"""

from google.cloud import bigquery

DATASET = 'dissertation_lebel'


def _get_project_id() -> str:
    """Load project ID from Colab Secrets, with fallback."""
    try:
        from google.colab import userdata
        return userdata.get('GCP_PROJECT_ID')
    except (ImportError, ModuleNotFoundError):
        # Not running in Colab — return placeholder for local linting/testing
        return 'YOUR-PROJECT-ID-HERE'


PROJECT_ID = _get_project_id()


def get_client(project_id: str = PROJECT_ID) -> bigquery.Client:
    """Return an authenticated BigQuery client."""
    return bigquery.Client(project=project_id)


def table_ref(table_name: str, project_id: str = PROJECT_ID) -> str:
    """Return fully-qualified table reference for a cached table."""
    return f"`{project_id}.{DATASET}.{table_name}`"
