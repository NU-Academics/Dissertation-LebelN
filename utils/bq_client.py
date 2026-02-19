"""BigQuery client factory."""

from google.cloud import bigquery

PROJECT_ID = 'YOUR-PROJECT-ID-HERE'
DATASET = 'dissertation_lebel'


def get_client(project_id: str = PROJECT_ID) -> bigquery.Client:
    """Return an authenticated BigQuery client."""
    return bigquery.Client(project=project_id)


def table_ref(table_name: str, project_id: str = PROJECT_ID) -> str:
    """Return fully-qualified table reference for a cached table."""
    return f"`{project_id}.{DATASET}.{table_name}`"
