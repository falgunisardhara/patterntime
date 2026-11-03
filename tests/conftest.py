import pytest

from patterntime.storage import DatabaseStorage


@pytest.fixture
def memory_storage():
    """Return an isolated in-memory DatabaseStorage instance."""
    return DatabaseStorage(
        db_url="sqlite:///:memory:",
        auto_create_tables=True,
    )