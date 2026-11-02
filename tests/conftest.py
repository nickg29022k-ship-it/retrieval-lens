import pytest

from rag_quality_evaluator.data import Document, load_corpus, load_qa


@pytest.fixture(scope="session")
def corpus():
    return load_corpus()


@pytest.fixture(scope="session")
def qa():
    return load_qa()


@pytest.fixture
def small_doc():
    return Document(
        doc_id="toy",
        title="Toy Policy",
        text=(
            "# Toy Policy\n\n## Leave\n\nStaff get 12 days of leave per year. Leave must be booked early.\n\n"
            "## Laptops\n\nLaptops are replaced every 4 years. Staff may pick a model.\n"
        ),
    )
