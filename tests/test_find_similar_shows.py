import pytest
import numpy as np
import sys
import os

# Add the root directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ShowSuggesterAI import cosine_similarity, find_similar_shows

@pytest.fixture
def mock_embeddings():
    """Mock embedding data for testing."""
    return {
        "Show A": np.array([1, 2, 3]),
        "Show B": np.array([4, 5, 6]),
        "Show C": np.array([7, 8, 9]),
        "Show D": np.array([10, 11, 12]),
        "Show E": np.array([13, 14, 15]),
        "Show F": np.array([16, 17, 18]),
        "Show G": np.array([19, 20, 21]),
    }

@pytest.fixture
def mock_average_vector():
    """Mock average vector."""
    return np.array([5, 5, 5])

def test_find_similar_shows(mock_embeddings, mock_average_vector):
    """Test finding the 5 most similar shows."""
    # Mock embeddings to skip loading from pickle
    similar_shows = find_similar_shows(mock_embeddings, mock_average_vector, input_shows=["Show A", "Show B"])
    
    # Check that the result excludes input shows
    for show in ["Show A", "Show B"]:
        assert show not in similar_shows
    
    # Check that 5 shows are returned
    assert len(similar_shows) == 5


def test_find_similar_shows_all_input_shows_in_embeddings(mock_embeddings, mock_average_vector):
    """Test when all input shows are in the embeddings."""
    similar_shows = find_similar_shows(mock_embeddings, mock_average_vector, input_shows=["Show A", "Show B"])
    assert "Show A" not in similar_shows
    assert "Show B" not in similar_shows
    assert len(similar_shows) == 5

def test_find_similar_shows_not_enough_shows(mock_embeddings, mock_average_vector):
    """Test when not enough shows are available to return 5."""
    # Remove shows to ensure fewer than 5 remain
    reduced_embeddings = {"Show A": mock_embeddings["Show A"], "Show B": mock_embeddings["Show B"]}
    similar_shows = find_similar_shows(reduced_embeddings, mock_average_vector, input_shows=[])
    assert len(similar_shows) == 2  # Only "Show A" and "Show B" remain

def test_find_similar_shows_empty_embeddings(mock_average_vector):
    """Test when the embeddings dictionary is empty."""
    similar_shows = find_similar_shows({}, mock_average_vector, input_shows=["Show A"])
    assert similar_shows == []

def test_cosine_similarity():
    """Test the cosine similarity function."""
    vector_a = np.array([1, 0, 0])
    vector_b = np.array([0, 1, 0])
    assert cosine_similarity(vector_a, vector_b) == pytest.approx(0, rel=1e-5)

    vector_c = np.array([1, 1, 0])
    vector_d = np.array([1, 1, 0])
    assert cosine_similarity(vector_c, vector_d) == pytest.approx(1, rel=1e-5)


def test_find_similar_shows_with_percentages(mock_embeddings, mock_average_vector):
    """Test that find_similar_shows returns percentages for the top 5 shows."""
    similar_shows = find_similar_shows(mock_embeddings, mock_average_vector, input_shows=["Show A"])
    
    # Check that each show has a percentage value
    for title, percentage in similar_shows:
        assert isinstance(title, str)
        assert isinstance(percentage, float)
        assert 0 <= percentage <= 100
