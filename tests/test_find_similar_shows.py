import pytest
import numpy as np
import sys
import os

# Add the root directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ShowSuggesterAI import load_embeddings_from_pickle, get_embedding_vectors_and_calc_average_vector, find_similar_shows, PICKLE_FILE

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
