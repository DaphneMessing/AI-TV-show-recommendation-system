import pytest
import pandas as pd
import sys
import os

# Add the root directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the file to be tested
from ShowSuggesterAI import match_shows, CSV_FILE

@pytest.fixture
def real_tv_shows():
    """Fixture to load real TV shows data from the CSV file."""
    return pd.read_csv(CSV_FILE)

def test_match_shows_all_correct(tv_shows):
    """Test when all user shows match using data from ."""
    user_input = ["Breaking Bad", "Game of Thrones"]  # Known titles in CSV
    available_shows = real_tv_shows["Title"].tolist()
    is_match, matched_shows = match_shows(user_input, available_shows)
    assert is_match is True
    assert matched_shows == ["Breaking Bad", "Game of Thrones"]

def test_match_shows_partial_match(tv_shows):
    """Test when some user shows don't match using real data."""
    user_input = ["Breaking Bad", "Nonexistent Show"]  # Mix of valid and invalid
    available_shows = real_tv_shows["Title"].tolist()
    is_match, matched_shows = match_shows(user_input, available_shows)
    assert is_match is False
    assert matched_shows is None

