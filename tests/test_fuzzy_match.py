import pytest
import pandas as pd
import sys
import os

# Add the root directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the file to be tested
from ShowSuggesterAI import match_shows, CSV_FILE

@pytest.fixture
def tv_shows():
    """Fixture to load real TV shows data from the CSV file."""
    return pd.read_csv(CSV_FILE)

def test_match_shows_all_correct(tv_shows):
    """Test when all user shows match data from the CSV file."""
    user_input = ["Breaking Bad", "Game of Thrones"]  # Known titles in CSV
    available_shows = tv_shows["Title"].tolist()
    is_match, matched_shows = match_shows(user_input, available_shows)
    assert is_match is True
    assert matched_shows == ["Breaking Bad", "Game of Thrones"]

def test_match_shows_partial_match(tv_shows):
    """Test when some user shows don't match data from the CSV file."""
    user_input = ["Breaking Bad", "Nonexistent Show1", "Nonexistent Show2"]  # Mix of valid and invalid
    available_shows = tv_shows["Title"].tolist()
    is_match, matched_shows = match_shows(user_input, available_shows)
    assert is_match is False
    assert matched_shows is None

def test_user_input_with_duplicates(tv_shows):
    """Test when the user enters the same show multiple times."""
    user_input = ["Breaking Bad", "Breaking Bad", "Game of Thrones"]  # Duplicate entries
    available_shows = tv_shows["Title"].tolist()
    is_match, matched_shows = match_shows(user_input, available_shows)
    assert is_match is True
    assert matched_shows == ["Breaking Bad", "Game of Thrones"]

def test_user_input_nonexistent_shows(tv_shows):
    """Test when the user enters shows that do not exist in the dataset."""
    user_input = ["Random Show 1", "Random Show 2"]  # Nonexistent shows
    available_shows = tv_shows["Title"].tolist()
    is_match, matched_shows = match_shows(user_input, available_shows)
    assert is_match is False
    assert matched_shows is None

def test_user_input_extra_whitespace(tv_shows):
    """Test when the user input has extra spaces or inconsistent capitalization."""
    user_input = ["  Breaking Bad  ", "  game OF thrones "]  # Extra spaces and casing
    available_shows = tv_shows["Title"].tolist()
    is_match, matched_shows = match_shows(user_input, available_shows)
    assert is_match is True
    assert matched_shows == ["Breaking Bad", "Game of Thrones"]


def test_fuzzy_matching_threshold(tv_shows):
    """Test fuzzy matching with a strict threshold."""
    user_input = ["gem of throns", " lupan", "witcher"]  # Slightly incorrect
    available_shows = tv_shows["Title"].tolist()
    is_match, matched_shows = match_shows(user_input, available_shows, threshold=10)
    assert is_match is True
    assert matched_shows == ["Game of Thrones", "Lupin", "The Witcher"]


def test_case_insensitivity(tv_shows):
    """Test when the user enters titles with different capitalization."""
    user_input = ["breaking bad", "GAME OF THRONES"]  # Different casing
    available_shows = tv_shows["Title"].tolist()
    is_match, matched_shows = match_shows(user_input, available_shows)
    assert is_match is True
    assert matched_shows == ["Breaking Bad", "Game of Thrones"]


def test_user_input_less_than_two(tv_shows):
    """Test when the user enters fewer than 2 shows."""
    user_input = ["Game of Thrones"]  
    available_shows = tv_shows["Title"].tolist()
    is_match, matched_shows = match_shows(user_input, available_shows)
    assert is_match is False
    assert matched_shows is None

def test_user_input_is_empty(tv_shows):
    """Test when the userdoesn't enter any show."""
    user_input = []  
    available_shows = tv_shows["Title"].tolist()
    is_match, matched_shows = match_shows(user_input, available_shows)
    assert is_match is False
    assert matched_shows is None





