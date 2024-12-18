
from dotenv import load_dotenv
from openai import OpenAI
from openai.types import CreateEmbeddingResponse
import pandas as pd
import pickle
import os
from thefuzz import process
import numpy as np

# Load API key from .env file
load_dotenv()
client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),  
)

# File paths
CSV_FILE = "imdb_tvshows.csv"
PICKLE_FILE = "tvshow_embeddings.pkl"


def generate_embeddings(descriptions, model="text-embedding-ada-002"):
    """Generate embeddings for a list of descriptions."""
    embeddings = {}
    for title, description in descriptions.items():
        try:
            response: CreateEmbeddingResponse = client.embeddings.create(
                input=description,
                model=model
            )
            embeddings[title] = response.data[0].embedding
        except Exception as e:
            print(f"Error generating embedding for {title}: {e}")
    return embeddings

def save_embeddings_to_pickle(embeddings, file_path):
    """Save embeddings dictionary to a pickle file."""
    with open(file_path, "wb") as file:
        pickle.dump(embeddings, file)

def load_embeddings_from_pickle(file_path):
    """Load embeddings dictionary from a pickle file."""
    with open(file_path, "rb") as file:
        return pickle.load(file)


def match_shows(user_shows, available_shows, threshold=80):
    """
    Match user input shows to available shows using fuzzy matching.
    
    Args:
        user_shows (list): List of TV shows provided by the user.
        available_shows (list): List of TV shows from the CSV file.
        threshold (int): Minimum similarity score to consider a match.

    Returns:
        tuple: (bool, list or None)
            - True and matched show titles if all user inputs match.
            - False and None if any input doesn't meet the threshold.
    """
    matched_shows = []
   
    for show in user_shows:
        match, score = process.extractOne(show, available_shows)
        if score >= threshold :
            if match not in matched_shows:
                matched_shows.append(match)
        
    # Check if fewer than 2 matches were found
    if len(matched_shows) < 2:
        return False, None
        
    return True, matched_shows


def get_embedding_vectors_and_calc_average_vector(tv_shows,embeddings):
    """
    Load embedding vectors from the pickle file for the given TV show titles.
    Calculate the average embedding vector for the given TV show titles.

    Args:
        tv_shows (list): List of the matched TV shows as strings. 

    Returns:
        np.ndarray: The average embedding vector as a NumPy array.
                    Returns None if no valid embeddings are found.
    """
    # Load embeddings from the pickle file
    # embeddings = load_embeddings_from_pickle(PICKLE_FILE)

    # Collect the embedding vectors for the given titles
    vectors = []
    for title in tv_shows:
        if title in embeddings:
            vectors.append(embeddings[title])
        else:
            print(f"Warning: No embedding found for '{title}'. Skipping.")

    # Check if any vectors were retrieved
    if not vectors:
        print("No valid embedding vectors found for the given titles.")
        return None

    # Convert to a NumPy array and calculate the average vector
    average_vector = np.mean(np.array(vectors), axis=0)

    return average_vector    


def main():
        
    # Check if embeddings pickle file exists
    if os.path.exists(PICKLE_FILE):
        print("Loading embeddings from pickle file...")
        embeddings = load_embeddings_from_pickle(PICKLE_FILE)
    else:
        print("Generating embeddings from CSV file...")
        # Load TV show descriptions from CSV
        data = pd.read_csv(CSV_FILE)
        descriptions = dict(zip(data["Title"], data["Description"]))
        
        # Generate embeddings
        embeddings = generate_embeddings(descriptions)
        
        # Save embeddings to a pickle file
        save_embeddings_to_pickle(embeddings, PICKLE_FILE)
        print("Embeddings saved to pickle file.")
    
    # Example usage of embeddings
    print(f"Loaded embeddings for {len(embeddings)} TV shows.")

     # Load TV show data from CSV
    data = pd.read_csv(CSV_FILE)
    available_shows = data["Title"].tolist()  # List of available TV show titles
    matched_shows=[]

    while True:
        # Prompt user for input
        user_input = input("Which TV shows did you really like watching? Separate them by a comma. Make sure to enter more than 1 show: ")
        user_shows = [show.strip() for show in user_input.split(",")]

        # Validate input length
        if len(user_shows) < 2:
            print("Please enter more than 1 show.")
            continue

        # Match shows using fuzzy matching
        is_match, matched_shows = match_shows(user_shows, available_shows)
        
        if is_match:
            res= ', '.join(matched_shows)
            user_input=input(f"Making sure, do you mean {res}?(y/n)")
            if user_input == "y":
                print("Great! Generating recommendations now…")
                break
        
        print("Sorry about that. Let's try again, please make sure to write the names of the TV shows correctly.")

    average_vector= get_embedding_vectors_and_calc_average_vector(matched_shows,embeddings)

    if average_vector is not None:
        print(f"Average vector calculated with shape: {average_vector}")
    else:
        print("Failed to calculate the average vector.")

        

if __name__ == "__main__":
    main()
