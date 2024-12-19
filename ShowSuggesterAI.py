from dotenv import load_dotenv
from openai import OpenAI
from openai.types import CreateEmbeddingResponse
import pandas as pd
import pickle
import os
from thefuzz import process
import numpy as np
from usearch.index import Index
import json



# Load API key from .env file
load_dotenv()
client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),  
)

# Retrieve the API key from environment variables
LIGHTX_API_KEY = os.getenv("LIGHTX_API_KEY")

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


def cosine_similarity(a, b):
    """Calculate the cosine similarity between two vectors."""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def find_similar_shows(embeddings, average_vector, input_shows):
    """
    Find the 5 most similar shows to the given average vector, excluding input shows.

    Args:
        embeddings (dict): Dictionary of {title: vector}.
        average_vector (np.ndarray): The average vector to compare against.
        input_shows (list): List of shows provided by the user.

    Returns:
        list: List of tuples containing titles and similarity percentages of the 5 most similar shows.
    """
    similarities = []
    
    for title, vector in embeddings.items():
        if title not in input_shows:  # Exclude input shows
            similarity = cosine_similarity(average_vector, vector)
            percentage = (similarity + 1) * 50  # Convert similarity (-1 to 1) to percentage (0 to 100)
            similarities.append((title, percentage))
    
    # Sort by similarity percentage in descending order and select top 5
    similarities.sort(key=lambda x: x[1], reverse=True)
    top_5_shows = similarities[:5]
    
    return top_5_shows

def build_index(embeddings):
    """
    Build a `usearch` index for fast similarity search.

    Args:
        embeddings (dict): Dictionary of {title: vector}.

    Returns:
        index: A `usearch` index populated with the embeddings.
        mapping: A dictionary mapping integer IDs to show titles.
    """

    vector_dim = len(next(iter(embeddings.values())))  # Get the vector dimension
    index = Index(ndim=vector_dim)  # Create the `usearch` index
    mapping = {}  # Map integer IDs to titles

    for idx, (title, vector) in enumerate(embeddings.items()):
        # Ensure the vector is a NumPy array
        vector = np.array(vector, dtype=np.float32)
        index.add(idx, vector)  # Add vector to the index
        mapping[idx] = title

    return index, mapping


def find_similar_shows_fast(embeddings, average_vector, input_shows, k=5):
    """
    Find the 5 most similar shows to the given average vector using `usearch`.

    Args:
        embeddings (dict): Dictionary of {title: vector}.
        average_vector (np.ndarray): The average vector to compare against.
        input_shows (list): List of shows provided by the user.
        k (int): Number of similar shows to return.

    Returns:
        list: List of tuples containing titles and similarity percentages of the most similar shows.
    """
    index, mapping = build_index(embeddings)

    # Search for top `k + len(input_shows)` similar vectors
    matches = index.search(average_vector, k + len(input_shows))

    # Filter out input shows and calculate percentages
    results = []
    for match in matches:
        title = mapping[match.key]
        if title not in input_shows:
            similarity_percentage = (1 - match.distance) * 100  # Convert distance to similarity percentage
            results.append((title, int(round(similarity_percentage))))

            # Stop after collecting `k` results
            if len(results) == k:
                break

    return results




def generate_new_shows(matched_shows, similar_shows, data):
    """
    Generate two new shows using OpenAI's GPT-4o mini based on input shows and recommendations.

    Args:
        matched_shows (list): List of matched input shows.
        similar_shows (list): List of recommended similar shows.
        data (pd.DataFrame): DataFrame containing the TV show titles and descriptions.

    Returns:
        tuple: Tuple containing two dictionaries, each with 'name' and 'description' keys.
    """
    # Prepare descriptions for matched and similar shows
    matched_descriptions = [
        f"{row['Title']}: {row['Description']}" for _, row in data.iterrows() if row['Title'] in matched_shows
    ]
    similar_descriptions = [
        f"{row['Title']}: {row['Description']}" for _, row in data.iterrows() if row['Title'] in [s[0] for s in similar_shows]
    ]

    print(f"matched desc:{matched_descriptions}\n")
    print(f"similar desc:{similar_descriptions}")

    # Add examples for clarity
    examples = """
    Examples:
    {
        "show1": {
            "name": "Empire of Shadows",
            "description": "A medieval power struggle full of intrigue and betrayal."
        },
        "show2": {
            "name": "Breaking Faith",
            "description": "A gritty crime drama about a detective walking a fine line between justice and vengeance."
        }
    }
    """

    # Create the prompt requesting JSON output
    prompt = f"""
    Based on the following TV shows and their descriptions:
    Matched shows: {', '.join(matched_descriptions)}
    Similar recommended shows: {', '.join(similar_descriptions)}

    Create two new TV shows in JSON format:
    {{
        "show1": {{
            "name": "Name of the first show",
            "description": "A one-line description of the first show"
        }},
        "show2": {{
            "name": "Name of the second show",
            "description": "A one-line description of the second show"
        }}
    }}

    {examples}
    """

    try:
        # Make the API call
        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "developer", "content": "You are a creative assistant."},
                {"role": "user", "content": prompt},
            ]
        )

        # Parse the JSON response
        response_content = completion.choices[0].message.content  # Correctly access the `content` attribute
        print(f"response before cleaning:\n{response_content}")

        # Remove the backticks and clean the response
        response_content = response_content.strip("```json").strip("```").strip()
        print(f"response after cleaning:\n{response_content}")

        # Parse JSON content
        response_json = json.loads(response_content)  # Parse JSON content

        show1 = response_json.get("show1", {})
        show2 = response_json.get("show2", {})

        return show1, show2

    except Exception as e:
        print("Error calling OpenAI API:", e)
        return None, None


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

    if average_vector is None:
        print("Error: Unable to calculate average vector.")
        return

    similar_shows= find_similar_shows_fast(embeddings, average_vector, matched_shows)    
    print("\nHere are the tv shows that I think you would love:")
    for title, percentage in similar_shows:
        print(f"{title}: {int(round(percentage))}% ")


    # Generate new shows using OpenAI
    show1, show2 = generate_new_shows(matched_shows, similar_shows, data)

    if show1 and show2:
        print("\nI have also created just for you two shows which I think you would love.")
        print(f"Show #1 is based on the fact that you loved the input shows that you gave me.")
        print(f"It's name is '{show1['name']}' and it is about {show1['description']}.")
        print(f"Show #2 is based on the shows that I recommended for you.")
        print(f"It's name is '{show2['name']}' and it is about {show2['description']}.")



if __name__ == "__main__":
    main()
