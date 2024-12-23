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
import requests
from PIL import Image
import time

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


def generate_embeddings(genres_with_descriptions, model="text-embedding-ada-002"):
    """Generate embeddings for a list of genres concatenated with  descriptions .

    Args:
        genres_with_descriptions (dict): Dictionary where the key is the title,
                                         and the value is the concatenated genres and description.
        model (str): OpenAI model to use for embeddings.

        dict: Dictionary where the key is the title and the value is the embedding vector.
    """
    embeddings = {}
    for title, genre_with_description in genres_with_descriptions.items():
        try:
            response: CreateEmbeddingResponse = client.embeddings.create(
                input=genre_with_description,
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


def calc_average_vector(tv_shows,embeddings):
    """
    Calculate the average embedding vector for the given TV show titles.

    Args:
        tv_shows (list): List of the matched TV shows as strings. 

    Returns:
        np.ndarray: The average embedding vector as a NumPy array.
                    Returns None if no valid embeddings are found.
    """
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

        # Remove the backticks and clean the response
        response_content = response_content.strip("```json").strip("```").strip()

        # Parse JSON content
        response_json = json.loads(response_content)  # Parse JSON content

        show1 = response_json.get("show1", {})
        show2 = response_json.get("show2", {})

        return show1, show2

    except Exception as e:
        print("Error calling OpenAI API:", e)
        return None, None
    
    
def generate_lightx_image(title, description, api_key):
    """
    Generate an image using the LightX image generator API.

    Args:
        title (str): The name of the generated tv show.
        description (str): The description of the generated tv show.
        api_key (str): The API key for authentication.

    Returns:
        The URL of the generated image.
    """
    prompt= f"Create a captivating TV show advertisement poster for a show named '{title}'. The show is about {description}. The poster should be visually striking, appealing, and reflective of the show's theme"

    # Step 1: Create an order with the prompt
    url = 'https://api.lightxeditor.com/external/api/v1/text2image'
    headers = {
        'Content-Type': 'application/json',
        'x-api-key': api_key
    }
    data = {
        "textPrompt": prompt
    }

    response = requests.post(url, headers=headers, json=data)
    if response.status_code != 200:
        print(f"Image generation request failed: {response.text}")
        return None

    response_json = response.json()
    order_id = response_json['body']['orderId']

    # Step 2: Poll the API for the status and retrieve the image URL
    status_url = 'https://api.lightxeditor.com/external/api/v1/order-status'
    for _ in range(5):  # Retry up to 5 times
        time.sleep(3)  # Wait 3 seconds between retries
        status_payload = {"orderId": order_id}
        status_response = requests.post(status_url, headers=headers, json=status_payload)

        if status_response.status_code != 200:
            print(f"Status check failed: {status_response.text}")
            continue

        status_json = status_response.json()
        status = status_json['body']['status']
        if status == "active":
            image_url = status_json['body']['output']
            return image_url
        
        elif status == "failed":
            print("Image generation failed.")
            return None

    print("Image generation timed out.")
    return None


def save_image_from_url(image_url, file_name):
    """
    Download and save an image from a URL to a local file as a PNG.

    Args:
        image_url (str): The URL of the image.
        file_name (str): The name of the file to save the image as (without extension).

    Returns:
        str: The file name if successful, None otherwise.
    """
    try:
        response = requests.get(image_url)
        response.raise_for_status()  # Raise an exception for HTTP errors

        # Save the file with a .png extension
        png_file_name = f"{file_name}.png"
        with open(png_file_name, 'wb') as file:
            file.write(response.content)
        
        return png_file_name  # Return the file name for further use
    except requests.RequestException as e:
        print(f"Failed to download image: {e}")
        return None
    except Exception as e:
        print(f"Error saving image: {e}")
        return None


def display_image(file_name):
    """
    Open and display an image using Pillow.

    Args:
        file_name (str): The path to the image file.
    """
    try:
        img = Image.open(file_name)
        img.show()  # Opens the image in the default image viewer
        
    except Exception as e:
        print(f"Error displaying image: {e}")


def load_or_generate_embeddings():
    """Load embeddings from a file or generate them from the CSV."""
    if os.path.exists(PICKLE_FILE):
        #print("Loading embeddings from pickle file...")
        return load_embeddings_from_pickle(PICKLE_FILE)
    
    #print("Generating embeddings from CSV file...")
    data = pd.read_csv(CSV_FILE)
    genres_with_descriptions = dict(zip(data["Title"], data["Genres"] + ". " + data["Description"]))
    embeddings = generate_embeddings(genres_with_descriptions)
    save_embeddings_to_pickle(embeddings, PICKLE_FILE)
    #print("Embeddings saved to pickle file.")
    return embeddings


def get_user_matched_shows(available_shows):
    """Prompt the user for shows and match them using fuzzy matching."""
    while True:
        user_input = input("\nWhich TV shows did you really like watching? Separate them by a comma. Make sure to enter more than 1 show: ")
        user_shows = [show.strip() for show in user_input.split(",")]

        if len(user_shows) < 2:
            print("\nPlease enter more than 1 show.\n")
            continue

        is_match, matched_shows = match_shows(user_shows, available_shows)
        if is_match:
            res = ', '.join(matched_shows)
            confirmation = input(f"\nMaking sure, do you mean {res}?(y/n): ")
            if confirmation.lower() == "y":
                print("\nGreat! Generating recommendations now…\n")
                return matched_shows
        
        print("\norry about that. Let's try again, please make sure to write the names of the TV shows correctly.\n")


def display_recommendations(similar_shows):
    """Print the list of recommended TV shows."""
    print("\nHere are the TV shows that I think you would love:\n")
    for title, percentage in similar_shows:
        print(f"{title}: {int(round(percentage))}% ")


def generate_and_display_new_shows(matched_shows, similar_shows, data):
    """Generate new shows and display their details."""
    show1, show2 = generate_new_shows(matched_shows, similar_shows, data)
    if not show1 or not show2:
        print("Error generating new shows.")
        return

    print("\nI have also created just for you two shows which I think you would love.\n")
    print(f"Show #1: '{show1['name']}' - {show1['description']}")
    print(f"Show #2: '{show2['name']}' - {show2['description']}")
    print("\nHere are also the 2 TV show ads. Hope you like them!\n")

    try:
        generated_url1 = generate_lightx_image(show1['name'], show1['description'], LIGHTX_API_KEY)
        generated_url2 = generate_lightx_image(show2['name'], show2['description'], LIGHTX_API_KEY)
        file_name1 = save_image_from_url(generated_url1, show1['name'])
        file_name2 = save_image_from_url(generated_url2, show2['name'])
        display_image(file_name1)
        display_image(file_name2)
    except Exception as e:
        print(f"Error displaying images: {e}")


def main():
    embeddings = load_or_generate_embeddings()
    data = pd.read_csv(CSV_FILE)
    available_shows = data["Title"].tolist()

    matched_shows = get_user_matched_shows(available_shows)
    average_vector = calc_average_vector(matched_shows, embeddings)

    if average_vector is None:
        print("Error: Unable to calculate average vector.")
        return

    similar_shows = find_similar_shows_fast(embeddings, average_vector, matched_shows)
    display_recommendations(similar_shows)
    generate_and_display_new_shows(matched_shows, similar_shows, data)

if __name__ == "__main__":
    main()
