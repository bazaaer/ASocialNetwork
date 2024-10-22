from flask import Blueprint

# Create a blueprint
api_blueprint = Blueprint('api', __name__)

@api_blueprint.route('/api', methods=['GET'])
def index():
    return 'ASocialNetwork API is running!'

@api_blueprint.route('/api/post', methods=['POST'])
def create_post():
    return 'Create post'


from typing import List, Optional
import spacy
from collections import Counter

def extract_keywords(prompt: str, amount_of_keywords: int = 10) -> Optional[List[str]]:
    """
    Extract and rank the most important keywords from a given prompt using spaCy.
    
    Args:
    prompt (str): The input prompt.
    amount_of_keywords (int): The number of top keywords to return.
    
    Returns:
    Optional[List[str]]: A list of the most important keywords.
    """
    # Load spaCy's language model
    nlp = spacy.load('en_core_web_sm')
    
    # Process the text with spaCy
    doc = nlp(prompt)
    
    # Priority 1: Named Entities (e.g., places, people, organizations)
    entities = [ent.text.lower() for ent in doc.ents if ent.label_ in ['PERSON', 'ORG', 'GPE', 'LOC']]
    
    # Priority 2: Nouns and Proper Nouns
    noun_phrases = [
        token.text.lower() for token in doc
        if token.pos_ in ['NOUN', 'PROPN']  # Nouns and proper nouns
        and not token.is_stop  # Exclude stop words
        and token.is_alpha  # Only include alphabetic tokens
    ]
    
    # Priority 3: Adjectives describing important nouns (only those with a direct relationship)
    adjectives = [
        token.text.lower() for token in doc
        if token.pos_ == 'ADJ'  # Adjectives
        and not token.is_stop  # Exclude stop words
        and token.is_alpha  # Only include alphabetic tokens
        and any(child for child in token.children if child.dep_ == "amod" and child.pos_ == "NOUN")
    ]
    
    # Combine all terms (giving entities higher priority)
    all_keywords = entities + noun_phrases + adjectives
    
    # Use Counter to calculate frequency of each keyword
    keyword_freq = Counter(all_keywords)
    
    # Get the most common keywords based on importance
    most_common_keywords = [item[0] for item in keyword_freq.most_common(amount_of_keywords)]
    
    return most_common_keywords

# Test the function
prompt = ("On a sunny afternoon, a young boy wearing a red jacket plays joyfully with his small, "
          "fluffy dog in the park. The trees sway gently in the breeze, and a few birds chirp melodiously "
          "from the branches above. In the distance, an old man sits on a wooden bench, reading a newspaper. "
          "Nearby, children laugh and chase each other near a sparkling fountain, while a street musician "
          "plays a soothing melody on his guitar. The vibrant colors of the flowers and the calm atmosphere "
          "make it a perfect day to relax and enjoy nature.")

extracted_keywords = extract_keywords(prompt)
print(extracted_keywords)
