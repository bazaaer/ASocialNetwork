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

def extract_keywords(prompt: str) -> Optional[List[str]]:
    
    nlp = spacy.load('en_core_web_sm')
    
    # Process the prompt with spaCy
    doc = nlp(prompt)
    
    # Filter out determiners, articles, and other irrelevant tokens
    keywords = [
        token.text for token in doc 
        if token.pos_ in ['NOUN', 'ADJ']  # Only nouns and adjectives
        and not token.is_stop  # Exclude stop words (like 'a', 'the', etc.)
        and token.is_alpha  # Only include alphabetic tokens
    ]
    
    # Return None if no keywords are found
    if not keywords:
        return None
    
    return keywords


# Test the function
prompt = "On a sunny afternoon, a young boy wearing a red jacket plays joyfully with his small, fluffy dog in the park. The trees sway gently in the breeze, and a few birds chirp melodiously from the branches above. In the distance, an old man sits on a wooden bench, reading a newspaper. Nearby, children laugh and chase each other near a sparkling fountain, while a street musician plays a soothing melody on his guitar. The vibrant colors of the flowers and the calm atmosphere make it a perfect day to relax and enjoy nature."
extracted_keywords = extract_keywords(prompt)
print(extracted_keywords)