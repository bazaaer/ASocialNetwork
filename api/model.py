from flask import Blueprint
from flask import Flask, request, jsonify

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


@api_blueprint.route('/api/advise', methods=['POST'])
def check_for_democraty():
    """
    Evaluates whether a given text aligns with the values of a militarized form of democracy found in the Helldivers 2 universe.

    The function uses a pre-trained language model (Meta LLaMA) to generate an analysis of whether the input text supports the democratic ideals of the Helldivers 2 universe, which prioritize unity, loyalty, sacrifice for the greater good, and support for authority and hierarchy over individual freedoms. The model will determine if the input text is "Democratic Enough" and provide suggestions for improvement if not.

    Parameters:
    ----------
    prompt : str
        The input text to be evaluated for its alignment with the democratic ideals of the Helldivers 2 universe.
    max_new_token : int, optional
        The maximum number of tokens the model should generate for the response. Defaults to 1000.

    Returns:
    -------
    str
        The model's analysis of whether the text is democratic enough, along with any necessary suggestions to improve its alignment with the Helldivers 2 democratic values.
    """
    import torch
    from transformers import pipeline

    # Get prompt from POST
    try:
        prompt = request.json['prompt']
    except:
        prompt = "The text is missing."
    try:
        int(max_new_token = request.json['max_new_token'])
    except:
        max_new_token = 1000

    # Check if CUDA is available
    if torch.cuda.is_available():
        device = 0  # Use GPU (cuda:0)
    else:
        device = -1  # Use CPU if GPU is not available

    model_id = "meta-llama/Llama-3.2-3B-Instruct"

    # Initialize the pipeline with GPU support (if available)
    pipe = pipeline(
        "text-generation",
        model=model_id,
        torch_dtype=torch.bfloat16,  # Use bfloat16 for better memory efficiency on GPU
        device_map="auto",  # Automatically map to available devices
    )

    messages = [
        {
            "role": "system",
            "content": """You are an expert in evaluating whether a given text aligns with the strict, militarized form of democracy found in the Helldivers 2 universe. 
            In this universe, democracy is centered around unity, loyalty, and the greater good, often prioritizing collective goals over individual freedoms. 
            Your task is to analyze any text and determine if it promotes the values of Helldivers 2's democracy, which include:
            
            1. Emphasizing unity and loyalty to the collective above all else.
            2. Promoting sacrifice for the greater good, where personal desires and individualism are secondary to the mission or democracy.
            3. Reinforcing duty to maintain order, discipline, and service to the democratic cause.
            4. Supporting the idea of a militaristic democracy where authority and hierarchy ensure collective success.
            5. Condemning dissent or individualism if it threatens the unity or mission of the democracy.
            
            For each text, provide:
            1. A detailed analysis explaining whether the text aligns with these values.
            2. A final judgment of 'Democratic Enough' or 'Not Democratic Enough'.
            3. If the text is 'Not Democratic Enough', suggest specific improvements or rephrasing to make it more aligned with the Helldivers 2 democratic ideals.
            """
        },
        {
            "role": "user", 
            "content": f"{prompt}"
        }
    ]

    # Generate text
    outputs = pipe(
        messages,
        max_new_tokens=max_new_token,
    )

    # Output the generated pirate-speak response
    response = outputs[0]["generated_text"][-1]['content']
    return jsonify(response)