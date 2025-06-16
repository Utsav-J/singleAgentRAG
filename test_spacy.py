import spacy

def test_spacy_model():
    try:
        # Load the model
        nlp = spacy.load("en_core_web_lg")
        
        # Test with a simple sentence
        doc = nlp("This is a test sentence to verify the spaCy model installation.")
        
        # Print some basic information
        print("Model loaded successfully!")
        print("\nTokenization test:")
        for token in doc:
            print(f"Token: {token.text}, POS: {token.pos_}, Lemma: {token.lemma_}")
            
        return True
    except Exception as e:
        print(f"Error testing spaCy model: {e}")
        return False

if __name__ == "__main__":
    test_spacy_model() 