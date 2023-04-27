# Run with python <= 3.7
import spacy
import errant

def get_error_tags(orig, corr):
    # Load the English spaCy model
    nlp = spacy.load('en')
    # Initialize ERRANT Annotator
    annotator = errant.load('en', nlp)
    # Tokenize and parse the original and corrected sentences with spaCy
    original = annotator.parse(orig, tokenise=True)
    corrupted = annotator.parse(corr, tokenise=True)
    # Perform the ERRANT comparison
    edits = annotator.annotate(original, corrupted)
    # Extract the error tags
    error_tags = [(edit.type)[2:] for edit in edits]
    return error_tags

if __name__ == "__main__":
    while True:
        orig = input("Enter original sentence: ")
        corr = input("Enter corrected sentence: ")
        error_tags = get_error_tags(orig, corr)
        print("Error tags: " + str(error_tags))
