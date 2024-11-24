import spacy

def extract_entities(text, model_path="model-best"):
    # Load your trained model
    nlp = spacy.load(model_path)

    targets = {}
    doc = nlp(text)
    
    # Iterate over the entities
    for ent in doc.ents:
        # Check if the entity label is already in the dictionary
        if ent.label_ in targets:
            # Check if already a list, if not, make it a list
            if not isinstance(targets[ent.label_], list):
                targets[ent.label_] = [targets[ent.label_]]
            # Append the new entity to the list
            targets[ent.label_].append(ent.text)
        else:
            # Add the entity text to the dictionary
            targets[ent.label_] = ent.text

    # Convert single-element lists to single values
    for key in targets:
        if isinstance(targets[key], list) and len(targets[key]) == 1:
            targets[key] = targets[key][0]

    return targets

# Optional: Test the function if the script is run directly
if __name__ == "__main__":
    text = "I am looking for a car with 25000 miles and a red and green exterior. It should have 4 doors and belong to small cars class."
    print(extract_entities(text))
