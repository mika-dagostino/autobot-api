from collections import defaultdict
import os
import faiss
import numpy as np
import pickle

import json

# Read the JSON file
with open(r'C:\Jeff_Documents\Programming\CodeJam14\autobot-api\Final_Vector_Embedding\creds.json', 'r') as file:
    data = json.load(file)
    CSV_DIR = data["vehicle_csv_path"]
    UNIQUE_VALUES_DIR = data["unique_values_path"]
    VECTOR_DATABASE_DIR = data["vector_database_path"]
    PARAMS = data["normal_columns"]
    DEFAULT_PREF_DICT = data["default_pref_dict"]



from sentence_transformers import SentenceTransformer

_model = None

def get_model():
    global _model
    if _model is None:
        # _model = SentenceTransformer(r"C:\Jeff_Documents\Programming\CodeJam14\Final_Vector_Embedding\fine_tuned_large_model")  # Load the model only once
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model

model = get_model()

def cosine_similarity(query, embeddings):
    query_norm = query / np.linalg.norm(query)
    embeddings_norm = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    return np.dot(embeddings_norm, query_norm)

def find_param(param_list_dir,param):

    for filename in os.listdir(param_list_dir):
        file_path = os.path.join(param_list_dir, filename)

        #filename = Body.txt
        if os.path.isfile(file_path):
            if filename[:-4] == param:
                with open(file_path, "r") as file:
                    loaded_list = file.read().split(",")
                return loaded_list


def encode_all_tokens(param_token:dict):

    incoming_preference_vector = DEFAULT_PREF_DICT
    for param in PARAMS:
        if param in param_token:
            query_vector = model.encode(param_token[param])  # Make sure it's a 2D array
            incoming_preference_vector[param] = query_vector
    return incoming_preference_vector

def normalize_vector(v):
    """Normalize a vector to have unit magnitude."""
    norm = np.linalg.norm(v)
    if norm == 0:  # Avoid division by zero
        return v
    return v / norm

def add_vectors_normalised(vNew,vOld):
    nv1 = normalize_vector(vNew)
    nv2 = normalize_vector(vOld)
    # Add the vectors (v1 + v1 + v2)
    result_vector = nv1 + nv1 + nv2

    # Re-normalize the resulting vector
    normalized_result_vector = normalize_vector(result_vector)

    
    # print("Normalized Sum Vector:", normalized_result_vector)
    # print("Magnitude:", np.linalg.norm(normalized_result_vector))  # Should be close to 1

    return normalized_result_vector

def update_pref_dict_MM(pref_dict_new,pref_dict_old):
    #Preference dictionary has the encodings
    current_pref = DEFAULT_PREF_DICT
    for param in PARAMS:
        if len(pref_dict_new[param]) == 384 and len(pref_dict_old[param]) == 384 :
            current_vector =  add_vectors_normalised(pref_dict_new[param],pref_dict_old[param])
            current_pref[param] = current_vector
        elif len(pref_dict_new[param]) == 384:
            current_pref[param] = pref_dict_new[param].copy()
        elif len(pref_dict_old[param]) == 384:
            current_pref[param] = pref_dict_old[param].copy()
        else:
            current_pref[param] = []
    return current_pref

def compare_one_param(encoded_query:str,param):

    index_path = r"C:\Jeff_Documents\Programming\CodeJam14\autobot-api\Final_Vector_Embedding\vector_database" + "\\" + param
    # print(index_path)
    if not os.path.exists(index_path):
        raise ValueError("Path to param does not exist param:" + str(param))

    with open(index_path, "rb") as file:  # "rb" mode reads in binary
        loaded_embedding = pickle.load(file)

    return cosine_similarity(encoded_query, loaded_embedding)
    

def disceern_similarites(param,similarities):
    param_list = find_param(UNIQUE_VALUES_DIR,param)
    #load Embedding Given a 
    best_match_index =  np.argsort(similarities)[-5:][::-1]
    # for i in best_match_index:
    #     print(f"Best match: Index {i}, Similarity {similarities[i]}, value matched {param_list[i]}")
    return param_list

def compare_all_params(token_dict):
    #Pref dict has the words
    current_rec_dict = DEFAULT_PREF_DICT

    for param in token_dict:
        encoded_query = model.encode(token_dict[param])
        similarity = compare_one_param(encoded_query,param)
        current_rec_dict[param] = disceern_similarites(param,similarity)[:2]

    return current_rec_dict


def compare_by_embedding(pref_dict):
    #Pref dict has the words
    current_rec_dict = DEFAULT_PREF_DICT

    for param in pref_dict:
        # print(pref_dict[param])
        if len(pref_dict[param]) == 384:
            encoded_query = pref_dict[param]
            similarity = compare_one_param(encoded_query,param)
            current_rec_dict[param] = disceern_similarites(param,similarity)[:2]

    return current_rec_dict

if __name__ == "__main__":
    example_dict = {"Year": "before 2019", "Make":"ok Honda", 
                    "PassengerCapacity": "ok seating 4 people",
                    "SellingPrice": "Less than 40000"}
    
    #Encode One Token Dictionary into a preference dictionary
    # encode_all_tokens()

    #Compare the new and old PREF vectors
    # update_pref_dict_MM()
    
    #Compare the token vector to all the parameter
    # print(compare_all_params(example_dict))

