import copy
from flask import Flask, jsonify, request
import random

import numpy as np
from intentClassification import queryGPT, queryGPTCustom
from running_final import extract_entities
from ContextualSentimentAnalysis import absaList
from Final_Vector_Embedding.data_processing.vector_search import compare_all_params, compare_by_embedding, update_pref_dict_MM, encode_all_tokens
from recommender import get_recommendation

app = Flask(__name__)

def translate_keys(ner_dict):
    ref_dict = {"MAKE":"Make", "MODEL":"Model", "BODY":"Body", "EXTERIOR COLOR":"ExteriorColor", "INTERIOR COLOR":"InteriorColor", "TRANSMISSION":"Transmission", "MARKET CLASS":"MarketClass", "DRIVETRAIN":"Drivetrain", "ENGINE DESCRIPTION":"Engine_Description", "DOORS":"Doors", "ENGINE CYLINDERS":"EngineCylinders", "MILES":"Miles", "SELLING PRICE":"SellingPrice", "PASSENGER CAPACITY":"PassengerCapacity", "CITYMPG":"CityMPG", "HIGHWAYMPG":"HighwayMPG", "YEAR":"Year"}
    token_dict = {
        "Year":[],
        "Body":[],
        "Make":[],
        "Model":[],
        "Doors":[],
        "ExteriorColor":[],
        "InteriorColor":[],
        "EngineCylinders":[],
        "Transmission":[],
        "Miles":[],
        "SellingPrice":[],
        "MarketClass":[],
        "PassengerCapacity":[],
        "Drivetrain":[],
        "Engine_Description":[],
        "CityMPG":[],
        "HighwayMPG":[]
    }

    for param in ner_dict:
        token_dict[ref_dict[param]] = ner_dict[param]
    return token_dict


@app.route('/api/chat', methods=['POST'])
def post_response():
    data = request.json

    user_input = data.get('prompt', 'No value provided')
    old_dict = data.get('preferenceVector', [])

    for k in old_dict:
        old_dict[k] = np.array(old_dict[k])
    old_preference_vector = old_dict

    # print("OLD PREF",old_preference_vector)

    chat_history = data.get('chatHistory', [])

    response = "Sorry, I didn't understand that. Can you please rephrase your question?"

	# Run the AI CODE here

	# Step 1
    try:
        intent = queryGPT(user_input) # Either compare, recommend, or conversational
        print("the intent detected is",intent)
    except Exception as e:
        print(f"Error querying GPT: {e}")
        return jsonify({"error": "Failed to get GPT response"}), 500

    if (intent == 'recommend'):
		# Step 2: Get NER's code for extracting entities
        entities = extract_entities(user_input)
        # print("Entites of the NER",entities)

		# Step 3: Sentiment Analysis
        # saOut = absaList(user_input, entities)

        saOut  = translate_keys(entities)
        print("SA out",saOut)

        #Step 4: Encode saOut into pref vector
        encoded_pref = encode_all_tokens(saOut)
        # print("THE ENCODED PREF VEC",encoded_pref)
        updated_preference_vector22 = update_pref_dict_MM(encoded_pref,old_preference_vector)
        # print("UPDATED PREF VEC",updated_preference_vector22)

        updated_preference_vector = {}
        for i in updated_preference_vector22:
            updated_preference_vector[i] = updated_preference_vector22[i].copy()

		# Step 4: Encoder
        encoderOut = compare_by_embedding(updated_preference_vector22)

		# Step 5: Query GPT for response formatting
        two_best_rows = get_recommendation(encoderOut)
        # print(two_best_rows)
        response = queryGPTCustom(two_best_rows, "You are given the 2 best cars that match the user's preferences and the first row just describes what each column means. Respond in a professional manner as if you are a car salesperson, make it short and detailled and DO NOT go off topic. Format the response in a nice and concise manner such that the options are easily distinguishable by the end user.")

    else:
        # Step 2: Query GPT for response
        gpt_response = queryGPTCustom(
            user_input,
            "Respond in a professional manner as if you are a car salesperson. Make it short and detailed, and DO NOT go off topic."
        )

        response = gpt_response

        # Updated preference vector logic (ensure this is defined elsewhere)
        updated_preference_vector = old_preference_vector
    
    # print("aaaaaaaaaaaaaa",updated_preference_vector)
    for i in updated_preference_vector:
        if isinstance(updated_preference_vector[i], np.ndarray):
            updated_preference_vector[i] = updated_preference_vector[i].tolist()

    body = {
		'response': response,
		'preferenceVector': updated_preference_vector
	}
    return jsonify(body)

if __name__ == '__main__':
    app.run(debug=True)
