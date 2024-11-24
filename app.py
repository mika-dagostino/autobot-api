from flask import Flask, jsonify
import random
from intentClassification import queryGPT, queryGPTCustom
from running_final import extract_entities
from ContextualSentimentAnalysis import absaList
from Final_Vector_Embedding.data_processing.vector_search import compare_all_params
from recommender import get_recommendation

app = Flask(__name__)

@app.route('/api/response', methods=['POST'])
def post_response():
    data = request.json

    user_input = data.get('prompt', 'No value provided')
    preference_vector = data.get('preferenceVector', [])
    chat_history = data.get('chatHistory', [])

    response = "Sorry, I didn't understand that. Can you please rephrase your question?"

	# Run the AI CODE here

	# Step 1: Get Stevens Code for Gemini
    intent = queryGPT(user_input) # Either compare, recommend, or conversational

    if (intent == 'recommend'):
		# Step 2: Get NER's code for extracting entities
        entities = extract_entities(user_input)

		# Step 3: Sentiment Analysis
        saOut = absaList(user_input, entities)

		# Step 4: Encoder
        encoderOut = compare_all_params(saOut)

		# Step 5: Query GPT for response formatting
        two_best_rows = get_recommendation(encoderOut)
        response = queryGPTCustom(two_best_rows, "You are given the 2 best cars that match the user's preferences and the first row just describes what each column means. Respond in a professional manner as if you are a car salesperson, make it short and detailled and DO NOT go off topic. Format the response in a nice and concise manner such that the options are easily distinguishable by the end user.")

    else:
		# Step 2: Query GPT for response
        response = queryGPTCustom(user_input, "Respond in a professional manner as if you are a car salesperson, make it short and detailled and DO NOT go off topic")


	# Get the response and updated preference vector
    updated_preference_vector = preference_vector

    body = {
		'response': response,
		'preferenceVector': updated_preference_vector
	}
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)
