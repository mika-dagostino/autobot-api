from flask import Flask, jsonify
import random

app = Flask(__name__)

# Dummy responses
dummy_responses = [
		'Absolutely! For daily commuting, I recommend something like the Toyota Corolla or Honda Civic. Both are known for their reliability, excellent fuel efficiency, and low maintenance costs. Are you looking for a sedan or would you consider a compact SUV like the Honda HR-V or Toyota RAV4 for a bit more space?',
		'Great question! On average, maintenance for this model will run around $400–$600 per year for regular oil changes, tire rotations, and inspections. Of course, this can vary depending on your driving habits and location, but these cars are known for their low upkeep costs.',
		'This car gets about 30 miles per gallon in the city and 40 on the highway, so it’s very economical if you’re driving a lot. If fuel efficiency is a priority, we could also look at hybrid options like the Toyota Prius, which goes up to 50 miles per gallon combined.',
		'Yes, this model comes with the full suite of advanced safety features, including adaptive cruise control, lane-keeping assist, automatic emergency braking, and blind-spot monitoring. It’s earned a 5-star safety rating from the NHTSA, so you can feel confident behind the wheel.',
		'That’s a great comparison! This model offers slightly better fuel efficiency and a more spacious interior than [Competitor’s Model]. However, the competitor has a slightly more powerful engine. If performance is key for you, we might consider that one. Otherwise, this car is a fantastic choice for everyday practicality.',
		'We have a range of financing options to suit different budgets. For example, we’re offering 0.9% APR for up to 60 months on select models right now. Alternatively, we can explore lease options if that’s more convenient. Would you like me to break down the monthly payments for this car?',
		'Yes, this model has a towing capacity of up to 3,500 pounds when properly equipped with the towing package. If you’re planning to tow frequently, I’d recommend the all-wheel-drive version for better stability. What kind of trailer or load will you be towing?',
		'Certainly! This car comes with a 3-year/36,000-mile basic warranty and a 5-year/60,000-mile powertrain warranty. Plus, you get 24/7 roadside assistance for the first three years. If you’d like, we can also discuss extended warranty options for added peace of mind.',
		'Of course! Let me grab the keys, and we’ll set you up for a test drive. Do you have a specific route you’d like to try, or should I suggest one that includes city and highway driving?',
		'Yes, this car is a great option for families. It has a spacious interior, plenty of legroom in the back for kids or car seats, and a large cargo area for strollers or groceries. Plus, the safety features make it a very secure choice for family trips.',
	]

@app.route('/api/response', methods=['POST'])
def post_response():
    data = request.json

    user_input = data.get('prompt', 'No value provided')
    preference_vector = data.get('preferenceVector', [])
    chat_history = data.get('chatHistory', [])

	# Run the AI CODE here

	# TODO: Implement the AI code here to generate a response based on the user input, preference vector, and chat history

	# Get the response and updated preference vector
    updated_preference_vector = preference_vector
    response = dummy_responses[0]

    body = {
		'response': response,
		'preferenceVector': updated_preference_vector
	}
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)
