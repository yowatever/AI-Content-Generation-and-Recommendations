from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/generate', methods=['POST'])
def generate():
    data = request.get_json()
    prompt = data.get('prompt', '')
    response = {"generated_text": f"Generated content for prompt: {prompt}"}
    return jsonify(response)

@app.route('/recommend', methods=['POST'])
def recommend():
    data = request.get_json()
    user_input = data.get('input', '')
    response = {"recommendations": [f"Recommendation for {user_input}"]}
    return jsonify(response)
