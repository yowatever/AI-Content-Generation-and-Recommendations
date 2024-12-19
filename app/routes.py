from flask import Flask, request, jsonify
from .models.generate_model import ContentGenerator
from .models.recommend_model import Recommender

app = Flask(__name__)
generator = ContentGenerator()
recommender = Recommender()

@app.route('/generate', methods=['POST'])
def generate():
    try:
        data = request.get_json()
        prompt = data.get('prompt')
        if not prompt:
            return jsonify({'error': 'No prompt provided'}), 400
            
        result = generator.generate(prompt)
        return jsonify({'generated_content': result})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/recommend', methods=['POST'])
def recommend():
    try:
        data = request.get_json()
        user_input = data.get('user_input')
        if not user_input:
            return jsonify({'error': 'No user input provided'}), 400
            
        result = recommender.recommend(user_input)
        return jsonify({'recommendations': result})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
