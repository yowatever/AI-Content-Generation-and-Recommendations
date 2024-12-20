from flask import Flask, request, jsonify
from app.models.generate_model import ContentGenerator
from app.models.recommend_model import Recommender

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
            
        recommendations = recommender.recommend(user_input)
        return jsonify({'recommendations': recommendations})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/add-knowledge', methods=['POST'])
def add_knowledge():
    try:
        data = request.get_json()
        text = data.get('text')
        if not text:
            return jsonify({'error': 'No text provided'}), 400
            
        generator.add_to_knowledge_base(text)
        return jsonify({'message': 'Knowledge base updated successfully'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/add-recommendation-items', methods=['POST'])
def add_recommendation_items():
    try:
        data = request.get_json()
        items = data.get('items')
        if not items:
            return jsonify({'error': 'No items provided'}), 400
            
        recommender.add_items(items)
        return jsonify({'message': 'Recommendation items added successfully'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
