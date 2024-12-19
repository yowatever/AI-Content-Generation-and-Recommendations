import pytest
from app.routes import app
import json

@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

def test_generate_endpoint(client):
    response = client.post('/generate',
                         data=json.dumps({'prompt': 'Test prompt'}),
                         content_type='application/json')
    assert response.status_code == 200
    assert 'generated_content' in response.get_json()

def test_recommend_endpoint(client):
    response = client.post('/recommend',
                         data=json.dumps({'user_input': 'Test input'}),
                         content_type='application/json')
    assert response.status_code == 200
    assert 'recommendations' in response.get_json()

def test_generate_no_prompt(client):
    response = client.post('/generate',
                         data=json.dumps({}),
                         content_type='application/json')
    assert response.status_code == 400

def test_recommend_no_input(client):
    response = client.post('/recommend',
                         data=json.dumps({}),
                         content_type='application/json')
    assert response.status_code == 400
