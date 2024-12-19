import pytest
from app.routes import app

@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

def test_generate(client):
    response = client.post('/generate', json={'prompt': 'Hello'})
    assert response.status_code == 200
    assert 'generated_text' in response.get_json()

def test_recommend(client):
    response = client.post('/recommend', json={'input': 'AI'})
    assert response.status_code == 200
    assert 'recommendations' in response.get_json()
