import sys
sys.path.insert(0, '..')  # Adds the parent directory to the Python path
from deliverable.main import app, nlp  # Adjust the import based on your structure

from fastapi.testclient import TestClient

client = TestClient(app)

def test_read_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "API is working"}

def test_model_loading():
    assert nlp is not None
    assert "textcat_multilabel" in nlp.pipe_names

def test_predict():
    response = client.post("/predict", json={"text": "koorts en hoofdpijn"})
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, dict)
    assert len(data) == 5  # since we expect top 5 predictions