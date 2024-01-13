from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
import spacy

app = FastAPI(
    title="Complaint Prediction API",
    description="This API predicts potential medical diagnosis based on user complaints (in form of Dutch text input) using a trained Dutch spaCy model.",
    version="1.0.0"
)

# Define a request model
class TextRequest(BaseModel):
    text: str

# Load your trained spaCy model
nlp = spacy.load("complaint_prediction_spacy")

@app.post("/predict", summary="Predict Complaints")
async def predict(request: TextRequest):
    """
    Predicts the top 5 medical complaints based on the provided text.
    To give a complete picture, we choose top 5, since according to our EDA, there are maximum 5 labels.
    Developer is expected to select their own threshold.

    Args:
        request (TextRequest): The request payload containing the text.
    
    Returns:
        dict: A dictionary of top 5 predicted complaints with their confidence scores.
    """
    try:
        doc = nlp(request.text)
        top_n = 5  # Top 5 labels
        return dict(sorted([(label, round(score, 2)) for label, score in doc.cats.items()], 
                           key=lambda item: item[1], reverse=True)[:top_n])
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/", summary="API Health Check")
async def read_root():
    """
    A simple health check endpoint that verifies if the API is running.
    
    Returns:
        dict: A confirmation message that the API is operational.
    """
    return {"message": "API is working"}

