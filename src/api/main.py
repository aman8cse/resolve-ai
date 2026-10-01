from fastapi import FastAPI, HTTPException

from src.api.schemas import AnalyzeRequest, AnalyzeResponse
from src.api.service import AgentService


app = FastAPI(
    title="Resolve AI",
    description="AI-powered customer support resolution engine",
    version="1.0.0",
)

service = AgentService()


@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "resolve-ai"
    }


@app.post(
    "/analyze",
    response_model=AnalyzeResponse
)
def analyze(request: AnalyzeRequest):

    try:
        return service.analyze(request.message)

    except Exception as e:
        print(f"Agent error: {e}")

        raise HTTPException(
            status_code=500,
            detail="Unable to analyze the customer request."
        )