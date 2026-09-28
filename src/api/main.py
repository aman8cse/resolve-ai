from fastapi import FastAPI

from src.agent import SupportAgent
from src.api.schemas import AnalyzeRequest, AnalyzeResponse


app = FastAPI(
    title="Hiver SDE Support Agent",
    description="AI-powered customer support analysis system",
    version="1.0.0"
)


print("Initializing Support Agent...")
agent = SupportAgent()


@app.get("/health")
def health():
    return {
        "status": "ok"
    }


@app.post("/analyze", response_model=AnalyzeResponse)
def analyze(request: AnalyzeRequest):

    result = agent.analyze(request.message)

    return result