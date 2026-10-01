from pydantic import BaseModel, Field


class AnalyzeRequest(BaseModel):
    message: str = Field(
        ...,
        min_length=1,
        max_length=2000
    )


class RetrievedCase(BaseModel):
    similarity: float
    tweet_id: int
    customer: str
    response: str


class Latency(BaseModel):
    classifier_ms: float
    retrieval_ms: float
    generation_ms: float
    escalation_ms: float
    total_ms: float


class AnalyzeResponse(BaseModel):
    message: str
    intent: str
    cases: list[RetrievedCase]
    reply: str
    decision: str
    reason: str
    latency: Latency