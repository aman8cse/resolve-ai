from pydantic import BaseModel


class AnalyzeRequest(BaseModel):
    message: str


class AnalyzeResponse(BaseModel):
    message: str
    intent: str
    reply: str
    decision: str
    reason: str
    cases: list