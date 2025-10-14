from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict, Any
import uvicorn
from pathlib import Path
import json

import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from attack_recommender import load_data, recommend_sequence

app = FastAPI(title="PitchSequence Recommender")


class Situation(BaseModel):
    risp: Optional[bool] = False
    outs: Optional[int] = 0
    late_inning: Optional[bool] = False


class RecommendRequest(BaseModel):
    batter: Optional[str]
    pitcher: Optional[str]
    count: Optional[str] = "0-0"
    situation: Optional[Situation] = Situation()
    seq_len: Optional[int] = 3


@app.on_event("startup")
def startup_event():
    global ARCHES, ARSENALS
    ARCHES, ARSENALS = load_data()


@app.post("/api/recommend")
def recommend(req: RecommendRequest):
    try:
        sit = req.situation.dict() if req.situation else {}
        res = recommend_sequence(ARCHES, ARSENALS, req.batter, req.pitcher, req.count, sit, seq_len=req.seq_len)
        return res
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    # local dev server
    uvicorn.run(app, host="127.0.0.1", port=8000)
