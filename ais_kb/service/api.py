from fastapi import FastAPI
from typing import Optional, List
from ais_kb.storage.db import ProductionKnowledgeStore
from ais_kb.core.schema import MechanisticClaim

app = FastAPI(title="AIS-KB Production Service")

@app.post("/ingest")
def ingest_claims(claims: List[MechanisticClaim], db_path: str = ":memory:"):
    store = ProductionKnowledgeStore(db_path=db_path)
    result = store.ingest_claims(claims)
    return {"status": "ingested", **result}

@app.get("/query")
def query_claims(source_id: Optional[str] = None, target_id: Optional[str] = None, model_family: Optional[str] = None, min_belief: float = 0.5):
    store = ProductionKnowledgeStore()
    claims = store.query_claims(source_id, target_id, model_family, min_belief)
    return {"count": len(claims), "claims": [c.model_dump() for c in claims]}

@app.get("/contradictions")
def contradictions():
    store = ProductionKnowledgeStore()
    return store.get_contradictions_and_tweaks()
