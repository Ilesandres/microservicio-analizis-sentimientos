# main.py

from fastapi import FastAPI
from pydantic import BaseModel
from typing import Dict, Any

from sentiment_service import analyze_review_sentiment

app = FastAPI(
    title="Microservicio de Análisis de Sentimientos",
    version="1.0.0",
    description="API mínima para evaluar el sentimiento de reseñas en español usando pysentimiento.",
    docs_url=None,
    redoc_url=None,
    openapi_url=None,
)


class Review(BaseModel):
    review_text: str
    
@app.post("/analyze-sentiment", response_model=Dict[str, Any])
async def analyze_sentiment(review: Review):
    result = analyze_review_sentiment(review.review_text)
    
    return result

@app.get("/health")
def health_check():
    return {"status": "ok", "service": "Sentiment Analyzer"}


if __name__ == "__main__":
    # Ejecutar con uvicorn cuando se lance directamente: `python main.py`
    import uvicorn

    # Nota: hemos desactivado la UI de documentación automática en la instancia
    # de FastAPI (docs_url/redoc_url/openapi_url = None), por lo que /docs y
    # /redoc no estarán disponibles.
    # Usamos reload=False para evitar problemas de recarga que importen
    # módulos en procesos hijos (que pueden forzar la carga de dependencias
    # pesadas como transformers tijdens el spawn).
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=False)