# sentiment_service.py

# No importamos pysentimiento a nivel de módulo para evitar cargas pesadas
# (p. ej. transformers) cuando se importa este módulo. Inicializaremos el
# analizador de forma perezosa (lazy) en la primera petición.
sentiment_analyzer = None

def get_analyzer():
    """Devuelve una instancia singleton del analizador, inicializándolo
    en la primera llamada."""
    global sentiment_analyzer
    if sentiment_analyzer is None:
        from pysentimiento import create_analyzer

        # Inicializar el analizador (costoso) sólo cuando se necesite.
        sentiment_analyzer = create_analyzer(task="sentiment", lang="es")
    return sentiment_analyzer

def map_sentiment_to_stars(probabilities: dict, label: str) -> int:
    if label == 'NEU':
        return 3
    if label == 'POS':
        prob_pos = probabilities.get('POS', 0.0)
        if prob_pos > 0.90:
            return 5
        return 4
        
    if label == 'NEG':
        prob_neg = probabilities.get('NEG', 0.0)
        if prob_neg > 0.90:
            return 1
        return 2
        
    return 3


def analyze_review_sentiment(review_text: str) -> dict:
    # Importamos el preprocesador localmente para no cargarlo en la
    # importación del módulo principal.
    from pysentimiento.preprocessing import preprocess_tweet

    analyzer = get_analyzer()
    processed_text = preprocess_tweet(review_text)
    analysis_result = analyzer.predict(processed_text)
    sentiment_label = analysis_result.output
    sentiment_probabilities = analysis_result.probas
    star_rating = map_sentiment_to_stars(sentiment_probabilities, sentiment_label)
    
    result = {
        "review_data": {
            "text_original": review_text,
            "text_processed": processed_text
        },
        "sentiment_analysis": {
            "label": sentiment_label,  # Ej: "POS", "NEG", "NEU"
            "score_1_to_5": star_rating, # Ej: 5
            "probabilities": {
                "positive": round(sentiment_probabilities.get('POS', 0.0) * 100, 2), # %
                "negative": round(sentiment_probabilities.get('NEG', 0.0) * 100, 2), # %
                "neutral": round(sentiment_probabilities.get('NEU', 0.0) * 100, 2)  # %
            }
        },
        "summary": {
            "puntuacion_estrellas": f"{star_rating}/5 ⭐",
            "polaridad_completa": {
                "POS": "Positiva" if sentiment_label == 'POS' else "No Positiva",
                "NEG": "Negativa" if sentiment_label == 'NEG' else "No Negativa",
                "NEU": "Neutral" if sentiment_label == 'NEU' else "No Neutral"
            }
        }
    }
    
    return result