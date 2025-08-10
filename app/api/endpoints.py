"""
Endpoints de la API para la plataforma de noticias
"""

from fastapi import APIRouter, HTTPException
from typing import Dict, List
import logging
import sys
import os

# Agregar el directorio padre al path para poder importar desde la raíz del proyecto
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Importar el agente LangGraph
from app.agent.langgraph_agent import NewsAgent
logger.info("✅ Usando agente LangGraph")
AGENT_TYPE = "langgraph"

router = APIRouter()

@router.get("/get-news")
async def get_news(filter_type: str = "both") -> Dict:
    """
    Obtiene noticias filtradas de IA y Marketing
    
    Args:
        filter_type: Tipo de filtro ('ai', 'marketing', 'both')
    
    Returns:
        JSON con las noticias filtradas
    """
    try:
        logger.info(f"Solicitando noticias con filtro: {filter_type}")
        
        # Crear instancia del agente
        agent = NewsAgent()
        
        # Ejecutar el agente para obtener noticias
        # Ambos agentes ahora usan solo filter_type
        news_data = await agent.get_filtered_news(filter_type.lower())
        
        # Filtrar por categoría específica (CORREGIDO: lógica más permisiva)
        if filter_type.lower() == "both":
            # Para "both": aceptar artículos de AI, Marketing o Both
            filtered_news = []
            for article in news_data:
                article_category = article.get("category", "")
                if article_category in ["ai", "marketing", "both"]:
                    filtered_news.append(article)
            news_data = filtered_news
        elif filter_type.lower() == "ai":
            # Para AI: aceptar artículos de AI o Both
            filtered_news = []
            for article in news_data:
                article_category = article.get("category", "")
                if article_category in ["ai", "both"]:
                    filtered_news.append(article)
            news_data = filtered_news
        elif filter_type.lower() == "marketing":
            # Para Marketing: aceptar artículos de Marketing o Both
            filtered_news = []
            for article in news_data:
                article_category = article.get("category", "")
                if article_category in ["marketing", "both"]:
                    filtered_news.append(article)
            news_data = filtered_news
        
        logger.info(f"Obtenidas {len(news_data)} noticias después del filtrado")
        
        return {
            "status": "success",
            "filter": filter_type,
            "count": len(news_data),
            "news": news_data
        }
        
    except Exception as e:
        logger.error(f"Error al obtener noticias: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error interno del servidor: {str(e)}"
        )

@router.get("/status")
async def api_status() -> Dict:
    """Verificar estado de la API"""
    return {
        "status": "active",
        "message": "API de noticias funcionando correctamente"
    }
