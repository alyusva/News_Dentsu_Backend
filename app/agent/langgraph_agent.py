"""
Agente LangGraph GRANULAR para obtener y filtrar noticias de IA y Marketing
Cada operación es un nodo independiente para máxima modularidad
"""

import os
import requests
import json
import re
from typing import List, Dict, Any, TypedDict
from datetime import datetime, timedelta
import logging
from urllib.parse import urlparse, parse_qs, urlencode, urlunparse

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END
from dotenv import load_dotenv

# Cargar variables de entorno
load_dotenv()

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AgentState(TypedDict):
    """Estado del agente LangGraph con granularidad máxima"""
    query: str
    filter_type: str
    raw_news: List[Dict]
    current_article_index: int
    current_article: Dict
    processed_articles: List[Dict]
    final_news: List[Dict]
    article_category: str
    is_duplicate: bool
    should_continue: bool
    error_message: str

class NewsAgent:
    """Agente granular para obtener y filtrar noticias usando LangGraph + OpenAI"""
    
    def __init__(self):
        self.news_api_key = os.getenv("NEWS_API_KEY")
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        
        if not self.news_api_key:
            raise ValueError("NEWS_API_KEY no encontrada en las variables de entorno")
        if not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY no encontrada en las variables de entorno")
        
        # Inicializar modelo OpenAI
        self.llm = ChatOpenAI(
            api_key=self.openai_api_key,
            model="gpt-4o-mini",  # Modelo más económico
            temperature=0.3
        )
        
        # Crear el grafo de LangGraph
        self.graph = self._create_langgraph()
        
        # Archivo para tracking de requests diarias
        self.requests_file = "daily_requests.json"
    
    def _get_today_key(self) -> str:
        """Obtener clave para el día actual"""
        return datetime.now().strftime("%Y-%m-%d")
    
    def _load_daily_requests(self) -> Dict[str, int]:
        """Cargar contador de requests diarias"""
        try:
            if os.path.exists(self.requests_file):
                with open(self.requests_file, 'r') as f:
                    return json.load(f)
            return {}
        except Exception as e:
            logger.warning(f"Error cargando requests diarias: {e}")
            return {}
    
    def _save_daily_requests(self, requests_data: Dict[str, int]):
        """Guardar contador de requests diarias"""
        try:
            with open(self.requests_file, 'w') as f:
                json.dump(requests_data, f, indent=2)
        except Exception as e:
            logger.error(f"Error guardando requests diarias: {e}")
    
    def _increment_daily_requests(self) -> int:
        """Incrementar y retornar contador de requests del día actual"""
        today = self._get_today_key()
        requests_data = self._load_daily_requests()
        
        # Limpiar requests de días anteriores (opcional, mantener solo últimos 7 días)
        cutoff_date = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")
        requests_data = {k: v for k, v in requests_data.items() if k >= cutoff_date}
        
        # Incrementar contador del día actual
        requests_data[today] = requests_data.get(today, 0) + 1
        self._save_daily_requests(requests_data)
        
        return requests_data[today]
    
    def _get_daily_requests_count(self) -> int:
        """Obtener número de requests realizadas hoy"""
        today = self._get_today_key()
        requests_data = self._load_daily_requests()
        return requests_data.get(today, 0)
    
    def _clean_url(self, url: str) -> str:
        """Limpiar URL removiendo parámetros UTM y tracking"""
        if not url:
            return ""
        
        try:
            parsed = urlparse(url)
            query_params = parse_qs(parsed.query)
            
            # Parámetros a remover (UTM y tracking)
            utm_params = ['utm_source', 'utm_medium', 'utm_campaign', 'utm_term', 'utm_content',
                         'fbclid', 'gclid', 'ref', 'source', 'medium', 'campaign']
            
            # Filtrar parámetros
            clean_params = {k: v for k, v in query_params.items() if k not in utm_params}
            
            # Reconstruir URL
            clean_query = urlencode(clean_params, doseq=True)
            clean_parsed = parsed._replace(query=clean_query)
            
            return urlunparse(clean_parsed)
        except:
            return url
    
    def _is_similar_title(self, title: str, seen_titles: set, threshold: float = 0.85) -> bool:
        """Verificar si el título es similar a alguno ya visto (Jaccard similarity)"""
        title_words = set(re.findall(r'\w+', title.lower()))
        
        for seen_title in seen_titles:
            seen_words = set(re.findall(r'\w+', seen_title.lower()))
            
            if title_words and seen_words:
                intersection = len(title_words.intersection(seen_words))
                union = len(title_words.union(seen_words))
                
                if union > 0:
                    similarity = intersection / union
                    if similarity >= threshold:
                        return True
        return False
    
    def _create_langgraph(self) -> StateGraph:
        """Crear el grafo de procesamiento GRANULAR con LangGraph"""
        
        # NODO 0: Verificar límite de requests diarias
        def check_daily_limit_node(state: AgentState) -> AgentState:
            """Verificar si podemos hacer requests hoy (límite: 100/día)"""
            logger.info("🔄 NODO 0: Verificando límite de requests diarias...")
            
            current_requests = self._get_daily_requests_count()
            daily_limit = 100
            
            if current_requests >= daily_limit:
                logger.error(f"🚫 LÍMITE ALCANZADO: {current_requests}/{daily_limit} requests hoy")
                state["raw_news"] = []
                state["error_message"] = f"Límite diario alcanzado ({current_requests}/{daily_limit})"
                return state
            else:
                logger.info(f"✅ LÍMITE OK: {current_requests}/{daily_limit} requests hoy")
                return state
        
        # NODO 1: Obtener noticias (una sola vez)
        def fetch_raw_news_node(state: AgentState) -> AgentState:
            """Obtener 100 noticias brutas de la API"""
            logger.info("🔄 NODO 1: Obteniendo noticias de NewsAPI...")
            
            # Si ya hay error de límite, no hacer request
            if state.get("error_message"):
                logger.error(f"❌ Saltando request: {state['error_message']}")
                return state
            
            query = state["query"]
            try:
                url = "https://newsapi.org/v2/everything"
                
                params = {
                    "q": query,
                    "language": "en", 
                    "sortBy": "publishedAt",
                    "pageSize": 100,  # ← MÁXIMO por request
                    "page": 1,        # ← SOLO página 1
                    "apiKey": self.news_api_key
                }
                
                logger.info(f"📡 LangGraph: Llamando a NewsAPI...")
                
                response = requests.get(url, params=params, timeout=15)
                
                # Solo incrementar contador si la request fue exitosa
                if response.status_code == 200:
                    current_requests = self._increment_daily_requests()
                    logger.info(f"✅ Request exitosa ({current_requests}/100 requests hoy)")
                
                # Manejo de errores
                if response.status_code == 426:
                    logger.error("💳 Plan gratuito agotado - necesita upgrade")
                    state["raw_news"] = []
                    return state
                elif response.status_code == 429:
                    logger.error("⏰ Rate limit - espera antes de la próxima request")
                    state["raw_news"] = []
                    return state
                elif response.status_code == 403:
                    logger.error("🔑 API Key inválida o bloqueada")
                    state["raw_news"] = []
                    return state
                
                response.raise_for_status()
                data = response.json()
                articles = data.get("articles", [])
                
                logger.info(f"✅ LangGraph: Obtenidos {len(articles)} artículos")
                
                state["raw_news"] = articles
                return state
                
            except Exception as e:
                logger.error(f"❌ Error API: {str(e)}")
                state["raw_news"] = []
                return state
        
        # NODO 2: Inicializar procesamiento
        def initialize_processing_node(state: AgentState) -> AgentState:
            """Inicializar variables para el procesamiento"""
            logger.info("🔄 NODO 2: Inicializando procesamiento...")
            state["current_article_index"] = 0
            state["processed_articles"] = []
            state["final_news"] = []
            state["should_continue"] = True
            return state
        
        # NODO 3: Seleccionar siguiente artículo
        def select_next_article_node(state: AgentState) -> AgentState:
            """Seleccionar el siguiente artículo para procesar"""
            raw_news = state.get("raw_news", [])
            index = state.get("current_article_index", 0)
            
            if index < len(raw_news):
                # Validación robusta contra None
                if raw_news[index] is not None and isinstance(raw_news[index], dict):
                    state["current_article"] = raw_news[index]
                else:
                    state["current_article"] = {}
                logger.info(f"🔄 NODO 3: Procesando artículo {index + 1}/{len(raw_news)}")
            else:
                state["current_article"] = {}
                state["should_continue"] = False
                logger.info("🔄 NODO 3: No hay más artículos")
            
            return state
        
        # NODO 4: Verificar categoría
        def check_category_node(state: AgentState) -> AgentState:
            """Verificar si el artículo pertenece a la categoría solicitada"""
            logger.info("🔄 NODO 4: Verificando categoría...")
            article = state.get("current_article", {})
            filter_type = state.get("filter_type", "both")
            
            if not article:
                state["article_category"] = "none"
                return state
            
            title = str(article.get("title", "") or "").lower()
            description = str(article.get("description", "") or "").lower()
            content = f"{title} {description}"
            
            # Palabras excluidas (noticias deportivas, entretenimiento, etc.)
            excluded_keywords = [
                "football", "soccer", "basketball", "tennis", "baseball", "golf", "sports",
                "athletic", "player", "team", "match", "game", "score", "league", "tournament",
                "championship", "fifa", "uefa", "nba", "nfl", "mlb", "olympics", "sport",
                "racing", "boxing", "wrestling", "swimming", "cycling", "running", "fitness"
            ]
            
            # Verificar exclusiones primero
            if any(keyword in content for keyword in excluded_keywords):
                state["article_category"] = "none"
                logger.info("🚫 Excluido: contiene palabras deportivas/entretenimiento")
                return state
            
            # Palabras clave específicas de IA (más restrictivas pero no tanto)
            ai_keywords = [
                "artificial intelligence", "machine learning", "deep learning", "neural network",
                "gpt", "llm", "language model", "chatgpt", "openai", "tensorflow", "pytorch",
                "computer vision", "natural language processing", "nlp", "generative ai",
                "ai model", "ai technology", "ai system", "ai platform", "ai solution",
                "automation", "algorithm", "data science", "predictive analytics",
                "cognitive computing", "ai development", "ai research", "ai startup",
                # Agregar más términos comunes pero específicos
                "ai", "artificial", "intelligent", "smart technology", "machine intelligence",
                "automated", "algorithmic", "tech innovation", "digital transformation"
            ]
            
            # Palabras clave específicas de Marketing (más restrictivas pero no tanto)
            marketing_keywords = [
                "digital marketing", "content marketing", "email marketing", "social media marketing",
                "marketing campaign", "advertising campaign", "brand strategy", "marketing automation",
                "conversion rate", "customer acquisition", "lead generation", "marketing roi",
                "programmatic advertising", "influencer marketing", "affiliate marketing",
                "marketing technology", "martech", "adtech", "marketing platform",
                "customer journey", "marketing analytics", "brand awareness", "marketing strategy",
                "performance marketing", "growth marketing", "marketing funnel",
                # Agregar más términos comunes
                "marketing", "advertising", "brand", "campaign", "digital strategy", "business growth",
                "customer engagement", "sales optimization", "media buying", "advertising technology"
            ]
            
            has_ai = any(keyword in content for keyword in ai_keywords)
            has_marketing = any(keyword in content for keyword in marketing_keywords)
            
            # Lógica más estricta
            if filter_type == "ai" and has_ai:
                state["article_category"] = "ai"
            elif filter_type == "marketing" and has_marketing:
                state["article_category"] = "marketing"
            elif filter_type == "both":
                if has_ai and has_marketing:
                    state["article_category"] = "both"
                elif has_ai:
                    state["article_category"] = "ai"
                elif has_marketing:
                    state["article_category"] = "marketing"
                else:
                    state["article_category"] = "none"
            else:
                state["article_category"] = "none"
            
            logger.info(f"🏷️ Categoría: {state['article_category']}")
            return state
        
        # NODO 5: Verificar duplicados
        def check_duplicate_node(state: AgentState) -> AgentState:
            """Verificar si el artículo es duplicado"""
            logger.info("🔄 NODO 5: Verificando duplicados...")
            article = state.get("current_article", {})
            processed_articles = state.get("processed_articles", [])
            
            if not article:
                state["is_duplicate"] = True
                return state
            
            url = article.get("url", "")
            title = str(article.get("title", "") or "").lower().strip()
            
            # Limpiar URL
            clean_url = self._clean_url(url)
            
            # Verificar URL duplicada
            for processed in processed_articles:
                processed_clean_url = self._clean_url(processed.get("url", ""))
                if clean_url and clean_url == processed_clean_url:
                    state["is_duplicate"] = True
                    logger.info("🚫 Duplicado por URL")
                    return state
            
            # Verificar título similar
            processed_titles = {p.get("title", "").lower().strip() for p in processed_articles}
            if self._is_similar_title(title, processed_titles):
                state["is_duplicate"] = True
                logger.info("🚫 Duplicado por título similar")
                return state
            
            state["is_duplicate"] = False
            logger.info("✅ No es duplicado")
            return state
        
        # NODO 6: Procesar artículo válido
        def process_valid_article_node(state: AgentState) -> AgentState:
            """Procesar un artículo que pasó todas las validaciones"""
            logger.info("🔄 NODO 6: Procesando artículo válido...")
            article = state.get("current_article", {})
            
            # Obtener fecha de publicación
            published_at = article.get("publishedAt", "")
            # Convertir fecha ISO a formato más legible si existe
            if published_at:
                try:
                    from datetime import datetime
                    dt = datetime.fromisoformat(published_at.replace('Z', '+00:00'))
                    formatted_date = dt.strftime('%Y-%m-%d')
                except:
                    formatted_date = published_at
            else:
                formatted_date = ""
            
            processed_article = {
                "title": article.get("title", ""),
                "description": article.get("description", "")[:200],
                "url": article.get("url", ""),
                "image": article.get("urlToImage") or "https://images.unsplash.com/photo-1488590528505-98d2b5aba04b?w=400&h=200&fit=crop",
                "category": state.get("article_category", "unknown"),
                "publishedAt": formatted_date
            }
            
            # Agregar a listas
            state["processed_articles"].append(processed_article)
            state["final_news"].append(processed_article)
            
            logger.info(f"✅ Artículo procesado. Total: {len(state['final_news'])}")
            return state
        
        # NODO 7: Incrementar índice
        def increment_index_node(state: AgentState) -> AgentState:
            """Incrementar el índice para el siguiente artículo"""
            state["current_article_index"] = state.get("current_article_index", 0) + 1
            return state
        
                # NODO 8: Verificar si necesitamos más
        def check_completion_node(state: AgentState) -> AgentState:
            """Verificar si hemos completado el objetivo"""
            final_count = len(state.get("final_news", []))
            raw_count = len(state.get("raw_news", []))
            current_index = state.get("current_article_index", 0)
            
            # Condiciones más flexibles para evitar bucles infinitos
            if final_count >= 12:  # Límite máximo: 12 artículos
                state["should_continue"] = False
                logger.info(f"🎯 NODO 8: Límite máximo alcanzado ({final_count} artículos)")
            elif final_count >= 6:  # Objetivo reducido: 6 artículos mínimo
                state["should_continue"] = False
                logger.info(f"🎯 NODO 8: Objetivo alcanzado con {final_count} artículos")
            elif current_index >= raw_count:
                state["should_continue"] = False
                logger.info(f"🎯 NODO 8: Procesados todos los artículos ({final_count} encontrados)")
            elif current_index >= 30:  # Límite de seguridad más bajo para evitar bucles
                state["should_continue"] = False
                logger.info(f"🎯 NODO 8: Límite de seguridad alcanzado ({final_count} artículos)")
            else:
                state["should_continue"] = True
                logger.info(f"🔄 NODO 8: Continuando... ({final_count}/6 artículos mínimo)")
            
            return state
        
        # NODO 9: Finalizar
        def finalize_results_node(state: AgentState) -> AgentState:
            """Finalizar y preparar resultados"""
            final_news = state.get("final_news", [])
            
            # Limitar a máximo 12
            final_news = final_news[:12]
            
            # Si tenemos menos de 3, agregar ejemplos (más permisivo)
            if len(final_news) < 3:
                sample_news = self._get_sample_news_by_filter(state.get("filter_type", "both"))
                needed = max(6 - len(final_news), 0)  # Completar hasta 6
                final_news.extend(sample_news[:needed])
                logger.info(f"🔄 Agregadas {needed} noticias de ejemplo para completar")
            
            state["final_news"] = final_news
            total_requests = self._get_daily_requests_count()
            logger.info(f"🏁 FINALIZADO: {len(final_news)} noticias (reales + ejemplos) - {total_requests}/100 requests hoy")
            return state
        
        # Crear el grafo
        workflow = StateGraph(AgentState)
        
        # Agregar todos los nodos
        workflow.add_node("check_daily_limit", check_daily_limit_node)
        workflow.add_node("fetch_raw_news", fetch_raw_news_node)
        workflow.add_node("initialize_processing", initialize_processing_node)
        workflow.add_node("select_next_article", select_next_article_node)
        workflow.add_node("check_category", check_category_node)
        workflow.add_node("check_duplicate", check_duplicate_node)
        workflow.add_node("process_valid_article", process_valid_article_node)
        workflow.add_node("increment_index", increment_index_node)
        workflow.add_node("check_completion", check_completion_node)
        workflow.add_node("finalize_results", finalize_results_node)
        
        # Flujo principal
        workflow.set_entry_point("check_daily_limit")
        workflow.add_edge("check_daily_limit", "fetch_raw_news")
        workflow.add_edge("fetch_raw_news", "initialize_processing")
        workflow.add_edge("initialize_processing", "select_next_article")
        
        # Después de seleccionar artículo, verificar categoría primero
        workflow.add_edge("select_next_article", "check_category")
        
        # Después de verificar categoría, verificar duplicados
        workflow.add_edge("check_category", "check_duplicate")
        
        # Función de decisión para procesar o saltar artículo
        def should_process_article(state: AgentState) -> str:
            """Decidir si procesar el artículo basado en categoría y duplicados"""
            category = state.get("article_category", "none")
            is_duplicate = state.get("is_duplicate", True)
            
            # Solo procesar si tiene la categoría correcta Y no es duplicado
            if category != "none" and not is_duplicate:
                return "process_valid_article"
            else:
                return "increment_index"
        
        # Función de decisión para continuar o finalizar
        def should_continue_processing(state: AgentState) -> str:
            """Decidir si continuar procesando más artículos"""
            should_continue = state.get("should_continue", False)
            
            if should_continue:
                return "select_next_article"
            else:
                return "finalize_results"
        
        # Edges condicionales (después de AMBAS verificaciones)
        workflow.add_conditional_edges(
            "check_duplicate",
            should_process_article,
            {
                "process_valid_article": "process_valid_article",
                "increment_index": "increment_index"
            }
        )
        
        workflow.add_edge("process_valid_article", "increment_index")
        workflow.add_edge("increment_index", "check_completion")
        
        workflow.add_conditional_edges(
            "check_completion",
            should_continue_processing,
            {
                "select_next_article": "select_next_article",
                "finalize_results": "finalize_results"
            }
        )
        
        workflow.add_edge("finalize_results", END)
        
        return workflow.compile()
    
    async def get_filtered_news(self, filter_type: str = "both") -> List[Dict[str, Any]]:
        """
        Método principal para obtener noticias filtradas usando el grafo granular
        """
        try:
            # Configurar consulta según el filtro
            if filter_type == "ai":
                query = "artificial intelligence OR machine learning OR AI OR deep learning OR neural network OR GPT OR ChatGPT"
            elif filter_type == "marketing":
                query = "digital marketing OR advertising OR social media marketing OR content marketing OR SEO OR campaign"
            else:  # both - Query más amplia para encontrar intersección
                query = "(artificial intelligence OR AI OR machine learning) AND (marketing OR advertising OR business OR campaign OR digital)"
            
            # Estado inicial
            initial_state = {
                "query": query,
                "filter_type": filter_type,
                "raw_news": [],
                "current_article_index": 0,
                "current_article": {},
                "processed_articles": [],
                "final_news": [],
                "article_category": "",
                "is_duplicate": False,
                "should_continue": True,
                "error_message": ""
            }
            
            logger.info(f"🚀 Iniciando LangGraph Agent para: {filter_type}")
            
            # VALIDACIÓN ROBUSTA: Envolver toda la ejecución en try-catch
            try:
                result = self.graph.invoke(initial_state, config={"recursion_limit": 200})
                
                final_news = result.get("final_news", [])
                logger.info(f"🎯 LangGraph Agent completado: {len(final_news)} noticias")
                
                return final_news
            except Exception as e:
                logger.error(f"❌ Error crítico en LangGraph: {str(e)}")
                return self._get_sample_news_by_filter(filter_type)
            
        except Exception as e:
            logger.error(f"Error en el agente de noticias: {str(e)}")
            return self._get_sample_news_by_filter(filter_type)
    
    def _get_sample_news_by_filter(self, filter_type: str) -> List[Dict[str, Any]]:
        """Obtener noticias de ejemplo específicas por filtro"""
        
        from datetime import datetime, timedelta
        today = datetime.now().strftime('%Y-%m-%d')
        yesterday = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
        
        ai_news = [
            {
                "title": "OpenAI Introduces Advanced GPT-4 Turbo with Enhanced Capabilities",
                "description": "New model features improved reasoning, longer context windows, and better instruction following for enterprise applications.",
                "url": "https://example.com/openai-gpt4-turbo",
                "image": "https://images.unsplash.com/photo-1677442136019-21780ecad995?w=400&h=200&fit=crop",
                "category": "ai",
                "publishedAt": today
            },
            {
                "title": "Google's Gemini Ultra Achieves Human-Level Performance on MMLU Benchmark", 
                "description": "Latest AI model demonstrates remarkable capabilities across diverse academic subjects and professional domains.",
                "url": "https://example.com/google-gemini-ultra",
                "image": "https://images.unsplash.com/photo-1488590528505-98d2b5aba04b?w=400&h=200&fit=crop",
                "category": "ai",
                "publishedAt": yesterday
            },
            {
                "title": "Microsoft Copilot Integration Transforms Workplace Productivity",
                "description": "AI assistant now embedded across Office suite, enabling natural language document creation and data analysis.",
                "url": "https://example.com/microsoft-copilot",
                "image": "https://images.unsplash.com/photo-1485827404703-89b55fcc595e?w=400&h=200&fit=crop",
                "category": "ai",
                "publishedAt": today
            }
        ]
        
        marketing_news = [
            {
                "title": "Programmatic Advertising Reaches $200B Milestone in 2024",
                "description": "Automated ad buying continues growth trajectory, driven by AI optimization and cross-platform integration.",
                "url": "https://example.com/programmatic-200b",
                "image": "https://images.unsplash.com/photo-1460925895917-afdab827c52f?w=400&h=200&fit=crop",
                "category": "marketing",
                "publishedAt": today
            },
            {
                "title": "Social Commerce Revenue Projected to Hit $1.2T by 2025",
                "description": "Integration of shopping features in social platforms drives unprecedented e-commerce growth rates.",
                "url": "https://example.com/social-commerce-1t",
                "image": "https://images.unsplash.com/photo-1556742049-0cfed4f6a45d?w=400&h=200&fit=crop",
                "category": "marketing",
                "publishedAt": yesterday
            },
            {
                "title": "Cookie-less Future: New Identity Solutions Gain Traction",
                "description": "Privacy-focused advertising technologies emerge as third-party cookies phase out across major browsers.",
                "url": "https://example.com/cookieless-future",
                "image": "https://images.unsplash.com/photo-1563013544-824ae1b704d3?w=400&h=200&fit=crop",
                "category": "marketing",
                "publishedAt": today
            }
        ]
        
        both_news = [
            {
                "title": "AI-Powered Personalization Drives 40% Increase in Marketing ROI",
                "description": "Machine learning algorithms revolutionize customer targeting, delivering unprecedented campaign performance metrics.",
                "url": "https://example.com/ai-personalization-roi",
                "image": "https://images.unsplash.com/photo-1551288049-bebda4e38f71?w=400&h=200&fit=crop",
                "category": "both",
                "publishedAt": today
            },
            {
                "title": "ChatGPT Integration Transforms Content Marketing Strategies",
                "description": "Brands leverage conversational AI for automated content creation, customer service, and lead generation.",
                "url": "https://example.com/chatgpt-content-marketing",
                "image": "https://images.unsplash.com/photo-1676299081847-824916de030a?w=400&h=200&fit=crop",
                "category": "both",
                "publishedAt": yesterday
            },
            {
                "title": "Computer Vision Technology Revolutionizes Retail Analytics",
                "description": "AI-powered visual recognition systems provide real-time insights into customer behavior and inventory optimization.",
                "url": "https://example.com/computer-vision-retail",
                "image": "https://images.unsplash.com/photo-1516110833967-0b5716ca1387?w=400&h=200&fit=crop",
                "category": "both",
                "publishedAt": today
            }
        ]
        
        if filter_type == "ai":
            return ai_news
        elif filter_type == "marketing":
            return marketing_news
        elif filter_type == "both":
            return both_news
        else:
            return ai_news + marketing_news + both_news
