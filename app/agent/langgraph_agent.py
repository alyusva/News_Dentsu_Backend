"""
Agente LangGraph PARALELO para obtener y filtrar noticias de IA y Marketing
Procesamiento en LOTES PARALELOS para máxima velocidad
"""

import os
import requests
import json
import re
import asyncio
from typing import List, Dict, Any, TypedDict
from datetime import datetime, timedelta
import logging
from urllib.parse import urlparse, parse_qs, urlencode, urlunparse
from concurrent.futures import ThreadPoolExecutor, as_completed

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
    """Estado del agente LangGraph con procesamiento paralelo"""
    query: str
    filter_type: str
    raw_news: List[Dict]
    current_batch: List[Dict]
    current_batch_index: int
    batch_size: int
    processed_articles: List[Dict]
    final_news: List[Dict]
    should_continue: bool
    error_message: str

class NewsAgent:
    """Agente paralelo para obtener y filtrar noticias usando LangGraph + OpenAI"""
    
    def __init__(self):
        self.news_api_key = os.getenv("NEWSAPI_KEY")
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        
        if not self.news_api_key:
            raise ValueError("NEWSAPI_KEY no encontrada en las variables de entorno")
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
        
        # Configuración de paralelización OPTIMIZADA para Cloud Run
        self.batch_size = 5  # Reducir de 10 a 5 artículos por lote
        self.max_workers = 3  # Reducir de 5 a 3 threads concurrentes
    
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
    
    def _is_similar_title(self, title: str, seen_titles: set, threshold: float = 0.60) -> bool:
        """Verificar si el título es similar a alguno ya visto (Jaccard similarity) - MUY ESTRICTO"""
        title_words = set(re.findall(r'\w+', title.lower()))
        
        for seen_title in seen_titles:
            seen_words = set(re.findall(r'\w+', seen_title.lower()))
            
            if title_words and seen_words:
                intersection = len(title_words.intersection(seen_words))
                union = len(title_words.union(seen_words))
                
                if union > 0:
                    similarity = intersection / union
                    # Umbral más bajo = MÁS estricto en detectar duplicados
                    if similarity >= threshold:
                        return True
        return False
    
    def _classify_by_keywords(self, title: str, description: str) -> str:
        """Clasificación de fallback usando palabras clave MÁS PERMISIVA"""
        # Asegurar que title y description son strings
        title = str(title or "")
        description = str(description or "")
        content = (title + " " + description).lower()
        
        # Palabras clave para AI - MÁS AMPLIAS Y PERMISIVAS
        ai_keywords = [
            # Términos directos de AI
            'artificial intelligence', 'machine learning', 'deep learning', 'neural network',
            'chatgpt', 'gpt', 'openai', 'llm', 'ai model', 'ai technology', 'ai system',
            'computer vision', 'natural language processing', 'nlp', 'automation',
            'ai development', 'ai research', 'generative ai', 'ai algorithm', 'ai software',
            # Términos relacionados con AI (más amplios)
            'ai-powered', 'ai powered', 'ai assistant', 'ai integration', 'copilot',
            'intelligent', 'smart technology', 'automated', 'algorithm', 'data science',
            'predictive', 'recommendation engine', 'voice assistant', 'chatbot',
            'tensorflow', 'pytorch', 'nvidia', 'gpu computing', 'ai startup'
        ]
        
        # Palabras clave para Marketing - MÁS AMPLIAS Y PERMISIVAS  
        marketing_keywords = [
            # Términos directos de marketing
            'digital marketing', 'content marketing', 'email marketing', 'social media marketing',
            'advertising campaign', 'brand strategy', 'marketing strategy', 'lead generation',
            'customer engagement', 'marketing analytics', 'conversion rate', 'marketing roi',
            'seo strategy', 'ppc advertising', 'marketing automation', 'influencer marketing',
            # Términos relacionados con marketing (más amplios)
            'advertising', 'campaign', 'branding', 'social media', 'instagram', 'facebook',
            'twitter', 'linkedin', 'tiktok', 'youtube', 'influencer', 'viral marketing',
            'customer acquisition', 'retention', 'loyalty program', 'personalization',
            'e-commerce', 'sales funnel', 'growth hacking', 'content creator',
            'affiliate marketing', 'programmatic', 'ad spend', 'ctr', 'cpm'
        ]
        
        # BUSCAR COINCIDENCIAS CON PALABRAS INDIVIDUALES TAMBIÉN
        ai_matches = 0
        marketing_matches = 0
        
        # Buscar términos completos
        for keyword in ai_keywords:
            if keyword in content:
                ai_matches += 2  # Peso mayor para términos completos
        
        for keyword in marketing_keywords:
            if keyword in content:
                marketing_matches += 2  # Peso mayor para términos completos
        
        # Buscar palabras individuales importantes (peso menor)
        ai_single_words = ['ai', 'ml', 'algorithm', 'intelligent', 'smart', 'automated', 'copilot', 'gpt']
        marketing_single_words = ['marketing', 'advertising', 'campaign', 'brand', 'social', 'seo', 'ads']
        
        for word in ai_single_words:
            if word in content.split():  # Buscar palabra completa, no substring
                ai_matches += 1
        
        for word in marketing_single_words:
            if word in content.split():  # Buscar palabra completa, no substring  
                marketing_matches += 1
        
        # CLASIFICACIÓN MÁS PERMISIVA
        if ai_matches >= 2 and marketing_matches >= 2:
            return "both"
        elif ai_matches >= 1:
            return "ai"
        elif marketing_matches >= 1:
            return "marketing"
        else:
            return "none"
    
    def _process_single_article(self, article: Dict, filter_type: str, processed_titles: set, processed_urls: set) -> Dict:
        """Procesar UN SOLO artículo (función para paralelización)"""
        try:
            if not article or not isinstance(article, dict):
                return {"status": "invalid", "article": None}
            
            title = str(article.get("title", "") or "")
            description = str(article.get("description", "") or "")
            url = article.get("url", "")
            
            # Validaciones básicas
            if len(title + description) < 10:
                return {"status": "too_short", "article": None}
            
            # Verificar duplicados
            clean_url = self._clean_url(url)
            if clean_url in processed_urls:
                return {"status": "duplicate_url", "article": None}
            
            if self._is_similar_title(title, processed_titles):
                return {"status": "duplicate_title", "article": None}
            
            # Clasificar con LLM
            try:
                classification_prompt = f"""Clasifica este artículo en una categoría:

TÍTULO: {title}
DESCRIPCIÓN: {description}

CATEGORÍAS:
- "ai": sobre inteligencia artificial, tecnología AI, machine learning, automatización
- "marketing": sobre marketing, publicidad, ventas, branding, social media
- "both": combina IA y marketing
- "none": otros temas

Responde solo: ai, marketing, both, o none"""

                messages = [
                    SystemMessage(content="Eres un clasificador de noticias experto. Analiza el contenido y responde solo con una palabra."),
                    HumanMessage(content=classification_prompt)
                ]
                
                response = self.llm.invoke(messages)
                llm_category = response.content.strip().lower()
                
                # Validar respuesta del LLM
                valid_categories = ["ai", "marketing", "both", "none"]
                if llm_category not in valid_categories:
                    llm_category = self._classify_by_keywords(title, description)
                
            except Exception as e:
                logger.warning(f"Error en clasificación LLM: {str(e)}")
                llm_category = self._classify_by_keywords(title, description)
            
            # Verificar si coincide con el filtro - LÓGICA ESPECÍFICA
            should_include = False
            if filter_type == "ai":
                # AI: solo artículos puros de AI + artículos mixtos
                should_include = llm_category in ["ai", "both"]
            elif filter_type == "marketing":
                # Marketing: SOLO artículos puros de marketing (no mixtos)
                should_include = llm_category == "marketing"
            elif filter_type == "both":
                # Both: artículos que específicamente combinen AI y Marketing
                should_include = llm_category == "both"
            
            if not should_include:
                return {"status": "filtered_out", "article": None}
            
            # Formatear fecha
            published_at = article.get("publishedAt", "")
            if published_at:
                try:
                    dt = datetime.fromisoformat(published_at.replace('Z', '+00:00'))
                    formatted_date = dt.strftime('%Y-%m-%d')
                except:
                    formatted_date = published_at
            else:
                formatted_date = ""
            
            processed_article = {
                "title": title,
                "description": description[:200],
                "url": url,
                "image": article.get("urlToImage") or "https://images.unsplash.com/photo-1488590528505-98d2b5aba04b?w=400&h=200&fit=crop",
                "category": llm_category,
                "publishedAt": formatted_date
            }
            
            return {"status": "processed", "article": processed_article}
            
        except Exception as e:
            logger.error(f"Error procesando artículo: {str(e)}")
            return {"status": "error", "article": None}
    
    def _create_langgraph(self) -> StateGraph:
        """Crear el grafo de procesamiento PARALELO con LangGraph"""
        
        # NODO 1: Obtener noticias
        def fetch_raw_news_node(state: AgentState) -> AgentState:
            """Obtener noticias brutas de la API"""
            logger.info("🔄 NODO 1: Obteniendo noticias de NewsAPI...")
            
            query = state["query"]
            try:
                url = "https://newsapi.org/v2/everything"
                params = {
                    "q": query,
                    "language": "en", 
                    "sortBy": "publishedAt",
                    "pageSize": 100,
                    "page": 1,
                    "apiKey": self.news_api_key
                }
                
                response = requests.get(url, params=params, timeout=15)
                
                if response.status_code == 200:
                    current_requests = self._increment_daily_requests()
                    logger.info(f"✅ Request exitosa ({current_requests}/100 requests hoy)")
                
                response.raise_for_status()
                data = response.json()
                articles = data.get("articles", [])
                
                logger.info(f"✅ Obtenidos {len(articles)} artículos")
                state["raw_news"] = articles
                return state
                
            except Exception as e:
                logger.error(f"❌ Error API: {str(e)}")
                state["raw_news"] = []
                return state
        
        # NODO 2: Inicializar procesamiento en lotes
        def initialize_batch_processing_node(state: AgentState) -> AgentState:
            """Inicializar el procesamiento en lotes"""
            logger.info("🔄 NODO 2: Inicializando procesamiento en lotes...")
            state["current_batch_index"] = 0
            state["batch_size"] = self.batch_size
            state["processed_articles"] = []
            state["final_news"] = []
            state["should_continue"] = True
            return state
        
        # NODO 3: Seleccionar siguiente lote
        def select_next_batch_node(state: AgentState) -> AgentState:
            """Seleccionar el siguiente lote de artículos para procesar"""
            raw_news = state.get("raw_news", [])
            batch_index = state.get("current_batch_index", 0)
            batch_size = state.get("batch_size", self.batch_size)
            
            start_idx = batch_index * batch_size
            end_idx = start_idx + batch_size
            
            if start_idx < len(raw_news):
                current_batch = raw_news[start_idx:end_idx]
                state["current_batch"] = current_batch
                logger.info(f"🔄 NODO 3: Seleccionando lote {batch_index + 1} ({len(current_batch)} artículos)")
            else:
                state["current_batch"] = []
                state["should_continue"] = False
                logger.info("🔄 NODO 3: No hay más lotes para procesar")
            
            return state
        
        # NODO 4: Procesar lote en paralelo
        def process_batch_parallel_node(state: AgentState) -> AgentState:
            """Procesar todo el lote de artículos EN PARALELO"""
            current_batch = state.get("current_batch", [])
            filter_type = state.get("filter_type", "both")
            processed_articles = state.get("processed_articles", [])
            
            if not current_batch:
                logger.info("🔄 NODO 4: Lote vacío, saltando")
                return state
            
            logger.info(f"🚀 NODO 4: Procesando {len(current_batch)} artículos EN PARALELO...")
            
            # Obtener títulos y URLs ya procesados para evitar duplicados
            processed_titles = {p.get("title", "").lower().strip() for p in processed_articles}
            processed_urls = {self._clean_url(p.get("url", "")) for p in processed_articles}
            
            # Procesar en paralelo usando ThreadPoolExecutor
            valid_articles = []
            
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                # Enviar todos los artículos del lote al pool de threads
                future_to_article = {
                    executor.submit(self._process_single_article, article, filter_type, processed_titles, processed_urls): article 
                    for article in current_batch
                }
                
                # Recoger resultados conforme van completando
                for future in as_completed(future_to_article):
                    try:
                        result = future.result(timeout=15)  # Reducir timeout de 30 a 15 segundos
                        
                        if result["status"] == "processed" and result["article"]:
                            valid_articles.append(result["article"])
                            # Actualizar conjuntos para evitar duplicados en el mismo lote
                            processed_titles.add(result["article"]["title"].lower().strip())
                            processed_urls.add(self._clean_url(result["article"]["url"]))
                        
                    except Exception as e:
                        logger.error(f"Error procesando artículo en paralelo: {str(e)}")
            
            # Agregar artículos válidos al estado
            state["processed_articles"].extend(valid_articles)
            state["final_news"].extend(valid_articles)
            
            logger.info(f"✅ NODO 4: Lote completado - {len(valid_articles)} artículos válidos agregados")
            logger.info(f"📊 Total acumulado: {len(state['final_news'])} artículos")
            
            return state
        
        # NODO 5: Incrementar índice de lote
        def increment_batch_index_node(state: AgentState) -> AgentState:
            """Incrementar el índice del lote para continuar"""
            state["current_batch_index"] = state.get("current_batch_index", 0) + 1
            return state
        
        # NODO 6: Verificar si continuar procesando
        def check_batch_completion_node(state: AgentState) -> AgentState:
            """Verificar si procesar más lotes o finalizar"""
            final_count = len(state.get("final_news", []))
            raw_count = len(state.get("raw_news", []))
            current_batch_index = state.get("current_batch_index", 0)
            batch_size = state.get("batch_size", self.batch_size)
            
            processed_count = current_batch_index * batch_size
            
            if final_count >= 20:  # Reducir límite máximo de 25 a 20 artículos
                state["should_continue"] = False
                logger.info(f"🎯 NODO 6: Límite máximo alcanzado ({final_count} artículos)")
            elif processed_count >= raw_count:
                state["should_continue"] = False
                logger.info(f"🎯 NODO 6: Todos los lotes procesados ({final_count} artículos finales)")
            elif current_batch_index >= 6:  # Reducir máximo de 8 a 6 lotes (30 artículos)
                state["should_continue"] = False
                logger.info(f"🎯 NODO 6: Límite de lotes alcanzado ({final_count} artículos)")
            else:
                state["should_continue"] = True
                logger.info(f"🔄 NODO 6: Continuando con siguiente lote... ({final_count} artículos hasta ahora)")
            
            return state
        
        # NODO 7: Finalizar
        def finalize_results_node(state: AgentState) -> AgentState:
            """Finalizar y preparar resultados"""
            final_news = state.get("final_news", [])
            logger.info(f"🔄 NODO 7: Finalizando - {len(final_news)} noticias procesadas")
            
            # Limitar a máximo 20 para Cloud Run
            final_news = final_news[:20]
            
            # Solo usar ejemplos si NO hay noticias reales
            if len(final_news) == 0:
                logger.warning("⚠️ No se encontraron noticias reales, usando fallback...")
                fallback_news = self._fallback_keyword_classification(
                    state.get("query", "technology business"), 
                    state.get("filter_type", "both")
                )
                if len(fallback_news) > 0:
                    final_news = fallback_news[:15]
                else:
                    sample_news = self._get_sample_news_by_filter(state.get("filter_type", "both"))
                    final_news = sample_news[:3]
            
            state["final_news"] = final_news
            total_requests = self._get_daily_requests_count()
            logger.info(f"🏁 FINALIZADO: {len(final_news)} noticias - {total_requests}/100 requests hoy")
            return state
        
        # Crear el grafo
        workflow = StateGraph(AgentState)
        
        # Agregar nodos
        workflow.add_node("fetch_raw_news", fetch_raw_news_node)
        workflow.add_node("initialize_batch_processing", initialize_batch_processing_node)
        workflow.add_node("select_next_batch", select_next_batch_node)
        workflow.add_node("process_batch_parallel", process_batch_parallel_node)
        workflow.add_node("increment_batch_index", increment_batch_index_node)
        workflow.add_node("check_batch_completion", check_batch_completion_node)
        workflow.add_node("finalize_results", finalize_results_node)
        
        # Definir flujo
        workflow.set_entry_point("fetch_raw_news")
        workflow.add_edge("fetch_raw_news", "initialize_batch_processing")
        workflow.add_edge("initialize_batch_processing", "select_next_batch")
        workflow.add_edge("select_next_batch", "process_batch_parallel")
        workflow.add_edge("process_batch_parallel", "increment_batch_index")
        workflow.add_edge("increment_batch_index", "check_batch_completion")
        
        # Función de decisión para continuar o finalizar
        def should_continue_batch_processing(state: AgentState) -> str:
            """Decidir si procesar más lotes"""
            should_continue = state.get("should_continue", False)
            
            if should_continue:
                return "select_next_batch"
            else:
                return "finalize_results"
        
        # Edge condicional
        workflow.add_conditional_edges(
            "check_batch_completion",
            should_continue_batch_processing,
            {
                "select_next_batch": "select_next_batch",
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
            # Queries BALANCEADAS para cada categoría
            if filter_type == "ai":
                query = "(\"artificial intelligence\" OR \"machine learning\" OR \"AI technology\" OR \"OpenAI\" OR \"ChatGPT\" OR \"deep learning\" OR \"neural network\" OR \"AI model\" OR \"copilot\" OR \"automation\")"
            elif filter_type == "marketing":
                query = "(\"digital marketing\" OR \"advertising campaign\" OR \"marketing strategy\" OR \"social media marketing\" OR \"brand management\" OR \"content marketing\" OR \"marketing analytics\" OR \"influencer\" OR \"social media\")"
            else:  # both - BALANCE FORZADO entre AI y Marketing
                query = "(\"artificial intelligence\" AND marketing) OR (\"machine learning\" AND advertising) OR (\"AI technology\" AND \"social media\") OR (\"ChatGPT\" AND campaign) OR (\"marketing automation\") OR (\"AI powered marketing\") OR (\"digital marketing\" AND \"automation\") OR (\"personalization\" AND algorithm)"
            
            # Estado inicial ADAPTADO para paralelización
            initial_state = {
                "query": query,
                "filter_type": filter_type,
                "raw_news": [],
                "current_batch": [],
                "current_batch_index": 0,
                "batch_size": self.batch_size,
                "processed_articles": [],
                "final_news": [],
                "should_continue": True,
                "error_message": ""
            }
            
            logger.info(f"🚀 Iniciando LangGraph Agent para: {filter_type}")
            
            # VALIDACIÓN ROBUSTA: Envolver toda la ejecución en try-catch
            try:
                result = self.graph.invoke(initial_state, config={"recursion_limit": 300})  # Aumentado de 200 a 300
                
                final_news = result.get("final_news", [])
                logger.info(f"🎯 LangGraph Agent completado: {len(final_news)} noticias")
                
                return final_news
            except Exception as e:
                logger.error(f"❌ Error crítico en LangGraph: {str(e)}")
                # Último recurso: usar clasificación por palabras clave de toda la lista
                return self._fallback_keyword_classification(initial_state["query"], filter_type)
            
        except Exception as e:
            logger.error(f"Error en el agente de noticias: {str(e)}")
            return self._fallback_keyword_classification(
                f"technology OR business OR digital OR innovation OR strategy OR marketing OR artificial intelligence", 
                filter_type
            )
    
    def _fallback_keyword_classification(self, query: str, filter_type: str) -> List[Dict[str, Any]]:
        """Método de fallback que usa solo clasificación por palabras clave"""
        try:
            logger.info("🔄 Ejecutando fallback con clasificación por palabras clave...")
            
            # Hacer request directa a NewsAPI
            url = "https://newsapi.org/v2/everything"
            params = {
                "q": query,
                "language": "en", 
                "sortBy": "publishedAt",
                "pageSize": 50,
                "apiKey": self.news_api_key
            }
            
            response = requests.get(url, params=params, timeout=15)
            
            if response.status_code == 200:
                data = response.json()
                articles = data.get("articles", [])
                
                processed_articles = []
                for article in articles[:50]:  # Procesar máximo 50
                    title = str(article.get("title") or "")
                    description = str(article.get("description") or "")
                    
                    if not title or len(title + description) < 10:
                        continue
                    
                    # Clasificar por palabras clave
                    category = self._classify_by_keywords(title, description)
                    
                    # Aplicar filtro - LÓGICA ESPECÍFICA
                    should_include = False
                    if filter_type == "ai":
                        # AI: artículos puros de AI + artículos mixtos
                        should_include = category in ["ai", "both"]
                    elif filter_type == "marketing":
                        # Marketing: SOLO artículos puros de marketing
                        should_include = category == "marketing"
                    elif filter_type == "both":
                        # Both: solo artículos que combinen AI y Marketing
                        should_include = category == "both"
                    
                    if should_include:
                        # Formatear fecha
                        published_at = article.get("publishedAt", "")
                        if published_at:
                            try:
                                dt = datetime.fromisoformat(published_at.replace('Z', '+00:00'))
                                formatted_date = dt.strftime('%Y-%m-%d')
                            except:
                                formatted_date = published_at
                        else:
                            formatted_date = ""
                        
                        processed_article = {
                            "title": title,
                            "description": description[:200],
                            "url": article.get("url", ""),
                            "image": article.get("urlToImage") or "https://images.unsplash.com/photo-1488590528505-98d2b5aba04b?w=400&h=200&fit=crop",
                            "category": category,
                            "publishedAt": formatted_date
                        }
                        processed_articles.append(processed_article)
                        
                        if len(processed_articles) >= 15:  # Límite para fallback
                            break
                
                logger.info(f"🔄 Fallback completado: {len(processed_articles)} artículos encontrados")
                return processed_articles
            
        except Exception as e:
            logger.error(f"❌ Error en fallback: {str(e)}")
        
        # Último recurso: ejemplos
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
