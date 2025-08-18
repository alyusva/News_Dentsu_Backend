# News Dentsu Backend

Backend API para plataforma de noticias de IA y Marketing con clasificación inteligente

## Características

- FastAPI con agente LangGraph para procesamiento inteligente
- Integración NewsAPI + OpenAI para clasificación automática
- Categorización de noticias (AI, Marketing, Both)
- Filtrado de duplicados y control de límites
- Deployment automático en Google Cloud Run

## Instalación Local

```bash
# Clonar repositorio
git clone https://github.com/alyusva/News_Dentsu_Backend.git
cd News_Dentsu_Backend

# Instalar dependencias
pip install -r requirements.txt

# Configurar variables de entorno
cp .env.example .env
# Editar .env con tus API keys

# Ejecutar servidor
python main.py
```

## Deployment

### Google Cloud Run
```bash
# Configurar variables
export GOOGLE_CLOUD_PROJECT="tu-proyecto"
export NEWSAPI_KEY="tu-newsapi-key"
export OPENAI_API_KEY="tu-openai-key"

# Usar script de deployment
chmod +x deploy-cloudrun.sh
./deploy-cloudrun.sh
```

### Deployment manual
```bash
gcloud run deploy news-dentsu-backend \
  --source . \
  --platform managed \
  --region europe-west1 \
  --allow-unauthenticated \
  --memory 1Gi \
  --cpu 1 \
  --set-env-vars="NEWSAPI_KEY=$NEWSAPI_KEY,OPENAI_API_KEY=$OPENAI_API_KEY"
```

## API Endpoints

- `GET /` - Status del servicio
- `GET /health` - Health check
- `GET /api/v1/get-news?filter_type={ai|marketing|both}` - Obtener noticias filtradas

### Ejemplos de uso
```bash
# Noticias de IA
curl "https://tu-servicio.run.app/api/v1/get-news?filter_type=ai"

# Noticias de Marketing
curl "https://tu-servicio.run.app/api/v1/get-news?filter_type=marketing"

# Ambas categorías
curl "https://tu-servicio.run.app/api/v1/get-news?filter_type=both"
```

## Estructura del Proyecto

```
News_Dentsu_Backend/
├── app/
│   ├── __init__.py
│   ├── api/
│   │   ├── __init__.py
│   │   └── endpoints.py          # Rutas de la API
│   ├── agent/
│   │   ├── __init__.py
│   │   └── langgraph_agent.py    # Agente LangGraph con procesamiento paralelo
│   └── core/
│       ├── __init__.py
│       └── config.py             # Configuración centralizada
├── main.py                       # Punto de entrada FastAPI
├── requirements.txt              # Dependencias Python
├── Dockerfile                    # Configuración de contenedor
├── deploy-cloudrun.sh           # Script de deployment automático
├── cloudbuild.yaml              # CI/CD para Google Cloud
├── .env.example                 # Plantilla de variables de entorno
├── .gitignore                   # Archivos excluidos del repo
└── README.md                    # Documentación del proyecto
```

## Configuración

Variables de entorno requeridas:
- `NEWSAPI_KEY` - API key de NewsAPI
- `OPENAI_API_KEY` - API key de OpenAI  
- `PORT` - Puerto del servidor (default: 8080)
- `HOST` - Host del servidor (default: 0.0.0.0)
- `DEBUG` - Modo debug (default: False)

## Arquitectura

### Agente LangGraph
El sistema utiliza un agente LangGraph con **7 nodos** especializados para procesamiento paralelo:

#### Nodos del Agente
1. **`fetch_raw_news_node`** - Obtiene noticias brutas de NewsAPI (hasta 100 artículos)
2. **`initialize_batch_processing_node`** - Configura procesamiento en lotes (5 artículos/lote, 3 threads)
3. **`select_next_batch_node`** - Selecciona siguiente lote de artículos para procesar
4. **`process_batch_parallel_node`**: Clasifica artículos en paralelo con OpenAI GPT-3.5
5. **`increment_batch_index_node`** - Avanza al siguiente lote de procesamiento
6. **`check_batch_completion_node`** - Decide si continuar procesando o finalizar
7. **`finalize_results_node`** - Prepara resultados finales (máximo 20 artículos)

#### Flujo de Procesamiento
```
1→2→3→4→5→6→7 (con loop 3↔6 hasta completar todos los lotes)
```

#### Características del Agente
- **Procesamiento paralelo**: 3 threads concurrentes con ThreadPoolExecutor
- **Control de límites**: Verificación de requests diarias (100/día NewsAPI)
- **Clasificación inteligente**: OpenAI GPT-3.5 turbo + fallback por palabras clave
- **Filtrado de duplicados**: Detección por URL limpia y similitud de títulos
- **Optimización Cloud Run**: Lotes pequeños para evitar timeouts

### Flujo de datos
```
NewsAPI → LangGraph Agent → OpenAI Classification → Duplicate Filter → JSON Response
```

## Tecnologías

- **FastAPI** - Framework web
- **LangGraph** - Orquestación de agentes IA
- **OpenAI GPT-3.5** - Clasificación de noticias
- **NewsAPI** - Fuente de noticias
- **Google Cloud Run** - Deployment serverless
- **Docker** - Containerización

## Límites y Control

- 100 requests/día a NewsAPI (plan gratuito)
- Hasta 25 noticias por respuesta
- Tracking diario de requests
- Fallback con clasificación por palabras clave

## Contacto

- GitHub: [@alyusva](https://github.com/alyusva)
- Proyecto: [News_Dentsu_Backend](https://github.com/alyusva/News_Dentsu_Backend)
