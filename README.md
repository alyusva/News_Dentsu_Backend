# News Platform Backend - Dentsu

Backend API para la plataforma de noticias de IA y Marketing.

## 🚀 Características

- FastAPI con LangGraph agent
- Integración con NewsAPI y OpenAI
- Categorización automática de noticias
- Deployment en Google Cloud Run

## 📦 Instalación Local

```bash
# Instalar dependencias
pip install -r requirements.txt

# Configurar variables de entorno
cp .env.example .env
# Editar .env con tus API keys

# Ejecutar servidor
python main.py
```

## �� Deployment

### Google Cloud Run
```bash
# Configurar proyecto
export GOOGLE_CLOUD_PROJECT="tu-proyecto-id"
export NEWS_API_KEY="tu-news-api-key"
export OPENAI_API_KEY="tu-openai-key"

# Desplegar
gcloud run deploy news-platform-api \
  --source . \
  --platform managed \
  --region europe-west1 \
  --allow-unauthenticated \
  --set-env-vars="NEWS_API_KEY=$NEWS_API_KEY,OPENAI_API_KEY=$OPENAI_API_KEY"
```

## 📚 API Endpoints

- `GET /` - Status del servicio
- `GET /health` - Health check
- `GET /api/news` - Obtener noticias filtradas

## 🔧 Configuración

Variables de entorno requeridas:
- `NEWS_API_KEY` - API key de NewsAPI
- `OPENAI_API_KEY` - API key de OpenAI
- `PORT` - Puerto del servidor (default: 8000)
