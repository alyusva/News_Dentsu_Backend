# News Dentsu Backend 📰🤖

Backend API para la plataforma de noticias de IA y Marketing con agente LangGraph inteligente.

## 🚀 Características

- **FastAPI** con agente **LangGraph** granular
- **Integración NewsAPI + OpenAI** para clasificación inteligente  
- **Categorización automática** de noticias (AI, Marketing, Both)
- **Filtrado de duplicados** avanzado
- **Deployment en Google Cloud Run** con escalado automático

## 📦 Instalación Local

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

## ☁️ Deployment en Google Cloud Run

```bash
# Configurar proyecto
export GOOGLE_CLOUD_PROJECT="news-dentsu"
export NEWSAPI_KEY="tu-newsapi-key"
export OPENAI_API_KEY="tu-openai-key"

# Desplegar
gcloud run deploy news-dentsu-backend \
  --source . \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --memory 1Gi \
  --cpu 1 \
  --set-env-vars="NEWSAPI_KEY=$NEWSAPI_KEY,OPENAI_API_KEY=$OPENAI_API_KEY"
```

## 📚 API Endpoints

- `GET /` - Status del servicio
- `GET /health` - Health check  
- `GET /api/v1/get-news?filter_type={ai|marketing|both}` - Obtener noticias filtradas

### Ejemplo de uso:
```bash
# Obtener noticias de IA
curl "https://tu-servicio.run.app/api/v1/get-news?filter_type=ai"

# Obtener noticias de Marketing  
curl "https://tu-servicio.run.app/api/v1/get-news?filter_type=marketing"

# Obtener ambas categorías
curl "https://tu-servicio.run.app/api/v1/get-news?filter_type=both"
```

## 🔧 Configuración

Variables de entorno requeridas:
- `NEWSAPI_KEY` - API key de NewsAPI
- `OPENAI_API_KEY` - API key de OpenAI
- `PORT` - Puerto del servidor (default: 8000)
- `HOST` - Host del servidor (default: 0.0.0.0)
- `DEBUG` - Modo debug (default: False)

## 🏗️ Arquitectura

### Agente LangGraph Granular
El backend utiliza un agente LangGraph con nodos especializados:

1. **Verificación de límites** - Control de requests diarias (100/día)
2. **Obtención de noticias** - Llamada a NewsAPI con query optimizada
3. **Clasificación inteligente** - LLM + palabras clave para categorización
4. **Filtrado de duplicados** - Detección por URL y similitud de títulos  
5. **Procesamiento final** - Formateo y entrega de resultados

### Flujo de Datos
```
NewsAPI → LangGraph Agent → OpenAI Classification → Duplicate Filter → JSON Response
```

## 🔄 Límites y Control

- **100 requests/día** a NewsAPI (plan gratuito)
- **Hasta 25 noticias** por respuesta para optimizar performance
- **Fallback inteligente** con clasificación por palabras clave
- **Tracking diario** de requests en archivo local

## 🚀 Tecnologías

- **FastAPI** - Framework web moderno y rápido
- **LangGraph** - Orquestación de agentes IA
- **OpenAI GPT-4o-mini** - Clasificación de noticias
- **NewsAPI** - Fuente de noticias en tiempo real
- **Google Cloud Run** - Deployment serverless
- **Docker** - Containerización

## 📄 Licencia

MIT License - ver archivo [LICENSE](LICENSE) para más detalles.

## 🤝 Contribuir

1. Fork el proyecto
2. Crea una feature branch (`git checkout -b feature/nueva-funcionalidad`)
3. Commit tus cambios (`git commit -am 'Añadir nueva funcionalidad'`)
4. Push a la branch (`git push origin feature/nueva-funcionalidad`)
5. Abre un Pull Request

## 📞 Contacto

- GitHub: [@alyusva](https://github.com/alyusva)
- Proyecto: [News_Dentsu_Backend](https://github.com/alyusva/News_Dentsu_Backend)
