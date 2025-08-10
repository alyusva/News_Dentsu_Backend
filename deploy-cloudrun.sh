#!/bin/bash

# Deploy script para Google Cloud Run
# Dentsu News Platform Backend

set -e

# Colores
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m'

# Configuración
PROJECT_ID=${GOOGLE_CLOUD_PROJECT:-"dentsu-news-platform"}
SERVICE_NAME="news-platform-api"
REGION="europe-west1"
IMAGE_NAME="gcr.io/$PROJECT_ID/$SERVICE_NAME"

echo -e "${BLUE}🚀 Deploying Dentsu News Platform Backend to Cloud Run...${NC}"

# Verificar variables de entorno
if [ -z "$NEWSAPI_KEY" ]; then
    echo -e "${RED}❌ Error: NEWSAPI_KEY not set${NC}"
    echo "Run: export NEWSAPI_KEY='your-api-key'"
    exit 1
fi

if [ -z "$OPENAI_API_KEY" ]; then
    echo -e "${RED}❌ Error: OPENAI_API_KEY not set${NC}"
    echo "Run: export OPENAI_API_KEY='your-api-key'"
    exit 1
fi

# Configurar proyecto
echo -e "${YELLOW}⚙️  Setting up Google Cloud project...${NC}"
gcloud config set project $PROJECT_ID

# Habilitar APIs
echo -e "${YELLOW}🔧 Enabling required APIs...${NC}"
gcloud services enable cloudbuild.googleapis.com
gcloud services enable run.googleapis.com

# Build y deploy en un solo comando
echo -e "${YELLOW}🏗️  Building and deploying...${NC}"
gcloud run deploy $SERVICE_NAME \
    --source . \
    --platform managed \
    --region $REGION \
    --allow-unauthenticated \
    --set-env-vars="NEWSAPI_KEY=$NEWSAPI_KEY,OPENAI_API_KEY=$OPENAI_API_KEY,DEBUG=False" \
    --memory=1Gi \
    --cpu=1 \
    --max-instances=10 \
    --timeout=300

# Obtener URL
SERVICE_URL=$(gcloud run services describe $SERVICE_NAME --platform managed --region $REGION --format 'value(status.url)')

echo -e "${GREEN}✅ Deployment successful!${NC}"
echo -e "${GREEN}🌐 Service URL: $SERVICE_URL${NC}"
echo -e "${GREEN}📋 Health check: $SERVICE_URL/health${NC}"
echo -e "${GREEN}📚 API docs: $SERVICE_URL/docs${NC}"

echo -e "${BLUE}📝 To view logs:${NC}"
echo "gcloud run services logs tail $SERVICE_NAME --region=$REGION"
