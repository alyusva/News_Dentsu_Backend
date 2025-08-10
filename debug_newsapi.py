#!/usr/bin/env python3
"""
Script para debuggear qué artículos están llegando realmente del NewsAPI
"""
import os
import httpx
import json
from datetime import datetime, timedelta

def fetch_real_news():
    """Obtener noticias reales del NewsAPI como lo hace el agente"""
    
    api_key = os.getenv("NEWS_API_KEY")
    if not api_key:
        print("❌ NEWS_API_KEY no encontrada")
        return
    
    # Los mismos queries que usa el agente
    queries = [
        "artificial intelligence OR machine learning OR AI technology OR automation OR neural networks OR deep learning",
        "digital marketing OR social media marketing OR content marketing OR brand strategy OR advertising technology",
        "AI marketing OR marketing automation OR personalization technology OR customer analytics OR programmatic advertising"
    ]
    
    # Fecha desde hace 7 días
    from_date = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')
    
    all_articles = []
    
    for i, query in enumerate(queries, 1):
        print(f"\n🔍 CONSULTA {i}: {query[:50]}...")
        
        url = "https://newsapi.org/v2/everything"
        params = {
            'q': query,
            'language': 'en',
            'sortBy': 'relevancy',
            'from': from_date,
            'pageSize': 10,  # Solo los primeros 10 por query
            'apiKey': api_key
        }
        
        try:
            response = httpx.get(url, params=params, timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                articles = data.get('articles', [])
                print(f"📰 Encontrados: {len(articles)} artículos")
                
                for j, article in enumerate(articles[:5], 1):  # Solo mostrar los primeros 5
                    print(f"\n   📄 ARTÍCULO {j}:")
                    print(f"   📰 Título: {article.get('title', 'Sin título')[:100]}")
                    print(f"   📝 Descripción: {article.get('description', 'Sin descripción')[:150]}")
                    print(f"   🏢 Fuente: {article.get('source', {}).get('name', 'Desconocida')}")
                    
                    # Verificar si tiene contenido útil
                    content = article.get('content', '')
                    if content and len(content) > 50:
                        print(f"   📖 Contenido: {content[:100]}...")
                    else:
                        print(f"   ⚠️  Contenido limitado o ausente")
                
                all_articles.extend(articles)
                
            else:
                print(f"❌ Error {response.status_code}: {response.text}")
                
        except Exception as e:
            print(f"❌ Error en consulta {i}: {e}")
    
    print(f"\n📊 RESUMEN:")
    print(f"Total de artículos obtenidos: {len(all_articles)}")
    
    # Analizar calidad de los artículos
    with_content = sum(1 for a in all_articles if a.get('content') and len(a.get('content', '')) > 100)
    with_description = sum(1 for a in all_articles if a.get('description') and len(a.get('description', '')) > 50)
    
    print(f"Con contenido sustancial: {with_content}")
    print(f"Con descripción buena: {with_description}")
    
    return all_articles

def test_classification_on_real_articles():
    """Probar la clasificación en artículos reales"""
    
    from openai import OpenAI
    
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    print("\n🧪 PROBANDO CLASIFICACIÓN EN ARTÍCULOS REALES")
    print("=" * 60)
    
    articles = fetch_real_news()
    
    if not articles:
        print("❌ No se obtuvieron artículos para probar")
        return
    
    # Probar clasificación en los primeros 5 artículos
    for i, article in enumerate(articles[:5], 1):
        print(f"\n🔬 PROBANDO ARTÍCULO {i}:")
        print(f"📰 Título: {article.get('title', 'Sin título')}")
        print(f"📝 Descripción: {article.get('description', 'Sin descripción')}")
        
        # Usar el mismo prompt que el agente
        title = article.get('title', '')
        description = article.get('description', '')
        content = article.get('content', '')
        
        if not title or not description:
            print("⚠️  Artículo sin título o descripción - SALTANDO")
            continue
        
        prompt = f"""Analiza este artículo de noticias y clasifícalo en una de estas categorías:

- "ai": Si el artículo habla principalmente sobre inteligencia artificial, machine learning, automatización, robots, o tecnología AI
- "marketing": Si el artículo habla principalmente sobre publicidad, marketing digital, redes sociales, branding, o estrategias comerciales  
- "both": Si el artículo combina ambos temas (AI aplicada al marketing, herramientas de marketing con IA, etc.)
- "none": Si no se relaciona con ninguna de las categorías anteriores

Título: {title}
Descripción: {description}
Contenido: {content}

Responde únicamente con una palabra: ai, marketing, both, o none"""

        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "Eres un clasificador de noticias experto. Debes responder únicamente con una palabra."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=10
            )
            
            result = response.choices[0].message.content.strip().lower()
            print(f"🤖 CLASIFICACIÓN: '{result}'")
            
            if result in ["ai", "marketing", "both"]:
                print("✅ ARTÍCULO RELEVANTE")
            else:
                print("❌ ARTÍCULO NO RELEVANTE")
                
        except Exception as e:
            print(f"❌ Error al clasificar: {e}")

if __name__ == "__main__":
    print("🚀 INICIANDO DEBUG DE NEWSAPI + LLM")
    
    # Verificar API keys
    if not os.getenv("NEWS_API_KEY"):
        print("❌ NEWS_API_KEY no encontrada")
        exit(1)
    
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ OPENAI_API_KEY no encontrada")
        exit(1)
    
    print("✅ API Keys encontradas")
    
    # Probar clasificación en artículos reales
    test_classification_on_real_articles()
    
    print("\n🏁 DEBUG COMPLETADO")
