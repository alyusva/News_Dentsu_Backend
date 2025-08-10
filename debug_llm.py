#!/usr/bin/env python3
"""
Script de debug para examinar cómo el LLM está clasificando las noticias
"""
import os
import json
from openai import OpenAI

# Configurar OpenAI
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def test_llm_classification():
    """Probar la clasificación del LLM con artículos de ejemplo"""
    
    # Artículo de ejemplo relacionado con AI/Marketing
    test_article = {
        "title": "Google Launches New AI-Powered Marketing Tools for Small Businesses",
        "description": "Google has announced a suite of artificial intelligence-powered marketing tools designed to help small businesses create more effective advertising campaigns. The new tools include automated ad copy generation and smart audience targeting.",
        "content": "Google today announced the launch of new AI-powered marketing tools specifically designed for small businesses. These tools leverage machine learning algorithms to analyze customer behavior and create personalized marketing campaigns. The platform includes features for automated content creation, smart audience segmentation, and performance optimization."
    }
    
    # El mismo prompt que usamos en el código
    prompt = f"""Analiza este artículo de noticias y clasifícalo en una de estas categorías:

- "ai": Si el artículo habla principalmente sobre inteligencia artificial, machine learning, automatización, robots, o tecnología AI
- "marketing": Si el artículo habla principalmente sobre publicidad, marketing digital, redes sociales, branding, o estrategias comerciales  
- "both": Si el artículo combina ambos temas (AI aplicada al marketing, herramientas de marketing con IA, etc.)
- "none": Si no se relaciona con ninguna de las categorías anteriores

Título: {test_article['title']}
Descripción: {test_article['description']}
Contenido: {test_article['content']}

Responde únicamente con una palabra: ai, marketing, both, o none"""

    print("🔍 PROMPT ENVIADO AL LLM:")
    print("=" * 50)
    print(prompt)
    print("=" * 50)
    
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
        print(f"\n🤖 RESPUESTA DEL LLM: '{result}'")
        
        # Verificar si la respuesta es válida
        valid_categories = ["ai", "marketing", "both", "none"]
        if result in valid_categories:
            print(f"✅ Categoría válida: {result}")
        else:
            print(f"❌ Categoría inválida: {result}")
            
        return result
        
    except Exception as e:
        print(f"❌ Error al llamar al LLM: {e}")
        return None

def test_with_multiple_articles():
    """Probar con varios artículos de diferentes tipos"""
    
    test_articles = [
        {
            "title": "OpenAI Releases ChatGPT-4 with Enhanced Language Capabilities",
            "description": "The latest version of ChatGPT features improved natural language processing",
            "expected": "ai"
        },
        {
            "title": "Nike's Social Media Strategy Boosts Brand Engagement by 40%",
            "description": "Nike's innovative social media campaigns have significantly increased customer engagement",
            "expected": "marketing"
        },
        {
            "title": "AI-Powered Chatbots Transform Customer Service Marketing",
            "description": "Companies are using artificial intelligence chatbots to improve their marketing outreach",
            "expected": "both"
        },
        {
            "title": "Local Restaurant Opens New Location Downtown",
            "description": "A popular local restaurant has expanded with a new location in the downtown area",
            "expected": "none"
        }
    ]
    
    print("\n🧪 PRUEBAS CON MÚLTIPLES ARTÍCULOS:")
    print("=" * 60)
    
    for i, article in enumerate(test_articles, 1):
        print(f"\n📰 ARTÍCULO {i}: {article['title']}")
        print(f"📄 Descripción: {article['description']}")
        print(f"🎯 Esperado: {article['expected']}")
        
        prompt = f"""Analiza este artículo de noticias y clasifícalo en una de estas categorías:

- "ai": Si el artículo habla principalmente sobre inteligencia artificial, machine learning, automatización, robots, o tecnología AI
- "marketing": Si el artículo habla principalmente sobre publicidad, marketing digital, redes sociales, branding, o estrategias comerciales  
- "both": Si el artículo combina ambos temas (AI aplicada al marketing, herramientas de marketing con IA, etc.)
- "none": Si no se relaciona con ninguna de las categorías anteriores

Título: {article['title']}
Descripción: {article['description']}

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
            print(f"🤖 Resultado: '{result}'")
            
            if result == article['expected']:
                print("✅ CORRECTO")
            else:
                print(f"❌ INCORRECTO (esperado: {article['expected']})")
                
        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    print("🚀 INICIANDO DEBUG DEL LLM")
    
    # Verificar API key
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ OPENAI_API_KEY no encontrada en variables de entorno")
        exit(1)
    
    print("✅ API Key encontrada")
    
    # Prueba básica
    test_llm_classification()
    
    # Pruebas múltiples
    test_with_multiple_articles()
    
    print("\n🏁 DEBUG COMPLETADO")
