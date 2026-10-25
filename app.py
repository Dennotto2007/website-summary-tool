import os
from flask import Flask, render_template, request, jsonify
import requests
from bs4 import BeautifulSoup
from openai import OpenAI
from dotenv import load_dotenv
import time
from typing import Dict, Any

# Lade Umgebungsvariablen
load_dotenv()

app = Flask(__name__)
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

def extract_content(url: str) -> (str, str, str, str):
    """Extrahiert den relevanten Textinhalt, Betreiber, Title und Meta-Description."""
    try:
        # Add https:// if not present
        if not url.startswith(('http://', 'https://')):
            url = 'https://' + url
            
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        for element in soup(['script', 'style', 'img', 'nav', 'footer', 'header']):
            element.decompose()
        text = ' '.join(soup.stripped_strings)
        owner = extract_owner(soup)
        # SEO: Title und Meta-Description
        title = soup.title.string.strip() if soup.title and soup.title.string else ''
        meta_desc_tag = soup.find('meta', attrs={'name': 'description'})
        meta_desc = meta_desc_tag['content'].strip() if meta_desc_tag and meta_desc_tag.get('content') else ''
        return text[:10000], owner, title, meta_desc
    except Exception as e:
        raise Exception(f"Fehler beim Laden der Webseite: {str(e)}")

def generate_summary(text: str, language: str = "en", owner: str = None, title: str = '', meta_desc: str = '') -> str:
    max_retries = 3
    retry_delay = 2

    # SEO-Abschnitte (jetzt ganz oben, mit Emoji und dreifacher Überschrift für mehr Fettung)
    seo_section = ''
    if title:
        if language == 'en':
            seo_section += f"### 🏷️ **SEO Title**\n{title}\n"
        elif language == 'de':
            seo_section += f"### 🏷️ **SEO-Titel**\n{title}\n"
        elif language == 'pl':
            seo_section += f"### 🏷️ **Tytuł SEO**\n{title}\n"
    if meta_desc:
        if language == 'en':
            seo_section += f"### 📝 **Meta Description**\n{meta_desc}\n"
        elif language == 'de':
            seo_section += f"### 📝 **Meta-Beschreibung**\n{meta_desc}\n"
        elif language == 'pl':
            seo_section += f"### 📝 **Opis meta**\n{meta_desc}\n"

    owner_section = ""
    if owner:
        if language == "en":
            owner_section = f"### 👤 **Website Owner**\n{owner}\n"
        elif language == "de":
            owner_section = f"### 👤 **Betreiber**\n{owner}\n"
        elif language == "pl":
            owner_section = f"### 👤 **Właściciel strony**\n{owner}\n"

    prompts = {
        "en": {
            "system": "You are an expert in website analysis. Answer in English.",
            "user": f"""Analyze the following text and create a structured summary in Markdown format:\n\n{text}\n\nThe summary should include the following elements. Please use the following formatting:\n- Place the SEO title and meta description at the very top as their own sections, with bold headings and a fitting emoji.\n- All section headings should be bold and have a fitting emoji (e.g. ### 📝 **Meta Description**).\n- Do NOT use emojis in the bullet points or in the text within the sections.\n- Start with a 'TL;DR' section (1-2 sentences) summarizing the page in a nutshell.\n- If possible, extract the website owner from the legal notice (impressum) or footer and add it as a 'Website Owner' section.\n\n{seo_section}### ⚡️ **TL;DR**\n(A very short summary in 1-2 sentences)\n\n### 📄 **Summary**\n(A short summary in 3-5 sentences)\n\n### 📚 **Main Topics**\n(Bullet points with the main topics)\n\n### #️⃣ **Tags**\n(Bullet points with 3-5 relevant keywords)\n\n### 🗂️ **Content Type**\n(Detected content type, e.g. blog, shop, portfolio)\n\n### 🌐 **Language**\n(Detected language of the page)\n{owner_section}\nFormat the answer as Markdown and reply in English."""
        },
        "de": {
            "system": "Du bist ein Experte für Webseiten-Analyse. Antworte auf Deutsch.",
            "user": f"""Analysiere den folgenden Text und erstelle eine strukturierte Zusammenfassung im Markdown-Format:\n\n{text}\n\nBitte verwende folgendes Format:\n- Platziere den SEO-Titel und die Meta-Beschreibung ganz oben als eigene Abschnitte, mit fetter Überschrift und passendem Emoji.\n- Alle Abschnittsüberschriften sollen fett und mit Emoji sein (z.B. ### 📝 **Meta-Beschreibung**).\n- Verwende KEINE Emojis in den Bullet-Points oder im Text innerhalb der Abschnitte.\n- Beginne mit einem 'TL;DR'-Abschnitt (1-2 Sätze), der die Seite auf den Punkt bringt.\n- Falls möglich, extrahiere den Betreiber aus dem Impressum oder Footer und gib ihn als eigenen Abschnitt 'Betreiber' aus.\n\n{seo_section}### ⚡️ **TL;DR**\n(Eine sehr kurze Zusammenfassung in 1-2 Sätzen)\n\n### 📄 **Zusammenfassung**\n(Eine kurze Zusammenfassung in 3-5 Sätzen)\n\n### 📚 **Hauptthemen**\n(Bullet-Points mit den wichtigsten Themen)\n\n### #️⃣ **Tags**\n(Bullet-Points mit 3-5 relevanten Keywords)\n\n### 🗂️ **Inhaltstyp**\n(Erkannter Inhaltstyp, z.B. Blog, Shop, Portfolio)\n\n### 🌐 **Sprache**\n(Erkannte Sprache der Seite)\n{owner_section}\nFormatiere die Antwort als Markdown und antworte auf Deutsch."""
        },
        "pl": {
            "system": "Jesteś ekspertem od analizy stron internetowych. Odpowiadaj po polsku.",
            "user": f"""Przeanalizuj poniższy tekst i stwórz uporządkowane podsumowanie w formacie Markdown:\n\n{text}\n\nUżyj następującego formatu:\n- Umieść tytuł SEO i meta opis na samej górze jako osobne sekcje, z pogrubionym nagłówkiem i emoji.\n- Wszystkie nagłówki sekcji powinny być pogrubione i mieć emoji (np. ### 📝 **Opis meta**).\n- NIE używaj emoji w punktach listy ani w treści sekcji.\n- Zacznij od sekcji 'TL;DR' (1-2 zdania), która podsumowuje stronę w skrócie.\n- Jeśli to możliwe, wyodrębnij właściciela strony z impressum lub stopki i dodaj sekcję 'Właściciel strony'.\n\n{seo_section}### ⚡️ **TL;DR**\n(Bardzo krótkie podsumowanie w 1-2 zdaniach)\n\n### 📄 **Podsumowanie**\n(Krótkie podsumowanie w 3-5 zdaniach)\n\n### 📚 **Główne tematy**\n(Punktory z głównymi tematami)\n\n### #️⃣ **Tagi**\n(Punktory z 3-5 odpowiednimi słowami kluczowymi)\n\n### 🗂️ **Typ treści**\n(Rozpoznany typ treści, np. blog, sklep, portfolio)\n\n### 🌐 **Język**\n(Rozpoznany język strony)\n{owner_section}\nSformatuj odpowiedź jako Markdown i odpowiedz po polsku."""
        }
    }
    prompt = prompts.get(language, prompts["en"])

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="gpt-5-mini",
                messages=[
                    {"role": "system", "content": prompt["system"]},
                    {"role": "user", "content": prompt["user"]}
                ]
            )
            return response.choices[0].message.content
        except Exception as e:
            if attempt == max_retries - 1:
                raise Exception(f"API error after {max_retries} attempts: {str(e)}")
            time.sleep(retry_delay * (attempt + 1))

def extract_owner(soup) -> str:
    # Suche nach Impressum/Legal Notice oder Footer
    # 1. Suche nach Linktexten
    impressum_links = soup.find_all('a', string=lambda s: s and ('impressum' in s.lower() or 'legal' in s.lower() or 'contact' in s.lower()))
    for link in impressum_links:
        href = link.get('href', '')
        if href and not href.startswith('#'):
            try:
                if href.startswith('http'):
                    resp = requests.get(href, timeout=5)
                else:
                    base = soup.find('base')
                    base_url = base['href'] if base and 'href' in base.attrs else ''
                    resp = requests.get(base_url + href, timeout=5)
                sub_soup = BeautifulSoup(resp.text, 'html.parser')
                owner = extract_owner_from_text(sub_soup.get_text(" ", strip=True))
                if owner:
                    return owner
            except Exception:
                continue
    # 2. Suche im Footer
    footer = soup.find('footer')
    if footer:
        owner = extract_owner_from_text(footer.get_text(" ", strip=True))
        if owner:
            return owner
    # 3. Suche im gesamten Text
    owner = extract_owner_from_text(soup.get_text(" ", strip=True))
    return owner

def extract_owner_from_text(text: str) -> str:
    # Einfache Heuristik für typische Betreiberangaben
    import re
    patterns = [
        r"(?:Inhaber|Betreiber|Owner|Herausgeber|Verantwortlich(?:er)?|Impressum)[:\s]*([A-ZÄÖÜ][\w\s,&.-]{3,100})",
        r"([A-Z][a-z]+ [A-Z][a-z]+)\s*(GmbH|AG|UG|Inc\.|LLC|Ltd\.|S\.A\.|S\.r\.l\.|Sp\.z o\.o\.|e\.K\.|KG|OHG)?"
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return match.group(0)
    return None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/summarize', methods=['POST'])
def summarize():
    try:
        url = request.form.get('url')
        language = request.form.get('language', 'en')
        if not url:
            return jsonify({'error': 'URL ist erforderlich'}), 400
        content, owner, title, meta_desc = extract_content(url)
        summary = generate_summary(content, language, owner, title, meta_desc)
        if request.form.get('save') == 'true':
            with open('summary.md', 'w', encoding='utf-8') as f:
                f.write(summary)
        return jsonify({'summary': summary})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port) 
