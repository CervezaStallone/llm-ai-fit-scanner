#!/usr/bin/env python3
"""
LLM & AI Fit Scanner
====================
Scans your system (CPU, RAM, GPU/VRAM) and scores each category and
individual model to show what you can run locally.

Usage:
    python llm_scanner.py                  # scan with built-in + cached catalog
    python llm_scanner.py --update         # update model catalog online
    python llm_scanner.py --json           # output as JSON
    python llm_scanner.py -l nl            # output in Dutch
    python llm_scanner.py -l de --update   # German + update catalog
"""

import argparse
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from urllib.request import urlopen, Request
from urllib.error import URLError

try:
    import psutil
except ImportError:
    sys.exit("psutil required: pip install psutil")

# ─── Paths ──────────────────────────────────────────────────────────────────

SCRIPT_DIR = Path(__file__).parent.resolve()
CATALOG_FILE = SCRIPT_DIR / "models_catalog.json"
REPORTS_DIR = SCRIPT_DIR / "reports"

# ─── Colors ─────────────────────────────────────────────────────────────────

GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
CYAN = "\033[96m"
BOLD = "\033[1m"
DIM = "\033[2m"
RESET = "\033[0m"

# ─── i18n ───────────────────────────────────────────────────────────────────

LANG = "en"  # set via --lang / -l
_LANG_IDX = {"en": 0, "nl": 1, "de": 2, "fr": 3}

def T(key, **kw):
    """Get translated string for current LANG."""
    vals = _S.get(key)
    if not vals:
        return key
    s = vals[_LANG_IDX.get(LANG, 0)]
    return s.format(**kw) if kw else s

# Translations: (EN, NL, DE, FR)
_S = {
    # ── Score labels
    "excellent":     ("Excellent", "Uitstekend", "Ausgezeichnet", "Excellent"),
    "good":          ("Good", "Goed", "Gut", "Bon"),
    "fair":          ("Fair (slower)", "Redelijk (langzamer)", "Mäßig (langsamer)", "Correct (plus lent)"),
    "poor":          ("Poor (slow)", "Matig (traag)", "Schwach (langsam)", "Faible (lent)"),
    "difficult":     ("Difficult", "Moeilijk", "Schwierig", "Difficile"),
    "not_possible":  ("Not possible", "Niet mogelijk", "Nicht möglich", "Pas possible"),
    # ── Modes
    "gpu_full":      ("GPU (full)", "GPU (volledig)", "GPU (vollständig)", "GPU (complet)"),
    "gpu_offload":   ("GPU+CPU offload ({pct}% VRAM)", "GPU+CPU offload ({pct}% VRAM)",
                      "GPU+CPU Offload ({pct}% VRAM)", "GPU+CPU offload ({pct}% VRAM)"),
    "cpu_only":      ("CPU-only (slow)", "CPU-only (langzaam)", "CPU-only (langsam)", "CPU uniquement (lent)"),
    "not_feasible":  ("Not feasible", "Niet haalbaar", "Nicht machbar", "Non réalisable"),
    "unknown":       ("Unknown", "Onbekend", "Unbekannt", "Inconnu"),
    # ── Terminal headers
    "hw_header":     ("HARDWARE (auto-detected)", "HARDWARE (automatisch gedetecteerd)",
                      "HARDWARE (automatisch erkannt)", "MATÉRIEL (détection automatique)"),
    "tools_header":  ("INSTALLED AI TOOLS", "GEINSTALLEERDE AI-TOOLS",
                      "INSTALLIERTE AI-TOOLS", "OUTILS IA INSTALLÉS"),
    "ollama_header": ("LOCALLY INSTALLED OLLAMA MODELS", "LOKAAL GEINSTALLEERDE OLLAMA-MODELLEN",
                      "LOKAL INSTALLIERTE OLLAMA-MODELLE", "MODÈLES OLLAMA INSTALLÉS LOCALEMENT"),
    "scores_header": ("SCORES PER CATEGORY", "SCORES PER CATEGORIE",
                      "SCORES PRO KATEGORIE", "SCORES PAR CATÉGORIE"),
    "top_header":    ("TOP INDIVIDUAL MODELS FOR YOUR HARDWARE",
                      "TOP INDIVIDUELE MODELLEN VOOR JOUW HARDWARE",
                      "TOP-MODELLE FÜR DEINE HARDWARE",
                      "TOP MODÈLES POUR VOTRE MATÉRIEL"),
    "summary_header":("SUMMARY & RECOMMENDATIONS", "SAMENVATTING & AANBEVELINGEN",
                      "ZUSAMMENFASSUNG & EMPFEHLUNGEN", "RÉSUMÉ & RECOMMANDATIONS"),
    # ── Labels
    "total":         ("total", "totaal", "gesamt", "total"),
    "available":     ("available", "beschikbaar", "verfügbar", "disponible"),
    "free":          ("free", "vrij", "frei", "libre"),
    "in_use":        ("in use", "in gebruik", "in Benutzung", "en utilisation"),
    "no_gpu":        ("No GPU detected", "Geen GPU gedetecteerd",
                      "Keine GPU erkannt", "Aucun GPU détecté"),
    "mode_label":    ("Mode", "Modus", "Modus", "Mode"),
    "examples_label":("Examples", "Voorbeelden", "Beispiele", "Exemples"),
    "runs_well":     ("Runs well:", "Goed draaibaar:", "Läuft gut:", "Fonctionne bien :"),
    "possible_slower":("Possible but slower:", "Mogelijk maar langzamer:",
                       "Möglich aber langsamer:", "Possible mais plus lent :"),
    "barely_feasible":("Not / barely feasible:", "Niet / nauwelijks haalbaar:",
                       "Nicht / kaum machbar:", "Non / à peine réalisable :"),
    # ── Tips (terminal)
    "tips_for":      ("Tips for your system ({ram}GB RAM, {vram}MB VRAM):",
                      "Tips voor jouw systeem ({ram}GB RAM, {vram}MB VRAM):",
                      "Tipps für dein System ({ram}GB RAM, {vram}MB VRAM):",
                      "Conseils pour votre système ({ram}GB RAM, {vram}MB VRAM) :"),
    "tip_vram_full_1": ("With {vram}MB VRAM you can run 7-8B models in Q4 quantization",
                      "Met {vram}MB VRAM kun je 7-8B modellen in Q4 kwantisatie",
                      "Mit {vram}MB VRAM kannst du 7-8B Modelle in Q4-Quantisierung",
                      "Avec {vram}MB VRAM vous pouvez exécuter des modèles 7-8B en Q4"),
    "tip_vram_full_2": ("fully on the GPU. Use --n-gpu-layers for offloading.",
                      "volledig op de GPU draaien. Gebruik --n-gpu-layers voor offload.",
                      "vollständig auf der GPU ausführen. Nutze --n-gpu-layers für Offloading.",
                      "entièrement sur le GPU. Utilisez --n-gpu-layers pour le déchargement."),
    "tip_vram_part_1": ("With {vram}MB VRAM you can run 7-8B models in Q4 with",
                      "Met {vram}MB VRAM kun je 7-8B modellen in Q4 draaien met",
                      "Mit {vram}MB VRAM kannst du 7-8B Modelle in Q4 ausführen mit",
                      "Avec {vram}MB VRAM vous pouvez exécuter des modèles 7-8B en Q4 avec"),
    "tip_vram_part_2": ("GPU+CPU offloading. Use --n-gpu-layers to configure.",
                      "GPU+CPU offload. Gebruik --n-gpu-layers om te configureren.",
                      "GPU+CPU-Offloading. Nutze --n-gpu-layers zur Konfiguration.",
                      "déchargement GPU+CPU. Utilisez --n-gpu-layers pour configurer."),
    "tip_ram_1":     ("With {ram}GB RAM you can run 30B+ models CPU-only",
                      "Met {ram}GB RAM kun je 30B+ modellen CPU-only draaien",
                      "Mit {ram}GB RAM kannst du 30B+ Modelle CPU-only ausführen",
                      "Avec {ram}GB RAM vous pouvez exécuter des modèles 30B+ en CPU"),
    "tip_ram_2":     ("in Q4 quantization (~2-5 tokens/sec).",
                      "in Q4 kwantisatie (~2-5 tokens/sec).",
                      "in Q4-Quantisierung (~2-5 Token/Sek).",
                      "en quantification Q4 (~2-5 tokens/sec)."),
    "tip_no_tools":  ("No AI tools detected! Start with Ollama:",
                      "Geen AI-tools gedetecteerd! Start met Ollama:",
                      "Keine AI-Tools erkannt! Starte mit Ollama:",
                      "Aucun outil IA détecté ! Commencez avec Ollama :"),
    "tip_ollama":    ("Ollama is installed! Try:",
                      "Ollama is geïnstalleerd! Probeer:",
                      "Ollama ist installiert! Probiere:",
                      "Ollama est installé ! Essayez :"),
    "tip_q4":        ("Tip: Q4_K_M quantization is the sweet spot between quality and speed.",
                      "Tip: Q4_K_M kwantisatie is de sweet spot tussen kwaliteit en snelheid.",
                      "Tipp: Q4_K_M ist der Sweet Spot zwischen Qualität und Geschwindigkeit.",
                      "Astuce : Q4_K_M est le compromis idéal entre qualité et vitesse."),
    "tip_gpu_cfg":   ("Tip: Use --num-gpu / -ngl to configure GPU offloading.",
                      "Tip: Gebruik --num-gpu / -ngl om GPU-offload te configureren.",
                      "Tipp: Nutze --num-gpu / -ngl um GPU-Offloading zu konfigurieren.",
                      "Astuce : Utilisez --num-gpu / -ngl pour configurer le déchargement GPU."),
    # ── Update messages
    "upd_updating":  ("Updating catalog...", "Catalogus updaten...",
                      "Katalog aktualisieren...", "Mise à jour du catalogue..."),
    "upd_hf":        ("Fetching HuggingFace API...", "HuggingFace API ophalen...",
                      "HuggingFace API abrufen...", "Récupération API HuggingFace..."),
    "upd_ollama":    ("Fetching Ollama registry...", "Ollama registry ophalen...",
                      "Ollama Registry abrufen...", "Récupération registre Ollama..."),
    "upd_found":     ("{n} models found", "{n} modellen gevonden",
                      "{n} Modelle gefunden", "{n} modèles trouvés"),
    "upd_error":     ("Error: {e}", "Fout: {e}", "Fehler: {e}", "Erreur : {e}"),
    "upd_no_models": ("No models fetched. Check your internet connection.",
                      "Geen modellen opgehaald. Controleer je internetverbinding.",
                      "Keine Modelle abgerufen. Prüfe deine Internetverbindung.",
                      "Aucun modèle récupéré. Vérifiez votre connexion internet."),
    "upd_saved":     ("{n} models saved in {file}", "{n} modellen opgeslagen in {file}",
                      "{n} Modelle gespeichert in {file}", "{n} modèles enregistrés dans {file}"),
    "upd_last":      ("Last updated: {ts}", "Laatst geüpdatet: {ts}",
                      "Zuletzt aktualisiert: {ts}", "Dernière mise à jour : {ts}"),
    "upd_catalog":   ("Catalog: {n} models (updated: {ts})",
                      "Catalogus: {n} modellen (geüpdatet: {ts})",
                      "Katalog: {n} Modelle (aktualisiert: {ts})",
                      "Catalogue : {n} modèles (mis à jour : {ts})"),
    "upd_cmd":       ("Update with: python llm_scanner.py --update",
                      "Update met: python llm_scanner.py --update",
                      "Aktualisieren mit: python llm_scanner.py --update",
                      "Mettre à jour : python llm_scanner.py --update"),
    "upd_no_cat":    ("No model catalog found. Fetch online data:",
                      "Geen model-catalogus gevonden. Haal online data op:",
                      "Kein Modellkatalog gefunden. Online-Daten abrufen:",
                      "Aucun catalogue trouvé. Récupérez les données en ligne :"),
    # ── Categories
    "cat_small":     ("Small LLMs (1-3B)", "Kleine LLMs (1-3B)",
                      "Kleine LLMs (1-3B)", "Petits LLMs (1-3B)"),
    "cat_small_d":   ("Fast, light models for chat, code-assist, classification",
                      "Snelle, lichte modellen voor chat, code-assist, classificatie",
                      "Schnelle, leichte Modelle für Chat, Code-Assist, Klassifizierung",
                      "Modèles rapides et légers pour chat, code, classification"),
    "cat_medium":    ("Medium LLMs (7-8B)", "Medium LLMs (7-8B)",
                      "Mittlere LLMs (7-8B)", "LLMs moyens (7-8B)"),
    "cat_medium_d":  ("Good chatbots, code generation, translation, summarization",
                      "Goede chatbots, code-generatie, vertaling, samenvatting",
                      "Gute Chatbots, Code-Generierung, Übersetzung, Zusammenfassung",
                      "Bons chatbots, génération de code, traduction, résumé"),
    "cat_large":     ("Large LLMs (13-14B)", "Grote LLMs (13-14B)",
                      "Große LLMs (13-14B)", "Grands LLMs (13-14B)"),
    "cat_large_d":   ("Higher quality reasoning, complex tasks",
                      "Hogere kwaliteit redenering, complexe taken",
                      "Höhere Qualität beim Reasoning, komplexe Aufgaben",
                      "Raisonnement de meilleure qualité, tâches complexes"),
    "cat_xl":        ("XL LLMs (30-34B)", "XL LLMs (30-34B)",
                      "XL LLMs (30-34B)", "LLMs XL (30-34B)"),
    "cat_xl_d":      ("Near-GPT-3.5 quality, complex reasoning",
                      "Near-GPT-3.5 kwaliteit, complexe redenering",
                      "GPT-3.5-nahe Qualität, komplexes Reasoning",
                      "Qualité proche de GPT-3.5, raisonnement complexe"),
    "cat_xxl":       ("XXL LLMs (65-72B)", "XXL LLMs (65-72B)",
                      "XXL LLMs (65-72B)", "LLMs XXL (65-72B)"),
    "cat_xxl_d":     ("Top-tier quality, comparable to GPT-4",
                      "Top-tier kwaliteit, vergelijkbaar met GPT-4",
                      "Top-Qualität, vergleichbar mit GPT-4",
                      "Qualité supérieure, comparable à GPT-4"),
    "cat_embed":     ("Embedding models", "Embedding-modellen",
                      "Embedding-Modelle", "Modèles d'embedding"),
    "cat_embed_d":   ("Text to vectors — for RAG, search, classification",
                      "Tekst naar vectoren – voor RAG, zoeken, classificatie",
                      "Text zu Vektoren – für RAG, Suche, Klassifizierung",
                      "Texte en vecteurs — pour RAG, recherche, classification"),
    "cat_stt":       ("Speech-to-text (STT)", "Spraak-naar-tekst (STT)",
                      "Sprache-zu-Text (STT)", "Reconnaissance vocale (STT)"),
    "cat_stt_d":     ("Audio transcription and translation",
                      "Audio transcriptie en vertaling",
                      "Audio-Transkription und Übersetzung",
                      "Transcription audio et traduction"),
    "cat_tts":       ("Text-to-speech (TTS)", "Tekst-naar-spraak (TTS)",
                      "Text-zu-Sprache (TTS)", "Synthèse vocale (TTS)"),
    "cat_tts_d":     ("Speech synthesis, voice cloning",
                      "Spraaksynthese, voice cloning",
                      "Sprachsynthese, Voice Cloning",
                      "Synthèse vocale, clonage de voix"),
    "cat_img_s":     ("Image Generation (small)", "Image Generation (klein)",
                      "Bildgenerierung (klein)", "Génération d'images (petit)"),
    "cat_img_s_d":   ("Generate images — smaller models",
                      "Afbeeldingen genereren – kleinere modellen",
                      "Bilder generieren – kleinere Modelle",
                      "Générer des images — petits modèles"),
    "cat_img_l":     ("Image Generation (large)", "Image Generation (groot)",
                      "Bildgenerierung (groß)", "Génération d'images (grand)"),
    "cat_img_l_d":   ("High-end image generation, Flux, SDXL",
                      "High-end beeldgeneratie, Flux, SDXL",
                      "High-End Bildgenerierung, Flux, SDXL",
                      "Génération d'images haut de gamme, Flux, SDXL"),
    "cat_vision":    ("Vision / Multimodal LLMs", "Vision / Multimodal LLMs",
                      "Vision / Multimodale LLMs", "LLMs Vision / Multimodaux"),
    "cat_vision_d":  ("LLMs that understand text + images",
                      "LLMs die tekst + afbeeldingen begrijpen",
                      "LLMs die Text + Bilder verstehen",
                      "LLMs comprenant texte + images"),
    "cat_code":      ("Code-specific LLMs", "Code-specifieke LLMs",
                      "Code-spezifische LLMs", "LLMs spécialisés code"),
    "cat_code_d":    ("Code generation, debugging, explanation",
                      "Code generatie, debugging, uitleg",
                      "Code-Generierung, Debugging, Erklärung",
                      "Génération de code, débogage, explication"),
    "cat_video":     ("Video Generation", "Videogeneratie",
                      "Videogenerierung", "Génération vidéo"),
    "cat_video_d":   ("Generate video clips from text or images",
                      "Genereer videoclips uit tekst of afbeeldingen",
                      "Videoclips aus Text oder Bildern generieren",
                      "Générer des clips vidéo à partir de texte ou d'images"),
    "cat_objdet":    ("Object Detection", "Objectdetectie",
                      "Objekterkennung", "Détection d'objets"),
    "cat_objdet_d":  ("Detect and classify objects in images",
                      "Objecten in afbeeldingen detecteren en classificeren",
                      "Objekte in Bildern erkennen und klassifizieren",
                      "Détecter et classifier des objets dans les images"),
    "cat_segment":   ("Image Segmentation", "Beeldsegmentatie",
                      "Bildsegmentierung", "Segmentation d'images"),
    "cat_segment_d": ("Segment objects, pixel-level scene understanding",
                      "Segmenteer objecten, pixelniveau scèneanalyse",
                      "Objekte segmentieren, pixelgenaue Szenenanalyse",
                      "Segmenter des objets, compréhension pixel par pixel"),
    "cat_audiogen":  ("Audio / Music Generation", "Audio- / Muziekgeneratie",
                      "Audio- / Musikgenerierung", "Génération audio / musique"),
    "cat_audiogen_d":("Generate music, sound effects, audio from text",
                      "Genereer muziek, geluidseffecten, audio uit tekst",
                      "Musik, Soundeffekte, Audio aus Text generieren",
                      "Générer musique, effets sonores, audio à partir de texte"),
    "cat_docai":     ("Document AI / OCR", "Document AI / OCR",
                      "Dokument-AI / OCR", "Document AI / OCR"),
    "cat_docai_d":   ("Extract text from documents, receipts, scanned pages",
                      "Tekst extraheren uit documenten, bonnen, scans",
                      "Text aus Dokumenten, Quittungen, Scans extrahieren",
                      "Extraire du texte de documents, reçus, pages scannées"),
    # ── Type labels
    "type_llm":      ("💬 Language Models (LLM)", "💬 Taal-modellen (LLM)",
                      "💬 Sprachmodelle (LLM)", "💬 Modèles de langage (LLM)"),
    "type_code":     ("💻 Code Models", "💻 Code-modellen",
                      "💻 Code-Modelle", "💻 Modèles de code"),
    "type_vision":   ("👁️ Vision / Multimodal", "👁️ Vision / Multimodal",
                      "👁️ Vision / Multimodal", "👁️ Vision / Multimodal"),
    "type_embed":    ("🔗 Embedding Models", "🔗 Embedding-modellen",
                      "🔗 Embedding-Modelle", "🔗 Modèles d'embedding"),
    "type_stt":      ("🎤 Speech-to-text", "🎤 Spraak-naar-tekst",
                      "🎤 Sprache-zu-Text", "🎤 Reconnaissance vocale"),
    "type_tts":      ("🔊 Text-to-speech", "🔊 Tekst-naar-spraak",
                      "🔊 Text-zu-Sprache", "🔊 Synthèse vocale"),
    "type_image":    ("🎨 Image Generation", "🎨 Image Generation",
                      "🎨 Bildgenerierung", "🎨 Génération d'images"),
    "type_video":    ("🎬 Video Generation", "🎬 Videogeneratie",
                      "🎬 Videogenerierung", "🎬 Génération vidéo"),
    "type_objdet":   ("🔍 Object Detection", "🔍 Objectdetectie",
                      "🔍 Objekterkennung", "🔍 Détection d'objets"),
    "type_segment":  ("✂️ Segmentation", "✂️ Segmentatie",
                      "✂️ Segmentierung", "✂️ Segmentation"),
    "type_audiogen": ("🎵 Audio / Music Gen", "🎵 Audio- / Muziekgen.",
                      "🎵 Audio- / Musikgen.", "🎵 Gén. audio / musique"),
    "type_docai":    ("📄 Document AI / OCR", "📄 Document AI / OCR",
                      "📄 Dokument-AI / OCR", "📄 Document AI / OCR"),
    # ── Markdown
    "md_title":      ("LLM & AI Fit Report", "LLM & AI Fit Report",
                      "LLM & AI Fit Bericht", "Rapport LLM & AI Fit"),
    "md_generated":  ("Generated", "Gegenereerd", "Generiert", "Généré"),
    "md_machine":    ("Machine", "Machine", "Maschine", "Machine"),
    "md_hw":         ("Hardware", "Hardware", "Hardware", "Matériel"),
    "md_component":  ("Component", "Component", "Komponente", "Composant"),
    "md_details":    ("Details", "Details", "Details", "Détails"),
    "md_ram_total":  ("RAM Total", "RAM Totaal", "RAM Gesamt", "RAM Total"),
    "md_ram_avail":  ("RAM Available", "RAM Beschikbaar", "RAM Verfügbar", "RAM Disponible"),
    "md_mem":        ("Memory Overview", "Geheugen Overzicht",
                      "Speicherübersicht", "Aperçu mémoire"),
    "md_tools":      ("Installed AI Tools", "Geïnstalleerde AI-Tools",
                      "Installierte AI-Tools", "Outils IA installés"),
    "md_no_tools":   ("No AI tools detected.", "Geen AI-tools gedetecteerd.",
                      "Keine AI-Tools erkannt.", "Aucun outil IA détecté."),
    "md_tool":       ("Tool", "Tool", "Tool", "Outil"),
    "md_desc":       ("Description", "Beschrijving", "Beschreibung", "Description"),
    "md_ollama_loc": ("Locally installed Ollama models",
                      "Lokaal geïnstalleerde Ollama-modellen",
                      "Lokal installierte Ollama-Modelle",
                      "Modèles Ollama installés localement"),
    "md_model":      ("Model", "Model", "Modell", "Modèle"),
    "md_size":       ("Size", "Grootte", "Größe", "Taille"),
    "md_scores_cat": ("Scores per Category", "Scores per Categorie",
                      "Scores pro Kategorie", "Scores par catégorie"),
    "md_chart_t":    ("AI Fit Scores", "AI Fit Scores",
                      "AI Fit Scores", "Scores AI Fit"),
    "md_detail_cat": ("Detail per category", "Detail per categorie",
                      "Detail pro Kategorie", "Détail par catégorie"),
    "md_category":   ("Category", "Categorie", "Kategorie", "Catégorie"),
    "md_score":      ("Score", "Score", "Score", "Score"),
    "md_rating":     ("Rating", "Beoordeling", "Bewertung", "Évaluation"),
    "md_mode":       ("Mode", "Modus", "Modus", "Mode"),
    "md_examples":   ("Examples", "Voorbeelden", "Beispiele", "Exemples"),
    "md_min_vram":   ("Minimum VRAM", "Minimaal VRAM", "Minimum VRAM", "VRAM minimum"),
    "md_min_ram":    ("Minimum RAM", "Minimaal RAM", "Minimum RAM", "RAM minimum"),
    "md_cpu_only":   ("CPU-only", "CPU-only", "CPU-only", "CPU uniquement"),
    "md_feasibility":("Feasibility Distribution", "Verdeling Haalbaarheid",
                      "Machbarkeitsverteilung", "Distribution de faisabilité"),
    "md_pie_t":      ("Categories per feasibility", "Categorieën per haalbaarheid",
                      "Kategorien nach Machbarkeit", "Catégories par faisabilité"),
    "md_pie_good":   ("Runs well (≥70)", "Goed draaibaar (≥70)",
                      "Läuft gut (≥70)", "Fonctionne bien (≥70)"),
    "md_pie_ok":     ("Possible (40-69)", "Mogelijk (40-69)",
                      "Möglich (40-69)", "Possible (40-69)"),
    "md_pie_no":     ("Not feasible (<40)", "Niet haalbaar (<40)",
                      "Nicht machbar (<40)", "Non réalisable (<40)"),
    "md_top_models": ("Top Models for your Hardware", "Top Modellen voor jouw Hardware",
                      "Top-Modelle für deine Hardware", "Top modèles pour votre matériel"),
    "md_top15":      ("Top 15 Models (chart)", "Top 15 Modellen (grafiek)",
                      "Top 15 Modelle (Diagramm)", "Top 15 modèles (graphique)"),
    "md_top15_t":    ("Top 15 Models — Score on your Hardware",
                      "Top 15 Modellen — Score op jouw Hardware",
                      "Top 15 Modelle — Score auf deiner Hardware",
                      "Top 15 modèles — Score sur votre matériel"),
    "md_params":     ("Parameters", "Parameters", "Parameter", "Paramètres"),
    "md_source":     ("Source", "Bron", "Quelle", "Source"),
    "md_summary":    ("Summary", "Samenvatting", "Zusammenfassung", "Résumé"),
    "md_good":       ("Runs well", "Goed draaibaar", "Läuft gut", "Fonctionne bien"),
    "md_possible":   ("Possible but slower", "Mogelijk maar langzamer",
                      "Möglich aber langsamer", "Possible mais plus lent"),
    "md_not":        ("Not / barely feasible", "Niet / nauwelijks haalbaar",
                      "Nicht / kaum machbar", "Non / à peine réalisable"),
    "md_recs":       ("Recommendations", "Aanbevelingen", "Empfehlungen", "Recommandations"),
    "md_vram_full":  ("With **{vram} MB VRAM** you can run **7-8B models in Q4** fully on GPU",
                      "Met **{vram} MB VRAM** kun je **7-8B modellen in Q4** volledig op de GPU draaien",
                      "Mit **{vram} MB VRAM** kannst du **7-8B Modelle in Q4** vollständig auf der GPU ausführen",
                      "Avec **{vram} MB VRAM** vous pouvez exécuter des **modèles 7-8B en Q4** sur le GPU"),
    "md_vram_part":  ("With **{vram} MB VRAM** you can run **7-8B models in Q4** with GPU+CPU offloading",
                      "Met **{vram} MB VRAM** kun je **7-8B modellen in Q4** draaien met GPU+CPU offload",
                      "Mit **{vram} MB VRAM** kannst du **7-8B Modelle in Q4** mit GPU+CPU-Offloading ausführen",
                      "Avec **{vram} MB VRAM** vous pouvez exécuter des **modèles 7-8B en Q4** avec déchargement GPU+CPU"),
    "md_offload_tip":("Use `--n-gpu-layers` to offload layers to GPU for larger models",
                      "Gebruik `--n-gpu-layers` om lagen naar GPU te offloaden voor grotere modellen",
                      "Nutze `--n-gpu-layers` um Schichten auf die GPU auszulagern",
                      "Utilisez `--n-gpu-layers` pour décharger les couches sur le GPU"),
    "md_ram_tip":    ("With **{ram} GB RAM** you can run **30B+ models CPU-only** in Q4 (~2-5 tok/s)",
                      "Met **{ram} GB RAM** kun je **30B+ modellen CPU-only** draaien in Q4 (~2-5 tok/s)",
                      "Mit **{ram} GB RAM** kannst du **30B+ Modelle CPU-only** in Q4 ausführen (~2-5 Tok/s)",
                      "Avec **{ram} GB RAM** vous pouvez exécuter **30B+ en CPU** en Q4 (~2-5 tok/s)"),
    "md_q4_tip":     ("**Q4_K_M** quantization is the sweet spot between quality and speed",
                      "**Q4_K_M** kwantisatie is de sweet spot tussen kwaliteit en snelheid",
                      "**Q4_K_M** ist der Sweet Spot zwischen Qualität und Geschwindigkeit",
                      "**Q4_K_M** est le compromis idéal entre qualité et vitesse"),
    "md_gpu_cfg":    ("Use `--num-gpu` / `-ngl` to configure GPU offloading",
                      "Gebruik `--num-gpu` / `-ngl` om GPU-offload te configureren",
                      "Nutze `--num-gpu` / `-ngl` um GPU-Offloading zu konfigurieren",
                      "Utilisez `--num-gpu` / `-ngl` pour configurer le déchargement GPU"),
    "md_quickstart": ("Quick start", "Snel starten", "Schnellstart", "Démarrage rapide"),
    "md_install_ol": ("Install Ollama", "Installeer Ollama",
                      "Ollama installieren", "Installer Ollama"),
    "md_dl_model":   ("Download a good 7B model", "Download een goed 7B model",
                      "Ein gutes 7B-Modell herunterladen", "Télécharger un bon modèle 7B"),
    "md_rec_ollama": ("Recommended Ollama models", "Aanbevolen Ollama modellen",
                      "Empfohlene Ollama-Modelle", "Modèles Ollama recommandés"),
    "md_vram_fit":   ("VRAM Fit – Which models fit?", "VRAM Fit – Welke modellen passen?",
                      "VRAM Fit – Welche Modelle passen?", "VRAM Fit – Quels modèles rentrent ?"),
    "md_fits":       ("Fits", "Past", "Passt", "Rentre"),
    "md_no_fit":     ("Doesn't fit", "Past niet", "Passt nicht", "Ne rentre pas"),
    "md_needed":     ("needed", "nodig", "benötigt", "nécessaire"),
    "md_catalog":    ("Catalog: {n} models (updated: {ts})",
                      "Catalogus: {n} modellen (geüpdatet: {ts})",
                      "Katalog: {n} Modelle (aktualisiert: {ts})",
                      "Catalogue : {n} modèles (mis à jour : {ts})"),
    "md_report_gen": ("Report generated: {ts} by LLM & AI Fit Scanner v1.2",
                      "Rapport gegenereerd: {ts} door LLM & AI Fit Scanner v1.2",
                      "Bericht erstellt: {ts} von LLM & AI Fit Scanner v1.2",
                      "Rapport généré : {ts} par LLM & AI Fit Scanner v1.2"),
    # ── Report file messages
    "report_saved":  ("📄 Report saved: {path}", "📄 Rapport opgeslagen: {path}",
                      "📄 Bericht gespeichert: {path}", "📄 Rapport enregistré : {path}"),
    "report_disable":("Use --no-report to disable", "Gebruik --no-report om dit uit te schakelen",
                      "Nutze --no-report zum Deaktivieren", "Utilisez --no-report pour désactiver"),
}


# ─── Hardware detection ─────────────────────────────────────────────────────

def get_cpu_info():
    info = {
        "name": platform.processor() or "Unknown",
        "cores": psutil.cpu_count(logical=False) or 1,
        "threads": psutil.cpu_count(logical=True) or 1,
        "freq_mhz": 0,
    }
    if os.path.exists("/proc/cpuinfo"):
        with open("/proc/cpuinfo") as f:
            for line in f:
                if "model name" in line:
                    info["name"] = line.split(":")[1].strip()
                    break
    freq = psutil.cpu_freq()
    if freq:
        info["freq_mhz"] = freq.max or freq.current
    return info


def get_ram_info():
    mem = psutil.virtual_memory()
    swap = psutil.swap_memory()
    return {
        "total_gb": mem.total / (1024 ** 3),
        "available_gb": mem.available / (1024 ** 3),
        "swap_gb": swap.total / (1024 ** 3),
    }


def get_gpu_info():
    gpus = []
    if shutil.which("nvidia-smi"):
        try:
            r = subprocess.run(
                ["nvidia-smi",
                 "--query-gpu=name,memory.total,memory.free,compute_cap",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=10,
            )
            for line in r.stdout.strip().splitlines():
                parts = [p.strip() for p in line.split(",")]
                if len(parts) >= 4:
                    gpus.append({
                        "name": parts[0],
                        "vram_mb": int(float(parts[1])),
                        "vram_free_mb": int(float(parts[2])),
                        "compute_cap": parts[3],
                        "vendor": "NVIDIA",
                    })
        except (subprocess.TimeoutExpired, Exception):
            pass
    if shutil.which("rocm-smi"):
        try:
            r = subprocess.run(
                ["rocm-smi", "--showmeminfo", "vram", "--csv"],
                capture_output=True, text=True, timeout=10,
            )
            for line in r.stdout.strip().splitlines()[1:]:
                parts = line.split(",")
                if len(parts) >= 2:
                    gpus.append({
                        "name": "AMD GPU",
                        "vram_mb": int(float(parts[0])) // (1024 * 1024),
                        "vram_free_mb": int(float(parts[0])) // (1024 * 1024),
                        "compute_cap": "N/A",
                        "vendor": "AMD",
                    })
        except (subprocess.TimeoutExpired, Exception):
            pass
    return gpus


def get_installed_tools():
    tools = {}
    check = {
        "ollama": "Ollama – LLM management",
        "llama-server": "llama.cpp server",
        "llama-cli": "llama.cpp CLI",
        "koboldcpp": "KoboldCpp – roleplay/creative",
        "text-generation-server": "TGI (Text Generation Inference)",
        "vllm": "vLLM – high-throughput serving",
        "whisper": "Whisper – speech-to-text",
        "stable-diffusion": "Stable Diffusion CLI",
    }
    for cmd, desc in check.items():
        path = shutil.which(cmd)
        if path:
            tools[cmd] = {"path": path, "desc": desc}
    py_tools = {
        "transformers": "HuggingFace Transformers",
        "torch": "PyTorch",
        "tensorflow": "TensorFlow",
        "onnxruntime": "ONNX Runtime",
        "ctransformers": "CTransformers (GGUF in Python)",
        "llama_cpp": "llama-cpp-python bindings",
        "diffusers": "HuggingFace Diffusers (image/video gen)",
        "whisper": "OpenAI Whisper",
        "faster_whisper": "Faster Whisper",
        "TTS": "Coqui TTS (text-to-speech)",
        "bark": "Bark (text-to-speech)",
        "sentence_transformers": "Sentence Transformers (embeddings)",
        "chromadb": "ChromaDB (vector store / RAG)",
        "langchain": "LangChain (LLM orchestration)",
        "auto_gptq": "AutoGPTQ (optimized quantization)",
        "exllama": "ExLlama (fast GPTQ inference)",
        "bitsandbytes": "BitsAndBytes (4/8-bit quantization)",
        "ultralytics": "Ultralytics (YOLO object detection)",
        "segment_anything": "SAM (Segment Anything)",
        "audiocraft": "AudioCraft (music/audio generation)",
        "pytesseract": "Tesseract OCR (document AI)",
        "easyocr": "EasyOCR (document AI)",
    }
    for mod, desc in py_tools.items():
        try:
            __import__(mod)
            tools[f"python:{mod}"] = {"path": "python", "desc": desc}
        except ImportError:
            pass
    return tools


def get_ollama_local_models():
    """Haal lijst van lokaal geïnstalleerde Ollama-modellen op."""
    if not shutil.which("ollama"):
        return []
    try:
        r = subprocess.run(
            ["ollama", "list"],
            capture_output=True, text=True, timeout=15,
        )
        models = []
        for line in r.stdout.strip().splitlines()[1:]:  # skip header
            parts = line.split()
            if parts:
                name = parts[0]
                size_str = ""
                # Zoek de size kolom (bv "4.1 GB")
                for i, p in enumerate(parts):
                    if p in ("GB", "MB", "KB") and i > 0:
                        size_str = f"{parts[i-1]} {p}"
                        break
                models.append({"name": name, "size": size_str})
        return models
    except (subprocess.TimeoutExpired, FileNotFoundError, Exception):
        return []


# ─── Online catalogus ───────────────────────────────────────────────────────

def _http_get_json(url, timeout=15):
    """Simpele JSON GET zonder externe dependencies."""
    req = Request(url, headers={"User-Agent": "llm-ai-fit-scanner/1.0"})
    with urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


def _guess_params_b(model_id, safetensors_params=None):
    """Probeer het aantal parameters (in miljarden) te schatten uit de naam."""
    if safetensors_params and isinstance(safetensors_params, dict):
        total = safetensors_params.get("total", 0)
        if total > 0:
            return total / 1e9
    # Regex op de naam: "7b", "13B", "70b", "1.5B", etc
    m = re.search(r"(\d+\.?\d*)\s*[bB]", model_id)
    if m:
        return float(m.group(1))
    return None


def _classify_model_type(model_id, tags, pipeline_tag=""):
    """Classificeer een model in een categorie op basis van tags en naam."""
    model_lower = model_id.lower()
    tags_lower = [t.lower() for t in tags] if tags else []
    pipe = pipeline_tag.lower()

    # Embedding
    if pipe in ("feature-extraction", "sentence-similarity") or "embed" in model_lower:
        return "embedding"
    # STT
    if pipe in ("automatic-speech-recognition",) or "whisper" in model_lower:
        return "stt"
    # TTS
    if pipe in ("text-to-speech",) or any(t in model_lower for t in ("tts", "bark", "piper")):
        return "tts"
    # Video gen
    if pipe in ("text-to-video",) or any(t in model_lower for t in (
            "cogvideo", "animatediff", "mochi", "stable-video", "videocrafter")):
        return "video-gen"
    # Image gen
    if pipe in ("text-to-image",) or any(t in model_lower for t in (
            "stable-diffusion", "sdxl", "flux", "diffusion")):
        return "image-gen"
    # Object detection
    if pipe in ("object-detection",) or any(t in model_lower for t in (
            "yolo", "detr", "grounding-dino", "owlvit")):
        return "object-detection"
    # Image segmentation
    if pipe in ("image-segmentation",) or any(t in model_lower for t in (
            "sam", "segment-anything", "mask2former", "oneformer")):
        return "segmentation"
    # Audio / music gen
    if pipe in ("text-to-audio",) or any(t in model_lower for t in (
            "musicgen", "audiocraft", "riffusion", "stable-audio")):
        return "audio-gen"
    # Document AI / OCR
    if pipe in ("document-question-answering", "image-to-text") or any(t in model_lower for t in (
            "trocr", "donut", "layoutlm", "florence", "got-ocr", "nougat")):
        return "document-ai"
    # Vision / multimodal
    if pipe in ("image-text-to-text", "visual-question-answering") or any(
            t in model_lower for t in ("llava", "vision", "vl-", "-vl", "multimodal", "minicpm-v")):
        return "vision"
    # Code
    if any(t in model_lower for t in ("code", "coder", "starcoder", "codellama", "deepseek-coder")):
        return "code-llm"
    # Generic LLM
    if pipe in ("text-generation", "text2text-generation", "conversational", ""):
        return "llm"
    return "llm"


def _param_to_size_bucket(params_b):
    """Map parameter count to size bucket."""
    if params_b is None:
        return "medium"
    if params_b <= 3.5:
        return "klein"
    elif params_b <= 9:
        return "medium"
    elif params_b <= 20:
        return "groot"
    elif params_b <= 45:
        return "xl"
    else:
        return "xxl"


def _vram_estimate_q4(params_b):
    """Schat VRAM-gebruik in MB voor Q4 kwantisatie (~0.6 GB per B params)."""
    if params_b is None:
        return 4000
    return int(params_b * 600)


def _ram_estimate_q4(params_b):
    """Schat RAM-gebruik in GB voor Q4 CPU-only (~0.75 GB per B params + overhead)."""
    if params_b is None:
        return 8
    return max(2, int(params_b * 0.75) + 2)


def fetch_huggingface_models(limit=100):
    """
    Haal trending/populaire GGUF- en text-generation-modellen op
    van de HuggingFace API.
    """
    models = []
    # Zoek op meerdere relevante categorieën
    searches = [
        ("gguf", "text-generation"),
        ("gguf", ""),
        ("", "text-to-image"),
        ("", "automatic-speech-recognition"),
        ("", "text-to-speech"),
        ("", "feature-extraction"),
        ("", "image-text-to-text"),
        ("", "text-to-video"),
        ("", "object-detection"),
        ("", "image-segmentation"),
        ("", "text-to-audio"),
        ("", "document-question-answering"),
        ("", "image-to-text"),
    ]
    seen = set()
    for tag_filter, pipeline in searches:
        params = [f"sort=downloads", f"direction=-1", f"limit={limit}"]
        if tag_filter:
            params.append(f"filter={tag_filter}")
        if pipeline:
            params.append(f"pipeline_tag={pipeline}")
        url = f"https://huggingface.co/api/models?{'&'.join(params)}"
        try:
            data = _http_get_json(url, timeout=20)
        except (URLError, Exception):
            continue
        for item in data:
            mid = item.get("id", "")
            if mid in seen:
                continue
            seen.add(mid)
            tags = item.get("tags", [])
            pipeline_tag = item.get("pipeline_tag", "")
            params_b = _guess_params_b(mid, item.get("safetensors", {}).get("parameters"))
            model_type = _classify_model_type(mid, tags, pipeline_tag)
            downloads = item.get("downloads", 0)
            likes = item.get("likes", 0)
            models.append({
                "id": mid,
                "type": model_type,
                "params_b": params_b,
                "size_bucket": _param_to_size_bucket(params_b) if model_type in ("llm", "code-llm", "vision") else None,
                "vram_q4_mb": _vram_estimate_q4(params_b) if params_b else None,
                "ram_q4_gb": _ram_estimate_q4(params_b) if params_b else None,
                "downloads": downloads,
                "likes": likes,
                "pipeline": pipeline_tag,
                "source": "huggingface",
                "has_gguf": "gguf" in [t.lower() for t in tags],
            })
    return models


def fetch_ollama_library():
    """
    Haal populaire modellen op via de Ollama API.
    Ollama's zoek-endpoint is beperkt; we gebruiken bekende model-namen.
    """
    # Ollama has no public "list all" API, but we can query known popular models
    known_models = [
        "llama3.1", "llama3.2", "llama3.3",
        "mistral", "mixtral", "gemma2", "gemma3",
        "qwen2.5", "qwen2.5-coder", "qwen3",
        "phi3", "phi4",
        "deepseek-r1", "deepseek-coder-v2", "deepseek-v2.5",
        "codellama", "starcoder2",
        "llava", "llava-llama3",
        "nomic-embed-text", "mxbai-embed-large", "snowflake-arctic-embed",
        "whisper",
        "stable-diffusion",
        "command-r",
        "yi",
        "moondream", "bakllava",
        "minicpm-v",
    ]
    models = []
    for name in known_models:
        url = f"https://ollama.com/api/show"
        # We proberen de tags-pagina te scrapen – als dat niet lukt, nemen we de bekende lijst
        try:
            # Ollama heeft geen publieke API voor metadata, maar we kunnen de
            # modelpagina proberen
            tag_url = f"https://registry.ollama.ai/v2/library/{name}/tags/list"
            data = _http_get_json(tag_url, timeout=10)
            tags_list = data.get("tags", [])
            params_b = _guess_params_b(name)
            model_type = _classify_model_type(name, [], "text-generation")
            for tag_info in tags_list[:8]:  # max 8 varianten per model
                tag_name = tag_info.get("name", "latest")
                full_name = f"{name}:{tag_name}"
                # Probeer de grootte uit de tag te halen
                tag_params_b = _guess_params_b(tag_name) or params_b
                models.append({
                    "id": full_name,
                    "type": model_type,
                    "params_b": tag_params_b,
                    "size_bucket": _param_to_size_bucket(tag_params_b) if model_type in ("llm", "code-llm", "vision") else None,
                    "vram_q4_mb": _vram_estimate_q4(tag_params_b) if tag_params_b else None,
                    "ram_q4_gb": _ram_estimate_q4(tag_params_b) if tag_params_b else None,
                    "downloads": None,
                    "likes": None,
                    "pipeline": "text-generation",
                    "source": "ollama",
                    "has_gguf": True,  # Ollama is altijd GGUF
                })
        except (URLError, Exception):
            # Voeg het model toe met alleen de basisnaam
            params_b = _guess_params_b(name)
            model_type = _classify_model_type(name, [], "text-generation")
            models.append({
                "id": name,
                "type": model_type,
                "params_b": params_b,
                "size_bucket": _param_to_size_bucket(params_b) if model_type in ("llm", "code-llm", "vision") else None,
                "vram_q4_mb": _vram_estimate_q4(params_b) if params_b else None,
                "ram_q4_gb": _ram_estimate_q4(params_b) if params_b else None,
                "downloads": None,
                "likes": None,
                "pipeline": "text-generation",
                "source": "ollama",
                "has_gguf": True,
            })
    return models


def update_catalog():
    """Fetch online models and save as local catalog."""
    print(f"\n{BOLD}{CYAN}  {T('upd_updating')}{RESET}\n")

    all_models = []

    # HuggingFace
    print(f"  {DIM}{T('upd_hf')}{RESET}", end=" ", flush=True)
    try:
        hf_models = fetch_huggingface_models(limit=50)
        print(f"{GREEN}{T('upd_found', n=len(hf_models))}{RESET}")
        all_models.extend(hf_models)
    except Exception as e:
        print(f"{RED}{T('upd_error', e=e)}{RESET}")

    # Ollama registry
    print(f"  {DIM}{T('upd_ollama')}{RESET}", end=" ", flush=True)
    try:
        ol_models = fetch_ollama_library()
        print(f"{GREEN}{T('upd_found', n=len(ol_models))}{RESET}")
        all_models.extend(ol_models)
    except Exception as e:
        print(f"{RED}{T('upd_error', e=e)}{RESET}")

    if not all_models:
        print(f"\n  {RED}{T('upd_no_models')}{RESET}\n")
        return False

    # Deduplicate on id
    seen = set()
    unique = []
    for m in all_models:
        if m["id"] not in seen:
            seen.add(m["id"])
            unique.append(m)

    catalog = {
        "updated": time.strftime("%Y-%m-%d %H:%M:%S"),
        "models": unique,
    }

    with open(CATALOG_FILE, "w") as f:
        json.dump(catalog, f, indent=2, ensure_ascii=False)

    print(f"\n  {GREEN}{T('upd_saved', n=len(unique), file=CATALOG_FILE.name)}{RESET}")
    print(f"  {DIM}{T('upd_last', ts=catalog['updated'])}{RESET}\n")
    return True


def load_catalog():
    """Laad de lokale catalogus (indien aanwezig)."""
    if CATALOG_FILE.exists():
        with open(CATALOG_FILE) as f:
            return json.load(f)
    return None


# ─── Built-in categories (fallback) ─────────────────────────────────────────

def get_categories():
    """Return built-in categories with translated names/descriptions."""
    return [
        {
            "categorie": T("cat_small"),
            "beschrijving": T("cat_small_d"),
            "vram_min_mb": 1500,
            "ram_min_gb": 4,
            "ram_cpu_only_gb": 4,
            "voorbeelden": [
                "Phi-3-mini (3.8B) Q4", "TinyLlama (1.1B)", "StableLM-2 (1.6B)",
                "Qwen2.5-1.5B", "Gemma-2B", "SmolLM2 (1.7B)",
            ],
            "match_types": ["llm", "code-llm"],
            "match_bucket": "klein",
        },
        {
            "categorie": T("cat_medium"),
            "beschrijving": T("cat_medium_d"),
            "vram_min_mb": 5000,
            "ram_min_gb": 8,
            "ram_cpu_only_gb": 8,
            "voorbeelden": [
                "Llama-3.1-8B Q4", "Mistral-7B Q4", "Gemma-2-9B Q4",
                "Qwen2.5-7B Q4", "DeepSeek-R1-8B Q4",
            ],
            "match_types": ["llm", "code-llm"],
            "match_bucket": "medium",
        },
        {
            "categorie": T("cat_large"),
            "beschrijving": T("cat_large_d"),
            "vram_min_mb": 8000,
            "ram_min_gb": 16,
            "ram_cpu_only_gb": 16,
            "voorbeelden": [
                "Llama-2-13B Q4", "Qwen2.5-14B Q4", "CodeLlama-13B Q4",
                "Phi-4 (14B) Q4",
            ],
            "match_types": ["llm", "code-llm"],
            "match_bucket": "groot",
        },
        {
            "categorie": T("cat_xl"),
            "beschrijving": T("cat_xl_d"),
            "vram_min_mb": 20000,
            "ram_min_gb": 32,
            "ram_cpu_only_gb": 24,
            "voorbeelden": [
                "DeepSeek-R1-32B Q4", "Qwen2.5-32B Q4",
                "CodeLlama-34B Q4", "Yi-34B Q4",
            ],
            "match_types": ["llm", "code-llm"],
            "match_bucket": "xl",
        },
        {
            "categorie": T("cat_xxl"),
            "beschrijving": T("cat_xxl_d"),
            "vram_min_mb": 40000,
            "ram_min_gb": 64,
            "ram_cpu_only_gb": 48,
            "voorbeelden": [
                "Llama-3.1-70B Q4", "Qwen2.5-72B Q4",
                "DeepSeek-V2.5 Q4", "Mixtral-8x22B Q2",
            ],
            "match_types": ["llm", "code-llm"],
            "match_bucket": "xxl",
        },
        {
            "categorie": T("cat_embed"),
            "beschrijving": T("cat_embed_d"),
            "vram_min_mb": 500,
            "ram_min_gb": 2,
            "ram_cpu_only_gb": 2,
            "voorbeelden": [
                "nomic-embed-text", "all-MiniLM-L6-v2",
                "bge-large-en", "mxbai-embed-large",
            ],
            "match_types": ["embedding"],
            "match_bucket": None,
        },
        {
            "categorie": T("cat_stt"),
            "beschrijving": T("cat_stt_d"),
            "vram_min_mb": 1500,
            "ram_min_gb": 4,
            "ram_cpu_only_gb": 6,
            "voorbeelden": [
                "Whisper-small", "Whisper-medium", "Whisper-large-v3",
                "Faster-Whisper",
            ],
            "match_types": ["stt"],
            "match_bucket": None,
        },
        {
            "categorie": T("cat_tts"),
            "beschrijving": T("cat_tts_d"),
            "vram_min_mb": 2000,
            "ram_min_gb": 4,
            "ram_cpu_only_gb": 6,
            "voorbeelden": [
                "Coqui XTTS-v2", "Bark", "Piper", "StyleTTS2",
            ],
            "match_types": ["tts"],
            "match_bucket": None,
        },
        {
            "categorie": T("cat_img_s"),
            "beschrijving": T("cat_img_s_d"),
            "vram_min_mb": 4000,
            "ram_min_gb": 8,
            "ram_cpu_only_gb": 16,
            "voorbeelden": [
                "Stable Diffusion 1.5", "SDXL-Turbo",
                "LCM (Latent Consistency)", "SD-Turbo",
            ],
            "match_types": ["image-gen"],
            "match_bucket": None,
        },
        {
            "categorie": T("cat_img_l"),
            "beschrijving": T("cat_img_l_d"),
            "vram_min_mb": 8000,
            "ram_min_gb": 16,
            "ram_cpu_only_gb": 32,
            "voorbeelden": [
                "SDXL (full)", "Flux.1-dev", "Flux.1-schnell",
            ],
            "match_types": ["image-gen-large"],
            "match_bucket": None,
        },
        {
            "categorie": T("cat_vision"),
            "beschrijving": T("cat_vision_d"),
            "vram_min_mb": 5000,
            "ram_min_gb": 10,
            "ram_cpu_only_gb": 12,
            "voorbeelden": [
                "LLaVA-1.6 (7B) Q4", "Qwen2-VL-7B Q4",
                "MiniCPM-V 2.6", "Phi-3-vision",
            ],
            "match_types": ["vision"],
            "match_bucket": None,
        },
        {
            "categorie": T("cat_code"),
            "beschrijving": T("cat_code_d"),
            "vram_min_mb": 5000,
            "ram_min_gb": 8,
            "ram_cpu_only_gb": 8,
            "voorbeelden": [
                "DeepSeek-Coder-V2-Lite Q4", "CodeLlama-7B Q4",
                "StarCoder2-7B Q4", "Qwen2.5-Coder-7B Q4",
            ],
            "match_types": ["code-llm"],
            "match_bucket": None,
        },
        {
            "categorie": T("cat_video"),
            "beschrijving": T("cat_video_d"),
            "vram_min_mb": 12000,
            "ram_min_gb": 16,
            "ram_cpu_only_gb": 32,
            "voorbeelden": [
                "CogVideoX-2B", "AnimateDiff", "Mochi-1",
                "Stable Video Diffusion",
            ],
            "match_types": ["video-gen"],
            "match_bucket": None,
        },
        {
            "categorie": T("cat_objdet"),
            "beschrijving": T("cat_objdet_d"),
            "vram_min_mb": 1000,
            "ram_min_gb": 2,
            "ram_cpu_only_gb": 4,
            "voorbeelden": [
                "YOLOv8", "YOLO11", "DETR", "RT-DETR",
                "Grounding DINO",
            ],
            "match_types": ["object-detection"],
            "match_bucket": None,
        },
        {
            "categorie": T("cat_segment"),
            "beschrijving": T("cat_segment_d"),
            "vram_min_mb": 2000,
            "ram_min_gb": 4,
            "ram_cpu_only_gb": 8,
            "voorbeelden": [
                "SAM (Segment Anything)", "SAM 2", "Mask2Former",
                "OneFormer",
            ],
            "match_types": ["segmentation"],
            "match_bucket": None,
        },
        {
            "categorie": T("cat_audiogen"),
            "beschrijving": T("cat_audiogen_d"),
            "vram_min_mb": 4000,
            "ram_min_gb": 8,
            "ram_cpu_only_gb": 16,
            "voorbeelden": [
                "MusicGen", "AudioCraft", "Riffusion",
                "Stable Audio Open",
            ],
            "match_types": ["audio-gen"],
            "match_bucket": None,
        },
        {
            "categorie": T("cat_docai"),
            "beschrijving": T("cat_docai_d"),
            "vram_min_mb": 1000,
            "ram_min_gb": 2,
            "ram_cpu_only_gb": 4,
            "voorbeelden": [
                "TrOCR", "Donut", "LayoutLMv3", "Florence-2",
                "GOT-OCR2",
            ],
            "match_types": ["document-ai"],
            "match_bucket": None,
        },
    ]


# ─── Catalogus → modellen per categorie ─────────────────────────────────────

def enrich_categories_with_catalog(categories, catalog):
    """Enrich categories with online models from the catalog."""
    if not catalog or "models" not in catalog:
        return categories
    for cat in categories:
        match_types = cat.get("match_types", [])
        match_bucket = cat.get("match_bucket")
        matched = []
        for m in catalog["models"]:
            mtype = m.get("type", "")
            mbucket = m.get("size_bucket")
            if mtype in match_types:
                if match_bucket is None or mbucket == match_bucket:
                    score = m.get("downloads", 0) or 0
                    matched.append((score, m["id"]))
        matched.sort(reverse=True)
        existing = set(cat["voorbeelden"])
        for _, mid in matched[:6]:
            short = mid.split("/")[-1] if "/" in mid else mid
            if short not in existing and mid not in existing:
                cat["voorbeelden"].append(short)
                existing.add(short)
    return categories


# ─── Scoring ────────────────────────────────────────────────────────────────

def compute_score(cat, gpu_vram_mb, ram_total_gb, ram_avail_gb, cpu_threads):
    vram_need = cat["vram_min_mb"]
    ram_need = cat["ram_min_gb"]
    ram_cpu_only = cat["ram_cpu_only_gb"]

    score = 0
    mode = ""

    if gpu_vram_mb > 0 and gpu_vram_mb >= vram_need * 0.5:
        vram_ratio = min(gpu_vram_mb / vram_need, 1.0)
        ram_ratio = min(ram_avail_gb / ram_need, 1.0)
        score = int(vram_ratio * 70 + ram_ratio * 20 + min(cpu_threads / 8, 1.0) * 10)
        if vram_ratio >= 1.0:
            mode = T("gpu_full")
        else:
            mode = T("gpu_offload", pct=f"{vram_ratio*100:.0f}")
    elif ram_total_gb >= ram_cpu_only * 0.75:
        ram_ratio = min(ram_total_gb / ram_cpu_only, 1.0)
        cpu_ratio = min(cpu_threads / 8, 1.0)
        score = int(ram_ratio * 55 + cpu_ratio * 15)
        mode = T("cpu_only")
        if ram_ratio >= 1.0:
            score = min(score + 5, 75)
    else:
        score = 0
        mode = T("not_feasible")

    return max(0, min(100, score)), mode


def compute_model_score(model, gpu_vram_mb, ram_total_gb, ram_avail_gb, cpu_threads):
    """Score an individual model based on estimated requirements."""
    vram = model.get("vram_q4_mb")
    ram = model.get("ram_q4_gb")
    if vram is None or ram is None:
        return None, T("unknown")

    cat_like = {
        "vram_min_mb": vram,
        "ram_min_gb": ram,
        "ram_cpu_only_gb": max(ram, 4),
    }
    return compute_score(cat_like, gpu_vram_mb, ram_total_gb, ram_avail_gb, cpu_threads)


def score_bar(score, width=30):
    filled = int(score / 100 * width)
    empty = width - filled
    if score >= 75:
        color = GREEN
    elif score >= 40:
        color = YELLOW
    else:
        color = RED
    return f"{color}{'█' * filled}{'░' * empty}{RESET} {color}{score:3d}/100{RESET}"


def score_label(score):
    if score >= 85:
        return f"{GREEN}★★★★★  {T('excellent')}{RESET}"
    elif score >= 70:
        return f"{GREEN}★★★★   {T('good')}{RESET}"
    elif score >= 50:
        return f"{YELLOW}★★★    {T('fair')}{RESET}"
    elif score >= 30:
        return f"{YELLOW}★★     {T('poor')}{RESET}"
    elif score >= 10:
        return f"{RED}★      {T('difficult')}{RESET}"
    else:
        return f"{RED}✗      {T('not_possible')}{RESET}"


# ─── Display ────────────────────────────────────────────────────────────────

def print_hardware(cpu, ram, gpus, tools):
    print(f"{BOLD}  {T('hw_header')}{RESET}")
    print(f"  {'─' * 50}")
    print(f"  CPU:    {cpu['name']}")
    print(f"          {cpu['cores']} cores / {cpu['threads']} threads @ {cpu['freq_mhz']:.0f} MHz")
    print(f"  RAM:    {ram['total_gb']:.1f} GB {T('total')} / {ram['available_gb']:.1f} GB {T('available')}")
    print(f"  Swap:   {ram['swap_gb']:.1f} GB")
    if gpus:
        for g in gpus:
            print(f"  GPU:    {g['name']} ({g['vendor']})")
            print(f"          {g['vram_mb']} MB VRAM ({g['vram_free_mb']} MB {T('free')})"
                  f" | Compute {g['compute_cap']}")
    else:
        print(f"  GPU:    {RED}{T('no_gpu')}{RESET}")
    print()

    if tools:
        print(f"{BOLD}  {T('tools_header')}{RESET}")
        print(f"  {'─' * 50}")
        for name, info in sorted(tools.items()):
            print(f"  {GREEN}✓{RESET} {info['desc']:40s} ({name})")
        print()


def print_ollama_local(ollama_models):
    if not ollama_models:
        return
    print(f"{BOLD}  {T('ollama_header')}{RESET}")
    print(f"  {'─' * 50}")
    for m in ollama_models:
        size = f" ({m['size']})" if m.get("size") else ""
        print(f"  {GREEN}●{RESET} {m['name']}{size}")
    print()


def print_category_scores(categories, total_vram, ram, cpu_threads):
    print(f"{BOLD}{'═' * 70}{RESET}")
    print(f"{BOLD}{CYAN}  {T('scores_header')}{RESET}")
    print(f"{BOLD}{'═' * 70}{RESET}\n")

    results = []
    for cat in categories:
        score, mode = compute_score(
            cat, total_vram, ram["total_gb"], ram["available_gb"], cpu_threads
        )
        results.append((cat, score, mode))

    results.sort(key=lambda x: x[1], reverse=True)

    for cat, score, mode in results:
        print(f"  {BOLD}{cat['categorie']}{RESET}")
        print(f"  {DIM}{cat['beschrijving']}{RESET}")
        print(f"  {score_bar(score)}  {score_label(score)}")
        print(f"  {DIM}{T('mode_label')}: {mode}{RESET}")
        print(f"  {DIM}{T('examples_label')}: {', '.join(cat['voorbeelden'][:6])}{RESET}")
        print()

    return results


def print_individual_models(catalog, total_vram, ram, cpu_threads):
    """Show individual model scores if a catalog exists."""
    if not catalog or "models" not in catalog:
        return

    models = catalog["models"]
    scored = []
    for m in models:
        score, mode = compute_model_score(
            m, total_vram, ram["total_gb"], ram["available_gb"], cpu_threads
        )
        if score is not None:
            scored.append((score, mode, m))

    if not scored:
        return

    scored.sort(key=lambda x: x[0], reverse=True)

    print(f"{BOLD}{'═' * 70}{RESET}")
    print(f"{BOLD}{CYAN}  {T('top_header')}{RESET}")
    print(f"{BOLD}{'═' * 70}{RESET}\n")

    type_labels = {
        "llm": T("type_llm"),
        "code-llm": T("type_code"),
        "vision": T("type_vision"),
        "embedding": T("type_embed"),
        "stt": T("type_stt"),
        "tts": T("type_tts"),
        "image-gen": T("type_image"),
        "video-gen": T("type_video"),
        "object-detection": T("type_objdet"),
        "segmentation": T("type_segment"),
        "audio-gen": T("type_audiogen"),
        "document-ai": T("type_docai"),
    }

    by_type = {}
    for score, mode, m in scored:
        mtype = m.get("type", "llm")
        by_type.setdefault(mtype, []).append((score, mode, m))

    for mtype, label in type_labels.items():
        items = by_type.get(mtype, [])
        if not items:
            continue

        print(f"  {BOLD}{label}{RESET}")
        for score, mode, m in items[:8]:
            params_str = f"{m['params_b']:.1f}B" if m.get('params_b') else "?"
            gguf = f" {GREEN}GGUF{RESET}" if m.get("has_gguf") else ""
            src = m.get("source", "")
            src_tag = f" [{src}]" if src else ""
            color = GREEN if score >= 70 else (YELLOW if score >= 40 else RED)
            print(f"    {color}{score:3d}/100{RESET} {m['id'][:50]:50s} "
                  f"{DIM}{params_str:>6s}{gguf}{DIM} {mode}{src_tag}{RESET}")
        print()


# ─── Markdown Report ────────────────────────────────────────────────────

def _md_score_bar(score, width=25):
    """Unicode balk voor in Markdown."""
    filled = int(score / 100 * width)
    empty = width - filled
    return f"`{'█' * filled}{'░' * empty}` **{score}/100**"


def _md_score_emoji(score):
    if score >= 85:
        return f"🟢 {T('excellent')}"
    elif score >= 70:
        return f"🟢 {T('good')}"
    elif score >= 50:
        return f"🟡 {T('fair')}"
    elif score >= 30:
        return f"🟠 {T('poor')}"
    elif score >= 10:
        return f"🔴 {T('difficult')}"
    else:
        return f"⛔ {T('not_possible')}"


def generate_markdown_report(cpu, ram, gpus, tools, ollama_models, categories,
                              catalog, total_vram):
    """Generate a full Markdown report with charts and details."""
    now = datetime.now()
    ts = now.strftime("%d-%m-%Y_%H:%M:%S")
    ts_display = now.strftime("%d-%m-%Y_%H:%M:%S")
    hostname = platform.node() or "unknown"

    # Score all categories
    cat_results = []
    for cat in categories:
        score, mode = compute_score(
            cat, total_vram, ram["total_gb"], ram["available_gb"], cpu["threads"]
        )
        cat_results.append((cat, score, mode))
    cat_results.sort(key=lambda x: x[1], reverse=True)

    # Score individual models
    model_results_by_type = {}
    if catalog and "models" in catalog:
        for m in catalog["models"]:
            score, mode = compute_model_score(
                m, total_vram, ram["total_gb"], ram["available_gb"], cpu["threads"]
            )
            if score is not None:
                mtype = m.get("type", "llm")
                model_results_by_type.setdefault(mtype, []).append((score, mode, m))
        for mtype in model_results_by_type:
            model_results_by_type[mtype].sort(key=lambda x: x[0], reverse=True)

    lines = []
    w = lines.append

    # ── Header
    w(f"# 🖥️ {T('md_title')}")
    w(f"")
    w(f"> **{T('md_generated')}:** {ts_display}  ")
    w(f"> **{T('md_machine')}:** {hostname}  ")
    w(f"> **CPU:** {cpu['name']}  ")
    w(f"> **OS:** {platform.system()} {platform.release()}")
    w(f"")
    w(f"---")
    w(f"")

    # ── Hardware
    w(f"## ⚙️ {T('md_hw')}")
    w(f"")
    w(f"| {T('md_component')} | {T('md_details')} |")
    w(f"|-----------|---------|")
    w(f"| **CPU** | {cpu['name']} |")
    w(f"| **Cores / Threads** | {cpu['cores']} / {cpu['threads']} |")
    w(f"| **CPU Freq** | {cpu['freq_mhz']:.0f} MHz |")
    w(f"| **{T('md_ram_total')}** | {ram['total_gb']:.1f} GB |")
    w(f"| **{T('md_ram_avail')}** | {ram['available_gb']:.1f} GB |")
    w(f"| **Swap** | {ram['swap_gb']:.1f} GB |")
    if gpus:
        for i, g in enumerate(gpus):
            w(f"| **GPU {i+1}** | {g['name']} ({g['vendor']}) |")
            w(f"| **VRAM** | {g['vram_mb']} MB {T('total')} / {g['vram_free_mb']} MB {T('free')} |")
            w(f"| **Compute Cap** | {g['compute_cap']} |")
    else:
        w(f"| **GPU** | ❌ {T('no_gpu')} |")
    w(f"")

    # ── RAM/VRAM visual
    w(f"### {T('md_mem')}")
    w(f"")
    w(f"```")
    ram_used = ram['total_gb'] - ram['available_gb']
    ram_pct = int(ram_used / ram['total_gb'] * 30)
    w(f"RAM  [{'█' * ram_pct}{'░' * (30 - ram_pct)}] "
      f"{ram_used:.1f} / {ram['total_gb']:.1f} GB ({ram_used/ram['total_gb']*100:.0f}% {T('in_use')})")
    if gpus:
        for g in gpus:
            vram_used = g['vram_mb'] - g['vram_free_mb']
            vram_pct = int(vram_used / g['vram_mb'] * 30) if g['vram_mb'] > 0 else 0
            w(f"VRAM [{'█' * vram_pct}{'░' * (30 - vram_pct)}] "
              f"{vram_used} / {g['vram_mb']} MB ({vram_used/g['vram_mb']*100:.0f}% {T('in_use')})")
    w(f"```")
    w(f"")

    # ── Installed tools
    w(f"## 🔧 {T('md_tools')}")
    w(f"")
    if tools:
        w(f"| {T('md_tool')} | {T('md_desc')} |")
        w(f"|------|-------------|")
        for name, info in sorted(tools.items()):
            w(f"| ✅ {name} | {info['desc']} |")
    else:
        w(f"❌ **{T('md_no_tools')}** [Ollama](https://ollama.ai)")
    w(f"")

    # ── Ollama local
    if ollama_models:
        w(f"### 📦 {T('md_ollama_loc')}")
        w(f"")
        w(f"| {T('md_model')} | {T('md_size')} |")
        w(f"|-------|---------|")
        for m in ollama_models:
            size = m.get('size', '-')
            w(f"| `{m['name']}` | {size} |")
        w(f"")

    # ── Category scores - Mermaid bar chart
    w(f"---")
    w(f"")
    w(f"## 📊 {T('md_scores_cat')}")
    w(f"")
    w(f"```mermaid")
    w(f"%%{{init: {{'theme': 'base', 'themeVariables': {{'primaryColor': '#4CAF50'}}}}}}%%")
    w(f"xychart-beta")
    w(f'    title "{T("md_chart_t")}"')
    w(f'    x-axis [{_mermaid_cat_labels(cat_results)}]')
    w(f'    y-axis "Score (0-100)" 0 --> 100')
    w(f'    bar [{_mermaid_cat_scores(cat_results)}]')
    w(f"```")
    w(f"")

    # ── Category detail table
    w(f"### {T('md_detail_cat')}")
    w(f"")
    w(f"| {T('md_category')} | {T('md_score')} | {T('md_rating')} | {T('md_mode')} | {T('md_examples')} |")
    w(f"|-----------|-------|-------------|-------|-------------|")
    for cat, score, mode in cat_results:
        emoji = _md_score_emoji(score)
        examples = ", ".join(cat["voorbeelden"][:4])
        w(f"| **{cat['categorie']}** | {score}/100 | {emoji} | {mode} | {examples} |")
    w(f"")

    # ── Category details (expandable per category)
    for cat, score, mode in cat_results:
        w(f"<details>")
        w(f"<summary><strong>{cat['categorie']}</strong> — {score}/100 {_md_score_emoji(score)}</summary>")
        w(f"")
        w(f"- **{T('md_desc')}:** {cat['beschrijving']}")
        w(f"- **{T('md_mode')}:** {mode}")
        w(f"- **{T('md_min_vram')}:** {cat['vram_min_mb']} MB")
        w(f"- **{T('md_min_ram')}:** {cat['ram_min_gb']} GB ({T('md_cpu_only')}: {cat['ram_cpu_only_gb']} GB)")
        w(f"- **{T('md_score')}:** {_md_score_bar(score)}")
        w(f"")
        w(f"**{T('md_examples')}:**")
        for ex in cat["voorbeelden"][:8]:
            w(f"- {ex}")
        w(f"")
        w(f"</details>")
        w(f"")

    # ── Mermaid pie chart - distribution
    w(f"### {T('md_feasibility')}")
    w(f"")
    top_count = sum(1 for _, s, _ in cat_results if s >= 70)
    ok_count = sum(1 for _, s, _ in cat_results if 40 <= s < 70)
    weak_count = sum(1 for _, s, _ in cat_results if s < 40)
    w(f"```mermaid")
    w(f"pie title {T('md_pie_t')}")
    if top_count:
        w(f'    "{T("md_pie_good")}" : {top_count}')
    if ok_count:
        w(f'    "{T("md_pie_ok")}" : {ok_count}')
    if weak_count:
        w(f'    "{T("md_pie_no")}" : {weak_count}')
    w(f"```")
    w(f"")

    # ── Individual models per type
    if model_results_by_type:
        w(f"---")
        w(f"")
        w(f"## 🤖 {T('md_top_models')}")
        w(f"")

        type_labels = {
            "llm": T("type_llm"),
            "code-llm": T("type_code"),
            "vision": T("type_vision"),
            "embedding": T("type_embed"),
            "stt": T("type_stt"),
            "tts": T("type_tts"),
            "image-gen": T("type_image"),
            "video-gen": T("type_video"),
            "object-detection": T("type_objdet"),
            "segmentation": T("type_segment"),
            "audio-gen": T("type_audiogen"),
            "document-ai": T("type_docai"),
        }

        for mtype, label in type_labels.items():
            items = model_results_by_type.get(mtype, [])
            if not items:
                continue

            w(f"### {label}")
            w(f"")
            w(f"| {T('md_score')} | {T('md_model')} | {T('md_params')} | GGUF | {T('md_mode')} | {T('md_source')} |")
            w(f"|-------|-------|------------|------|-------|------|")
            for score, mode, m in items[:10]:
                params_str = f"{m['params_b']:.1f}B" if m.get('params_b') else "?"
                gguf = "✅" if m.get("has_gguf") else "❌"
                emoji = _md_score_emoji(score).split()[0]
                src = m.get("source", "")
                w(f"| {emoji} {score}/100 | `{m['id'][:55]}` | {params_str} | {gguf} | {mode} | {src} |")
            w(f"")

        # Mermaid chart top models (top 15 overall)
        all_scored = []
        for items in model_results_by_type.values():
            all_scored.extend(items[:5])
        all_scored.sort(key=lambda x: x[0], reverse=True)
        top15 = all_scored[:15]
        if top15:
            w(f"### {T('md_top15')}")
            w(f"")
            w(f"```mermaid")
            w(f"xychart-beta")
            w(f'    title "{T("md_top15_t")}"')
            w(f'    x-axis [{_mermaid_model_labels(top15)}]')
            w(f'    y-axis "Score" 0 --> 100')
            w(f'    bar [{_mermaid_model_scores(top15)}]')
            w(f"```")
            w(f"")

    # ── Summary
    w(f"---")
    w(f"")
    w(f"## ✅ {T('md_summary')}")
    w(f"")

    top = [(c, s, m) for c, s, m in cat_results if s >= 70]
    ok = [(c, s, m) for c, s, m in cat_results if 40 <= s < 70]
    weak = [(c, s, m) for c, s, m in cat_results if s < 40]

    if top:
        w(f"### 🟢 {T('md_good')}")
        w(f"")
        for cat, score, mode in top:
            w(f"- **{cat['categorie']}** — {score}/100 ({mode})")
        w(f"")
    if ok:
        w(f"### 🟡 {T('md_possible')}")
        w(f"")
        for cat, score, mode in ok:
            w(f"- **{cat['categorie']}** — {score}/100 ({mode})")
        w(f"")
    if weak:
        w(f"### 🔴 {T('md_not')}")
        w(f"")
        for cat, score, mode in weak:
            w(f"- **{cat['categorie']}** — {score}/100 ({mode})")
        w(f"")

    # ── Tips
    w(f"## 💡 {T('md_recs')}")
    w(f"")
    if total_vram >= 5000 and total_vram < 8000:
        w(f"- {T('md_vram_full', vram=total_vram)}")
        w(f"- {T('md_offload_tip')}")
    elif total_vram >= 3000 and total_vram < 5000:
        w(f"- {T('md_vram_part', vram=total_vram)}")
        w(f"- {T('md_offload_tip')}")
    ram_gb_str = f"{ram['total_gb']:.0f}"
    if ram["total_gb"] >= 24:
        w(f"- {T('md_ram_tip', ram=ram_gb_str)}")
    w(f"- {T('md_q4_tip')}")
    w(f"- {T('md_gpu_cfg')}")
    w(f"")
    if not tools:
        w(f"### {T('md_quickstart')}")
        w(f"")
        w(f"```bash")
        w(f"# {T('md_install_ol')}")
        w(f"curl -fsSL https://ollama.ai/install.sh | sh")
        w(f"")
        w(f"# {T('md_dl_model')}")
        w(f"ollama pull llama3.1:8b-instruct-q4_K_M")
        w(f"```")
    elif "ollama" in tools:
        w(f"### {T('md_rec_ollama')}")
        w(f"")
        w(f"```bash")
        w(f"ollama pull llama3.1:8b       # chat")
        w(f"ollama pull qwen2.5-coder:7b  # code")
        w(f"ollama pull nomic-embed-text   # embeddings/RAG")
        w(f"```")
    w(f"")

    # ── VRAM fit chart
    if gpus:
        w(f"### {T('md_vram_fit')}")
        w(f"")
        vram = total_vram
        sizes = [
            ("1B Q4", 600), ("3B Q4", 1800), ("7B Q4", 4200),
            ("8B Q4", 4800), ("13B Q4", 7800), ("14B Q4", 8400),
            ("32B Q4", 19200), ("70B Q4", 42000),
        ]
        w(f"```")
        for label, need in sizes:
            pct = min(vram / need, 1.0)
            bar_len = int(pct * 30)
            if pct >= 1.0:
                fit = f"✅ {T('md_fits')}"
            elif pct >= 0.5:
                fit = f"⚠️ {pct*100:.0f}%"
            else:
                fit = f"❌ {T('md_no_fit')}"
            w(f"{label:8s} [{'█' * bar_len}{'░' * (30 - bar_len)}] {fit} ({need} MB {T('md_needed')} / {vram} MB)")
        w(f"```")
        w(f"")

    # ── Footer
    w(f"---")
    w(f"")
    if catalog:
        w(f"*{T('md_catalog', n=len(catalog.get('models', [])), ts=catalog.get('updated', '?'))}*  ")
    w(f"*{T('md_report_gen', ts=ts_display)}*")
    w(f"")

    # Write to file
    REPORTS_DIR.mkdir(exist_ok=True)
    filename = f"llm_report_{hostname}_{ts}.md"
    filepath = REPORTS_DIR / filename
    with open(filepath, "w") as f:
        f.write("\n".join(lines))

    return filepath


def _mermaid_cat_labels(cat_results):
    """Generate short Mermaid x-axis labels for categories."""
    labels = []
    for cat, score, mode in cat_results:
        name = cat["categorie"]
        # Shorten known size patterns for the chart
        matched = False
        for pattern in ("1-3B", "7-8B", "13-14B", "30-34B", "65-72B"):
            if pattern in name:
                name = pattern
                matched = True
                break
        if not matched:
            lower = name.lower()
            if "embed" in lower:
                name = "Embed"
            elif "(tts)" in lower or "tekst-naar-s" in lower:
                name = "TTS"
            elif "(stt)" in lower or "spraak-naar-t" in lower:
                name = "STT"
            elif "video" in lower:
                name = "VideoGen"
            elif "segment" in lower:
                name = "Segment"
            elif "image" in lower or "bild" in lower:
                if "klein" in lower or "small" in lower or "petit" in lower:
                    name = "ImgGen-S"
                else:
                    name = "ImgGen-L"
            elif "object" in lower or "objekt" in lower or "objet" in lower or "detect" in lower:
                name = "ObjDet"
            elif "audio" in lower or "music" in lower or "muziek" in lower or "musik" in lower or "musique" in lower:
                name = "AudioGen"
            elif "document" in lower or "ocr" in lower or "dokument" in lower:
                name = "DocAI"
            elif "vision" in lower or "multimod" in lower:
                name = "Vision"
            elif "code" in lower:
                name = "Code"
        labels.append(f'"{name}"')
    return ", ".join(labels)


def _mermaid_cat_scores(cat_results):
    return ", ".join(str(s) for _, s, _ in cat_results)


def _mermaid_model_labels(top_models):
    labels = []
    for score, mode, m in top_models:
        name = m["id"].split("/")[-1] if "/" in m["id"] else m["id"]
        # Kort af tot max 15 chars
        if len(name) > 15:
            name = name[:14] + "…"
        labels.append(f'"{name}"')
    return ", ".join(labels)


def _mermaid_model_scores(top_models):
    return ", ".join(str(s) for s, _, _ in top_models)


def print_summary(results, ram, total_vram, tools, catalog):
    print(f"{BOLD}{'═' * 70}{RESET}")
    print(f"{BOLD}{CYAN}  {T('summary_header')}{RESET}")
    print(f"{BOLD}{'═' * 70}{RESET}\n")

    top = [r for r in results if r[1] >= 70]
    ok = [r for r in results if 40 <= r[1] < 70]
    weak = [r for r in results if r[1] < 40]

    if top:
        print(f"  {GREEN}{BOLD}{T('runs_well')}{RESET}")
        for cat, score, mode in top:
            print(f"    {GREEN}✓{RESET} {cat['categorie']} ({score}/100 – {mode})")
        print()
    if ok:
        print(f"  {YELLOW}{BOLD}{T('possible_slower')}{RESET}")
        for cat, score, mode in ok:
            print(f"    {YELLOW}~{RESET} {cat['categorie']} ({score}/100 – {mode})")
        print()
    if weak:
        print(f"  {RED}{BOLD}{T('barely_feasible')}{RESET}")
        for cat, score, mode in weak:
            print(f"    {RED}✗{RESET} {cat['categorie']} ({score}/100 – {mode})")
        print()

    # Tips
    ram_str = f"{ram['total_gb']:.0f}"
    print(f"  {BOLD}{T('tips_for', ram=ram_str, vram=str(total_vram))}{RESET}\n")

    if total_vram >= 5000 and total_vram < 8000:
        print(f"  • {T('tip_vram_full_1', vram=total_vram)}")
        print(f"    {T('tip_vram_full_2')}")
        print()
    elif total_vram >= 3000 and total_vram < 5000:
        print(f"  • {T('tip_vram_part_1', vram=total_vram)}")
        print(f"    {T('tip_vram_part_2')}")
        print()
    if ram["total_gb"] >= 24:
        print(f"  • {T('tip_ram_1', ram=ram_str)}")
        print(f"    {T('tip_ram_2')}")
        print()

    if not tools:
        print(f"  {YELLOW}• {T('tip_no_tools')}{RESET}")
        print(f"    curl -fsSL https://ollama.ai/install.sh | sh")
        print(f"    ollama pull llama3.1:8b-instruct-q4_K_M")
        print()
    elif "ollama" in tools:
        print(f"  • {T('tip_ollama')}")
        print(f"    ollama pull llama3.1:8b       # chat")
        print(f"    ollama pull qwen2.5-coder:7b  # code")
        print(f"    ollama pull nomic-embed-text   # embeddings/RAG")
        print()

    print(f"  {DIM}{T('tip_q4')}{RESET}")
    print(f"  {DIM}{T('tip_gpu_cfg')}{RESET}")
    print()

    # Catalog status
    if catalog:
        print(f"  {DIM}{T('upd_catalog', n=len(catalog.get('models', [])), ts=catalog.get('updated', '?'))}{RESET}")
        print(f"  {DIM}{T('upd_cmd')}{RESET}")
    else:
        print(f"  {YELLOW}• {T('upd_no_cat')}{RESET}")
        print(f"    python llm_scanner.py --update")
    print()


# ─── Main ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="LLM & AI Fit Scanner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python llm_scanner.py              # scan with auto-detection
  python llm_scanner.py --update     # update model catalog online
  python llm_scanner.py --json       # output as JSON
  python llm_scanner.py -l nl        # output in Dutch
  python llm_scanner.py --update --json
        """,
    )
    parser.add_argument("--update", action="store_true",
                        help="Update model catalog via HuggingFace & Ollama (internet required)")
    parser.add_argument("--json", action="store_true",
                        help="Output as JSON instead of terminal colors")
    parser.add_argument("--no-report", action="store_true",
                        help="Do NOT generate a Markdown report (default: report is generated)")
    parser.add_argument("--lang", "-l", choices=["en", "nl", "de", "fr"], default="en",
                        help="Language for output: en (default), nl, de, fr")
    args = parser.parse_args()

    # ── Set language
    global LANG
    LANG = args.lang

    # ── Update catalog if requested
    if args.update:
        update_catalog()
        return

    # ── Hardware detection (always automatic)
    cpu = get_cpu_info()
    ram = get_ram_info()
    gpus = get_gpu_info()
    tools = get_installed_tools()
    total_vram = sum(g["vram_mb"] for g in gpus) if gpus else 0
    ollama_models = get_ollama_local_models()

    # ── Load catalog
    catalog = load_catalog()

    # ── Enrich categories with online data
    categories = get_categories()
    categories = enrich_categories_with_catalog(categories, catalog)

    # ── JSON output
    if args.json:
        results = []
        for cat in categories:
            score, mode = compute_score(
                cat, total_vram, ram["total_gb"], ram["available_gb"], cpu["threads"]
            )
            results.append({
                "categorie": cat["categorie"],
                "beschrijving": cat["beschrijving"],
                "score": score,
                "modus": mode,
                "voorbeelden": cat["voorbeelden"][:8],
            })

        individual = []
        if catalog and "models" in catalog:
            for m in catalog["models"]:
                score, mode = compute_model_score(
                    m, total_vram, ram["total_gb"], ram["available_gb"], cpu["threads"]
                )
                if score is not None:
                    individual.append({
                        "id": m["id"],
                        "type": m.get("type"),
                        "params_b": m.get("params_b"),
                        "score": score,
                        "modus": mode,
                        "source": m.get("source"),
                        "has_gguf": m.get("has_gguf"),
                    })
            individual.sort(key=lambda x: x["score"], reverse=True)

        output = {
            "hardware": {
                "cpu": cpu,
                "ram": ram,
                "gpus": gpus,
            },
            "tools": {k: v["desc"] for k, v in tools.items()},
            "ollama_local": ollama_models,
            "categorie_scores": results,
            "individuele_modellen": individual[:50],
            "catalogus_datum": catalog.get("updated") if catalog else None,
        }
        print(json.dumps(output, indent=2, ensure_ascii=False))
        return

    # ── Terminal output
    print(f"\n{BOLD}{'═' * 70}{RESET}")
    print(f"{BOLD}{CYAN}  🖥️  LLM & AI Fit Scanner{RESET}")
    print(f"{BOLD}{'═' * 70}{RESET}\n")

    print_hardware(cpu, ram, gpus, tools)
    print_ollama_local(ollama_models)

    results = print_category_scores(categories, total_vram, ram, cpu["threads"])
    print_individual_models(catalog, total_vram, ram, cpu["threads"])
    print_summary(results, ram, total_vram, tools, catalog)

    # ── Markdown rapport genereren
    if not args.no_report:
        report_path = generate_markdown_report(
            cpu, ram, gpus, tools, ollama_models, categories,
            catalog, total_vram
        )
        print(f"  {GREEN}{BOLD}{T('report_saved', path=report_path)}{RESET}")
        print(f"  {DIM}{T('report_disable')}{RESET}\n")


if __name__ == "__main__":
    main()
