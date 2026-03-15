#!/usr/bin/env python3
"""
LLM & AI Capability Scanner
============================
Scant je systeem (CPU, RAM, GPU/VRAM) en geeft per categorie en per
individueel model een score die aangeeft wat je lokaal kunt draaien.

Gebruik:
    python llm_scanner.py              # scan met ingebouwde + gecachte catalogus
    python llm_scanner.py --update     # update model-catalogus online (HuggingFace + Ollama)
    python llm_scanner.py --json       # output als JSON
"""

import argparse
import copy
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
    sys.exit("psutil is vereist: pip install psutil")

# ─── Paden ──────────────────────────────────────────────────────────────────

SCRIPT_DIR = Path(__file__).parent.resolve()
CATALOG_FILE = SCRIPT_DIR / "models_catalog.json"
REPORTS_DIR = SCRIPT_DIR / "reports"

# ─── Kleuren ────────────────────────────────────────────────────────────────

GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
CYAN = "\033[96m"
BOLD = "\033[1m"
DIM = "\033[2m"
RESET = "\033[0m"


# ─── Hardware detectie ──────────────────────────────────────────────────────

def get_cpu_info():
    info = {
        "name": platform.processor() or "Onbekend",
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
        "ollama": "Ollama – eenvoudig LLM-beheer",
        "llama-server": "llama.cpp server",
        "llama-cli": "llama.cpp CLI",
        "koboldcpp": "KoboldCpp – roleplay/creatief",
        "text-generation-server": "TGI (Text Generation Inference)",
        "vllm": "vLLM – high-throughput serving",
        "whisper": "Whisper – spraak-naar-tekst",
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
        "diffusers": "HuggingFace Diffusers (image gen)",
        "whisper": "OpenAI Whisper",
        "faster_whisper": "Faster Whisper",
        "TTS": "Coqui TTS (tekst-naar-spraak)",
        "bark": "Bark (tekst-naar-spraak)",
        "sentence_transformers": "Sentence Transformers (embeddings)",
        "chromadb": "ChromaDB (vector store / RAG)",
        "langchain": "LangChain (LLM orchestratie)",
        "auto_gptq": "AutoGPTQ (geoptimaliseerde quantisatie)",
        "exllama": "ExLlama (snelle GPTQ inference)",
        "bitsandbytes": "BitsAndBytes (4/8-bit quantisatie)",
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
    req = Request(url, headers={"User-Agent": "llm-scanner/1.0"})
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
    # Image gen
    if pipe in ("text-to-image",) or any(t in model_lower for t in (
            "stable-diffusion", "sdxl", "flux", "diffusion")):
        return "image-gen"
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
    """Haal online modellen op en sla op als lokale catalogus."""
    print(f"\n{BOLD}{CYAN}  Catalogus updaten...{RESET}\n")

    all_models = []

    # HuggingFace
    print(f"  {DIM}HuggingFace API ophalen...{RESET}", end=" ", flush=True)
    try:
        hf_models = fetch_huggingface_models(limit=50)
        print(f"{GREEN}{len(hf_models)} modellen gevonden{RESET}")
        all_models.extend(hf_models)
    except Exception as e:
        print(f"{RED}Fout: {e}{RESET}")

    # Ollama registry
    print(f"  {DIM}Ollama registry ophalen...{RESET}", end=" ", flush=True)
    try:
        ol_models = fetch_ollama_library()
        print(f"{GREEN}{len(ol_models)} modellen gevonden{RESET}")
        all_models.extend(ol_models)
    except Exception as e:
        print(f"{RED}Fout: {e}{RESET}")

    if not all_models:
        print(f"\n  {RED}Geen modellen opgehaald. Controleer je internetverbinding.{RESET}\n")
        return False

    # Deduplicate op basis van id
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

    print(f"\n  {GREEN}✓ {len(unique)} modellen opgeslagen in {CATALOG_FILE.name}{RESET}")
    print(f"  {DIM}Laatst geüpdatet: {catalog['updated']}{RESET}\n")
    return True


def load_catalog():
    """Laad de lokale catalogus (indien aanwezig)."""
    if CATALOG_FILE.exists():
        with open(CATALOG_FILE) as f:
            return json.load(f)
    return None


# ─── Ingebouwde categorieën (fallback) ──────────────────────────────────────

BUILTIN_CATEGORIES = [
    {
        "categorie": "Kleine LLMs (1-3B)",
        "beschrijving": "Snelle, lichte modellen voor chat, code-assist, classificatie",
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
        "categorie": "Medium LLMs (7-8B)",
        "beschrijving": "Goede chatbots, code-generatie, vertaling, samenvatting",
        "vram_min_mb": 4000,
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
        "categorie": "Grote LLMs (13-14B)",
        "beschrijving": "Hogere kwaliteit redenering, complexe taken",
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
        "categorie": "XL LLMs (30-34B)",
        "beschrijving": "Near-GPT-3.5 kwaliteit, complexe redenering",
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
        "categorie": "XXL LLMs (65-72B)",
        "beschrijving": "Top-tier kwaliteit, vergelijkbaar met GPT-4",
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
        "categorie": "Embedding-modellen",
        "beschrijving": "Tekst naar vectoren – voor RAG, zoeken, classificatie",
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
        "categorie": "Spraak-naar-tekst (STT)",
        "beschrijving": "Audio transcriptie en vertaling",
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
        "categorie": "Tekst-naar-spraak (TTS)",
        "beschrijving": "Spraaksynthese, voice cloning",
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
        "categorie": "Image Generation (klein)",
        "beschrijving": "Afbeeldingen genereren – kleinere modellen",
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
        "categorie": "Image Generation (groot)",
        "beschrijving": "High-end beeldgeneratie, Flux, SDXL",
        "vram_min_mb": 8000,
        "ram_min_gb": 16,
        "ram_cpu_only_gb": 32,
        "voorbeelden": [
            "SDXL (volledig)", "Flux.1-dev", "Flux.1-schnell",
        ],
        "match_types": ["image-gen-large"],
        "match_bucket": None,
    },
    {
        "categorie": "Vision / Multimodal LLMs",
        "beschrijving": "LLMs die tekst + afbeeldingen begrijpen",
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
        "categorie": "Code-specifieke LLMs",
        "beschrijving": "Code generatie, debugging, uitleg",
        "vram_min_mb": 4000,
        "ram_min_gb": 8,
        "ram_cpu_only_gb": 8,
        "voorbeelden": [
            "DeepSeek-Coder-V2-Lite Q4", "CodeLlama-7B Q4",
            "StarCoder2-7B Q4", "Qwen2.5-Coder-7B Q4",
        ],
        "match_types": ["code-llm"],
        "match_bucket": None,
    },
]


# ─── Catalogus → modellen per categorie ─────────────────────────────────────

def enrich_categories_with_catalog(categories, catalog):
    """Voeg online modellen toe aan de 'voorbeelden' per categorie."""
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
        # Top modellen op downloads, voeg toe als ze nog niet in voorbeelden staan
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
            mode = "GPU (volledig)"
        else:
            mode = f"GPU+CPU offload ({vram_ratio*100:.0f}% VRAM)"
    elif ram_total_gb >= ram_cpu_only * 0.5:
        ram_ratio = min(ram_total_gb / ram_cpu_only, 1.0)
        cpu_ratio = min(cpu_threads / 8, 1.0)
        score = int(ram_ratio * 55 + cpu_ratio * 15)
        mode = "CPU-only (langzaam)"
        if ram_ratio >= 1.0:
            score = min(score + 5, 75)
    else:
        score = 0
        mode = "Niet haalbaar"

    return max(0, min(100, score)), mode


def compute_model_score(model, gpu_vram_mb, ram_total_gb, ram_avail_gb, cpu_threads):
    """Score een individueel model op basis van geschatte vereisten."""
    vram = model.get("vram_q4_mb")
    ram = model.get("ram_q4_gb")
    if vram is None or ram is None:
        return None, "Onbekend"

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
        return f"{GREEN}★★★★★  Uitstekend{RESET}"
    elif score >= 70:
        return f"{GREEN}★★★★   Goed{RESET}"
    elif score >= 50:
        return f"{YELLOW}★★★    Redelijk (langzamer){RESET}"
    elif score >= 30:
        return f"{YELLOW}★★     Matig (traag){RESET}"
    elif score >= 10:
        return f"{RED}★      Moeilijk{RESET}"
    else:
        return f"{RED}✗      Niet mogelijk{RESET}"


# ─── Display ────────────────────────────────────────────────────────────────

def print_hardware(cpu, ram, gpus, tools):
    print(f"{BOLD}  HARDWARE (automatisch gedetecteerd){RESET}")
    print(f"  ─────────────────────────────────────────")
    print(f"  CPU:    {cpu['name']}")
    print(f"          {cpu['cores']} cores / {cpu['threads']} threads @ {cpu['freq_mhz']:.0f} MHz")
    print(f"  RAM:    {ram['total_gb']:.1f} GB totaal / {ram['available_gb']:.1f} GB beschikbaar")
    print(f"  Swap:   {ram['swap_gb']:.1f} GB")
    if gpus:
        for g in gpus:
            print(f"  GPU:    {g['name']} ({g['vendor']})")
            print(f"          {g['vram_mb']} MB VRAM ({g['vram_free_mb']} MB vrij)"
                  f" | Compute {g['compute_cap']}")
    else:
        print(f"  GPU:    {RED}Geen GPU gedetecteerd{RESET}")
    print()

    if tools:
        print(f"{BOLD}  GEINSTALLEERDE AI-TOOLS{RESET}")
        print(f"  ─────────────────────────────────────────")
        for name, info in sorted(tools.items()):
            print(f"  {GREEN}✓{RESET} {info['desc']:40s} ({name})")
        print()


def print_ollama_local(ollama_models):
    if not ollama_models:
        return
    print(f"{BOLD}  LOKAAL GEINSTALLEERDE OLLAMA-MODELLEN{RESET}")
    print(f"  ─────────────────────────────────────────")
    for m in ollama_models:
        size = f" ({m['size']})" if m.get("size") else ""
        print(f"  {GREEN}●{RESET} {m['name']}{size}")
    print()


def print_category_scores(categories, total_vram, ram, cpu_threads):
    print(f"{BOLD}{'═' * 70}{RESET}")
    print(f"{BOLD}{CYAN}  SCORES PER CATEGORIE{RESET}")
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
        print(f"  {DIM}Modus: {mode}{RESET}")
        print(f"  {DIM}Voorbeelden: {', '.join(cat['voorbeelden'][:6])}{RESET}")
        print()

    return results


def print_individual_models(catalog, total_vram, ram, cpu_threads):
    """Toon individuele model-scores als er een catalogus is."""
    if not catalog or "models" not in catalog:
        return

    models = catalog["models"]
    # Filter op modellen met bekende parameters
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
    print(f"{BOLD}{CYAN}  TOP INDIVIDUELE MODELLEN VOOR JOUW HARDWARE{RESET}")
    print(f"{BOLD}{'═' * 70}{RESET}\n")

    # Groepeer per type
    type_labels = {
        "llm": "Taal-modellen (LLM)",
        "code-llm": "Code-modellen",
        "vision": "Vision / Multimodal",
        "embedding": "Embedding-modellen",
        "stt": "Spraak-naar-tekst",
        "tts": "Tekst-naar-spraak",
        "image-gen": "Image Generation",
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
        # Toon top 8 per type
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
        return "🟢 Uitstekend"
    elif score >= 70:
        return "🟢 Goed"
    elif score >= 50:
        return "🟡 Redelijk"
    elif score >= 30:
        return "🟠 Matig"
    elif score >= 10:
        return "🔴 Moeilijk"
    else:
        return "⛔ Niet mogelijk"


def generate_markdown_report(cpu, ram, gpus, tools, ollama_models, categories,
                              catalog, total_vram):
    """Genereer een volledig Markdown-rapport met grafieken en details."""
    now = datetime.now()
    ts = now.strftime("%d:%m:%Y_%H:%M")
    ts_display = now.strftime("%d-%m-%Y %H:%M")

    # Score alle categorieën
    cat_results = []
    for cat in categories:
        score, mode = compute_score(
            cat, total_vram, ram["total_gb"], ram["available_gb"], cpu["threads"]
        )
        cat_results.append((cat, score, mode))
    cat_results.sort(key=lambda x: x[1], reverse=True)

    # Score individuele modellen
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
    w = lines.append  # shorthand

    # ── Header
    w(f"# 🖥️ LLM & AI Capability Report")
    w(f"")
    w(f"> **Gegenereerd:** {ts_display}  ")
    w(f"> **Machine:** {cpu['name']}  ")
    w(f"> **OS:** {platform.system()} {platform.release()}")
    w(f"")
    w(f"---")
    w(f"")

    # ── Hardware
    w(f"## ⚙️ Hardware")
    w(f"")
    w(f"| Component | Details |")
    w(f"|-----------|---------|")
    w(f"| **CPU** | {cpu['name']} |")
    w(f"| **Cores / Threads** | {cpu['cores']} / {cpu['threads']} |")
    w(f"| **CPU Freq** | {cpu['freq_mhz']:.0f} MHz |")
    w(f"| **RAM Totaal** | {ram['total_gb']:.1f} GB |")
    w(f"| **RAM Beschikbaar** | {ram['available_gb']:.1f} GB |")
    w(f"| **Swap** | {ram['swap_gb']:.1f} GB |")
    if gpus:
        for i, g in enumerate(gpus):
            w(f"| **GPU {i+1}** | {g['name']} ({g['vendor']}) |")
            w(f"| **VRAM** | {g['vram_mb']} MB totaal / {g['vram_free_mb']} MB vrij |")
            w(f"| **Compute Cap** | {g['compute_cap']} |")
    else:
        w(f"| **GPU** | ❌ Geen GPU gedetecteerd |")
    w(f"")

    # ── RAM/VRAM visueel
    w(f"### Geheugen Overzicht")
    w(f"")
    w(f"```")
    ram_used = ram['total_gb'] - ram['available_gb']
    ram_pct = int(ram_used / ram['total_gb'] * 30)
    w(f"RAM  [{('█' * ram_pct) + ('░' * (30 - ram_pct))}] "
      f"{ram_used:.1f} / {ram['total_gb']:.1f} GB ({ram_used/ram['total_gb']*100:.0f}% in gebruik)")
    if gpus:
        for g in gpus:
            vram_used = g['vram_mb'] - g['vram_free_mb']
            vram_pct = int(vram_used / g['vram_mb'] * 30) if g['vram_mb'] > 0 else 0
            w(f"VRAM [{('█' * vram_pct) + ('░' * (30 - vram_pct))}] "
              f"{vram_used} / {g['vram_mb']} MB ({vram_used/g['vram_mb']*100:.0f}% in gebruik)")
    w(f"```")
    w(f"")

    # ── Geïnstalleerde tools
    w(f"## 🔧 Geïnstalleerde AI-Tools")
    w(f"")
    if tools:
        w(f"| Tool | Beschrijving |")
        w(f"|------|-------------|")
        for name, info in sorted(tools.items()):
            w(f"| ✅ {name} | {info['desc']} |")
    else:
        w(f"❌ **Geen AI-tools gedetecteerd.** Installeer [Ollama](https://ollama.ai) om te starten.")
    w(f"")

    # ── Ollama lokaal
    if ollama_models:
        w(f"### 📦 Lokaal geïnstalleerde Ollama-modellen")
        w(f"")
        w(f"| Model | Grootte |")
        w(f"|-------|---------|")
        for m in ollama_models:
            size = m.get('size', '-')
            w(f"| `{m['name']}` | {size} |")
        w(f"")

    # ── Categorie scores - Mermaid bar chart
    w(f"---")
    w(f"")
    w(f"## 📊 Scores per Categorie")
    w(f"")
    w(f"```mermaid")
    w(f"%%{{init: {{'theme': 'base', 'themeVariables': {{'primaryColor': '#4CAF50'}}}}}}%%")
    w(f"xychart-beta")
    w(f'    title "AI Capability Scores"')
    w(f'    x-axis [{_mermaid_cat_labels(cat_results)}]')
    w(f'    y-axis "Score (0-100)" 0 --> 100')
    w(f'    bar [{_mermaid_cat_scores(cat_results)}]')
    w(f"```")
    w(f"")

    # ── Categorie detail tabel
    w(f"### Detail per categorie")
    w(f"")
    w(f"| Categorie | Score | Beoordeling | Modus | Voorbeelden |")
    w(f"|-----------|-------|-------------|-------|-------------|")
    for cat, score, mode in cat_results:
        emoji = _md_score_emoji(score)
        examples = ", ".join(cat["voorbeelden"][:4])
        w(f"| **{cat['categorie']}** | {score}/100 | {emoji} | {mode} | {examples} |")
    w(f"")

    # ── Categorie details (uitgebreid per categorie)
    for cat, score, mode in cat_results:
        w(f"<details>")
        w(f"<summary><strong>{cat['categorie']}</strong> — {score}/100 {_md_score_emoji(score)}</summary>")
        w(f"")
        w(f"- **Beschrijving:** {cat['beschrijving']}")
        w(f"- **Modus:** {mode}")
        w(f"- **Minimaal VRAM:** {cat['vram_min_mb']} MB")
        w(f"- **Minimaal RAM:** {cat['ram_min_gb']} GB (CPU-only: {cat['ram_cpu_only_gb']} GB)")
        w(f"- **Score:** {_md_score_bar(score)}")
        w(f"")
        w(f"**Voorbeelden:**")
        for ex in cat["voorbeelden"][:8]:
            w(f"- {ex}")
        w(f"")
        w(f"</details>")
        w(f"")

    # ── Mermaid pie chart - verdeling
    w(f"### Verdeling Haalbaarheid")
    w(f"")
    top_count = sum(1 for _, s, _ in cat_results if s >= 70)
    ok_count = sum(1 for _, s, _ in cat_results if 40 <= s < 70)
    weak_count = sum(1 for _, s, _ in cat_results if s < 40)
    w(f"```mermaid")
    w(f"pie title Categorieën per haalbaarheid")
    if top_count:
        w(f'    "Goed draaibaar (≥70)" : {top_count}')
    if ok_count:
        w(f'    "Mogelijk (40-69)" : {ok_count}')
    if weak_count:
        w(f'    "Niet haalbaar (<40)" : {weak_count}')
    w(f"```")
    w(f"")

    # ── Individuele modellen per type
    if model_results_by_type:
        w(f"---")
        w(f"")
        w(f"## 🤖 Top Modellen voor jouw Hardware")
        w(f"")

        type_labels = {
            "llm": "💬 Taal-modellen (LLM)",
            "code-llm": "💻 Code-modellen",
            "vision": "👁️ Vision / Multimodal",
            "embedding": "🔗 Embedding-modellen",
            "stt": "🎤 Spraak-naar-tekst",
            "tts": "🔊 Tekst-naar-spraak",
            "image-gen": "🎨 Image Generation",
        }

        for mtype, label in type_labels.items():
            items = model_results_by_type.get(mtype, [])
            if not items:
                continue

            w(f"### {label}")
            w(f"")
            w(f"| Score | Model | Parameters | GGUF | Modus | Bron |")
            w(f"|-------|-------|------------|------|-------|------|")
            for score, mode, m in items[:10]:
                params_str = f"{m['params_b']:.1f}B" if m.get('params_b') else "?"
                gguf = "✅" if m.get("has_gguf") else "❌"
                emoji = _md_score_emoji(score).split()[0]  # just the emoji
                src = m.get("source", "")
                w(f"| {emoji} {score}/100 | `{m['id'][:55]}` | {params_str} | {gguf} | {mode} | {src} |")
            w(f"")

        # Mermaid chart top modellen (top 15 overall)
        all_scored = []
        for items in model_results_by_type.values():
            all_scored.extend(items[:5])
        all_scored.sort(key=lambda x: x[0], reverse=True)
        top15 = all_scored[:15]
        if top15:
            w(f"### Top 15 Modellen (grafiek)")
            w(f"")
            w(f"```mermaid")
            w(f"xychart-beta")
            w(f'    title "Top 15 Modellen — Score op jouw Hardware"')
            w(f'    x-axis [{_mermaid_model_labels(top15)}]')
            w(f'    y-axis "Score" 0 --> 100')
            w(f'    bar [{_mermaid_model_scores(top15)}]')
            w(f"```")
            w(f"")

    # ── Samenvatting
    w(f"---")
    w(f"")
    w(f"## ✅ Samenvatting")
    w(f"")

    top = [(c, s, m) for c, s, m in cat_results if s >= 70]
    ok = [(c, s, m) for c, s, m in cat_results if 40 <= s < 70]
    weak = [(c, s, m) for c, s, m in cat_results if s < 40]

    if top:
        w(f"### 🟢 Goed draaibaar")
        w(f"")
        for cat, score, mode in top:
            w(f"- **{cat['categorie']}** — {score}/100 ({mode})")
        w(f"")
    if ok:
        w(f"### 🟡 Mogelijk maar langzamer")
        w(f"")
        for cat, score, mode in ok:
            w(f"- **{cat['categorie']}** — {score}/100 ({mode})")
        w(f"")
    if weak:
        w(f"### 🔴 Niet / nauwelijks haalbaar")
        w(f"")
        for cat, score, mode in weak:
            w(f"- **{cat['categorie']}** — {score}/100 ({mode})")
        w(f"")

    # ── Tips
    w(f"## 💡 Aanbevelingen")
    w(f"")
    if total_vram >= 3000 and total_vram < 8000:
        w(f"- Met **{total_vram} MB VRAM** kun je **7-8B modellen in Q4** volledig op de GPU draaien")
        w(f"- Gebruik `--n-gpu-layers` om lagen naar GPU te offloaden voor grotere modellen")
    if ram["total_gb"] >= 24:
        w(f"- Met **{ram['total_gb']:.0f} GB RAM** kun je **30B+ modellen CPU-only** draaien in Q4 (~2-5 tok/s)")
    w(f"- **Q4_K_M** kwantisatie is de sweet spot tussen kwaliteit en snelheid")
    w(f"- Gebruik `--num-gpu` / `-ngl` om GPU-offload te configureren")
    w(f"")
    if not tools:
        w(f"### Snel starten")
        w(f"")
        w(f"```bash")
        w(f"# Installeer Ollama")
        w(f"curl -fsSL https://ollama.ai/install.sh | sh")
        w(f"")
        w(f"# Download een goed 7B model")
        w(f"ollama pull llama3.1:8b-instruct-q4_K_M")
        w(f"```")
    elif "ollama" in tools:
        w(f"### Aanbevolen Ollama modellen")
        w(f"")
        w(f"```bash")
        w(f"ollama pull llama3.1:8b       # chat")
        w(f"ollama pull qwen2.5-coder:7b  # code")
        w(f"ollama pull nomic-embed-text   # embeddings/RAG")
        w(f"```")
    w(f"")

    # ── VRAM fit chart
    if gpus:
        w(f"### VRAM Fit – Welke modellen passen?")
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
            fit = "✅ Past" if pct >= 1.0 else f"⚠️ {pct*100:.0f}%" if pct >= 0.5 else "❌ Past niet"
            w(f"{label:8s} [{'█' * bar_len}{'░' * (30 - bar_len)}] {fit} ({need} MB nodig / {vram} MB)")
        w(f"```")
        w(f"")

    # ── Footer
    w(f"---")
    w(f"")
    if catalog:
        w(f"*Catalogus: {len(catalog.get('models', []))} modellen "
          f"(geüpdatet: {catalog.get('updated', '?')})*  ")
    w(f"*Rapport gegenereerd: {ts_display} door LLM Scanner v1.1*")
    w(f"")

    # Write to file
    REPORTS_DIR.mkdir(exist_ok=True)
    filename = f"llm_report_{ts}.md"
    filepath = REPORTS_DIR / filename
    with open(filepath, "w") as f:
        f.write("\n".join(lines))

    return filepath


def _mermaid_cat_labels(cat_results):
    """Genereer Mermaid x-as labels voor categorieën."""
    labels = []
    for cat, score, mode in cat_results:
        # Kort het label in voor de grafiek
        name = cat["categorie"]
        name = name.replace("Kleine LLMs (1-3B)", "1-3B")
        name = name.replace("Medium LLMs (7-8B)", "7-8B")
        name = name.replace("Grote LLMs (13-14B)", "13-14B")
        name = name.replace("XL LLMs (30-34B)", "30-34B")
        name = name.replace("XXL LLMs (65-72B)", "65-72B")
        name = name.replace("Embedding-modellen", "Embed")
        name = name.replace("Spraak-naar-tekst (STT)", "STT")
        name = name.replace("Tekst-naar-spraak (TTS)", "TTS")
        name = name.replace("Image Generation (klein)", "ImgGen-S")
        name = name.replace("Image Generation (groot)", "ImgGen-L")
        name = name.replace("Vision / Multimodal LLMs", "Vision")
        name = name.replace("Code-specifieke LLMs", "Code")
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
    print(f"{BOLD}{CYAN}  SAMENVATTING & AANBEVELINGEN{RESET}")
    print(f"{BOLD}{'═' * 70}{RESET}\n")

    top = [r for r in results if r[1] >= 70]
    ok = [r for r in results if 40 <= r[1] < 70]
    weak = [r for r in results if r[1] < 40]

    if top:
        print(f"  {GREEN}{BOLD}Goed draaibaar:{RESET}")
        for cat, score, mode in top:
            print(f"    {GREEN}✓{RESET} {cat['categorie']} ({score}/100 – {mode})")
        print()
    if ok:
        print(f"  {YELLOW}{BOLD}Mogelijk maar langzamer:{RESET}")
        for cat, score, mode in ok:
            print(f"    {YELLOW}~{RESET} {cat['categorie']} ({score}/100 – {mode})")
        print()
    if weak:
        print(f"  {RED}{BOLD}Niet / nauwelijks haalbaar:{RESET}")
        for cat, score, mode in weak:
            print(f"    {RED}✗{RESET} {cat['categorie']} ({score}/100 – {mode})")
        print()

    # Tips
    print(f"  {BOLD}Tips voor jouw systeem ({ram['total_gb']:.0f}GB RAM, "
          f"{total_vram}MB VRAM):{RESET}\n")

    if total_vram >= 3000 and total_vram < 8000:
        print(f"  • Met {total_vram}MB VRAM kun je 7-8B modellen in Q4 kwantisatie")
        print(f"    volledig op de GPU draaien. Gebruik --n-gpu-layers voor offload.")
        print()
    if ram["total_gb"] >= 24:
        print(f"  • Met {ram['total_gb']:.0f}GB RAM kun je 30B+ modellen CPU-only draaien")
        print(f"    in Q4 kwantisatie (~2-5 tokens/sec).")
        print()

    if not tools:
        print(f"  {YELLOW}• Geen AI-tools gedetecteerd! Start met Ollama:{RESET}")
        print(f"    curl -fsSL https://ollama.ai/install.sh | sh")
        print(f"    ollama pull llama3.1:8b-instruct-q4_K_M")
        print()
    elif "ollama" in tools:
        print(f"  • Ollama is geïnstalleerd! Probeer:")
        print(f"    ollama pull llama3.1:8b       # chat")
        print(f"    ollama pull qwen2.5-coder:7b  # code")
        print(f"    ollama pull nomic-embed-text   # embeddings/RAG")
        print()

    # Catalogus status
    if catalog:
        print(f"  {DIM}Catalogus: {len(catalog.get('models', []))} modellen"
              f" (geüpdatet: {catalog.get('updated', '?')}){RESET}")
        print(f"  {DIM}Update met: python llm_scanner.py --update{RESET}")
    else:
        print(f"  {YELLOW}• Geen model-catalogus gevonden. Haal online data op:{RESET}")
        print(f"    python llm_scanner.py --update")
    print()


# ─── Main ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="LLM & AI Capability Scanner – scant je hardware en toont wat je kunt draaien",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Voorbeelden:
  python llm_scanner.py              # scan met auto-detectie
  python llm_scanner.py --update     # update model-catalogus online
  python llm_scanner.py --json       # output als JSON
  python llm_scanner.py --update --json
        """,
    )
    parser.add_argument("--update", action="store_true",
                        help="Update de model-catalogus via HuggingFace & Ollama (internet vereist)")
    parser.add_argument("--json", action="store_true",
                        help="Output als JSON in plaats van terminal-kleuren")
    parser.add_argument("--no-report", action="store_true",
                        help="Genereer GEEN Markdown-rapport (standaard wordt er wel eentje gemaakt)")
    args = parser.parse_args()

    # ── Update catalogus indien gewenst
    if args.update:
        update_catalog()

    # ── Hardware detectie (altijd automatisch)
    cpu = get_cpu_info()
    ram = get_ram_info()
    gpus = get_gpu_info()
    tools = get_installed_tools()
    total_vram = sum(g["vram_mb"] for g in gpus) if gpus else 0
    ollama_models = get_ollama_local_models()

    # ── Catalogus laden
    catalog = load_catalog()

    # ── Categorieën verrijken met online data
    categories = copy.deepcopy(BUILTIN_CATEGORIES)
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
    print(f"{BOLD}{CYAN}  🖥️  LLM & AI Capability Scanner{RESET}")
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
        print(f"  {GREEN}{BOLD}📄 Rapport opgeslagen: {report_path}{RESET}")
        print(f"  {DIM}Gebruik --no-report om dit uit te schakelen{RESET}\n")


if __name__ == "__main__":
    main()
