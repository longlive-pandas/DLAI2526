import os
import json
import torch
import logging
from collections import Counter
from PIL import Image, ImageOps
from transformers import CLIPProcessor, CLIPModel
from datasets import load_dataset
from tqdm import tqdm

# --- CONFIGURAZIONI ---
CUSTOM_DATA_DIR = "data/botanical_garden_512x896"
OUTPUT_DIR = "data/pretraining_botanical_128x256"
LOG_DIR = "./checkpoints_botanical"
LOG_FILE = os.path.join(LOG_DIR, "training.log")
QUOTAS_FILE = os.path.join(LOG_DIR, "quotas_target.json")

TOTAL_TARGET_IMAGES = 60000
IMAGE_WIDTH = 128
IMAGE_HEIGHT = 256

# Categorie target botaniche PULITE (Rimosse le "stopwords visive" generiche)
BOTANICAL_CLASSES = [
    "cactus", "succulent plant", "palm tree", "fern", "japanese maple", 
    "yellow wildflowers", "pink flowers", "cycad", "agave", "aloe vera", 
    "conifer tree", "bamboo", "moss", "ivy vine", 
    "red flowers", "white flowers", "purple wildflowers"
]
TOP_K_TAGS = len(BOTANICAL_CLASSES) 

# Setup Logging doppio
os.makedirs(LOG_DIR, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, mode='a'),
        logging.StreamHandler()
    ]
)

def get_clip_model(device):
    logging.info(f"Caricamento CLIP-ViT su {device}...")
    model_id = "openai/clip-vit-base-patch32"
    processor = CLIPProcessor.from_pretrained(model_id)
    model = CLIPModel.from_pretrained(model_id, torch_dtype=torch.float16).eval().to(device)
    return processor, model

def load_or_calculate_quotas(device):
    """Gestisce la Fase 1 e 2, con meccanismo di resume tramite JSON."""
    if os.path.exists(QUOTAS_FILE):
        logging.info(f"[Resume] File {QUOTAS_FILE} trovato. Salto analisi CLIP e carico le quote esistenti.")
        with open(QUOTAS_FILE, "r") as f:
            return json.load(f)

    processor, model = get_clip_model(device)
    image_paths = [
        os.path.join(CUSTOM_DATA_DIR, f) for f in os.listdir(CUSTOM_DATA_DIR) 
        if f.lower().endswith(('.png', '.jpg', '.jpeg'))
    ]

    tags_counter = Counter()
    logging.info(f"[Fase 1] Avvio classificazione su {len(image_paths)} immagini custom...")
    
    with torch.no_grad():
        for path in tqdm(image_paths, desc="Analisi Custom (CLIP)"):
            try:
                img = Image.open(path).convert("RGB")
                inputs = processor(text=BOTANICAL_CLASSES, images=img, return_tensors="pt", padding=True).to(device, torch.float16)
                outputs = model(**inputs)
                
                probs = outputs.logits_per_image.softmax(dim=1)
                best_class = BOTANICAL_CLASSES[probs.argmax().item()]
                tags_counter.update([best_class])
            except Exception as e:
                logging.warning(f"Errore immagine custom {path}: {e}")
                continue

    # Pulizia VRAM
    del model
    del processor
    torch.cuda.empty_cache()

    # Calcolo Quote
    top_tags = tags_counter.most_common(TOP_K_TAGS)
    total_top_occurrences = sum(count for _, count in top_tags)
    
    quotas = {}
    logging.info("\n[Fase 2] Quote target calcolate:")
    for tag, count in top_tags:
        proportion = count / total_top_occurrences
        target_amount = int(proportion * TOTAL_TARGET_IMAGES)
        quotas[tag] = target_amount
        logging.info(f" - {tag}: {target_amount} immagini")
        
    # Salvataggio per resume futuri
    with open(QUOTAS_FILE, "w") as f:
        json.dump(quotas, f, indent=4)
        
    return quotas

def filter_and_download_plantnet(device, quotas):
    """Fase 3: Streaming di PlantNet300K con Resume Check su disco."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    collected_counts = {kw: 0 for kw in quotas.keys()}
    
    # --- MECCANISMO DI RESUME DISCO ---
    logging.info("[Resume] Controllo immagini già scaricate in locale...")
    existing_files = os.listdir(OUTPUT_DIR)
    for filename in existing_files:
        # Estraiamo la classe dal nome del file (es: "palm_tree_00123.jpg" -> "palm tree")
        for kw in quotas.keys():
            safe_kw = kw.replace(' ', '_')
            if filename.startswith(safe_kw):
                collected_counts[kw] += 1
                break
                
    already_downloaded = sum(collected_counts.values())
    total_to_download = sum(quotas.values())
    
    if already_downloaded >= total_to_download:
        logging.info("\n[OK] Tutte le quote sono già state raggiunte! Dataset completo.")
        return

    if already_downloaded > 0:
        logging.info(f"[Resume] Ripresa dal download di {already_downloaded}/{total_to_download} immagini.")

    # Carichiamo CLIP per il filtraggio
    processor, model = get_clip_model(device)

    logging.info("\n[Fase 3] Inizializzazione streaming del dataset PlantNet300K...")
    dataset = load_dataset("mikehemberger/plantnet300K", split="train", streaming=True)
    
    with torch.no_grad():
        with tqdm(total=total_to_download, initial=already_downloaded, desc="Estrazione PlantNet") as pbar:
            for item in dataset:
                if all(collected_counts[kw] >= quotas[kw] for kw in quotas):
                    break
                
                try:
                    img = item.get('image') or item.get('img') or item.get('jpg')
                    if img is None:
                        continue
                    
                    if img.mode != "RGB":
                        img = img.convert("RGB")

                    inputs = processor(text=list(quotas.keys()), images=img, return_tensors="pt", padding=True).to(device, torch.float16)
                    outputs = model(**inputs)
                    
                    probs = outputs.logits_per_image.softmax(dim=1)
                    best_class_idx = probs.argmax().item()
                    best_class = list(quotas.keys())[best_class_idx]
                    confidence = probs.max().item()
                    
                    # Salvataggio se la confidenza è buona e c'è ancora spazio nella quota
                    if confidence > 0.15 and collected_counts[best_class] < quotas[best_class]:
                        img_resized = ImageOps.fit(img, (IMAGE_WIDTH, IMAGE_HEIGHT), Image.Resampling.LANCZOS)
                        
                        filename = f"{best_class.replace(' ', '_')}_{collected_counts[best_class]:05d}.jpg"
                        img_resized.save(os.path.join(OUTPUT_DIR, filename), "JPEG", quality=95)
                        
                        collected_counts[best_class] += 1
                        pbar.update(1)
                        
                except Exception:
                    continue

    logging.info(f"\n[OK] Pipeline completata! Dataset botanico salvato in {OUTPUT_DIR}")

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    logging.info("=== Avvio Pipeline Resumable di Filtro Semantico su PlantNet300K ===")
    target_quotas = load_or_calculate_quotas(device)
    filter_and_download_plantnet(device, target_quotas)