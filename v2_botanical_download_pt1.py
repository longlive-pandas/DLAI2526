import os
from tqdm import tqdm
from datasets import load_dataset
from PIL import Image, ImageOps

# --- CONFIGURAZIONI ---
OUTPUT_DIR = "data/pretraining_botanical_128x256"
TARGET_IMAGES = 60000  # Target per la Fase 1 di Pre-training
IMAGE_WIDTH = 128
IMAGE_HEIGHT = 256

def process_and_save():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("Inizializzazione streaming di mikehemberger/plantnet300K...")
    # Streaming attivo per processare le immagini senza scaricare l'intero dataset su disco
    dataset = load_dataset("mikehemberger/plantnet300K", split="train", streaming=True)
    
    saved_count = 0
    
    with tqdm(total=TARGET_IMAGES, desc="Download & Crop Immagini") as pbar:
        for item in dataset:
            if saved_count >= TARGET_IMAGES:
                break
            
            try:
                # Estrazione dinamica della chiave immagine
                img = item.get("image") or item.get("jpg") or item.get("img")
                if img is None:
                    continue
                
                # Conversione standard in RGB
                if img.mode != "RGB":
                    img = img.convert("RGB")
                
                # Crop centrale mantenendo l'aspect ratio e ridimensionamento a 128x256
                img_resized = ImageOps.fit(
                    img, 
                    (IMAGE_WIDTH, IMAGE_HEIGHT), 
                    Image.Resampling.LANCZOS
                )
                
                # Salvataggio
                file_path = os.path.join(OUTPUT_DIR, f"plantnet_{saved_count:05d}.jpg")
                img_resized.save(file_path, "JPEG", quality=95)
                
                saved_count += 1
                pbar.update(1)
                
            except Exception:
                # Ignora eventuali errori di streaming/decrittografia su singole immagini
                continue
                
    print(f"\n[OK] Completato! {saved_count} immagini salvate e ritagliate a {IMAGE_WIDTH}x{IMAGE_HEIGHT} in {OUTPUT_DIR}")

if __name__ == "__main__":
    process_and_save()