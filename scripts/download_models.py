#!/usr/bin/env python3
"""
download_models.py
------------------
Downloads all 5 S2S Translator models from Hugging Face.

Usage:
  python download_models.py
  python download_models.py --model whisper
"""

import argparse
import os
import json
from pathlib import Path
from datetime import datetime

# ── config ──────────────────────────────────────────────────────────
MODELS_DIR = Path("./models_cache")

MODELS = {
    "whisper": {
        "hf_id":       "openai/whisper-base",
        "type":        "asr",
        "description": "Whisper Base - English ASR",
        "loader":      "whisper",
        "subdir":      "asr/whisper-base",
    },
    "opus-en-fr": {
        "hf_id":       "Helsinki-NLP/opus-mt-tc-big-en-fr",
        "type":        "mt",
        "description": "OPUS-MT English → French",
        "loader":      "marian",
        "subdir":      "mt/opus-mt-tc-big-en-fr",
    },
    "opus-en-tw": {
        "hf_id":       "Helsinki-NLP/opus-mt-en-tw",
        "type":        "mt",
        "description": "OPUS-MT English → Twi",
        "loader":      "marian",
        "subdir":      "mt/opus-mt-en-tw",
    },
    "mms-tts-fra": {
        "hf_id":       "facebook/mms-tts-fra",
        "type":        "tts",
        "description": "MMS TTS French",
        "loader":      "vits",
        "subdir":      "tts/mms-tts-fra",
    },
    "mms-tts-aka": {
        "hf_id":       "facebook/mms-tts-aka",
        "type":        "tts",
        "description": "MMS TTS Akan/Twi",
        "loader":      "vits",
        "subdir":      "tts/mms-tts-aka",
    },
}

# ── helpers ──────────────────────────────────────────────────────────
def log(msg):
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}")

def get_save_dir(cfg, models_dir):
    return Path(models_dir) / cfg["subdir"]

def download_whisper(save_dir, hf_id):
    import whisper
    log(f"  Downloading Whisper base model...")
    whisper.load_model("base", download_root=str(save_dir))
    log(f"  ✓ Whisper saved to {save_dir}")
    return str(save_dir)

def download_marian(save_dir, hf_id):
    from transformers import MarianMTModel, MarianTokenizer
    log(f"  Downloading tokenizer for {hf_id} ...")
    tokenizer = MarianTokenizer.from_pretrained(hf_id)
    tokenizer.save_pretrained(str(save_dir))
    log(f"  Downloading model weights for {hf_id} ...")
    model = MarianMTModel.from_pretrained(hf_id)
    model.save_pretrained(str(save_dir))
    log(f"  ✓ MarianMT saved to {save_dir}")
    return str(save_dir)

def download_vits(save_dir, hf_id):
    from transformers import VitsModel, AutoTokenizer
    log(f"  Downloading tokenizer for {hf_id} ...")
    tokenizer = AutoTokenizer.from_pretrained(hf_id)
    tokenizer.save_pretrained(str(save_dir))
    log(f"  Downloading model weights for {hf_id} ...")
    model = VitsModel.from_pretrained(hf_id)
    model.save_pretrained(str(save_dir))
    log(f"  ✓ MMS VITS saved to {save_dir}")
    return str(save_dir)

LOADERS = {
    "whisper": download_whisper,
    "marian":  download_marian,
    "vits":    download_vits,
}

def update_db(hf_id, model_type, local_path, db_url):
    if not db_url:
        return
    try:
        import psycopg2
        table = {"asr": "asr_models", "mt": "mt_models", "tts": "tts_models"}[model_type]
        conn = psycopg2.connect(db_url)
        cur = conn.cursor()
        cur.execute(
            f"UPDATE {table} SET local_path=%s, is_downloaded=TRUE WHERE hf_model_id=%s",
            (local_path, hf_id)
        )
        conn.commit()
        cur.close()
        conn.close()
        log(f"  ✓ Database updated for {hf_id}")
    except Exception as e:
        log(f"  ⚠ DB update skipped: {e}")

def main():
    parser = argparse.ArgumentParser(description="Download S2S Translator models")
    parser.add_argument("--model", default="all",
                        help="Model to download: all, whisper, opus-en-fr, opus-en-tw, mms-tts-fra, mms-tts-aka")
    parser.add_argument("--db-url", default=os.getenv("DATABASE_URL"))
    parser.add_argument("--models-dir", default="./models_cache")
    args = parser.parse_args()

    models_dir = Path(args.models_dir)
    targets = MODELS if args.model == "all" else {args.model: MODELS[args.model]}

    log(f"Starting download of {len(targets)} model(s)...")
    log(f"Saving to: {models_dir.resolve()}\n")

    results = {}
    for key, cfg in targets.items():
        log(f"▶ [{key}] {cfg['description']}")
        save_dir = get_save_dir(cfg, models_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        try:
            local_path = LOADERS[cfg["loader"]](save_dir, cfg["hf_id"])
            update_db(cfg["hf_id"], cfg["type"], local_path, args.db_url)
            results[key] = {"status": "✓ success", "path": local_path}
        except Exception as e:
            log(f"  ✗ FAILED: {e}")
            results[key] = {"status": "✗ failed", "error": str(e)}
        print()

    print("=" * 60)
    print("DOWNLOAD SUMMARY")
    print("=" * 60)
    for key, r in results.items():
        print(f"  {r['status']}  {key:15s}  {r.get('path', r.get('error',''))}")

    manifest_path = models_dir / "manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps({
        "downloaded_at": datetime.now().isoformat(),
        "models": {k: {"hf_id": MODELS[k]["hf_id"], **v} for k, v in results.items()}
    }, indent=2))
    log(f"Manifest written to {manifest_path}")

if __name__ == "__main__":
    main()
