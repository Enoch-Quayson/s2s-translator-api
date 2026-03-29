#!/usr/bin/env python3
import os
import time
import torch
import numpy as np
import scipy.io.wavfile as wavfile
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Literal

# ── model paths ──────────────────────────────────────────────────────
MODELS_DIR = Path(os.getenv("MODELS_DIR", "./models_cache"))

MODEL_PATHS = {
    "asr": {
        "en": MODELS_DIR / "asr" / "whisper-base",
    },
    "mt": {
        "en-fr": MODELS_DIR / "mt" / "opus-mt-tc-big-en-fr",
        "en-tw": MODELS_DIR / "mt" / "opus-mt-en-tw",
    },
    "tts": {
        "fr": MODELS_DIR / "tts" / "mms-tts-fra",
        "tw": MODELS_DIR / "tts" / "mms-tts-aka",
    },
}

FFMPEG_PATH = r"C:\Users\kbenz\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-8.1-full_build\bin"

# ── result type ───────────────────────────────────────────────────────
@dataclass
class TranslationResult:
    source_language: str
    target_language: str
    asr_transcript: str
    asr_confidence: float
    translated_text: str
    audio_path: Optional[str]
    asr_duration_ms: int
    mt_duration_ms: int
    tts_duration_ms: int
    total_duration_ms: int
    error: Optional[str] = None

    @property
    def success(self) -> bool:
        return self.error is None


# ── ASR module ────────────────────────────────────────────────────────
class ASRModule:
    def __init__(self, model_path: Optional[str] = None):
        os.environ["PATH"] += os.pathsep + FFMPEG_PATH
        import whisper
        path = model_path or str(MODEL_PATHS["asr"]["en"])
        self._model_path = path
        if Path(path).exists():
            self.model = whisper.load_model("base", download_root=path)
        else:
            self.model = whisper.load_model("base")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def transcribe(self, audio_path: str) -> tuple[str, float]:
        result = self.model.transcribe(audio_path, language="en", fp16=False)
        transcript = result["text"].strip()
        segments = result.get("segments", [])
        if segments:
            avg_logprob = np.mean([s.get("avg_logprob", -0.5) for s in segments])
            confidence = float(np.clip(1 + avg_logprob, 0, 1))
        else:
            confidence = 0.0
        return transcript, confidence


# ── MT module ─────────────────────────────────────────────────────────
class MTModule:
    def __init__(self, source_lang: str, target_lang: str, model_path: Optional[str] = None):
        from transformers import MarianMTModel, MarianTokenizer
        pair = f"{source_lang}-{target_lang}"
        path = model_path or str(MODEL_PATHS["mt"].get(pair, ""))
        if Path(path).exists():
            self.tokenizer = MarianTokenizer.from_pretrained(path)
            self.model = MarianMTModel.from_pretrained(path)
        else:
            hf_ids = {
                "en-fr": "Helsinki-NLP/opus-mt-tc-big-en-fr",
                "en-tw": "Helsinki-NLP/opus-mt-en-tw",
            }
            hf_id = hf_ids.get(pair)
            if not hf_id:
                raise ValueError(f"No MT model for language pair: {pair}")
            self.tokenizer = MarianTokenizer.from_pretrained(hf_id)
            self.model = MarianMTModel.from_pretrained(hf_id)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)

    def translate(self, text: str, max_length: int = 512) -> str:
        inputs = self.tokenizer(
            [text], return_tensors="pt", padding=True,
            truncation=True, max_length=max_length
        ).to(self.device)
        with torch.no_grad():
            translated_ids = self.model.generate(**inputs, max_length=max_length)
        return self.tokenizer.decode(translated_ids[0], skip_special_tokens=True)


# ── TTS module ────────────────────────────────────────────────────────
class TTSModule:
    SAMPLE_RATE = 16000

    def __init__(self, language: str, model_path: Optional[str] = None):
        from transformers import VitsModel, AutoTokenizer
        path = model_path or str(MODEL_PATHS["tts"].get(language, ""))
        if Path(path).exists():
            self.tokenizer = AutoTokenizer.from_pretrained(path)
            self.model = VitsModel.from_pretrained(path)
        else:
            hf_ids = {
                "fr": "facebook/mms-tts-fra",
                "tw": "facebook/mms-tts-aka",
            }
            hf_id = hf_ids.get(language)
            if not hf_id:
                raise ValueError(f"No TTS model for language: {language}")
            self.tokenizer = AutoTokenizer.from_pretrained(hf_id)
            self.model = VitsModel.from_pretrained(hf_id)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        self.language = language

    def synthesize(self, text: str, output_path: str) -> str:
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        torch.manual_seed(42)
        with torch.no_grad():
            waveform = self.model(**inputs).waveform
        waveform_np = waveform.squeeze().cpu().float().numpy()
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        wavfile.write(output_path, self.SAMPLE_RATE, waveform_np)
        return output_path


# ── Main Pipeline ─────────────────────────────────────────────────────
class S2SPipeline:
    def __init__(
        self,
        target_language: str = "fr",
        asr_model_path: Optional[str] = None,
        mt_model_path: Optional[str] = None,
        tts_model_path: Optional[str] = None,
        lazy_load: bool = True,
    ):
        if target_language not in ("fr", "tw"):
            raise ValueError("target_language must be 'fr' or 'tw'")
        self.source_language = "en"
        self.target_language = target_language
        self._asr_model_path = asr_model_path
        self._mt_model_path = mt_model_path
        self._tts_model_path = tts_model_path
        self._asr: Optional[ASRModule] = None
        self._mt: Optional[MTModule] = None
        self._tts: Optional[TTSModule] = None
        if not lazy_load:
            self._load_all()

    def _load_all(self):
        self._get_asr()
        self._get_mt()
        self._get_tts()

    def _get_asr(self) -> ASRModule:
        if self._asr is None:
            self._asr = ASRModule(model_path=self._asr_model_path)
        return self._asr

    def _get_mt(self) -> MTModule:
        if self._mt is None:
            self._mt = MTModule(
                source_lang="en",
                target_lang=self.target_language,
                model_path=self._mt_model_path
            )
        return self._mt

    def _get_tts(self) -> TTSModule:
        if self._tts is None:
            self._tts = TTSModule(
                language=self.target_language,
                model_path=self._tts_model_path
            )
        return self._tts

    def translate_audio(
        self,
        audio_path: str,
        output_dir: str = "./outputs",
        output_filename: Optional[str] = None,
    ) -> TranslationResult:
        pipeline_start = time.time()
        try:
            asr_start = time.time()
            asr = self._get_asr()
            transcript, confidence = asr.transcribe(audio_path)
            asr_ms = int((time.time() - asr_start) * 1000)

            mt_start = time.time()
            mt = self._get_mt()
            translated_text = mt.translate(transcript)
            mt_ms = int((time.time() - mt_start) * 1000)

            tts_start = time.time()
            tts = self._get_tts()
            if output_filename is None:
                stem = Path(audio_path).stem
                output_filename = f"{stem}_{self.target_language}_translated.wav"
            output_path = str(Path(output_dir) / output_filename)
            tts.synthesize(translated_text, output_path)
            tts_ms = int((time.time() - tts_start) * 1000)
            total_ms = int((time.time() - pipeline_start) * 1000)

            return TranslationResult(
                source_language="en", target_language=self.target_language,
                asr_transcript=transcript, asr_confidence=confidence,
                translated_text=translated_text, audio_path=output_path,
                asr_duration_ms=asr_ms, mt_duration_ms=mt_ms,
                tts_duration_ms=tts_ms, total_duration_ms=total_ms,
            )
        except Exception as e:
            total_ms = int((time.time() - pipeline_start) * 1000)
            return TranslationResult(
                source_language="en", target_language=self.target_language,
                asr_transcript="", asr_confidence=0.0, translated_text="",
                audio_path=None, asr_duration_ms=0, mt_duration_ms=0,
                tts_duration_ms=0, total_duration_ms=total_ms, error=str(e),
            )

    def translate_text(
        self,
        text: str,
        output_dir: str = "./outputs",
        output_filename: Optional[str] = None,
    ) -> TranslationResult:
        pipeline_start = time.time()
        try:
            mt_start = time.time()
            mt = self._get_mt()
            translated_text = mt.translate(text)
            mt_ms = int((time.time() - mt_start) * 1000)

            tts_start = time.time()
            tts = self._get_tts()
            if output_filename is None:
                import hashlib
                h = hashlib.md5(text.encode()).hexdigest()[:8]
                output_filename = f"tts_{self.target_language}_{h}.wav"
            output_path = str(Path(output_dir) / output_filename)
            tts.synthesize(translated_text, output_path)
            tts_ms = int((time.time() - tts_start) * 1000)
            total_ms = int((time.time() - pipeline_start) * 1000)

            return TranslationResult(
                source_language="en", target_language=self.target_language,
                asr_transcript=text, asr_confidence=1.0,
                translated_text=translated_text, audio_path=output_path,
                asr_duration_ms=0, mt_duration_ms=mt_ms,
                tts_duration_ms=tts_ms, total_duration_ms=total_ms,
            )
        except Exception as e:
            return TranslationResult(
                source_language="en", target_language=self.target_language,
                asr_transcript="", asr_confidence=0.0, translated_text="",
                audio_path=None, asr_duration_ms=0, mt_duration_ms=0,
                tts_duration_ms=0, total_duration_ms=0, error=str(e),
            )
