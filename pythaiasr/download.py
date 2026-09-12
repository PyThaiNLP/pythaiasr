# -*- coding: utf-8 -*-
import os
import sys
import shutil
import urllib.request
from pathlib import Path
from typing import Dict, Optional, Tuple

DEFAULT_MODEL_URLS = {
    "encoder": "https://huggingface.co/wannaphong/asr_cat_model/resolve/main/encoder-fastconformer-quran-ar.onnx",
    "decoder": "https://huggingface.co/wannaphong/asr_cat_model/resolve/main/decoder_joint-fastconformer-quran-ar.onnx",
    "vocab": "https://huggingface.co/wannaphong/asr_cat_model/resolve/main/tokenizer/vocab.json",
    "metadata": "https://huggingface.co/wannaphong/asr_cat_model/resolve/main/export_metadata.json",
}

DEFAULT_FILENAMES = {
    "encoder": "encoder-fastconformer-quran-ar.onnx",
    "decoder": "decoder_joint-fastconformer-quran-ar.onnx",
    "vocab": "vocab.json",
    "metadata": "export_metadata.json",
}


def get_pythaiasr_path() -> str:
    """
    Get root user data path for pythaiasr (~/pythaiasr-data).

    Can be customized via environment variable `PYTHAIASR_DATA_DIR` or `PYTHAIASR_CACHE_DIR`.
    """
    env_path = os.environ.get("PYTHAIASR_DATA_DIR") or os.environ.get("PYTHAIASR_CACHE_DIR")
    if env_path:
        path = os.path.abspath(os.path.expanduser(env_path))
    else:
        path = os.path.join(os.path.expanduser("~"), "pythaiasr-data")
    try:
        os.makedirs(path, exist_ok=True)
    except OSError:
        pass
    return path


def download_file(url: str, dest_path: str, chunk_size: int = 1024 * 1024) -> str:
    """
    Download a file from `url` to `dest_path` with progress indication.
    Downloads to a temporary file first and renames upon completion.
    """
    dest_path = os.path.abspath(dest_path)
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)

    temp_path = f"{dest_path}.tmp"
    filename = os.path.basename(dest_path)

    print(f"Downloading {filename} from {url} ...")

    try:
        # Try using requests if available
        try:
            import requests
            with requests.get(url, stream=True, timeout=60) as resp:
                resp.raise_for_status()
                total_size = int(resp.headers.get("content-length", 0))
                downloaded = 0
                with open(temp_path, "wb") as f:
                    for chunk in resp.iter_content(chunk_size=chunk_size):
                        if chunk:
                            f.write(chunk)
                            downloaded += len(chunk)
                            if total_size > 0:
                                percent = (downloaded / total_size) * 100
                                mb_down = downloaded / (1024 * 1024)
                                mb_total = total_size / (1024 * 1024)
                                sys.stdout.write(f"\r  [{percent:5.1f}%] {mb_down:6.1f} / {mb_total:6.1f} MB")
                                sys.stdout.flush()
            if total_size > 0:
                sys.stdout.write("\n")
        except ImportError:
            # Fallback to urllib.request
            req = urllib.request.Request(
                url,
                headers={"User-Agent": "PyThaiASR/1.3"}
            )
            with urllib.request.urlopen(req, timeout=60) as resp, open(temp_path, "wb") as f:
                total_size = int(resp.headers.get("content-length", 0))
                downloaded = 0
                while True:
                    chunk = resp.read(chunk_size)
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        percent = (downloaded / total_size) * 100
                        mb_down = downloaded / (1024 * 1024)
                        mb_total = total_size / (1024 * 1024)
                        sys.stdout.write(f"\r  [{percent:5.1f}%] {mb_down:6.1f} / {mb_total:6.1f} MB")
                        sys.stdout.flush()
            if total_size > 0:
                sys.stdout.write("\n")

        shutil.move(temp_path, dest_path)
        print(f"Saved: {dest_path}")
        return dest_path
    except Exception as e:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        raise RuntimeError(f"Failed to download {url} to {dest_path}: {e}") from e


def get_typhoon_model_files(
    model_dir: Optional[str] = None,
    urls: Optional[Dict[str, str]] = None,
) -> Tuple[str, str, str]:
    """
    Ensure Typhoon FastConformer RNN-T ONNX model files and vocab are present.
    Downloads them to `~/pythaiasr-data/typhoon-asr-realtime/` if missing.

    :param model_dir: Custom directory to store or load model files.
    :param urls: Custom dictionary with URLs for 'encoder', 'decoder', 'vocab'.
    :return: Tuple of (encoder_path, decoder_path, vocab_path)
    """
    if model_dir is None:
        root_data = get_pythaiasr_path()
        model_dir = os.path.join(root_data, "typhoon-asr-realtime")
    else:
        model_dir = os.path.abspath(os.path.expanduser(model_dir))

    try:
        os.makedirs(model_dir, exist_ok=True)
    except OSError:
        pass

    urls = urls or DEFAULT_MODEL_URLS

    encoder_path = os.path.join(model_dir, DEFAULT_FILENAMES["encoder"])
    decoder_path = os.path.join(model_dir, DEFAULT_FILENAMES["decoder"])
    vocab_path = os.path.join(model_dir, DEFAULT_FILENAMES["vocab"])

    # If not in target model_dir, check if already present in a local typhoon_asr directory
    if not (os.path.exists(encoder_path) and os.path.exists(decoder_path) and os.path.exists(vocab_path)):
        local_candidates = [
            os.path.abspath("typhoon_asr"),
            os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "typhoon_asr")),
        ]
        for candidate in local_candidates:
            cand_enc = os.path.join(candidate, DEFAULT_FILENAMES["encoder"])
            cand_dec = os.path.join(candidate, DEFAULT_FILENAMES["decoder"])
            cand_voc = os.path.join(candidate, "tokenizer", DEFAULT_FILENAMES["vocab"])
            if not os.path.exists(cand_voc):
                cand_voc = os.path.join(candidate, DEFAULT_FILENAMES["vocab"])
            if os.path.exists(cand_enc) and os.path.exists(cand_dec) and os.path.exists(cand_voc):
                try:
                    os.makedirs(model_dir, exist_ok=True)
                    if not os.path.exists(encoder_path):
                        shutil.copy2(cand_enc, encoder_path)
                    if not os.path.exists(decoder_path):
                        shutil.copy2(cand_dec, decoder_path)
                    if not os.path.exists(vocab_path):
                        shutil.copy2(cand_voc, vocab_path)
                except OSError:
                    # If target directory is not writable (e.g. sandbox), return local candidate paths directly
                    return cand_enc, cand_dec, cand_voc
                break

    # Download missing files
    try:
        os.makedirs(model_dir, exist_ok=True)
    except OSError:
        pass

    if not os.path.exists(encoder_path):
        download_file(urls["encoder"], encoder_path)
    if not os.path.exists(decoder_path):
        download_file(urls["decoder"], decoder_path)
    if not os.path.exists(vocab_path):
        download_file(urls["vocab"], vocab_path)

    return encoder_path, decoder_path, vocab_path
