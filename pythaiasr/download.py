# -*- coding: utf-8 -*-
import os
import sys
import shutil
import urllib.request
from pathlib import Path
from typing import Dict, Optional, Tuple

DEFAULT_MODEL_URLS = {
    "encoder": "https://huggingface.co/wannaphong/typhoon-asr-realtime-onnx/resolve/main/encoder-fastconformer-quran-ar.onnx",
    "decoder": "https://huggingface.co/wannaphong/typhoon-asr-realtime-onnx/resolve/main/decoder_joint-fastconformer-quran-ar.onnx",
    "vocab": "https://huggingface.co/wannaphong/typhoon-asr-realtime-onnx/resolve/main/tokenizer/vocab.json",
    "metadata": "https://huggingface.co/wannaphong/typhoon-asr-realtime-onnx/resolve/main/export_metadata.json",
}

DEFAULT_FILENAMES = {
    "encoder": "encoder-fastconformer-quran-ar.onnx",
    "decoder": "decoder_joint-fastconformer-quran-ar.onnx",
    "vocab": "vocab.json",
    "metadata": "export_metadata.json",
}

DEFAULT_DIARIZATION_URLS = {
    "segmentation": "https://huggingface.co/onnx-community/pyannote-segmentation-3.0/resolve/main/onnx/model.onnx",
}

DEFAULT_DIARIZATION_FILENAMES = {
    "segmentation": "segmentation-3.0.onnx",
}

DEFAULT_NEMOTRON_DIARIZATION_URLS = {
    "preprocessor": "https://huggingface.co/joosthel/Nemotron-3-Diarization-ONNX/resolve/main/preprocessor_core.onnx",
    "model_int8": "https://huggingface.co/joosthel/Nemotron-3-Diarization-ONNX/resolve/main/model.int8.onnx",
    "model_fp32": "https://huggingface.co/joosthel/Nemotron-3-Diarization-ONNX/resolve/main/model.onnx",
    "constants": "https://huggingface.co/joosthel/Nemotron-3-Diarization-ONNX/resolve/main/constants.npz",
}

DEFAULT_NEMOTRON_DIARIZATION_FILENAMES = {
    "preprocessor": "preprocessor_core.onnx",
    "model_int8": "model.int8.onnx",
    "model_fp32": "model.onnx",
    "constants": "constants.npz",
}

DEFAULT_NEMOTRON_ASR_URLS_INT4 = {
    "encoder": "https://huggingface.co/wannaphong/typhoon-asr-streaming-nemotron-0.6b-int4-onnx/resolve/main/encoder.onnx",
    "decoder": "https://huggingface.co/wannaphong/typhoon-asr-streaming-nemotron-0.6b-int4-onnx/resolve/main/decoder.onnx",
    "joint": "https://huggingface.co/wannaphong/typhoon-asr-streaming-nemotron-0.6b-int4-onnx/resolve/main/joint.onnx",
    "vocab": "https://huggingface.co/wannaphong/typhoon-asr-streaming-nemotron-0.6b-int4-onnx/resolve/main/tokens.txt",
    "config": "https://huggingface.co/wannaphong/typhoon-asr-streaming-nemotron-0.6b-int4-onnx/resolve/main/nemotron_onnx_config.json",
}

DEFAULT_NEMOTRON_ASR_URLS_FP32 = {
    "encoder": "https://huggingface.co/wannaphong/typhoon-asr-streaming-nemotron-0.6b-fp32-onnx/resolve/main/encoder.onnx",
    "encoder_data": "https://huggingface.co/wannaphong/typhoon-asr-streaming-nemotron-0.6b-fp32-onnx/resolve/main/encoder.onnx.data",
    "decoder": "https://huggingface.co/wannaphong/typhoon-asr-streaming-nemotron-0.6b-fp32-onnx/resolve/main/decoder.onnx",
    "joint": "https://huggingface.co/wannaphong/typhoon-asr-streaming-nemotron-0.6b-fp32-onnx/resolve/main/joint.onnx",
    "vocab": "https://huggingface.co/wannaphong/typhoon-asr-streaming-nemotron-0.6b-fp32-onnx/resolve/main/tokens.txt",
    "config": "https://huggingface.co/wannaphong/typhoon-asr-streaming-nemotron-0.6b-fp32-onnx/resolve/main/nemotron_onnx_config.json",
}

DEFAULT_NEMOTRON_ASR_URLS = DEFAULT_NEMOTRON_ASR_URLS_INT4

DEFAULT_NEMOTRON_ASR_FILENAMES = {
    "encoder": "encoder.onnx",
    "encoder_data": "encoder.onnx.data",
    "decoder": "decoder.onnx",
    "joint": "joint.onnx",
    "vocab": "tokens.txt",
    "config": "nemotron_onnx_config.json",
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


def get_diarization_model_files(
    model_dir: Optional[str] = None,
    urls: Optional[Dict[str, str]] = None,
) -> str:
    """
    Ensure ONNX diarization segmentation model is present.
    Downloads to `~/pythaiasr-data/diarization/` if missing.

    :param model_dir: Custom directory to store or load model files.
    :param urls: Custom dictionary with URLs for 'segmentation'.
    :return: Path to segmentation model file.
    """
    if model_dir is None:
        root_data = get_pythaiasr_path()
        model_dir = os.path.join(root_data, "diarization")
    else:
        model_dir = os.path.abspath(os.path.expanduser(model_dir))

    try:
        os.makedirs(model_dir, exist_ok=True)
    except OSError:
        pass

    urls = urls or DEFAULT_DIARIZATION_URLS
    seg_filename = DEFAULT_DIARIZATION_FILENAMES["segmentation"]
    seg_path = os.path.join(model_dir, seg_filename)

    # Check local candidate paths if not present
    if not os.path.exists(seg_path):
        local_candidates = [
            os.path.abspath("diarization"),
            os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "diarization")),
            os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models")),
        ]
        for candidate in local_candidates:
            cand_seg = os.path.join(candidate, seg_filename)
            if os.path.exists(cand_seg):
                try:
                    os.makedirs(model_dir, exist_ok=True)
                    if not os.path.exists(seg_path):
                        shutil.copy2(cand_seg, seg_path)
                except OSError:
                    return cand_seg
                break

    # Download missing files
    try:
        os.makedirs(model_dir, exist_ok=True)
    except OSError:
        pass

    if not os.path.exists(seg_path):
        download_file(urls["segmentation"], seg_path)

    return seg_path


def get_nemotron_diarization_model_files(
    model_dir: Optional[str] = None,
    urls: Optional[Dict[str, str]] = None,
    precision: str = "int8",
) -> Tuple[str, str, str]:
    """
    Ensure Nemotron-3 Diarization ONNX model files are present.
    Downloads to `~/pythaiasr-data/nemotron-3-diarization-onnx/` if missing.

    :param model_dir: Custom directory to store or load model files.
    :param urls: Custom dictionary with URLs for 'preprocessor', 'model_int8', 'model_fp32', 'constants'.
    :param precision: Model precision: 'int8' (default, ~104 MB) or 'fp32' (~397 MB).
    :return: Tuple of (preprocessor_path, model_path, constants_path).
    """
    if model_dir is None:
        root_data = get_pythaiasr_path()
        model_dir = os.path.join(root_data, "nemotron-3-diarization-onnx")
    else:
        model_dir = os.path.abspath(os.path.expanduser(model_dir))

    try:
        os.makedirs(model_dir, exist_ok=True)
    except OSError:
        pass

    urls = urls or DEFAULT_NEMOTRON_DIARIZATION_URLS
    model_key = "model_fp32" if precision == "fp32" else "model_int8"
    model_filename = DEFAULT_NEMOTRON_DIARIZATION_FILENAMES[model_key]
    prep_filename = DEFAULT_NEMOTRON_DIARIZATION_FILENAMES["preprocessor"]
    const_filename = DEFAULT_NEMOTRON_DIARIZATION_FILENAMES["constants"]

    prep_path = os.path.join(model_dir, prep_filename)
    model_path = os.path.join(model_dir, model_filename)
    const_path = os.path.join(model_dir, const_filename)

    # Check if files exist in target model_dir
    if os.path.exists(prep_path) and os.path.exists(model_path) and os.path.exists(const_path):
        return prep_path, model_path, const_path

    # Check local candidate paths
    local_candidates = [
        os.path.abspath("nemotron_diarization"),
        os.path.abspath("nemotron-3-diarization-onnx"),
        os.path.abspath("Nemotron-3-Diarization-ONNX"),
        os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "nemotron_diarization")),
        os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "nemotron-3-diarization-onnx")),
        os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "Nemotron-3-Diarization-ONNX")),
    ]

    for candidate in local_candidates:
        if os.path.isdir(candidate):
            cand_prep = os.path.join(candidate, prep_filename)
            cand_model = os.path.join(candidate, model_filename)
            cand_const = os.path.join(candidate, const_filename)
            if os.path.exists(cand_prep) and os.path.exists(cand_model) and os.path.exists(cand_const):
                try:
                    os.makedirs(model_dir, exist_ok=True)
                    if not os.path.exists(prep_path):
                        shutil.copy2(cand_prep, prep_path)
                    if not os.path.exists(model_path):
                        shutil.copy2(cand_model, model_path)
                    if not os.path.exists(const_path):
                        shutil.copy2(cand_const, const_path)
                except OSError:
                    return cand_prep, cand_model, cand_const
                return prep_path, model_path, const_path

    # Download missing files
    try:
        os.makedirs(model_dir, exist_ok=True)
    except OSError:
        pass

    if not os.path.exists(prep_path):
        download_file(urls["preprocessor"], prep_path)
    if not os.path.exists(model_path):
        download_file(urls[model_key], model_path)
    if not os.path.exists(const_path):
        download_file(urls["constants"], const_path)

    return prep_path, model_path, const_path


def get_nemotron_asr_model_files(
    model_dir: Optional[str] = None,
    precision: str = "int4",
    urls: Optional[Dict[str, str]] = None,
) -> Tuple[str, str, str, str, str]:
    """
    Ensure Nemotron 3.5 Streaming ASR ONNX model files are present.
    Downloads to `~/pythaiasr-data/typhoon-asr-streaming-nemotron-0.6b-{precision}-onnx/` if missing.

    :param model_dir: Custom directory to store or load model files.
    :param precision: Model precision ('int4' or 'fp32', default: 'int4').
    :param urls: Custom dictionary with URLs for 'encoder', 'encoder_data' (for fp32), 'decoder', 'joint', 'vocab', 'config'.
    :return: Tuple of (encoder_path, decoder_path, joint_path, vocab_path, config_path).
    """
    precision = precision.lower()
    if urls is None:
        urls = DEFAULT_NEMOTRON_ASR_URLS_FP32 if precision == "fp32" else DEFAULT_NEMOTRON_ASR_URLS_INT4

    if model_dir is None:
        root_data = get_pythaiasr_path()
        model_dir = os.path.join(root_data, f"typhoon-asr-streaming-nemotron-0.6b-{precision}-onnx")
    else:
        model_dir = os.path.abspath(os.path.expanduser(model_dir))

    try:
        os.makedirs(model_dir, exist_ok=True)
    except OSError:
        pass

    enc_path = os.path.join(model_dir, DEFAULT_NEMOTRON_ASR_FILENAMES["encoder"])
    enc_data_path = os.path.join(model_dir, DEFAULT_NEMOTRON_ASR_FILENAMES["encoder_data"])
    dec_path = os.path.join(model_dir, DEFAULT_NEMOTRON_ASR_FILENAMES["decoder"])
    joint_path = os.path.join(model_dir, DEFAULT_NEMOTRON_ASR_FILENAMES["joint"])
    vocab_path = os.path.join(model_dir, DEFAULT_NEMOTRON_ASR_FILENAMES["vocab"])
    config_path = os.path.join(model_dir, DEFAULT_NEMOTRON_ASR_FILENAMES["config"])

    all_needed = [enc_path, dec_path, joint_path, vocab_path]
    if "encoder_data" in urls:
        all_needed.append(enc_data_path)

    # Check local candidate paths if any file is missing
    if not all(os.path.exists(p) for p in all_needed):
        local_candidates = [
            os.path.abspath(f"typhoon-asr-streaming-nemotron-0.6b-{precision}-onnx"),
            os.path.abspath(f"typhoon-asr-streaming-nemotron-0.6b-onnx/{precision}"),
            os.path.abspath("typhoon-asr-streaming-nemotron-0.6b-onnx"),
            os.path.abspath(os.path.join(os.path.dirname(__file__), "..", f"typhoon-asr-streaming-nemotron-0.6b-{precision}-onnx")),
            os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "typhoon-asr-streaming-nemotron-0.6b-onnx", precision)),
            os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "typhoon-asr-streaming-nemotron-0.6b-onnx")),
        ]
        for candidate in local_candidates:
            if os.path.isdir(candidate):
                cand_files = {
                    key: os.path.join(candidate, DEFAULT_NEMOTRON_ASR_FILENAMES[key])
                    for key in DEFAULT_NEMOTRON_ASR_FILENAMES
                }
                # Also accept joiner.onnx if joint.onnx is missing
                if not os.path.exists(cand_files["joint"]):
                    cand_joiner = os.path.join(candidate, "joiner.onnx")
                    if os.path.exists(cand_joiner):
                        cand_files["joint"] = cand_joiner
                # Also accept tokens.txt / vocab.txt
                if not os.path.exists(cand_files["vocab"]):
                    cand_voc = os.path.join(candidate, "vocab.txt")
                    if os.path.exists(cand_voc):
                        cand_files["vocab"] = cand_voc

                # Check if core files exist
                needed_cand = [cand_files["encoder"], cand_files["decoder"], cand_files["joint"], cand_files["vocab"]]
                if "encoder_data" in urls:
                    needed_cand.append(cand_files["encoder_data"])

                if all(os.path.exists(p) for p in needed_cand):
                    try:
                        os.makedirs(model_dir, exist_ok=True)
                        for key, dest in [
                            ("encoder", enc_path),
                            ("encoder_data", enc_data_path),
                            ("decoder", dec_path),
                            ("joint", joint_path),
                            ("vocab", vocab_path),
                            ("config", config_path),
                        ]:
                            src = cand_files.get(key)
                            if src and os.path.exists(src) and not os.path.exists(dest):
                                shutil.copy2(src, dest)
                    except OSError:
                        return cand_files["encoder"], cand_files["decoder"], cand_files["joint"], cand_files["vocab"], cand_files.get("config", config_path)
                    break

    # Download missing files
    try:
        os.makedirs(model_dir, exist_ok=True)
    except OSError:
        pass

    for key, path in [
        ("encoder", enc_path),
        ("encoder_data", enc_data_path),
        ("decoder", dec_path),
        ("joint", joint_path),
        ("vocab", vocab_path),
        ("config", config_path),
    ]:
        if not os.path.exists(path) and key in urls:
            download_file(urls[key], path)

    return enc_path, dec_path, joint_path, vocab_path, config_path

