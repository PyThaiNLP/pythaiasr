# -*- coding: utf-8 -*-
import torch
import torchaudio
import numpy as np
import logging
from transformers.utils import logging
logging.set_verbosity(40)
import numpy as np


class ASR:
    def __init__(self, model: str="airesearch/wav2vec2-large-xlsr-53-th", lm: bool=False, device: str=None) -> None:
        """
        :param str model: The ASR model name
        :param bool lm: Use language model (default is False and except *airesearch/wav2vec2-large-xlsr-53-th* model)
        :param str device: device

        **Options for model**
            * *airesearch/wav2vec2-large-xlsr-53-th* (default) - AI RESEARCH - PyThaiNLP model
            * *wannaphong/wav2vec2-large-xlsr-53-th-cv8-newmm* - Thai Wav2Vec2 with CommonVoice V8 (newmm tokenizer) + language model 
            * *wannaphong/wav2vec2-large-xlsr-53-th-cv8-deepcut* - Thai Wav2Vec2 with CommonVoice V8 (deepcut tokenizer) + language model
            * *scb10x/typhoon-asr-realtime* - Typhoon ASR Real-Time model
        """
        self.model_name = model
        self.support_model =[
            "airesearch/wav2vec2-large-xlsr-53-th",
            "wannaphong/wav2vec2-large-xlsr-53-th-cv8-newmm",
            "wannaphong/wav2vec2-large-xlsr-53-th-cv8-deepcut",
            "scb10x/typhoon-asr-realtime"
        ]
        assert self.model_name in self.support_model
        self.lm = lm
        self.is_typhoon = "typhoon" in model.lower()
        
        if device!=None:
            self.device = torch.device(device) if not self.is_typhoon else device
        else:
            if self.is_typhoon:
                self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
            else:
                self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load model based on type
        if self.is_typhoon:
            try:
                import nemo.collections.asr as nemo_asr
            except ImportError:
                raise ImportError(
                    "nemo-toolkit is required for Typhoon ASR models. "
                    "Install it with: pip install pythaiasr[typhoon]"
                )
            self.model = nemo_asr.models.ASRModel.from_pretrained(
                model_name=self.model_name,
                map_location=self.device
            )
            self.processor = None
        elif not self.lm:
            from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
            self.processor = Wav2Vec2Processor.from_pretrained(self.model_name)
            self.model = Wav2Vec2ForCTC.from_pretrained(self.model_name).to(self.device)
        else:
            from transformers import AutoProcessor, AutoModelForCTC
            self.processor = AutoProcessor.from_pretrained(self.model_name)
            self.model = AutoModelForCTC.from_pretrained(self.model_name).to(self.device)

    def speech_file_to_array_fn(self, batch: dict) -> dict:
        speech_array, sampling_rate = torchaudio.load(batch["path"])
        batch["speech"] = speech_array[0]
        batch["sampling_rate"] = sampling_rate
        return batch

    def resample(self, batch: dict) -> dict:
        resampler=torchaudio.transforms.Resample(batch['sampling_rate'], 16_000)
        batch["speech"] = resampler(batch["speech"]).numpy()
        batch["sampling_rate"] = 16_000
        return batch

    def prepare_dataset(self, batch: dict) -> dict:
        # check that all files have the correct sampling rate
        batch["input_values"] = self.processor(batch["speech"], sampling_rate=batch["sampling_rate"]).input_values
        return batch
    
    def __call__(self, data: str, sampling_rate: int=16_000) -> str:
        """
        :param str data: path of sound file or numpy array of the voice
        :param int sampling_rate: The sample rate
        """
        # Typhoon ASR uses NeMo and has different API
        if self.is_typhoon:
            import librosa
            import soundfile as sf
            import tempfile
            import os
            
            # Handle numpy array or file path
            if isinstance(data, np.ndarray):
                # Save numpy array to temporary file for Typhoon ASR
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
                    temp_path = tmp_file.name
                    # Normalize audio
                    normalized_data = data / (np.max(np.abs(data)) + 1e-8)
                    sf.write(temp_path, normalized_data, sampling_rate)
                
                try:
                    transcriptions = self.model.transcribe(audio=[temp_path])
                    txt = transcriptions[0] if transcriptions else ""
                finally:
                    if os.path.exists(temp_path):
                        os.remove(temp_path)
            else:
                # File path - prepare audio for Typhoon ASR
                y, sr = librosa.load(str(data), sr=None)
                
                # Resample if needed
                if sr != 16000:
                    y = librosa.resample(y, orig_sr=sr, target_sr=16000)
                    sr = 16000
                
                # Normalize and save to temporary file
                y = y / (np.max(np.abs(y)) + 1e-8)
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
                    temp_path = tmp_file.name
                    sf.write(temp_path, y, sr)
                
                try:
                    transcriptions = self.model.transcribe(audio=[temp_path])
                    txt = transcriptions[0] if transcriptions else ""
                finally:
                    if os.path.exists(temp_path):
                        os.remove(temp_path)
            
            return txt
        
        # Wav2Vec2 models processing (existing code)
        b = {}
        if isinstance(data,np.ndarray):
            b["speech"] = data
            b["sampling_rate"] = sampling_rate
            _preprocessing = b
        else:
            b["path"] = data
            _preprocessing = self.speech_file_to_array_fn(b)
        a = self.prepare_dataset(b)
        input_dict = self.processor(a["input_values"][0], return_tensors="pt", padding=True).to(self.device)
        logits = self.model(input_dict.input_values).logits
        pred_ids = torch.argmax(logits, dim=-1)[0]
        if self.model_name == "airesearch/wav2vec2-large-xlsr-53-th":
            txt = self.processor.decode(pred_ids)
        elif self.lm:
            txt = self.processor.batch_decode(logits.detach().numpy()).text[0]
        else:
            txt = self.processor.decode(pred_ids)
        return txt

_model_name = "airesearch/wav2vec2-large-xlsr-53-th"
_model = None


def asr(data: str, model: str = _model_name, lm: bool=False, device: str=None, sampling_rate: int=16_000) -> str:
    """
    :param str data: path of sound file or numpy array of the voice
    :param str model: The ASR model name
    :param bool lm: Use language model (except *airesearch/wav2vec2-large-xlsr-53-th* model)
    :param str device: device
    :param int sampling_rate: The sample rate
    :return: Thai text from ASR
    :rtype: str

    **Options for model**
        * *airesearch/wav2vec2-large-xlsr-53-th* (default) - AI RESEARCH - PyThaiNLP model
        * *wannaphong/wav2vec2-large-xlsr-53-th-cv8-newmm* - Thai Wav2Vec2 with CommonVoice V8 (newmm tokenizer) (+ language model)
        * *wannaphong/wav2vec2-large-xlsr-53-th-cv8-deepcut* - Thai Wav2Vec2 with CommonVoice V8 (deepcut tokenizer) (+ language model)
        * *scb10x/typhoon-asr-realtime* - Typhoon ASR Real-Time model (requires nemo-toolkit)
    """
    global _model, _model_name
    if model!=_model or _model == None:
        _model = ASR(model, lm=lm, device=device)
        _model_name = model

    return _model(data=data, sampling_rate=sampling_rate)


def transcribe(audio_file: str, model: str = "scb10x/typhoon-asr-realtime", with_timestamps: bool = False, device: str = None) -> dict:
    """
    Real-time ASR inference with detailed output (Typhoon ASR models only).
    
    :param str audio_file: Path to audio file
    :param str model: The ASR model name (must be a Typhoon model)
    :param bool with_timestamps: Whether to return word-level timestamps
    :param str device: Device to run inference on ('cpu', 'cuda', or None for auto)
    :return: Dictionary containing transcription, timestamps (if requested), processing time, and audio duration
    :rtype: dict
    
    **Supported models**
        * *scb10x/typhoon-asr-realtime* - Typhoon ASR Real-Time model
    
    **Example**
        >>> result = transcribe("audio.wav", with_timestamps=True)
        >>> print(result['text'])
        >>> print(result['processing_time'])
        >>> for timestamp in result.get('timestamps', []):
        ...     print(f"{timestamp['word']}: {timestamp['start']:.2f}s - {timestamp['end']:.2f}s")
    """
    if "typhoon" not in model.lower():
        raise ValueError(
            f"transcribe() function only supports Typhoon ASR models. "
            f"Got model: {model}. Use asr() function for other models."
        )
    
    try:
        import nemo.collections.asr as nemo_asr
        import librosa
        import soundfile as sf
        import time
        from pathlib import Path
    except ImportError as e:
        raise ImportError(
            "nemo-toolkit and librosa are required for Typhoon ASR models. "
            "Install with: pip install pythaiasr[typhoon]"
        ) from e
    
    # Determine device
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Load model
    asr_model = nemo_asr.models.ASRModel.from_pretrained(
        model_name=model,
        map_location=device
    )
    
    # Prepare audio
    audio_path = Path(audio_file)
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_file}")
    
    # Load and preprocess audio
    y, sr = librosa.load(str(audio_path), sr=None)
    audio_duration = len(y) / sr
    
    # Resample if needed
    if sr != 16000:
        y = librosa.resample(y, orig_sr=sr, target_sr=16000)
        sr = 16000
    
    # Normalize
    y = y / (np.max(np.abs(y)) + 1e-8)
    
    # Save to temporary file
    import tempfile
    import os
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
        temp_path = tmp_file.name
        sf.write(temp_path, y, sr)
    
    try:
        start_time = time.time()
        
        result_data = {}
        
        if with_timestamps:
            # Get transcription with hypotheses for timestamp estimation
            hypotheses = asr_model.transcribe(audio=[temp_path], return_hypotheses=True)
            processing_time = time.time() - start_time
            
            transcription = ""
            if hypotheses and len(hypotheses) > 0 and hasattr(hypotheses[0], 'text'):
                transcription = hypotheses[0].text
            
            result_data['text'] = transcription
            
            # Generate estimated timestamps
            timestamps = []
            if transcription and audio_duration > 0:
                words = transcription.split()
                if len(words) > 0:
                    avg_duration = audio_duration / len(words)
                    for i, word in enumerate(words):
                        timestamps.append({
                            'word': word,
                            'start': i * avg_duration,
                            'end': (i + 1) * avg_duration
                        })
            result_data['timestamps'] = timestamps
        else:
            # Basic transcription
            transcriptions = asr_model.transcribe(audio=[temp_path])
            processing_time = time.time() - start_time
            result_data['text'] = transcriptions[0] if transcriptions else ""
        
        result_data['processing_time'] = processing_time
        result_data['audio_duration'] = audio_duration
        
        return result_data
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)
