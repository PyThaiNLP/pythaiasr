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
            * *biodatlab/whisper-small-th-combined* - Thai Whisper small model
            * *biodatlab/whisper-th-medium-combined* - Thai Whisper medium model
            * *biodatlab/whisper-th-large-combined* - Thai Whisper large model
        """
        self.model_name = model
        self.support_model =[
            "airesearch/wav2vec2-large-xlsr-53-th",
            "wannaphong/wav2vec2-large-xlsr-53-th-cv8-newmm",
            "wannaphong/wav2vec2-large-xlsr-53-th-cv8-deepcut",
            "biodatlab/whisper-small-th-combined",
            "biodatlab/whisper-th-medium-combined",
            "biodatlab/whisper-th-large-combined"
        ]
        self.whisper_models = [
            "biodatlab/whisper-small-th-combined",
            "biodatlab/whisper-th-medium-combined",
            "biodatlab/whisper-th-large-combined"
        ]
        assert self.model_name in self.support_model
        self.lm = lm
        if device!=None:
            self.device = torch.device(device)
        else:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Check if model is a Whisper model
        self.is_whisper = self.model_name in self.whisper_models
        
        if self.is_whisper:
            from transformers import WhisperProcessor, WhisperForConditionalGeneration
            self.processor = WhisperProcessor.from_pretrained(self.model_name)
            self.model = WhisperForConditionalGeneration.from_pretrained(self.model_name).to(self.device)
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
        b = {}
        if isinstance(data,np.ndarray):
            b["speech"] = data
            b["sampling_rate"] = sampling_rate
            _preprocessing = b
        else:
            b["path"] = data
            _preprocessing = self.speech_file_to_array_fn(b)
        
        if self.is_whisper:
            # Whisper model processing
            # Resample if needed
            if b["sampling_rate"] != 16_000:
                # Convert to tensor if needed
                speech_tensor = b["speech"] if isinstance(b["speech"], torch.Tensor) else torch.tensor(b["speech"])
                resampler = torchaudio.transforms.Resample(b['sampling_rate'], 16_000)
                b["speech"] = resampler(speech_tensor).numpy()
                b["sampling_rate"] = 16_000
            
            # Process audio with Whisper processor
            input_features = self.processor(b["speech"], sampling_rate=16_000, return_tensors="pt").input_features
            input_features = input_features.to(self.device)
            
            # Generate transcription
            predicted_ids = self.model.generate(input_features)
            txt = self.processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
        else:
            # Wav2Vec2 model processing
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
        * *biodatlab/whisper-small-th-combined* - Thai Whisper small model
        * *biodatlab/whisper-th-medium-combined* - Thai Whisper medium model
        * *biodatlab/whisper-th-large-combined* - Thai Whisper large model
    """
    global _model, _model_name
    if model!=_model or _model == None:
        _model = ASR(model, lm=lm, device=device)
        _model_name = model

    return _model(data=data, sampling_rate=sampling_rate)


def stream_asr(model: str = _model_name, lm: bool=False, device: str=None, 
               chunk_duration: float=5.0, sampling_rate: int=16_000):
    """
    Stream audio from microphone/soundcard and perform real-time ASR.
    
    :param str model: The ASR model name
    :param bool lm: Use language model (except *airesearch/wav2vec2-large-xlsr-53-th* model)
    :param str device: device
    :param float chunk_duration: Duration of each audio chunk in seconds (default: 5.0)
    :param int sampling_rate: The sample rate (default: 16000)
    :yield: Thai text transcription from each audio chunk
    
    **Options for model**
        * *airesearch/wav2vec2-large-xlsr-53-th* (default) - AI RESEARCH - PyThaiNLP model
        * *wannaphong/wav2vec2-large-xlsr-53-th-cv8-newmm* - Thai Wav2Vec2 with CommonVoice V8 (newmm tokenizer) (+ language model)
        * *wannaphong/wav2vec2-large-xlsr-53-th-cv8-deepcut* - Thai Wav2Vec2 with CommonVoice V8 (deepcut tokenizer) (+ language model)
        * *biodatlab/whisper-small-th-combined* - Thai Whisper small model
        * *biodatlab/whisper-th-medium-combined* - Thai Whisper medium model
        * *biodatlab/whisper-th-large-combined* - Thai Whisper large model
    
    **Example:**
        .. code-block:: python
        
            from pythaiasr import stream_asr
            
            # Stream audio and print transcriptions
            for transcription in stream_asr(chunk_duration=5.0):
                print(transcription)
                # Press Ctrl+C to stop
    """
    try:
        import pyaudio
    except ImportError:
        raise ImportError(
            "pyaudio is required for audio streaming. "
            "Install it with: pip install pyaudio"
        )
    
    global _model, _model_name
    if model!=_model or _model == None:
        _model = ASR(model, lm=lm, device=device)
        _model_name = model
    
    # Initialize PyAudio
    audio = pyaudio.PyAudio()
    
    # Calculate chunk size
    chunk_size = int(sampling_rate * chunk_duration)
    
    try:
        # Open stream
        stream = audio.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=sampling_rate,
            input=True,
            frames_per_buffer=chunk_size
        )
        
        print(f"Recording audio from microphone... (chunk duration: {chunk_duration}s)")
        print("Press Ctrl+C to stop.")
        
        while True:
            # Read audio chunk
            audio_data = stream.read(chunk_size, exception_on_overflow=False)
            
            # Convert to numpy array
            audio_array = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32)
            
            # Normalize to [-1, 1]
            audio_array = audio_array / 32768.0
            
            # Perform ASR on chunk
            transcription = _model(data=audio_array, sampling_rate=sampling_rate)
            
            if transcription.strip():  # Only yield non-empty transcriptions
                yield transcription
                
    except KeyboardInterrupt:
        print("\nStopping audio stream...")
    finally:
        # Clean up
        if 'stream' in locals():
            stream.stop_stream()
            stream.close()
        audio.terminate()
