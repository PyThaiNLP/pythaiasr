# -*- coding: utf-8 -*-
import torch
from transformers import AutoProcessor, AutoModelForCTC
import torchaudio
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class ASR:
    def __init__(self, model: str="airesearch/wav2vec2-large-xlsr-53-th", device=None) -> None:
        """
        :param str model: The ASR model
        :param str device: device

        **Options for model**
            * *airesearch/wav2vec2-large-xlsr-53-th* (default) - AI RESEARCH - PyThaiNLP model
            * *wannaphong/wav2vec2-large-xlsr-53-th-cv8-newmm* - Thai Wav2Vec2 with CommonVoice V8 (newmm tokenizer) + language model 
            * *wannaphong/wav2vec2-large-xlsr-53-th-cv8-deepcut* - Thai Wav2Vec2 with CommonVoice V8 (deepcut tokenizer) + language model 
        """
        self.processor = AutoProcessor.from_pretrained(model)
        self.model_name = model
        self.model = AutoModelForCTC.from_pretrained(model)
        if device!=None:
            self.device = torch.device(device)

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
    
    def __call__(self, file: str) -> str:
        """
        :param str file: path of sound file
        :param str model: The ASR model
        """
        b = {}
        b['path'] = file
        a = self.prepare_dataset(self.resample(self.speech_file_to_array_fn(b)))
        input_dict = self.processor(a["input_values"][0], return_tensors="pt", padding=True)
        logits = self.model(input_dict.input_values).logits
        pred_ids = torch.argmax(logits, dim=-1)[0]

        txt = self.processor.batch_decode(logits.detach().numpy()).text[0]
        return txt

_model_name = "airesearch/wav2vec2-large-xlsr-53-th"
_model = None


def asr(file: str, model: str = _model_name) -> str:
    """
    :param str file: path of sound file
    :param str model: The ASR model
    :return: thai text from ASR
    :rtype: str

    **Options for model**
        * *airesearch/wav2vec2-large-xlsr-53-th* (default) - AI RESEARCH - PyThaiNLP model
        * *wannaphong/wav2vec2-large-xlsr-53-th-cv8-newmm* - Thai Wav2Vec2 with CommonVoice V8 (newmm tokenizer) + language model 
        * *wannaphong/wav2vec2-large-xlsr-53-th-cv8-deepcut* - Thai Wav2Vec2 with CommonVoice V8 (deepcut tokenizer) + language model 
    """
    global _model, _model_name
    if model!=_model or _model == None:
        _model = ASR(model)
        _model_name = model

    return _model(file=file)
