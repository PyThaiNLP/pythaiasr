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
        """
        self.model_name = model
        self.support_model =[
            "airesearch/wav2vec2-large-xlsr-53-th",
            "wannaphong/wav2vec2-large-xlsr-53-th-cv8-newmm",
            "wannaphong/wav2vec2-large-xlsr-53-th-cv8-deepcut"
        ]
        assert self.model_name in self.support_model
        self.lm = lm
        if device!=None:
            self.device = torch.device(device)
        else:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if not self.lm:
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
    """
    global _model, _model_name
    if model!=_model or _model == None:
        _model = ASR(model, lm=lm, device=device)
        _model_name = model

    return _model(data=data, sampling_rate=sampling_rate)
