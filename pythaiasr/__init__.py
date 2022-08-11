# -*- coding: utf-8 -*-
import torch
from transformers import AutoProcessor, AutoModelForCTC
import torchaudio
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class ASR:
    def __init__(self, model: str="airesearch/wav2vec2-large-xlsr-53-th", device=None) -> None:
        self.processor = AutoProcessor.from_pretrained(model)
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
    
    def __call__(self, file: str, tokenized: bool = False) -> str:
        b = {}
        b['path'] = file
        a = self.prepare_dataset(self.resample(self.speech_file_to_array_fn(b)))
        input_dict = self.processor(a["input_values"][0], return_tensors="pt", padding=True)
        logits = self.model(input_dict.input_values).logits
        pred_ids = torch.argmax(logits, dim=-1)[0]

        if tokenized:
            txt = self.processor.decode(pred_ids)
        else:
            txt = self.processor.decode(pred_ids).replace(' ','')
        return txt

_model_name = "airesearch/wav2vec2-large-xlsr-53-th"
_model = ASR(model=_model_name)


def asr(file: str, tokenized: bool = False, model: str = "airesearch/wav2vec2-large-xlsr-53-th") -> str:
    """
    :param str file: path of sound file
    :param bool show_pad: show [PAD] in output
    :param str model: The ASR model
    :return: thai text from ASR
    :rtype: str
    """
    global _model, _model_name
    if model!=_model:
        _model = ASR(model)
        _model_name = model

    return _model(file=file, tokenized=tokenized)
