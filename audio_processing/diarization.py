import torch
from pyannote.audio import Pipeline

class SpeakerDiarization:
    def __init__(self, model_name, use_auth_token):
        self.pipeline = Pipeline.from_pretrained(model_name, use_auth_token=use_auth_token)
        self.pipeline.to(torch.device("cuda"))

    def diarize(self, file_path):
        return self.pipeline(file_path)
