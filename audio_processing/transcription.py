import whisper

class AudioTranscriber:
    def __init__(self, model_name):
        self.model = whisper.load_model(model_name)

    def transcribe(self, audio_segment):
        result = self.model.transcribe(audio_segment, language='ru')
        return result['text'].strip()

    def summarize(self, text, num_sentences=20):
        parser = PlaintextParser.from_string(text, Tokenizer(self.language))
        summary = self.summarizer(parser.document, num_sentences)
        return summary

