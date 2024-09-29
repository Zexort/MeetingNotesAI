from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer

class TextSummarizer:
    def init(self, language='russian'):
        self.language = language
        self.summarizer = LsaSummarizer()