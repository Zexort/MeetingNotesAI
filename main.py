import torch
from audio_processing.audio_loader import load_audio
from audio_processing.diarization import SpeakerDiarization
from audio_processing.transcription import AudioTranscriber
from summarization.summarizer import TextSummarizer
from question_answering.qa_model import QuestionAnsweringModel

def main(file_path):
    # Загрузка аудио
    audio = load_audio(file_path)

    # Диаризация
    diarization_model = SpeakerDiarization("pyannote/speaker-diarization-3.1", use_auth_token="HG_TOKEN")  # Вписать токен с Hugging face
    diarization = diarization_model.diarize(file_path)

    # Транскрипция
    transcriber = AudioTranscriber("large")
    dialogue = ""

    for segment, _, speaker in diarization.itertracks(yield_label=True):
        start_time = segment.start
        end_time = segment.end
        audio_segment = audio[int(start_time * 16000):int(end_time * 16000)]
        text_segment = transcriber.transcribe(audio_segment)

        if text_segment and text_segment != "Продолжение следует...":
            speaker_number = speaker.split('_')[-1]
            dialogue += f"Speaker{speaker_number}: {text_segment}\n"
            
    print(dialogue)

    # Суммаризация
    summarizer = TextSummarizer()
    summary = summarizer.summarize(dialogue, num_sentences=20)

    # Вывод суммаризации
    print("Суммаризация:")
    for sentence in summary:
        print(sentence)

    # Вопросы и ответы
    qa_model = QuestionAnsweringModel("Qwen/Qwen2.5-Coder-7B-Instruct")
    prompt = (
        f"На основе следующего текста: {dialogue}, "
        f"ответьте на следующие вопросы:\n"
        f"1. Какие вопросы были решены?\n"
        f"2. Какие вопросы остались открытыми?"
    )
    
    response = qa_model.generate_response(prompt)
    print("\nОтветы на вопросы:")
    print(response)


file_path = input("Введите путь до аудиофайла:")  # Укажите путь к вашему аудиофайлу
main(file_path)
torch.cuda.empty_cache()  # Очистка видеопамяти для дальнейшей работы с LLM

