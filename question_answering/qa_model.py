from transformers import AutoModelForCausalLM, AutoTokenizer

class QuestionAnsweringModel:
    def __init__(self, model_name):
        self.model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto", device_map="auto")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def generate_response(self, prompt):
        messages = [
            {"role": "system", "content": "Вы бот, который отвечает на вопросы на основе предоставленного текста."},
            {"role": "user", "content": prompt}
        ]

        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

        generated_ids = self.model.generate(**model_inputs, max_new_tokens=512)
        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return response