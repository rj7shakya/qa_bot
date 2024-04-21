import os

import dotenv
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer

dotenv.load_dotenv()

CKPT_PATH = os.getenv('MODEL_PATH')

def load_and_predict(question):
    tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-base")
    model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-base")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.load_state_dict(torch.load(CKPT_PATH, map_location=device))

    inputs = tokenizer.encode("answer the question:" + question, return_tensors="pt")
    output_ids = model.generate(inputs, max_length=1024, num_return_sequences=1)
    output = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return output
