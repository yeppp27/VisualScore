import argparse
import os
import torch
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2_5_VLProcessor
from PIL import Image, ImageOps
import re

SYSTEM_PROMPT = (
    "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. "
    "The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. "
    "The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags."
)

def load_image(image_path: str) -> "PIL.Image.Image":
    image = Image.open(image_path)
    image = ImageOps.exif_transpose(image).convert("RGB")
    return image

def strip_sequence(text, pad_token, eos_token):
    pad_token_escaped = re.escape(pad_token)
    eos_token_escaped = re.escape(eos_token)
    text = re.sub(f"^({eos_token_escaped}|{pad_token_escaped})+", "", text)
    text = re.sub(f"({eos_token_escaped}|{pad_token_escaped})+$", "", text)
    return text

class RewardModelProxy:
    def __init__(self, args):
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            args.reward_pretrain, device_map="auto", torch_dtype=torch.bfloat16
        )
        self.processor = Qwen2_5_VLProcessor.from_pretrained(args.reward_pretrain)
        self.max_length = args.max_len
        self.batch_size = args.batch_size

    def evaluate_correctness(self, generated: str) -> float:
        return 1.0 if "correct" in generated.lower() else 0.5

    def get_reward(self, queries, image_paths):
        batch_size = self.batch_size or len(queries)
        rewards = []

        for idx in range(0, len(queries), batch_size):
            batch_queries = queries[idx:idx+batch_size]
            batch_images = image_paths[idx:idx+batch_size]

            cleaned_queries = [
                strip_sequence(q, self.processor.pad_token, self.processor.eos_token) + self.processor.eos_token
                for q in batch_queries
            ]

            for i, query in enumerate(cleaned_queries):
                image = load_image(batch_images[i])
                text = self.processor.apply_chat_template(
                    [{"role": "system", "content": SYSTEM_PROMPT},
                     {"role": "user", "content": query}],
                    tokenize=False, add_generation_prompt=True
                )
                inputs = self.processor(
                    text=[text],
                    images=[image],
                    padding=True,
                    return_tensors="pt"
                ).to('cuda')

                output_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=2048,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.8,
                    top_k=20,
                    num_return_sequences=5
                )

                decoded_outputs = self.processor.batch_decode(
                    output_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
                )

                if any("<answer>" in text and "</answer>" in text for text in decoded_outputs):
                    rewards.append(-1.0)
                else:
                    scores = [self.evaluate_correctness(text) for text in decoded_outputs]
                    avg_score = sum(scores) / len(scores)
                    rewards.append(avg_score)

        return rewards

app = FastAPI()

@app.post("/get_reward")
async def get_reward(request: Request):
    data = await request.json()
    queries = data.get("queries")
    image_paths = data.get("image_paths")
    rewards = reward_model.get_reward(queries, image_paths)
    return JSONResponse({"rewards": rewards})

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--reward_pretrain", type=str, required=True, help="Model path")
    parser.add_argument("--max_len", type=int, default=2048)
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--batch_size", type=int, default=None)
    args = parser.parse_args()

    reward_model = RewardModelProxy(args)
    uvicorn.run(app, host=args.host, port=args.port)
