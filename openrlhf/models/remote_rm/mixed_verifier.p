import argparse
import os
import torch
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2_5_VLProcessor
from PIL import Image, ImageOps
import re
import json
import Levenshtein
import random

# Define the system prompt
SYSTEM_PROMPT = (
    "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. "
    "The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. "
    "The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags."
)

def load_image(image_path: str) -> "PIL.Image.Image":
    """Load image and convert to RGB."""
    image = Image.open(image_path)
    image = ImageOps.exif_transpose(image).convert("RGB")
    return image

def strip_sequence(text, pad_token, eos_token):
    pad_token_escaped = re.escape(pad_token)
    eos_token_escaped = re.escape(eos_token)

    pattern = f"^({eos_token_escaped}|{pad_token_escaped})+"
    text = re.sub(pattern, "", text)

    pattern = f"({eos_token_escaped}|{pad_token_escaped})+$"
    text = re.sub(pattern, "", text)
    return text

class RewardModelProxy:
    def __init__(self, args):
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            "/mnt/petrelfs/luyiting/ckt/Qwen2.5-VL-7B-Instruct/Qwen2.5-VL-7B-Instruct/",
            device_map="auto",
            torch_dtype=torch.bfloat16
        )
        
        self.processor = Qwen2_5_VLProcessor.from_pretrained(
            "/mnt/petrelfs/luyiting/ckt/Qwen2.5-VL-7B-Instruct/Qwen2.5-VL-7B-Instruct/",
            min_pixels=256 * 28 * 28,
            max_pixels=4900 * 28 * 28
        )
        self.max_length = args.max_len
        self.batch_size = args.batch_size

    def get_reward(self, queries, image_paths):
        if self.batch_size is None:
            batch_size = len(queries)
        else:
            batch_size = self.batch_size

        # Remove pad_token from queries
        for i in range(len(queries)):
            queries[i] = (
                strip_sequence(queries[i], self.processor.pad_token, self.processor.eos_token)
                + self.processor.eos_token
            )

        rewards = []
        # Process each query with the associated image
        with torch.no_grad():
            for i in range(0, len(queries), batch_size):
                images = [load_image(image_paths[i]) for i in range(i, min(len(image_paths), i + batch_size))]
                text = self.processor.apply_chat_template(
                    [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": queries[i]}],
                    tokenize=False, add_generation_prompt=True
                )
                inputs = self.processor(
                    text=[text],
                    images=images,
                    padding=True,
                    return_tensors="pt"
                ).to('cuda')

                output_ids = self.model.generate(
                    **inputs, 
                    max_new_tokens=2048, 
                    do_sample=True, 
                    temperature=0.7, 
                    top_p=0.8,
                    top_k=20
                )

                generated_text = self.processor.batch_decode(output_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
                rewards.append(generated_text[0])  # Assuming single output per batch
        return rewards

def accuracy_reward(completion, solution, epsilon=0.5, **kwargs):
    reward = 0.0

    if reward == 0.0:
        try:
            sol_match = re.search(r'<answer>(.*?)</answer>', solution, re.DOTALL)
            ground_truth = sol_match.group(1).strip() if sol_match else solution.strip()

            pred_match = re.search(r'<answer>(.*?)</answer>', completion, re.DOTALL)
            student_answer = pred_match.group(1).strip() if pred_match else completion.strip()

            # 判断ground_truth是否为纯数字
            if re.match(r'^[-+]?[0-9]*\.?[0-9]+$', ground_truth):
                numbers = re.findall(r'[-+]?[0-9]*\.?[0-9]+', student_answer)
                if numbers:
                    pred_value = float(numbers[-1])  # 取最后一个出现的数字
                    gt_value = float(ground_truth)
                    reward = 1.0 if abs(pred_value - gt_value) < epsilon else 0.0
                else:
                    reward = 0.0
            else:
                # 非数字内容直接比较字符串
                if student_answer == ground_truth:
                    reward = 1.0
                else:
                    # 比较最后一个字母/字符
                    student_chars = extract_letters(student_answer)
                    if student_chars and student_chars[-1] == ground_truth:
                        reward = 1.0
        except Exception as e:
            print(f"Error in reward calculation: {e}")
            reward = 0.0
    return reward

app = FastAPI()

@app.post("/get_reward")
async def get_reward(request: Request):
    data = await request.json()
    queries = data.get("queries")
    image_paths = data.get("image_paths")
    groundtruth = data.get("groundtruth")

    # Generate response using the model
    responses = reward_model.get_reward(queries, image_paths)
    
    # Compute accuracy reward
    acc_rewards = [accuracy_reward(response, gt) for response, gt in zip(responses, groundtruth)]
    
    result = {
        "rewards": acc_rewards
    }
    
    return JSONResponse(result)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--reward_pretrain", type=str, default=None, help="HF model name or path")
    parser.add_argument("--max_len", type=int, default=2048, help="Maximum sequence length")
    parser.add_argument("--port", type=int, default=5000, help="Port number for the server")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="IP for the server")
    parser.add_argument("--batch_size", type=int, default=None)
    args = parser.parse_args()

    # Initialize the reward model
    reward_model = RewardModelProxy(args)
    
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
