import json
import os
import random
import re
from argparse import ArgumentParser
from multiprocessing import Process, Queue

import Levenshtein
from flask import Flask, jsonify, request
from latex2sympy2_extended import NormalizationConfig
from math_verify import LatexExtractionConfig, parse, verify

from loguru import logger
from concurrent import futures
app = Flask(__name__)

problem_to_answer = {}


def get_response_from_query(q: str):
    ends_of_sentence = ["<|im_end|>", "<｜end▁of▁sentence｜>", "<|endoftext|>"]
    pos = re.search(response_prefix, q)
    if pos is None:
        return None
    response = q[pos.end() :]
    for e in ends_of_sentence:
        response = response.replace(e, "")
    return response.strip()


'''def verify_format(content):
    """
    Verify if the string meets the format requirements:
    - Must start with <think> and end with </answer>
    - Must contain exactly one pair of <think>...</think> and <answer>...</answer> tags
    - No extra characters allowed between </think> and <answer> tags
    """
    think_count = content.count("<think>")
    answer_count = content.count("<answer>")
    return bool(re.match(format_pattern, content, re.DOTALL)) and think_count == 1 and answer_count == 1
'''
def verify_format(completion):
    pattern = (
        r"^(?=(?:.*<think>){1})(?=(?:.*<\/think>){1})"
        r"(?=(?:.*<answer>){1})(?=(?:.*<\/answer>){1})"
        r"(?!.*<think>.*<think>)"
        r"(?!.*<\/think>.*<\/think>)"
        r"(?!.*<answer>.*<answer>)"
        r"(?!.*<\/answer>.*<\/answer>)"
        r".*<think>(.+?)</think>\s*<answer>.+?</answer>.*$"
    )
    matches = re.search(pattern, completion, re.DOTALL)
    return 0.5 if matches else 0.0


def find_similar_problem(problem):
    max_sim = -1
    target_problem = None
    for p in problem_to_answer.keys():
        sim = Levenshtein.ratio(problem, p)
        if sim > max_sim:
            max_sim = sim
            target_problem = p
    return target_problem


import random

def verify_math(content, sol, epsilon):
    try:
        debug = random.random() < 0.3  # 30% 概率打印 debug 信息

        # 提取 <answer>...</answer> 内容
        match = re.search(r"<answer>(.*?)</answer>", content, re.DOTALL)
        if not match:
            
            print("[Error] No <answer>...</answer> found in content.")
            return 0.0

        pred_str = match.group(1).strip()
        gt_str = sol.strip()

        # 转换为 float
        pred_val = float(pred_str)
        gt_val = float(gt_str)

        diff = abs(pred_val - gt_val)
        result = 1.0 if diff < epsilon else 0.0

        if debug:
            print("----- Debug: Verifying Math -----")
            print("Raw Content:", content)
            print("Ground Truth:", sol)
        
        if debug:
            print("Extracted Predicted Answer:", pred_str)
            print("Expected Ground Truth:", gt_str)

        if debug:
            print("Computed Difference:", diff)
            print("Reward Result:", result)
            print("---------------------------------")

        return result

    except Exception as e:
        print("[Exception] Failed to verify math content:", e)
        return 0.0


@app.route("/get_reward", methods=["POST"])
def get_reward():
    # 获取请求中的 JSON 数据
    data = request.get_json()
    # 检查是否有 'query' 字段
    if "query" not in data:
        return jsonify({"error": "queries field is required"}), 400
    rewards = []
    format_rewards = []
    acc_rewards_futures = []
    for q,problem in zip(data["query"],data["prompts"]):
        if problem is None:
            return jsonify({"error": f"problem not found from {q}"}), 400
        if problem not in problem_to_answer:
            # This should not happen
            print(f"problem not exists: {problem}")
            problem = find_similar_problem(problem)
        answer = problem_to_answer[problem]
        response = get_response_from_query(q) or q
        #print('answer',answer, 'response',response)
        if response is None:
            return jsonify({"error": f"response not found from {q}"}), 400
        format_reward = float(verify_format(response)) * 1
        acc_reward_future = math_verify_executor.submit(verify_math, response, answer, args.epsilon)
       
        do_print = random.randint(1, 20) == 1
        if do_print:
            info=f"Query: {q}\n\nProblem: {problem}\n\n Answer: {answer}\n\n Response: {response}\n\n Format Reward: {format_reward}\n\n Acc Reward: {acc_reward_future.result()}\n\n"
            info = re.sub(r"<\|.*?\|>","",info)
            #logger.info(info)
            print(info)
            
        format_rewards.append(format_reward)
        acc_rewards_futures.append(acc_reward_future)
    acc_rewards = [f.result() for f in acc_rewards_futures]
    rewards = [f + a for f, a in zip(format_rewards, acc_rewards)]
    # 返回包含 rewards 的响应
    return jsonify({"rewards": rewards,"format_rewards":format_rewards,"acc_rewards":acc_rewards})


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--dataset", type=str, default=None, help="Datasets to use (comma separated)", required=True
    )
    parser.add_argument(
        "--prompt-template", type=str, default=None, help="Prompt template", required=True
    )
    parser.add_argument(
        "--input_key", type=str, default="prompt", help="The key name of prompt."
    )
    parser.add_argument("--log_file", type=str, default="remote_rm.log", help="Log file path")
    
    parser.add_argument("--port", type=int, default=5003, help="Port to run the reward server on")

    parser.add_argument("--epsilon", type=float, default=0.5, help="Tolerance for float comparison")

    args = parser.parse_args()
    logger.remove()
    logger.add(args.log_file)
    # Split dataset paths and load all datasets
    dataset = []
    for dataset_path in args.dataset.split(','):
        dataset_path = dataset_path.strip()
        if dataset_path.endswith("json"):
            with open(dataset_path, "r") as f:
                dataset.extend(json.load(f))
        elif dataset_path.endswith("jsonl"):
            with open(dataset_path, "r") as f:
                dataset.extend([json.loads(l) for l in f.readlines()])
        else:
            raise ValueError(f"Unsupported file format for dataset: {dataset_path}")

    format_pattern = r"^<think>(?:(?!</think>).)*</think><answer>(?:(?!</answer>).)*</answer>\Z"

    if args.prompt_template=="chatml":
        problem_pattern = r"<\|im_start\|>user\n(.*?)<\|im_end\|>"
        response_prefix = r"<\|im_start\|>assistant\n"
    elif args.prompt_template=="qwen1":
        problem_pattern = r"｜User｜>(.*?)<｜Assistant｜>"
        response_prefix = r"<｜Assistant｜>"
    elif args.prompt_template=="base":
        problem_pattern = r"User: (.*?)\n\nAssistant:"
        response_prefix = r"Assistant: "
    else:
        raise ValueError(f"Unknown chat format: {args.dataset}")
    print("load dataset success")
    for item in dataset:
        problem = item[args.input_key]
        answer = item["answer"].strip()
        
        
        problem_to_answer[problem] = answer

    # math_verify can only run in main thread
    math_verify_executor = futures.ProcessPoolExecutor(max_workers=16)

    app.run(host="0.0.0.0", port=args.port, debug=False, use_reloader=False)
    math_verify_executor.shutdown()