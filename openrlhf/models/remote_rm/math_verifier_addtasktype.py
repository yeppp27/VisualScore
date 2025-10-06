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
import difflib

def get_response_from_query(q: str):
    ends_of_sentence = ["<|im_end|>", "<｜end▁of▁sentence｜>", "<|endoftext|>"]
    pos = re.search(response_prefix, q)
    if pos is None:
        return None
    response = q[pos.end() :]
    for e in ends_of_sentence:
        response = response.replace(e, "")
    return response.strip()


def verify_format(content):
    """
    Verify if the string meets the format requirements:
    - Must start with <think> and end with </answer>
    - Must contain exactly one pair of <think>...</think> and <answer>...</answer> tags
    - No extra characters allowed between </think> and <answer> tags
    """
    think_count = content.count("<think>")
    answer_count = content.count("<answer>")
    return bool(re.match(format_pattern, content, re.DOTALL)) and think_count == 1 and answer_count == 1



def find_similar_problem(problem):
    max_sim = -1
    target_problem = None
    for p in problem_to_answer.keys():
        sim = Levenshtein.ratio(problem, p)
        if sim > max_sim:
            max_sim = sim
            target_problem = p
    return target_problem


choices = ["a", "b", "c", "d"]

def extract_answer_tag(content):
    match = re.search(r"<answer>(.*?)</answer>", content, re.DOTALL)
    return match.group(1).strip() if match else content.strip()

def normalize(text):
    return re.sub(r"\s+", " ", text.lower().strip())

def verify_choice(response, gold):
    response = normalize(response)
    gold = normalize(gold)
    if gold in choices and gold in response:
        if all(c not in response for c in choices if c != gold):
            return 1.0
    return 0.0

def verify_phrase(response, gold):
    response = normalize(response)
    gold = normalize(gold)
    if response == gold:
        return 1.0
    ratio = difflib.SequenceMatcher(None, response, gold).ratio()
    if ratio >= 0.85:
        return 1.0
    # overlap keyword heuristic
    r_set = set(response.split())
    g_set = set(gold.split())
    if len(r_set & g_set) / max(len(g_set), 1) >= 0.6:
        return 1.0
    return 0.0

def verify_math(response, gold, epsilon):
    try:
        raw_answer = extract_answer_tag(response)
        if not (raw_answer.startswith("$") and raw_answer.endswith("$")):
            raw_answer = f"${raw_answer}$"
        raw_answer = raw_answer.replace("\n", "").replace("\r", "").strip()

        gold_parsed = parse(
            gold,
            extraction_mode="first_match",
            extraction_config=[LatexExtractionConfig()],
        )
        if not gold_parsed:
            print("Failed to parse gold solution:", gold)
            return 1.0

        answer_parsed = parse(
            raw_answer,
            extraction_config=[
                LatexExtractionConfig(
                    normalization_config=NormalizationConfig(
                        nits=False,
                        malformed_operators=False,
                        basic_latex=True,
                        equations=True,
                        boxed=True,
                        units=True,
                    ),
                    boxed_match_priority=0,
                    try_extract_without_anchor=False,
                )
            ],
            extraction_mode="first_match",
        )

        if not answer_parsed:
            number_match = re.search(r"\$?\s*([0-9.]+)\s*\$?", raw_answer)
            if number_match:
                answer_expr = float(number_match.group(1))
            else:
                return 0.0
        else:
            answer_obj = answer_parsed[0]
            answer_expr = answer_obj.expression.evalf() if hasattr(answer_obj, "expression") else float(answer_obj)

        gold_expr = gold_parsed[0].expression.evalf() if hasattr(gold_parsed[0], "expression") else gold_parsed[0]
        reward = float(1.0 if abs(float(answer_expr) - float(gold_expr)) < epsilon else 0.0)
    except Exception as e:
        print("Error verifying math latex:", e)
        reward = 0.0
    return reward

def verify_answer_by_type(response, gold, qtype, epsilon):
    if qtype == "math":
        return verify_math(response, gold, epsilon)
    elif qtype == "choice":
        return verify_choice(extract_answer_tag(response), gold)
    elif qtype == "phrase":
        return verify_phrase(extract_answer_tag(response), gold)
    else:
        # fallback: try math first, then choice, then phrase
        math_reward = verify_math(response, gold, epsilon)
        if math_reward == 1.0:
            return math_reward
        if verify_choice(extract_answer_tag(response), gold):
            return 1.0
        return verify_phrase(extract_answer_tag(response), gold)

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
    '''for q,problem in zip(data["query"],data["prompts"]):
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
        format_reward = float(verify_format(response)) * 0.5
        acc_reward_future = math_verify_executor.submit(verify_math, response, answer, args.epsilon)'''
    for q,problem in zip(data["query"],data["prompts"]):
        if problem is None:
            return jsonify({"error": f"problem not found from {q}"}), 400
        if problem not in problem_to_answer:
            print(f"problem not exists: {problem}")
            problem = find_similar_problem(problem)

        answer_data = problem_to_answer[problem]
        answer = answer_data["answer"]
        qtype = answer_data.get("type", None)

        response = get_response_from_query(q) or q
        if response is None:
            return jsonify({"error": f"response not found from {q}"}), 400

        format_reward = float(verify_format(response)) * 0.5
        acc_reward_future = math_verify_executor.submit(verify_answer_by_type, response, answer, qtype, args.epsilon)

       
        do_print = random.randint(1, 20) == 1
        if do_print:
            info=f"Query: {q}\n\nProblem: {problem}\n\n Answer: {answer}\n\n Response: {response}\n\n Format Reward: {format_reward}\n\n Acc Reward: {acc_reward_future.result()}\n\n"
            info = re.sub(r"<\|.*?\|>","",info)
            logger.info(info)
            #print(info)
            
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
        qtype = item.get("type", None)
        if qtype == "math" and not answer.startswith("$"):
            answer = "$" + answer + "$"
        problem_to_answer[problem] = {"answer": answer, "type": qtype}
    '''for item in dataset:
        problem = item[args.input_key]
        answer = item["answer"].strip()
        # we require the answer to be in latex format
        if answer[0] != "$":
            answer = "$" + answer + "$"
        problem_to_answer[problem] = answer'''

    # math_verify can only run in main thread
    math_verify_executor = futures.ProcessPoolExecutor(max_workers=16)

    app.run(host="0.0.0.0", port=args.port, debug=False, use_reloader=False)
    math_verify_executor.shutdown()