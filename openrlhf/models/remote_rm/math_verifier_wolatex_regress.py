import json
import os
import random
import re
import math
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

# 集成 RewardCalculator 类
class RewardCalculator:
    def __init__(self, initial_sigma=1.0, initial_epsilon_bonus=0.3, max_possible_error=None):
        """
        Initializes the RewardCalculator.

        Args:
            initial_sigma (float): Initial value for sigma in Gaussian RBF reward.
                                   This can be annealed during curriculum learning.
            initial_epsilon_bonus (float): Initial epsilon for binary rewards or bonus thresholds.
                                           This can be annealed during curriculum learning.
            max_possible_error (float, optional): The maximum possible absolute error.
                                                  Used for 'bounded_reward'.
                                                  Can be a dictionary mapping dataset_name to value.
        """
        self.initial_sigma = initial_sigma
        self.initial_epsilon_bonus = initial_epsilon_bonus
        
        self.current_sigma = initial_sigma
        self.current_epsilon_bonus = initial_epsilon_bonus
        
        self.max_possible_error = max_possible_error 
        self.training_step = 0 # Represents current training iteration, step, or epoch

    def update_curriculum_parameters(self, current_step_or_epoch, total_steps_or_epochs=None):
        """
        Updates curriculum-dependent parameters like sigma or epsilon_bonus.
        This method should be called periodically during training (e.g., every epoch).

        Args:
            current_step_or_epoch (int): The current training step or epoch.
            total_steps_or_epochs (int, optional): Total steps/epochs for linear annealing.
        """
        self.training_step = current_step_or_epoch

        # --- Implement your annealing logic here ---
        # Example: Linearly anneal sigma for Gaussian RBF
        # min_sigma = 0.1 # Target minimum sigma
        # if total_steps_or_epochs and total_steps_or_epochs > 0:
        #     progress = min(1.0, float(current_step_or_epoch) / total_steps_or_epochs)
        #     self.current_sigma = self.initial_sigma - (self.initial_sigma - min_sigma) * progress
        #     self.current_sigma = max(self.current_sigma, min_sigma) # Ensure it doesn't go below min

        # Example: Linearly anneal epsilon_bonus for binary/hybrid rewards
        # min_epsilon = 0.05 # Target minimum epsilon
        # if total_steps_or_epochs and total_steps_or_epochs > 0:
        #     progress = min(1.0, float(current_step_or_epoch) / total_steps_or_epochs)
        #     self.current_epsilon_bonus = self.initial_epsilon_bonus - (self.initial_epsilon_bonus - min_epsilon) * progress
        #     self.current_epsilon_bonus = max(self.current_epsilon_bonus, min_epsilon)

        # For now, this is a placeholder. You'll need to define how these parameters change.
        # If no annealing is needed for a particular run, these values will just stay initial.
        pass

    def get_reward(self, content, sol, reward_type="negative_mae", dataset_name=None, debug_print=False):
        """
        Calculates the reward based on the predicted content and the ground truth solution.

        Args:
            content (str): The model's output string, expected to contain <answer>...</answer>.
            sol (str): The ground truth solution string.
            reward_type (str): Type of reward to calculate. Options:
                               - "negative_mae": R = -|Spred - Sgt|
                               - "negative_mse": R = -(Spred - Sgt)^2
                               - "gaussian_rbf": R = exp(- (Spred - Sgt)^2 / (2 * sigma^2) )
                               - "bounded_reward": R = 1 - (|Spred - Sgt| / max_possible_error)
                               - "original_binary": Your previous binary reward (uses current_epsilon_bonus)
                               - "hybrid_mae_bonus": Negative MAE + bonus if error < current_epsilon_bonus
            dataset_name (str, optional): Name or identifier for the dataset,
                                          can be used for dataset-specific reward adjustments.
            debug_print (bool): If True, prints debug information.

        Returns:
            float: The calculated reward.
        """
        try:
            match = re.search(r"<answer>(.*?)</answer>", content, re.DOTALL)
            if not match:
                if debug_print: print(f"[Error] No <answer>...</answer> found in content: {content[:100]}...")
                return 0.0 

            pred_str = match.group(1).strip()
            gt_str = sol.strip()

            pred_val = float(pred_str)
            gt_val = float(gt_str)

            error = pred_val - gt_val
            abs_error = abs(error)

            # --- Determine parameters to use (potentially dataset-specific or curriculum-driven) ---
            sigma_to_use = self.current_sigma
            epsilon_bonus_to_use = self.current_epsilon_bonus
            
            # Example of dataset-specific max_possible_error for bounded_reward
            max_error_for_bounded = self.max_possible_error
            if isinstance(self.max_possible_error, dict):
                max_error_for_bounded = self.max_possible_error.get(dataset_name, None) # Fallback to None or a default

            # You could add more dataset-specific logic here if needed:
            # if dataset_name == "hard_dataset":
            #     sigma_to_use = self.current_sigma * 1.2 # Be more lenient

            reward = 0.0
            if reward_type == "negative_mae":
                # R = -|Spred - Sgt|
                reward = -abs_error
            elif reward_type == "negative_mse":
                # R = -(Spred - Sgt)^2
                reward = -(error**2)
            elif reward_type == "gaussian_rbf":
                # R = exp(- (Spred - Sgt)^2 / (2 * sigma^2) )
                if sigma_to_use <= 0:
                    if debug_print: print(f"[Error] Sigma for Gaussian RBF reward must be positive. Got {sigma_to_use}")
                    return 0.0 
                reward = math.exp(-(error**2) / (2 * sigma_to_use**2))
            elif reward_type == "bounded_reward":
                # R = 1 - (|Spred - Sgt| / max_possible_error)
                if max_error_for_bounded is None or max_error_for_bounded <= 0:
                    if debug_print: print(f"[Error] max_possible_error for bounded reward is invalid or not provided. Got {max_error_for_bounded}")
                    return 0.0 
                reward = 1.0 - (abs_error / max_error_for_bounded)
                # Optional: clip reward if it can go extensively negative, e.g., reward = max(reward, -1.0)
            elif reward_type == "original_binary":
                # Your previous logic: R = 1.0 if diff < epsilon else 0.0
                # Uses the potentially annealed self.current_epsilon_bonus as epsilon
                reward = 1.0 if abs_error < epsilon_bonus_to_use else 0.0
            elif reward_type == "hybrid_mae_bonus":
                bonus_amount = 0.5 # Example fixed bonus, could also be annealed or dataset-specific
                current_bonus = 0.0
                if abs_error < epsilon_bonus_to_use:
                    current_bonus = bonus_amount
                reward = -abs_error + current_bonus
            else:
                if debug_print: print(f"[Error] Unknown reward type: {reward_type}")
                return 0.0

            if debug_print:
                print(f"----- Debug: Calculating Reward (Dataset: {dataset_name}, Step: {self.training_step}) -----")
                print(f"Content: '{content[:50]}...', Sol: '{sol}'")
                print(f"Pred_val: {pred_val:.4f}, GT_val: {gt_val:.4f}, Abs Error: {abs_error:.4f}")
                print(f"Reward Type: {reward_type}, Params: (sigma={sigma_to_use if 'gaussian' in reward_type else 'N/A'}, eps_bonus={epsilon_bonus_to_use if 'binary' in reward_type or 'hybrid' in reward_type else 'N/A'}, max_err_bound={max_error_for_bounded if 'bounded' in reward_type else 'N/A'})")
                print(f"Calculated Reward: {reward:.4f}")
                print("---------------------------------")

            return reward

        except ValueError: # Handle cases where conversion to float fails
            if debug_print: print(f"[Exception] ValueError: Could not convert pred_str '{pred_str if 'pred_str' in locals() else 'UNKNOWN'}' or gt_str '{gt_str if 'gt_str' in locals() else 'UNKNOWN'}' to float.")
            return 0.0 
        except Exception as e:
            if debug_print: print(f"[Exception] Failed to calculate reward: {e}")
            return 0.0


def get_response_from_query(q: str):
    ends_of_sentence = ["<|im_end|>", "<｜end▁of▁sentence｜>", "<|endoftext|>"]
    pos = re.search(response_prefix, q)
    if pos is None:
        return None
    response = q[pos.end() :]
    for e in ends_of_sentence:
        response = response.replace(e, "")
    return response.strip()


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


# 使用 RewardCalculator 进行奖励计算
def calculate_math_reward(content, sol, reward_calculator, reward_type="original_binary", dataset_name=None):
    """
    使用 RewardCalculator 计算数学奖励
    
    Args:
        content (str): 模型输出
        sol (str): 标准答案
        reward_calculator (RewardCalculator): 奖励计算器实例
        reward_type (str): 奖励类型
        dataset_name (str): 数据集名称
    
    Returns:
        float: 计算得到的奖励值
    """
    debug = random.random() < 0.3  # 30% 概率打印 debug 信息
    return reward_calculator.get_reward(
        content=content,
        sol=sol,
        reward_type=reward_type,
        dataset_name=dataset_name,
        debug_print=debug
    )


@app.route("/get_reward", methods=["POST"])
def get_reward():
    # 获取请求中的 JSON 数据
    data = request.get_json()
    # 检查是否有 'query' 字段
    if "query" not in data:
        return jsonify({"error": "queries field is required"}), 400
    
    # 获取奖励类型和数据集名称（可选参数）
    reward_type = data.get("reward_type", "original_binary")
    dataset_name = data.get("dataset_name", None)
    
    rewards = []
    format_rewards = []
    acc_rewards_futures = []
    
    for q, problem in zip(data["query"], data["prompts"]):
        if problem is None:
            return jsonify({"error": f"problem not found from {q}"}), 400
        if problem not in problem_to_answer:
            # This should not happen
            print(f"problem not exists: {problem}")
            problem = find_similar_problem(problem)
        answer = problem_to_answer[problem]
        response = get_response_from_query(q) or q
        
        if response is None:
            return jsonify({"error": f"response not found from {q}"}), 400
        
        format_reward = float(verify_format(response)) * 1
        
        # 使用 RewardCalculator 进行奖励计算
        acc_reward_future = math_verify_executor.submit(
            calculate_math_reward, 
            response, 
            answer, 
            reward_calculator,
            reward_type,
            dataset_name
        )
       
        do_print = random.randint(1, 20) == 1
        if do_print:
            info = f"Query: {q}\n\nProblem: {problem}\n\n Answer: {answer}\n\n Response: {response}\n\n Format Reward: {format_reward}\n\n Acc Reward: {acc_reward_future.result()}\n\n"
            info = re.sub(r"<\|.*?\|>", "", info)
            print(info)
            
        format_rewards.append(format_reward)
        acc_rewards_futures.append(acc_reward_future)
    
    acc_rewards = [f.result() for f in acc_rewards_futures]
    rewards = [f + a for f, a in zip(format_rewards, acc_rewards)]
    
    # 返回包含 rewards 的响应
    return jsonify({
        "rewards": rewards,
        "format_rewards": format_rewards,
        "acc_rewards": acc_rewards
    })


@app.route("/update_curriculum", methods=["POST"])
def update_curriculum():
    """
    更新课程学习参数的接口
    """
    data = request.get_json()
    current_step = data.get("current_step", 0)
    total_steps = data.get("total_steps", None)
    
    reward_calculator.update_curriculum_parameters(current_step, total_steps)
    
    return jsonify({
        "success": True,
        "current_step": current_step,
        "current_sigma": reward_calculator.current_sigma,
        "current_epsilon_bonus": reward_calculator.current_epsilon_bonus
    })


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
    
    # RewardCalculator 相关参数
    parser.add_argument("--reward_type", type=str, default="original_binary", 
                        help="Type of reward to use (negative_mae, negative_mse, gaussian_rbf, bounded_reward, original_binary, hybrid_mae_bonus)")
    parser.add_argument("--initial_sigma", type=float, default=1.0, 
                        help="Initial sigma value for Gaussian RBF reward")
    parser.add_argument("--initial_epsilon_bonus", type=float, default=0.3, 
                        help="Initial epsilon bonus for binary/hybrid rewards")
    parser.add_argument("--max_possible_error", type=float, default=None, 
                        help="Maximum possible error for bounded reward")

    args = parser.parse_args()
    logger.remove()
    logger.add(args.log_file)
    
    # 初始化 RewardCalculator
    # 使用 args.epsilon 作为 initial_epsilon_bonus 如果没有单独指定
    epsilon_bonus = args.initial_epsilon_bonus if hasattr(args, 'initial_epsilon_bonus') else args.epsilon
    
    reward_calculator = RewardCalculator(
        initial_sigma=args.initial_sigma,
        initial_epsilon_bonus=epsilon_bonus,
        max_possible_error=args.max_possible_error
    )
    
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

    if args.prompt_template == "chatml":
        problem_pattern = r"<\|im_start\|>user\n(.*?)<\|im_end\|>"
        response_prefix = r"<\|im_start\|>assistant\n"
    elif args.prompt_template == "qwen1":
        problem_pattern = r"｜User｜>(.*?)<｜Assistant｜>"
        response_prefix = r"<｜Assistant｜>"
    elif args.prompt_template == "base":
        problem_pattern = r"User: (.*?)\n\nAssistant:"
        response_prefix = r"Assistant: "
    else:
        raise ValueError(f"Unknown chat format: {args.prompt_template}")
    
    print("load dataset success")
    for item in dataset:
        problem = item[args.input_key]
        answer = item["answer"].strip()
        problem_to_answer[problem] = answer

    # math_verify can only run in main thread
    math_verify_executor = futures.ProcessPoolExecutor(max_workers=16)

    print(f"Starting reward server with reward type: {args.reward_type}")
    print(f"RewardCalculator initialized with sigma={reward_calculator.current_sigma}, epsilon_bonus={reward_calculator.current_epsilon_bonus}")
    
    app.run(host="0.0.0.0", port=args.port, debug=False, use_reloader=False)
    math_verify_executor.shutdown()