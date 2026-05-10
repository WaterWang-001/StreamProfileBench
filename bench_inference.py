"""StreamProfileBench inference & evaluation runner.

Calls an OpenAI-compatible LLM API for each user step, threading the persona
forward, then scores predictions with the metrics from the paper:

- Recall (overall)            R̄
- Recall_Novelty              R_N
- Recall_Stability            R_S
- F1^NS = HM(R_N, R_S)
- Error_Decay  (E_D)
- Error_Peer   (E_P)
- Error_Viral  (E_V)
- Error_Random (E_R)

Required env vars:
- ``LLM_API_KEY``  : API key for your provider.
- ``LLM_API_BASE`` : Base URL of an OpenAI-compatible endpoint
                     (e.g. ``https://api.openai.com/v1``).

Optional:
- ``LLM_INSECURE_TLS=1`` : skip TLS verification (off by default).
- ``BENCH_MAX_WORKERS``  : parallel inference threads (default 16).

CLI:
    python bench_inference.py --model gpt-4o-mini --platform weibo
    python bench_inference.py --model gpt-4o-mini                 # all 5
"""

import argparse
import concurrent.futures
import json
import logging
import os
import re
import time
from collections import defaultdict

import httpx
import numpy as np
from openai import OpenAI
from tqdm import tqdm

from bench_task import BenchTaskBuilder

# ================= Config =================

logging.basicConfig(level=logging.CRITICAL)
logging.getLogger("httpx").setLevel(logging.CRITICAL)
logging.getLogger("httpcore").setLevel(logging.CRITICAL)

API_KEY = os.environ.get("LLM_API_KEY")
API_BASE = os.environ.get("LLM_API_BASE")
INSECURE_TLS = os.environ.get("LLM_INSECURE_TLS", "0") == "1"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
PLATFORMS = ["weibo", "xiaohongshu", "toutiao", "zhihu", "douban"]

MAX_WORKERS = int(os.environ.get("BENCH_MAX_WORKERS", "16"))

# ===========================================


def get_paths(platform, model_name):
    input_file = os.path.join(DATA_DIR, f"{platform}_inference_tasks.jsonl")
    output_dir = os.path.join(RESULTS_DIR, os.path.basename(model_name))
    pred_file = os.path.join(output_dir, f"inference_results_{platform}.jsonl")
    report_file = os.path.join(output_dir, f"eval_report_{platform}.txt")
    return input_file, output_dir, pred_file, report_file


class LLMClient:
    def __init__(self, model_name, api_key=None, api_url=None):
        key = api_key or API_KEY
        url = api_url or API_BASE
        if not key:
            raise RuntimeError(
                "LLM_API_KEY is not set. Export it in your shell or pass --api_key."
            )
        if not url:
            raise RuntimeError(
                "LLM_API_BASE is not set. Export it in your shell or pass --api_url."
            )
        self.client = OpenAI(
            api_key=key,
            base_url=url,
            http_client=httpx.Client(verify=not INSECURE_TLS),
        )
        self.model_name = model_name

    def generate(self, prompt, candidate_pool):
        return self._real_inference(prompt)

    def _real_inference(self, prompt):
        try:
            for attempt in range(3):
                try:
                    response = self.client.chat.completions.create(
                        model=self.model_name,
                        messages=[
                            {"role": "system", "content": "You are a user profiling system that maintains evolving user personas from streaming social media data. Output valid JSON only."},
                            {"role": "user", "content": prompt}
                        ],
                        temperature=0.0,
                        response_format={"type": "json_object"}
                    )
                    content = response.choices[0].message.content
                    content = re.sub(r"```json|```", "", content).strip()
                    return json.loads(content)
                except Exception as e:
                    if attempt == 2:
                        raise e
                    time.sleep(1)
        except Exception as e:
            return {"predicted_tags": [], "reasoning": f"Error: {str(e)}"}


# Metrics reported in the paper, in display order.
METRIC_KEYS = ['Recall', 'Recall_Novelty', 'Recall_Stability',
               'Error_Decay', 'Error_Peer', 'Error_Viral', 'Error_Random']


class BenchEvaluator:
    def __init__(self):
        self.user_steps = []  # each element: list of (step_id, metrics)

    @staticmethod
    def calculate_metrics(pred_tags, gt_tags, meta_data):
        pred_set = set(pred_tags)

        target_new = set(meta_data.get('T_new', []))
        target_keep = set(meta_data.get('T_keep', []))
        gt_all = target_new | target_keep

        if not gt_all:
            return None

        d_cluster = set(meta_data.get('D_cluster', []))
        d_viral = set(meta_data.get('D_viral', []))
        d_decay = set(meta_data.get('D_decay', []))
        d_random = set(meta_data.get('D_random', []))

        # 1. Overall recall
        recall = len(pred_set & gt_all) / len(gt_all)

        # 2. Decomposed recall: novelty (T_new) and stability (T_keep)
        recall_novelty = len(pred_set & target_new) / len(target_new) if target_new else np.nan
        recall_stability = len(pred_set & target_keep) / len(target_keep) if target_keep else np.nan

        # 3. Distractor error rates (lower is better)
        error_peer = len(pred_set & d_cluster) / len(d_cluster) if d_cluster else np.nan
        error_viral = len(pred_set & d_viral) / len(d_viral) if d_viral else np.nan
        error_decay = len(pred_set & d_decay) / len(d_decay) if d_decay else np.nan
        error_random = len(pred_set & d_random) / len(d_random) if d_random else np.nan

        return {
            'Recall': recall,
            'Recall_Novelty': recall_novelty,
            'Recall_Stability': recall_stability,
            'Error_Decay': error_decay,
            'Error_Peer': error_peer,
            'Error_Viral': error_viral,
            'Error_Random': error_random,
        }

    def add_user(self, step_metrics_with_id):
        if step_metrics_with_id:
            self.user_steps.append(step_metrics_with_id)

    # ---- aggregation ----

    @staticmethod
    def _nanmean(values):
        valid = [v for v in values if not np.isnan(v)]
        return np.mean(valid) if valid else np.nan

    def _aggregate_macro(self):
        """Two-level macro average: mean within user across steps, then mean across users."""
        user_means = defaultdict(list)
        for user_steps in self.user_steps:
            per_user = defaultdict(list)
            for _, m in user_steps:
                for k in METRIC_KEYS:
                    per_user[k].append(m[k])
            for k in METRIC_KEYS:
                if per_user[k]:
                    user_means[k].append(self._nanmean(per_user[k]))
        return {k: self._nanmean(user_means[k]) for k in METRIC_KEYS}

    def _compute_f1ns(self):
        """F1^NS: per user, average R_N and R_S over its steps then take HM; macro-avg across users."""
        f1ns_values = []
        for user_steps in self.user_steps:
            nov = self._nanmean([m['Recall_Novelty'] for _, m in user_steps])
            stab = self._nanmean([m['Recall_Stability'] for _, m in user_steps])
            if not np.isnan(nov) and not np.isnan(stab) and (nov + stab) > 0:
                f1ns_values.append(2 * nov * stab / (nov + stab))
        return np.mean(f1ns_values) if f1ns_values else np.nan

    # ---- reporting ----

    def print_report(self):
        report_lines = []

        def emit(line):
            print(line)
            report_lines.append(line)

        def section(title):
            emit(f"\n{'=' * 50}\n{title}\n{'=' * 50}")

        def fmt(v):
            return f"{v:.2%}" if not np.isnan(v) else "N/A"

        m_bar = self._aggregate_macro()
        section("Overall Macro Average (M̄)")
        for k in METRIC_KEYS:
            emit(f"  {k:<25}: {fmt(m_bar[k])}")

        n_users = len(self.user_steps)
        n_steps = sum(len(us) for us in self.user_steps)
        emit(f"\nUsers: {n_users}  |  Total Steps: {n_steps}")

        f1ns = self._compute_f1ns()
        section("F1^NS — Novelty-Stability Harmonic Mean")
        emit(f"  F1^NS                    : {fmt(f1ns)}")

        return "\n".join(report_lines)


def rebuild_prompt_with_persona(user_task, step, prev_persona):
    """Re-assemble the prompt with the persona from the previous step."""
    builder = BenchTaskBuilder(user_task.get('platform', ''))
    return builder.format_prompt(
        username=user_task['username'],
        bio=user_task.get('bio', ''),
        batch_n_posts=step['posts_text'],
        candidate_pool=step['candidate_pool'],
        step_id=step['step_id'],
        total_steps=step.get('total_steps', len(user_task['prediction_tasks'])),
        prev_persona=prev_persona,
    )


def process_single_user(user_task, llm_client):
    user_result = {
        "user_id": user_task['user_id'],
        "platform": user_task.get('platform', ''),
        "username": user_task['username'],
        "steps": []
    }
    step_metrics_collected = []
    prev_persona = None  # chain persona across steps

    for step in user_task['prediction_tasks']:
        prompt = rebuild_prompt_with_persona(user_task, step, prev_persona)

        pool = step['candidate_pool']
        gt_data = step['ground_truth']
        meta = step['meta']

        llm_output = llm_client.generate(prompt, pool)
        pred_tags = llm_output.get("predicted_tags", [])
        reasoning = llm_output.get("reasoning", "")
        persona_summary = llm_output.get("persona_summary", "")

        prev_persona = persona_summary if persona_summary else prev_persona

        gt_tags = gt_data['new_tags'] + gt_data['keep_tags']
        metrics = BenchEvaluator.calculate_metrics(pred_tags, gt_tags, meta)

        if metrics:
            step_metrics_collected.append((step['step_id'], metrics))

        user_result['steps'].append({
            "step_id": step['step_id'],
            "prediction": pred_tags,
            "ground_truth": gt_tags,
            "metrics": metrics,
            "reasoning": reasoning,
            "persona_summary": persona_summary,
        })

    return user_result, step_metrics_collected


def run_platform(platform, model_name, api_url=None, api_key=None):
    input_file, output_dir, pred_file, report_file = get_paths(platform, model_name)

    if not os.path.exists(input_file):
        print(f"Input file not found: {input_file}")
        return

    tasks_buffer = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                tasks_buffer.append(json.loads(line))

    total_tasks = len(tasks_buffer)
    print(f"[{platform}] Starting inference: {total_tasks} users | Workers: {MAX_WORKERS}")

    llm = LLMClient(model_name=model_name, api_key=api_key, api_url=api_url)
    evaluator = BenchEvaluator()
    results = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_user = {
            executor.submit(process_single_user, task, llm): task
            for task in tasks_buffer
        }

        for future in tqdm(concurrent.futures.as_completed(future_to_user),
                           total=total_tasks, desc=f"[{platform}] Processing"):
            try:
                user_res, step_metrics = future.result()
                results.append(user_res)
                evaluator.add_user(step_metrics)
            except Exception as e:
                print(f"Task generated an exception: {e}")

    os.makedirs(output_dir, exist_ok=True)
    with open(pred_file, 'w', encoding='utf-8') as f:
        for res in results:
            f.write(json.dumps(res, ensure_ascii=False) + "\n")

    print(f"\n[{platform}] Inference done! Results saved to: {pred_file}")

    report_text = evaluator.print_report()
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report_text)


def main():
    parser = argparse.ArgumentParser(description="Run LLM inference for StreamProfileBench")
    parser.add_argument("--platform", type=str, choices=PLATFORMS, default=None,
                        help="Process a single platform. If not set, process all.")
    parser.add_argument("--model", type=str, required=True,
                        help="Model name passed to the LLM API")
    parser.add_argument("--api_url", type=str, default=None,
                        help="OpenAI-compatible base URL (overrides $LLM_API_BASE)")
    parser.add_argument("--api_key", type=str, default=None,
                        help="API key (overrides $LLM_API_KEY)")
    args = parser.parse_args()

    platforms = [args.platform] if args.platform else PLATFORMS
    for platform in platforms:
        run_platform(platform, args.model, api_url=args.api_url, api_key=args.api_key)


if __name__ == "__main__":
    main()
