# StreamProfileBench

Streaming user-interest profiling benchmark for Chinese social media platforms.
Each task asks an LLM to (a) maintain a rolling persona summary from a stream
of a user's posts, and (b) predict which tags from a curated candidate pool the
user will engage with in the next time window.


## Repository layout

```
StreamProfileBench/
├── bench_inference.py    # main runner: inference + scoring
├── bench_task.py         # prompt formatter (PLATFORM_CONTEXT + format_prompt)
├── stats.py              # aggregate eval reports across models into Excel
├── data/
│   ├── weibo_inference_tasks.jsonl
│   ├── xiaohongshu_inference_tasks.jsonl
│   ├── toutiao_inference_tasks.jsonl
│   ├── zhihu_inference_tasks.jsonl
│   └── douban_inference_tasks.jsonl
├── requirements.txt
├── LICENSE                # Apache-2.0
└── README.md
```

## Install

```bash
pip install -r requirements.txt
```

Python 3.9+.

## Quick start

```bash
export LLM_API_KEY="sk-..."
export LLM_API_BASE="https://api.openai.com/v1"   # any OpenAI-compatible endpoint

# Single platform
python bench_inference.py --model gpt-4o-mini --platform weibo

# All five platforms
python bench_inference.py --model gpt-4o-mini
```

Outputs go to `results/<model>/`:
- `inference_results_<platform>.jsonl` — per-user, per-step predictions, ground truth, persona summaries, and metrics.
- `eval_report_<platform>.txt` — text report (M̄ block + F1^NS).

### Aggregate results across models into Excel tables

After running multiple models:

```bash
python stats.py
# writes metric_excels/{Recall,Recall_Novelty,Recall_Stability,Error_*,F1^NS}.xlsx
```

Each Excel file has one row per model and one column per platform (plus an
`Avg.` column).

## Configuration

| Variable               | Purpose                                                       | Default |
|------------------------|---------------------------------------------------------------|---------|
| `LLM_API_KEY`          | API key for your LLM provider                                 | (required) |
| `LLM_API_BASE`         | OpenAI-compatible base URL                                    | (required) |
| `LLM_INSECURE_TLS`     | Set `1` to skip TLS verification (e.g. for self-signed dev)   | `0` |
| `BENCH_MAX_WORKERS`    | Parallel inference threads                                    | `16` |
| `--model`              | Model name passed to the API                                  | (required) |
| `--api_url`/`--api_key`| Override the env vars                                         | unset |
| `--platform`           | One of `weibo / xiaohongshu / toutiao / zhihu / douban`       | all |

## Data format

One JSON object per line in each `data/<platform>_inference_tasks.jsonl`:

```json
{
  "user_id": "we_16d82b9a92",
  "platform": "weibo",
  "username": "U_010ada10",
  "bio": "...",
  "total_steps": 4,
  "prediction_tasks": [
    {
      "step_id": 1,
      "total_steps": 4,
      "date_input":  "2025-06-12",
      "date_target": "2025-06-17",
      "posts_text":  "[1] Content: ...\nTags: ...\n[2] ...",
      "candidate_pool": ["tag1", "tag2", ...],
      "ground_truth": {
        "all_tags":  ["..."],
        "new_tags":  ["..."],
        "keep_tags": ["..."]
      },
      "meta": {
        "T_keep":   ["..."],
        "T_new":    ["..."],
        "D_decay":  ["..."],
        "D_cluster":["..."],
        "D_viral":  ["..."],
        "D_random": ["..."]
      }
    }
  ]
}
```

Per platform / step the candidate pool fixes the positive ratio at
`|C⁺| / |C| = 25%`, with the negative side composed of four distractor types:

- **D_decay**   — tags the user engaged with in the *current* batch but does **not** carry forward (interests on the way out).
- **D_cluster** — tags from the same semantic cluster as the GT (peer-cluster distractors).
- **D_viral**   — tags trending platform-wide on the target date (popular but irrelevant).
- **D_random**  — tags sampled uniformly from the platform's tag vocabulary.

Ground truth is `T_new ∪ T_keep`, where `T_keep = current ∩ future` and
`T_new = future \ current`.

## Metrics

The evaluator reports the metrics from the StreamProfileBench paper:

| Metric              | Definition                                                                 |
|---------------------|----------------------------------------------------------------------------|
| **Recall** (R̄)      | `|pred ∩ GT| / |GT|`, averaged within user across steps then across users  |
| **Recall_Novelty**  | Recall restricted to `T_new` (plasticity)                                  |
| **Recall_Stability**| Recall restricted to `T_keep` (stability)                                  |
| **F1^NS**           | Harmonic mean of `Recall_Novelty` and `Recall_Stability` (plasticity-stability balance) |
| **Error_Decay** (E_D)  | False-positive rate on `D_decay`  (lower is better)                     |
| **Error_Peer** (E_P)   | False-positive rate on `D_cluster`                                      |
| **Error_Viral** (E_V)  | False-positive rate on `D_viral`                                        |
| **Error_Random** (E_R) | False-positive rate on `D_random`                                       |

Aggregation uses a two-level macro average: within-user mean across steps,
then macro mean across users. F1^NS is computed per user (HM of the user-level
R_N and R_S) and then macro-averaged.

## Leaderboard

Macro-averaged Recall (R̄) and F1^NS across the five Chinese social media
platforms. **Bold** = best, _italic_ = second-best. XHS denotes Xiaohongshu.

### Average Recall (R̄, %)

| Model              | Avg.      | Zhihu     | Weibo     | Toutiao   | XHS       | Douban    |
|--------------------|-----------|-----------|-----------|-----------|-----------|-----------|
| _Closed-source_    |           |           |           |           |           |           |
| GPT-4o-mini        | 33.32     | 29.97     | 34.63     | 37.42     | 34.46     | 30.13     |
| GPT-5-mini         | 35.08     | 29.09     | 35.77     | 38.47     | 38.62     | 33.45     |
| GPT-5.1            | 38.87     | 44.52     | 38.65     | 41.50     | 41.69     | 28.01     |
| Gemini-3-Flash     | **52.26** | **62.95** | **46.76** | **50.09** | **48.37** | **53.15** |
| _Open-source_      |           |           |           |           |           |           |
| MiniMax-M2.5       | 35.61     | 36.90     | 35.53     | 38.06     | 38.37     | 29.17     |
| GLM-4.7            | 41.69     | 49.56     | _40.03_   | 39.49     | _40.73_   | 38.65     |
| DeepSeek-v3.2      | _43.05_   | _51.50_   | 39.63     | _42.65_   | 40.30     | _41.19_   |
| Llama-3.1-8B       | 25.18     | 17.36     | 29.36     | 30.68     | 30.81     | 17.69     |
| Llama-3.1-70B      | 38.08     | 42.43     | 38.07     | 38.96     | 36.36     | 34.58     |
| Qwen3-8B           | 30.97     | 25.04     | 34.10     | 37.52     | 36.09     | 22.10     |
| Qwen3-14B          | 39.44     | 48.93     | 37.00     | 40.09     | 37.30     | 33.88     |
| Qwen3-32B          | 38.26     | 45.65     | 37.26     | 40.54     | 37.37     | 30.46     |
| GPT-oss-20B        | 34.08     | 26.66     | 34.36     | 35.73     | 35.88     | 37.75     |
| GPT-oss-120B       | 35.25     | 31.64     | 35.76     | 37.36     | 36.57     | 34.90     |

### F1^NS (%)

| Model              | Overall   | Zhihu     | Weibo     | Toutiao   | XHS       | Douban    |
|--------------------|-----------|-----------|-----------|-----------|-----------|-----------|
| _Closed-source_    |           |           |           |           |           |           |
| GPT-4o-mini        | 33.53     | 36.03     | 31.24     | 38.73     | 27.94     | 9.71      |
| GPT-5-mini         | 35.44     | 35.70     | 35.97     | 36.87     | 29.02     | 33.81     |
| GPT-5.1            | 42.73     | _52.63_   | 41.97     | 43.05     | _38.75_   | 28.88     |
| Gemini-3-Flash     | **54.97** | 49.30     | **52.75** | **53.71** | **48.78** | **52.65** |
| _Open-source_      |           |           |           |           |           |           |
| MiniMax-M2.5       | 37.90     | 41.56     | 35.79     | 39.09     | 31.26     | 26.18     |
| GLM-4.7            | 44.21     | 47.79     | _43.62_   | 40.24     | 35.72     | 26.10     |
| DeepSeek-v3.2      | _44.63_   | 48.47     | 40.40     | _41.44_   | 33.59     | _31.12_   |
| Llama-3.1-8B       | 18.85     | 19.43     | 21.21     | 20.29     | 14.52     | 18.80     |
| Llama-3.1-70B      | 40.38     | 42.76     | 40.46     | 40.80     | 32.91     | 17.55     |
| Qwen3-8B           | 31.54     | 28.99     | 32.09     | 36.89     | 29.37     | 25.05     |
| Qwen3-14B          | 41.33     | _48.68_   | 37.25     | 39.68     | 30.37     | 28.64     |
| Qwen3-32B          | 40.82     | 48.21     | 38.54     | 41.30     | 32.39     | 27.43     |
| GPT-oss-20B        | 34.68     | 32.07     | 31.78     | 32.43     | 27.21     | 29.53     |
| GPT-oss-120B       | 37.12     | 39.17     | 35.09     | 36.77     | 30.22     | 33.68     |

For per-platform `Recall_Stability` / `Recall_Novelty` and the four
distractor error rates, see the paper appendix (Tables 17 and 18).

## Anonymization notes

This release uses anonymized identifiers:

- `user_id` is a hash prefixed with the platform code (`we_` weibo, `xi_` xiaohongshu, `to_` toutiao, `zh_` zhihu, `do_` douban).
- `username` values are replaced with placeholders of the form `U_xxxxxxx`.
- Free-text fields (`bio`, `posts_text`) are passed through a span-detection +
  regex pipeline that masks ten PI categories (phone, ID, bank, email, contact,
  plate, IP, GEO, device, self-name) into fixed placeholders. See the paper
  appendix C.3 for details.

If you find any residual PII please open an issue.


