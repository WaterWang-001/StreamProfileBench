import re
from pathlib import Path
from collections import defaultdict
import pandas as pd


def parse_report_file(file_path: Path):
    """Parse a per-platform eval report and return {metric_name: pct_value}.

    Extracts the M̄ block (Recall, Recall_Novelty, Recall_Stability, Error_*) and the
    F1^NS line. The two sections in the report are delimited by `=` rules.
    """
    text = file_path.read_text(encoding="utf-8")
    metrics = {}

    # M̄ block: from "Overall Macro Average" up to the "Users:" summary line.
    mbar_match = re.search(
        r"Overall Macro Average.*?\n=+\n(.*?)\n\s*Users:",
        text,
        re.DOTALL,
    )
    if mbar_match:
        metric_pattern = re.compile(
            r'^\s*([A-Za-z0-9_\-\^\.\(\)]+)\s*:\s*([0-9.]+)%\s*$',
            re.MULTILINE,
        )
        for name, value in metric_pattern.findall(mbar_match.group(1)):
            metrics[name] = float(value)

    # F1^NS line.
    f1_match = re.search(r'F1\^NS\s*:\s*([0-9.]+)%', text)
    if f1_match:
        metrics["F1^NS"] = float(f1_match.group(1))

    return metrics


def collect_all_results(results_root="results"):
    """
    收集所有结果:
    all_data[metric][model][task] = value
    """
    results_root = Path(results_root)
    all_data = defaultdict(lambda: defaultdict(dict))

    for file_path in results_root.rglob("eval_report_*.txt"):
        model_name = file_path.parent.name

        match = re.match(r"eval_report_(.+)\.txt$", file_path.name)
        if not match:
            continue
        task_name = match.group(1).strip().lower()

        metrics = parse_report_file(file_path)

        for metric_name, value in metrics.items():
            all_data[metric_name][model_name][task_name] = value

    return all_data


def prettify_task_name(task_name: str) -> str:
    mapping = {
        "zhihu": "Zhihu",
        "weibo": "Weibo",
        "toutiao": "Toutiao",
        "xiaohongshu": "Xiaohongshu",
        "douban": "Douban",
    }
    return mapping.get(task_name.lower(), task_name.capitalize())


def build_metric_dataframe(metric_data, task_order=None):
    """
    构造单个指标的 DataFrame
    metric_data: {model: {task: value}}
    """
    all_tasks = set()
    for _, task_dict in metric_data.items():
        all_tasks.update(task_dict.keys())

    if task_order is None:
        task_order = sorted(all_tasks)
    else:
        existing = [t for t in task_order if t in all_tasks]
        remaining = sorted(all_tasks - set(existing))
        task_order = existing + remaining

    rows = []
    for model in sorted(metric_data.keys()):
        task_dict = metric_data[model]

        values = [task_dict.get(task) for task in task_order]
        valid_values = [v for v in values if v is not None]
        avg_value = sum(valid_values) / len(valid_values) if valid_values else None

        row = {
            "Model": model,
            "Avg.": avg_value
        }
        for task in task_order:
            row[prettify_task_name(task)] = task_dict.get(task)

        rows.append(row)

    columns = ["Model", "Avg."] + [prettify_task_name(task) for task in task_order]
    df = pd.DataFrame(rows, columns=columns)
    return df


def sanitize_filename(name: str) -> str:
    return re.sub(r'[\\/*?:"<>|]', "_", name)


def save_one_metric_one_excel(all_data, output_dir="metric_excels", task_order=None):
    """
    一个指标保存成一个 Excel 文件
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for metric_name, metric_data in all_data.items():
        df = build_metric_dataframe(metric_data, task_order=task_order)

        safe_metric_name = sanitize_filename(metric_name)
        output_path = output_dir / f"{safe_metric_name}.xlsx"

        with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
            df.to_excel(writer, sheet_name="Sheet1", index=False)
            ws = writer.sheets["Sheet1"]

            # 列宽
            for col in ws.columns:
                max_len = 0
                col_letter = col[0].column_letter
                for cell in col:
                    val = "" if cell.value is None else str(cell.value)
                    max_len = max(max_len, len(val))
                ws.column_dimensions[col_letter].width = min(max_len + 2, 20)

            # 两位小数
            for row in ws.iter_rows(min_row=2):
                for cell in row[1:]:
                    if isinstance(cell.value, (int, float)):
                        cell.number_format = "0.00"

        print(f"Saved: {output_path}")


if __name__ == "__main__":
    task_order = ["zhihu", "weibo", "toutiao", "xiaohongshu", "douban"]

    all_data = collect_all_results("results")
    save_one_metric_one_excel(
        all_data,
        output_dir="metric_excels",
        task_order=task_order
    )