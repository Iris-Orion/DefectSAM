"""
Aggregate SwanLab cloud runs: mean ± std over repeated runs.

Usage:
    # Basic: aggregate given projects
    python -m utils.swanlab_stats --projects sd900_bench --user bttb
    
    # Several projects
    python -m utils.swanlab_stats --projects neu_bench magnetic_bench --user bttb
    
    # Filter by method name
    python -m utils.swanlab_stats --projects sd900_bench --filter lora
    
    # Export CSV
    python -m utils.swanlab_stats --projects sd900_bench --output results.csv
    
    # Export a Markdown table
    python -m utils.swanlab_stats --projects sd900_bench --format markdown
"""

import argparse
import sys
import time
from typing import Optional
from collections import defaultdict

import numpy as np
import pandas as pd
from swanlab.api import OpenApi


SWANLAB_API_KEY = ""

DEFAULT_TEST_METRICS = ['test/test_dice', 'test/test_iou', 'test/test_hd95']
DEFAULT_TRAIN_METRICS = ['train/dice', 'train/iou']
DEFAULT_VAL_METRICS = ['val/dice']


def get_api(username: Optional[str] = None) -> tuple[OpenApi, str]:
    api = OpenApi(api_key=__import__("os").environ["SWANLAB_API_KEY"])
    user = username or api.username
    return api, user


def list_experiments(api: OpenApi, project: str, user: str,
                     filter_kw: Optional[str] = None) -> list:
    resp = api.list_experiments(project, user)
    if resp.code != 200:
        print(f"[WARN] 无法获取 {user}/{project}: {resp.errmsg}")
        return []
    
    exps = resp.data
    filtered_exps = []
    for exp in exps:
        exp_dict = exp.model_dump() if hasattr(exp, 'model_dump') else (
            exp.dict() if hasattr(exp, 'dict') else exp
        )
        name = exp_dict.get('name') or (exp['name'] if hasattr(exp, '__getitem__') else getattr(exp, 'name', ''))
        if filter_kw and filter_kw.lower() not in name.lower():
            continue
        filtered_exps.append(exp)
    
    return filtered_exps


def fetch_metrics(api: OpenApi, exp_id: str, keys: list[str], retries: int = 3) -> dict[str, pd.DataFrame]:
    result = {}
    for key in keys:
        for attempt in range(retries):
            try:
                r = api.get_metrics(exp_id, key)
                if r.code == 200 and not r.data.empty:
                    result[key] = r.data
                break
            except Exception as e:
                if attempt < retries - 1:
                    time.sleep(1)
        time.sleep(0.2)
    return result


def extract_ft_type(exp_dict: dict) -> str:
    ft_type = "unknown"
    try:
        profile = exp_dict.get('profile', {})
        if profile and 'config' in profile:
            config = profile['config']
            if 'ft_type' in config:
                ft_val = config['ft_type']
                if isinstance(ft_val, dict) and 'value' in ft_val:
                    ft_type = str(ft_val['value'])
                else:
                    ft_type = str(ft_val)
    except Exception:
        pass
    return ft_type


def find_best_epoch(metrics: dict) -> tuple[int, float]:
    if 'val/dice' not in metrics or metrics['val/dice'].empty:
        return -1, 0.0
    
    val_df = metrics['val/dice']
    # idxmax returns the index label (the step), not the position.
    best_step = val_df['val/dice'].idxmax()
    best_val = val_df.loc[val_df.index == best_step, 'val/dice'].values[0]
    return int(best_step), float(best_val)


def get_metric_at_step(metrics: dict, metric_key: str, target_step: int) -> Optional[float]:
    if metric_key not in metrics or metrics[metric_key].empty:
        return None
    
    df = metrics[metric_key]
    closest_step = min(df.index, key=lambda s: abs(s - target_step))
    closest_idx = df.index.get_loc(closest_step)
    return float(df.iloc[closest_idx][metric_key])


def get_final_metric(metrics: dict, metric_key: str) -> Optional[float]:
    if metric_key not in metrics or metrics[metric_key].empty:
        return None
    
    df = metrics[metric_key]
    return float(df[metric_key].iloc[-1])


def collect_experiment_data(api: OpenApi, projects: list[str], user: str,
                           filter_kw: Optional[str] = None) -> dict:
    """Returns {ft_type: [run_data, ...]}."""
    all_runs = defaultdict(list)
    
    for proj in projects:
        exps = list_experiments(api, proj, user, filter_kw)
        if not exps:
            print(f"[INFO] {user}/{proj}: 无实验" + (f" 匹配 '{filter_kw}'" if filter_kw else ""))
            continue
        
        print(f"[INFO] {user}/{proj}: 找到 {len(exps)} 个实验")
        
        for exp in exps:
            exp_dict = exp.model_dump() if hasattr(exp, 'model_dump') else (
                exp.dict() if hasattr(exp, 'dict') else exp
            )
            
            if not isinstance(exp_dict, dict) and hasattr(exp, '__getitem__'):
                exp_dict = exp
            
            eid = exp_dict.get('cuid') or (exp['cuid'] if hasattr(exp, '__getitem__') else getattr(exp, 'cuid'))
            raw_name = exp_dict.get('name') or (exp['name'] if hasattr(exp, '__getitem__') else getattr(exp, 'name'))
            
            ft_type = extract_ft_type(exp_dict)
            
            all_metric_keys = DEFAULT_TEST_METRICS + DEFAULT_TRAIN_METRICS + DEFAULT_VAL_METRICS
            metrics = fetch_metrics(api, eid, all_metric_keys)
            
            best_step, best_val_dice = find_best_epoch(metrics)
            
            run_data = {
                'name': raw_name,
                'project': proj,
                'ft_type': ft_type,
                'best_step': best_step,
                'best_val_dice': best_val_dice,
            }
            
            for key in DEFAULT_TEST_METRICS:
                metric_name = key.split('/')[-1]
                run_data[metric_name] = get_final_metric(metrics, key)
            
            if best_step > 0:
                run_data['train_dice@best'] = get_metric_at_step(metrics, 'train/dice', best_step)
                run_data['train_iou@best'] = get_metric_at_step(metrics, 'train/iou', best_step)
            else:
                run_data['train_dice@best'] = None
                run_data['train_iou@best'] = None
            
            all_runs[ft_type].append(run_data)
            print(f"  [OK] {raw_name} | ft_type={ft_type} | best_val_dice={best_val_dice:.4f}")
    
    return dict(all_runs)


def compute_stats(values: list) -> tuple[float, float]:
    valid_values = [v for v in values if v is not None and not np.isnan(v)]
    if len(valid_values) == 0:
        return np.nan, np.nan
    return np.mean(valid_values), np.std(valid_values)


def format_mean_std(mean: float, std: float, precision: int = 4) -> str:
    if np.isnan(mean):
        return "N/A"
    return f"{mean:.{precision}f}±{std:.{precision}f}"


def generate_summary_table(grouped_data: dict) -> pd.DataFrame:
    summary_rows = []
    
    for ft_type, runs in grouped_data.items():
        if not runs:
            continue
        
        n_runs = len(runs)
        
        test_dice_vals = [r['test_dice'] for r in runs]
        test_iou_vals = [r['test_iou'] for r in runs]
        test_hd95_vals = [r['test_hd95'] for r in runs]
        train_dice_vals = [r['train_dice@best'] for r in runs]
        train_iou_vals = [r['train_iou@best'] for r in runs]
        val_dice_vals = [r['best_val_dice'] for r in runs]
        
        test_dice_mean, test_dice_std = compute_stats(test_dice_vals)
        test_iou_mean, test_iou_std = compute_stats(test_iou_vals)
        test_hd95_mean, test_hd95_std = compute_stats(test_hd95_vals)
        train_dice_mean, train_dice_std = compute_stats(train_dice_vals)
        train_iou_mean, train_iou_std = compute_stats(train_iou_vals)
        val_dice_mean, val_dice_std = compute_stats(val_dice_vals)
        
        row = {
            'Method': ft_type,
            'Runs': n_runs,
            'Test_Dice': format_mean_std(test_dice_mean, test_dice_std),
            'Test_Dice_mean': test_dice_mean,
            'Test_Dice_std': test_dice_std,
            'Test_IoU': format_mean_std(test_iou_mean, test_iou_std),
            'Test_IoU_mean': test_iou_mean,
            'Test_IoU_std': test_iou_std,
            'Test_HD95': format_mean_std(test_hd95_mean, test_hd95_std),
            'Test_HD95_mean': test_hd95_mean,
            'Test_HD95_std': test_hd95_std,
            'Train_Dice@Best': format_mean_std(train_dice_mean, train_dice_std),
            'Train_Dice@Best_mean': train_dice_mean,
            'Train_Dice@Best_std': train_dice_std,
            'Train_IoU@Best': format_mean_std(train_iou_mean, train_iou_std),
            'Train_IoU@Best_mean': train_iou_mean,
            'Train_IoU@Best_std': train_iou_std,
            'Val_Dice@Best': format_mean_std(val_dice_mean, val_dice_std),
            'Val_Dice@Best_mean': val_dice_mean,
            'Val_Dice@Best_std': val_dice_std,
        }
        summary_rows.append(row)
    
    summary_rows.sort(key=lambda x: x['Test_Dice_mean'] if not np.isnan(x['Test_Dice_mean']) else -1, reverse=True)
    
    return pd.DataFrame(summary_rows)


def print_pretty_table(df: pd.DataFrame):
    display_cols = ['Method', 'Runs', 'Test_Dice', 'Test_IoU', 'Test_HD95', 
                    'Train_Dice@Best', 'Train_IoU@Best', 'Val_Dice@Best']
    
    display_df = df[display_cols].copy()
    
    print("\n" + "="*100)
    print("SwanLab实验统计结果")
    print("="*100)
    
    header = f"{'Method':<20} {'Runs':>6} {'Test_Dice':>14} {'Test_IoU':>14} {'Test_HD95':>14} {'Train_Dice@B':>14} {'Train_IoU@B':>14}"
    print(header)
    print("-"*100)
    
    for _, row in display_df.iterrows():
        line = f"{row['Method']:<20} {row['Runs']:>6} {row['Test_Dice']:>14} {row['Test_IoU']:>14} {row['Test_HD95']:>14} {row['Train_Dice@Best']:>14} {row['Train_IoU@Best']:>14}"
        print(line)
    
    print("="*100)


def export_markdown(df: pd.DataFrame, output_path: str):
    """Hand-built Markdown so pandas' tabulate dependency is not needed."""
    display_cols = ['Method', 'Runs', 'Test_Dice', 'Test_IoU', 'Test_HD95', 
                    'Train_Dice@Best', 'Train_IoU@Best']
    display_df = df[display_cols].copy()
    
    lines = []
    lines.append("# SwanLab实验统计结果\n")
    lines.append("| Method | Runs | Test_Dice | Test_IoU | Test_HD95 | Train_Dice@Best | Train_IoU@Best |")
    lines.append("|--------|------|-----------|----------|-----------|-----------------|----------------|")
    
    for _, row in display_df.iterrows():
        lines.append(f"| {row['Method']} | {row['Runs']} | {row['Test_Dice']} | {row['Test_IoU']} | {row['Test_HD95']} | {row['Train_Dice@Best']} | {row['Train_IoU@Best']} |")
    
    with open(output_path, 'w') as f:
        f.write('\n'.join(lines))
        f.write('\n')
    
    print(f"[OK] Markdown表格已保存至: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='SwanLab云端实验统计工具 - 计算多次运行的mean±std',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument('--projects', '-p', nargs='+', required=True,
                        help='SwanLab项目名称(s)')
    parser.add_argument('--user', '-u', default=None,
                        help='SwanLab用户名 (默认: auto-detect)')
    parser.add_argument('--filter', '-f', default=None,
                        help='按关键词过滤实验名称')
    parser.add_argument('--output', '-o', default=None,
                        help='导出CSV文件路径')
    parser.add_argument('--format', choices=['csv', 'markdown', 'both'], default='csv',
                        help='输出格式 (默认: csv)')
    args = parser.parse_args()
    
    api, user = get_api(args.user)
    print(f"[INFO] 使用用户: {user}")
    print(f"[INFO] 项目: {', '.join(args.projects)}")
    
    grouped_data = collect_experiment_data(api, args.projects, user, args.filter)
    
    if not grouped_data:
        print("[ERROR] 未获取到任何数据，请检查项目名称和过滤器")
        sys.exit(1)
    
    summary_df = generate_summary_table(grouped_data)
    
    print_pretty_table(summary_df)
    
    if args.output:
        base_name = args.output.replace('.csv', '').replace('.md', '')
        
        if args.format in ['csv', 'both']:
            csv_path = base_name + '.csv'
            csv_cols = ['Method', 'Runs'] + [c for c in summary_df.columns if c.endswith(('_mean', '_std'))]
            summary_df[csv_cols].to_csv(csv_path, index=False)
            print(f"[OK] CSV已保存至: {csv_path}")
        
        if args.format in ['markdown', 'both']:
            md_path = base_name + '.md'
            export_markdown(summary_df, md_path)
    
    print("\n[详细数据]")
    for ft_type, runs in grouped_data.items():
        print(f"\n{ft_type} ({len(runs)} runs):")
        for run in runs:
            td = f"{run['test_dice']:.4f}" if run['test_dice'] is not None else 'N/A'
            print(f"  - {run['name']}: test_dice={td}, best_val_dice={run['best_val_dice']:.4f}")


if __name__ == '__main__':
    main()
