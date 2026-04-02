"""
SwanLab experiment visualization tool.

Usage examples:
    # Basic: visualize a single project
    python -m utils.swanlab_viz --projects sd900_ada

    # Multiple projects on one figure
    python -m utils.swanlab_viz --projects sd900_ada sd900_real

    # Filter experiments by keyword
    python -m utils.swanlab_viz --projects sd900_ada --filter adalora

    # Custom metrics
    python -m utils.swanlab_viz --projects sd900_ada --metrics train/loss val/dice val/iou val/hd95

    # Custom output path
    python -m utils.swanlab_viz --projects sd900_ada -o my_plot

    # Only print summary table, no plot
    python -m utils.swanlab_viz --projects sd900_ada --table-only

    # Specify username (default: auto-detect from login)
    python -m utils.swanlab_viz --projects sd900_ada --user bttb
"""

import argparse
import time
import sys
from typing import Optional

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from swanlab.api import OpenApi


DEFAULT_METRICS = ['train/loss', 'val/dice', 'val/iou']


def get_api(username: Optional[str] = None) -> tuple[OpenApi, str]:
    api = OpenApi()
    user = username or api.username
    return api, user


def list_experiments(api: OpenApi, project: str, user: str,
                     filter_kw: Optional[str] = None) -> list:
    resp = api.list_experiments(project, user)
    if resp.code != 200:
        print(f"[WARN] Failed to list experiments for {user}/{project}: {resp.errmsg}")
        return []
    exps = resp.data
    
    filtered_exps = []
    for exp in exps:
        exp_dict = exp.model_dump() if hasattr(exp, 'model_dump') else (exp.dict() if hasattr(exp, 'dict') else exp)
        name = exp_dict.get('name') or (exp['name'] if hasattr(exp, '__getitem__') else getattr(exp, 'name', ''))
        if filter_kw and filter_kw.lower() not in name.lower():
            continue
        filtered_exps.append(exp)
        
    return filtered_exps


def fetch_metrics(api: OpenApi, exp_id: str, keys: list[str],
                  retries: int = 3) -> dict[str, pd.DataFrame]:
    result = {}
    for key in keys:
        for attempt in range(retries):
            try:
                r = api.get_metrics(exp_id, key)
                if r.code == 200 and not r.data.empty:
                    result[key] = r.data
                break
            except Exception:
                if attempt < retries - 1:
                    time.sleep(1)
        time.sleep(0.2)
    return result


def shorten_name(name: str, project: str) -> str:
    """Remove common project prefix from experiment name for cleaner labels."""
    for sep in ['_sam_base_', '_sam_large_', '_sam_huge_']:
        if sep in name:
            idx = name.index(sep) + len(sep)
            return name[idx:]
    if name.startswith(project.replace('_', '')):
        return name[len(project.replace('_', '')):]
    return name


def collect_all_data(api: OpenApi, projects: list[str], user: str,
                     metric_keys: list[str],
                     filter_kw: Optional[str] = None) -> dict:
    """
    Returns grouped data by ft_type:
        {ft_type: {display_name: {project: str, metrics: {key: DataFrame}}}}
    """
    grouped_data = {}
    for proj in projects:
        exps = list_experiments(api, proj, user, filter_kw)
        if not exps:
            print(f"[INFO] No experiments found in {user}/{proj}"
                  + (f" matching '{filter_kw}'" if filter_kw else ""))
            continue
        print(f"[INFO] {user}/{proj}: {len(exps)} experiment(s)")
        
        for exp in exps:
            exp_dict = exp.model_dump() if hasattr(exp, 'model_dump') else (exp.dict() if hasattr(exp, 'dict') else exp)
            
            # fallback for dict access
            if not isinstance(exp_dict, dict) and hasattr(exp, '__getitem__'):
                exp_dict = exp
                
            eid = exp_dict.get('cuid') or (exp['cuid'] if hasattr(exp, '__getitem__') else getattr(exp, 'cuid'))
            raw_name = exp_dict.get('name') or (exp['name'] if hasattr(exp, '__getitem__') else getattr(exp, 'name'))
            
            # Extract ft_type
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
                
            short = shorten_name(raw_name, proj)
            label = f"[{proj}] {short}" if len(projects) > 1 else short
            
            metrics = fetch_metrics(api, eid, metric_keys)
            
            if ft_type not in grouped_data:
                grouped_data[ft_type] = {}
                
            grouped_data[ft_type][label] = {'project': proj, 'metrics': metrics}
            
    return grouped_data


def print_summary_table(grouped_data: dict, metric_keys: list[str]):
    display_keys = metric_keys
    header = f"{'Experiment':<45}" + "".join(f"{k:>14}" for k in display_keys)
    
    for ft_type, all_data in grouped_data.items():
        print(f"\n[ FT Type: {ft_type} ]")
        print("=" * len(header))
        print(header)
        print("-" * len(header))

        rows = []
        for name, info in all_data.items():
            vals = []
            for key in display_keys:
                if key in info['metrics']:
                    df = info['metrics'][key]
                    vals.append(df[key].iloc[-1])
                else:
                    vals.append(None)
            rows.append((name, vals))

        sort_idx = next((i for i, k in enumerate(display_keys) if 'dice' in k),
                        next((i for i, k in enumerate(display_keys) if 'loss' not in k), 0))
        rows.sort(key=lambda r: r[1][sort_idx] if r[1][sort_idx] is not None else -1,
                  reverse=('loss' not in display_keys[sort_idx] if sort_idx < len(display_keys) else False))

        for name, vals in rows:
            val_strs = []
            for v in vals:
                val_strs.append(f"{v:>14.4f}" if v is not None else f"{'N/A':>14}")
            print(f"{name:<45}{''.join(val_strs)}")
        print("=" * len(header))


def plot_metrics(grouped_data: dict, metric_keys: list[str], output_prefix: str):
    for ft_type, all_data in grouped_data.items():
        if not all_data:
            continue
            
        n = len(metric_keys)
        fig, axes = plt.subplots(1, n, figsize=(6 * n, 5))
        if n == 1:
            axes = [axes]

        for label, info in all_data.items():
            for i, key in enumerate(metric_keys):
                if key in info['metrics']:
                    df = info['metrics'][key]
                    axes[i].plot(df.index, df[key], label=label, alpha=0.8, linewidth=1.5)

        for i, key in enumerate(metric_keys):
            axes[i].set_title(key, fontsize=14)
            axes[i].set_xlabel('Step')
            axes[i].set_ylabel(key)
            axes[i].legend(fontsize=7, loc='best')
            axes[i].grid(True, alpha=0.3)

        plt.suptitle(f"FT Type: {ft_type}", fontsize=16)
        plt.tight_layout()
        
        safe_ft_type = ft_type.replace('/', '_').replace(' ', '_')
        out_path = f"{output_prefix}_{safe_ft_type}.png"
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        print(f"[OK] Plot saved to {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Visualize SwanLab experiments',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument('--projects', '-p', nargs='+', required=True,
                        help='SwanLab project name(s)')
    parser.add_argument('--user', '-u', default=None,
                        help='SwanLab username (default: auto-detect)')
    parser.add_argument('--metrics', '-m', nargs='+', default=None,
                        help=f'Metric keys to plot (default: {DEFAULT_METRICS})')
    parser.add_argument('--filter', '-f', default=None,
                        help='Filter experiments by keyword in name')
    parser.add_argument('--output', '-o', default=None,
                        help='Output image path prefix (default: <projects>_viz)')
    parser.add_argument('--table-only', action='store_true',
                        help='Only print summary table, skip plotting')
    args = parser.parse_args()

    metric_keys = args.metrics or DEFAULT_METRICS
    api, user = get_api(args.user)

    grouped_data = collect_all_data(api, args.projects, user, metric_keys, args.filter)
    
    has_data = any(len(data) > 0 for data in grouped_data.values())
    if not has_data:
        print("[ERROR] No data collected. Check project names and filters.")
        sys.exit(1)

    print_summary_table(grouped_data, metric_keys)

    if not args.table_only:
        output_prefix = args.output or "_".join(args.projects) + "_viz"
        # If user passed an extension, strip it so we can append _<ft_type>.png
        if output_prefix.endswith('.png'):
            output_prefix = output_prefix[:-4]
        plot_metrics(grouped_data, metric_keys, output_prefix)


if __name__ == '__main__':
    main()
