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
    python -m utils.swanlab_viz --projects sd900_ada -o my_plot.png

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
                     filter_kw: Optional[str] = None) -> list[dict]:
    resp = api.list_experiments(project, user)
    if resp.code != 200:
        print(f"[WARN] Failed to list experiments for {user}/{project}: {resp.errmsg}")
        return []
    exps = resp.data
    if filter_kw:
        exps = [e for e in exps if filter_kw.lower() in e['name'].lower()]
    return exps


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
    # Try removing project-based prefix patterns
    for sep in ['_sam_base_', '_sam_large_', '_sam_huge_']:
        if sep in name:
            idx = name.index(sep) + len(sep)
            return name[idx:]
    # Fallback: remove leading project name if present
    if name.startswith(project.replace('_', '')):
        return name[len(project.replace('_', '')):]
    return name


def collect_all_data(api: OpenApi, projects: list[str], user: str,
                     metric_keys: list[str],
                     filter_kw: Optional[str] = None) -> dict:
    """
    Returns:
        {display_name: {project: str, metrics: {key: DataFrame}}}
    """
    all_data = {}
    for proj in projects:
        exps = list_experiments(api, proj, user, filter_kw)
        if not exps:
            print(f"[INFO] No experiments found in {user}/{proj}"
                  + (f" matching '{filter_kw}'" if filter_kw else ""))
            continue
        print(f"[INFO] {user}/{proj}: {len(exps)} experiment(s)")
        for exp in exps:
            eid = exp['cuid']
            raw_name = exp['name']
            short = shorten_name(raw_name, proj)
            # Prefix with project name when comparing multiple projects
            label = f"[{proj}] {short}" if len(projects) > 1 else short
            metrics = fetch_metrics(api, eid, metric_keys)
            all_data[label] = {'project': proj, 'metrics': metrics}
    return all_data


def print_summary_table(all_data: dict, metric_keys: list[str]):
    # Use the non-loss metrics for the table, plus loss
    display_keys = metric_keys
    header = f"{'Experiment':<45}" + "".join(f"{k:>14}" for k in display_keys)
    print("\n" + "=" * len(header))
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

    # Sort by first val metric (usually dice) descending, None last
    sort_idx = next((i for i, k in enumerate(display_keys) if 'dice' in k),
                    next((i for i, k in enumerate(display_keys) if 'loss' not in k), 0))
    rows.sort(key=lambda r: r[1][sort_idx] if r[1][sort_idx] is not None else -1,
              reverse=('loss' not in display_keys[sort_idx]))

    for name, vals in rows:
        val_strs = []
        for v in vals:
            val_strs.append(f"{v:>14.4f}" if v is not None else f"{'N/A':>14}")
        print(f"{name:<45}{''.join(val_strs)}")
    print("=" * len(header))


def plot_metrics(all_data: dict, metric_keys: list[str],
                 title: str, output_path: str):
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

    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n[OK] Plot saved to {output_path}")


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
                        help='Output image path (default: <projects>_viz.png)')
    parser.add_argument('--table-only', action='store_true',
                        help='Only print summary table, skip plotting')
    parser.add_argument('--title', '-t', default=None,
                        help='Custom plot title')
    args = parser.parse_args()

    metric_keys = args.metrics or DEFAULT_METRICS
    api, user = get_api(args.user)

    all_data = collect_all_data(api, args.projects, user, metric_keys, args.filter)
    if not all_data:
        print("[ERROR] No data collected. Check project names and filters.")
        sys.exit(1)

    print_summary_table(all_data, metric_keys)

    if not args.table_only:
        title = args.title or " vs ".join(args.projects)
        output = args.output or "_".join(args.projects) + "_viz.png"
        plot_metrics(all_data, metric_keys, title, output)


if __name__ == '__main__':
    main()
