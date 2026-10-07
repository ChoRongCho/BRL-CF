"""Pool all scenes within each domain; regenerate figures and exact statistics.

Run: python3 collect_exp/vlm_comparison_final_20261005/plot_comparison.py
Requires matplotlib and numpy. Input CSVs are never modified.
"""

import csv
import json
import math
import os
from pathlib import Path
from statistics import mean, stdev

os.environ.setdefault('MPLCONFIGDIR', '/tmp/brl-vlm-matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

OUT = Path(__file__).resolve().parent
BASE = OUT.parent
DOMAINS = {'Tomato': BASE / 'tomato_vlm_final_20261005',
           'Waste': BASE / 'waste_vlm_final_20261005'}
DOMAINS = {name: folder if (folder / 'runs.csv').exists() else OUT / folder.name
           for name, folder in DOMAINS.items()}
METHODS = {'pomdp': 'POMDP', 'knowno': 'KnowNo'}
COLORS = {'pomdp': '#287DA8', 'knowno': '#E59448'}
METRICS = [
    ('success_rate', '1. 성공률', '%', 100),
    ('questions', '2. 성공 시 질의 횟수', '회', 1),
    ('query_probability', '3. 성공 시 질의확률', '%', 100),
    ('steps', '4. 성공 시 planLength', 'steps', 1),
    ('total_time_s', '5. 성공 시 전체 시간', '초', 1),
    ('interaction_time_s', '6. 성공 시 상호작용 시간', '초', 1),
    ('planning_residual_time_s', '7. 성공 시 전체 − (상호작용 + 실행) 시간', '초', 1),
]


def load_data(all_runs=False):
    groups = {}
    for domain, folder in DOMAINS.items():
        with (folder / 'runs.csv').open(encoding='utf-8-sig', newline='') as f:
            rows = list(csv.DictReader(f))
        assert len({r['run'] for r in rows}) == len(rows)
        for row in rows:
            assert row['method'] in METHODS
            assert row['success'] in ('True', 'False')
            for key in ('steps', 'questions', 'query_steps', 'query_probability',
                        'total_time_s', 'interaction_time_s', 'execute_time_s',
                        'planning_residual_time_s'):
                row[key] = float(row[key])
                assert math.isfinite(row[key])
            assert row['steps'] > 0
            assert math.isclose(row['query_probability'], row['query_steps'] / row['steps'])
            residual = row['total_time_s'] - (row['interaction_time_s'] + row['execute_time_s'])
            assert math.isclose(residual, row['planning_residual_time_s'], abs_tol=1e-8)
            row['planning_residual_time_s'] = residual
        for method in METHODS:
            subset = [r for r in rows if r['method'] == method]
            successes = [r for r in subset if r['success'] == 'True']
            assert successes
            selected = subset if all_runs else successes
            groups[domain, method] = {
                'total': len(subset), 'success': len(successes),
                'scenes': ', '.join(sorted({r['scene'] for r in subset})),
                'success_rate': (len(successes) / len(subset), None),
                **{key: (mean(r[key] for r in selected),
                         stdev(r[key] for r in selected) if len(selected) > 1 else None)
                   for key, *_ in METRICS[1:]},
            }
    return groups


def save_statistics(groups, destination=OUT, all_runs=False):
    destination.mkdir(parents=True, exist_ok=True)
    with (destination / 'pooled_statistics.csv').open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'domain', 'method', 'scenes', 'total_n', 'success_n', 'metric',
            'metric_n', 'mean', 'sample_sd', 'unit'])
        writer.writeheader()
        for (domain, method), group in groups.items():
            for key, _, unit, scale in METRICS:
                avg, sd = group[key]
                writer.writerow(dict(domain=domain, method=METHODS[method], scenes=group['scenes'],
                                     total_n=group['total'], success_n=group['success'], metric=key,
                                     metric_n=group['total'] if key == 'success_rate' or all_runs else group['success'],
                                     mean=avg * scale, sample_sd='' if sd is None else sd * scale,
                                     unit=unit))
    definitions = {
        'aggregation': 'Pool individual runs across all available scenes within each domain and method. No scene balancing or sample-size matching.',
        'success_rate': 'Number of successful runs / all runs; no error bar.',
        'other_metrics': 'Arithmetic mean over successful runs; error bars = sample standard deviation (ddof=1), not confidence intervals.',
        'query_probability': 'Mean of per-run distinct question-bearing steps / all steps; plotted and exported as percent.',
        'planLength': 'runs.csv steps = Plan Summary.steps (executed/logged plan length, including detection/navigation/check_done where present).',
        'total_time_s': 'Timing.total_time, not GUI wall-clock duration.',
        'interaction_time_s': 'Timing.interaction_time.total.',
        'planning_residual_time_s': 'Per-run total_time - (interaction_time.total + execute_time.total); includes search and other overhead.',
        'inputs': [str(p.relative_to(BASE) / 'runs.csv') for p in DOMAINS.values()],
    }
    if all_runs:
        definitions['other_metrics'] = 'Arithmetic mean over ALL runs (success and failure); sample standard deviation ddof=1. Failed runs contribute their recorded values up to termination.'
    (destination / 'definitions.json').write_text(json.dumps(definitions, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def draw(groups, domains, destination, stem, all_runs=False):
    plt.rcParams.update({'font.family': 'Noto Sans CJK JP', 'font.size': 10,
                         'axes.unicode_minus': False, 'axes.spines.top': False,
                         'axes.spines.right': False, 'axes.spines.left': False,
                         'pdf.fonttype': 42, 'svg.fonttype': 'none'})
    fig, axes = plt.subplots(2, 4, figsize=(18, 9))
    fig.patch.set_facecolor('#FAFBFD')
    fig.suptitle(' / '.join(domains) + '  |  전체 scene 통합 비교',
                 x=.055, y=.965, ha='left', fontsize=23, fontweight='bold', color='#192B3D')
    fig.text(.055, .911, '성공·실패 전체 시행 평균 ± 표본 표준편차' if all_runs else '성공률: 전체 시행 기준   ·   나머지 지표: 성공 시행 평균 ± 표본 표준편차',
             fontsize=12, color='#556575')
    fig.legend(handles=[Patch(color=COLORS[m], label=label) for m, label in METHODS.items()],
               loc='upper right', bbox_to_anchor=(.965, .974), ncol=2, frameon=False, fontsize=12)
    x = np.arange(len(domains))
    for ax, (key, title, unit, scale) in zip(axes.flat, METRICS):
        ax.set_facecolor('#FAFBFD')
        ax.set_title(title.replace('성공 시 ', '') if all_runs else title, loc='left', fontsize=11, fontweight='bold', pad=15)
        upper = 0
        for j, method in enumerate(METHODS):
            stats = [groups[d, method][key] for d in domains]
            values = [s[0] * scale for s in stats]
            errors = [s[1] * scale if s[1] is not None else 0 for s in stats]
            pos = x + (j - .5) * .32
            ax.bar(pos, values, width=.28, color=COLORS[method], zorder=3,
                   yerr=None if key == 'success_rate' else errors,
                   error_kw={'ecolor': '#374553', 'capsize': 4, 'elinewidth': 1.1})
            upper = max(upper, max(v + e for v, e in zip(values, errors)))
            for p, v, e in zip(pos, values, errors):
                label = f'{v:.1f}%' if scale == 100 else f'{v:.2f}'
                ax.annotate(label, (p, v + e), xytext=(0, 6), textcoords='offset points',
                            ha='center', fontsize=10, fontweight='bold', color='#263B4D')
        ax.set_xticks(x, domains)
        ax.set_ylabel(unit, color='#556575')
        ax.set_ylim(0, 114 if key == 'success_rate' else max(upper * 1.25, 1))
        if key == 'success_rate':
            ax.set_yticks([0, 25, 50, 75, 100])
        ax.set_xlim(-.65, len(domains) - .35)
        ax.yaxis.grid(True, color='#DEE4EB', linewidth=.8, zorder=0)
        ax.tick_params(axis='both', length=0, pad=8)
    note = axes.flat[-1]
    note.axis('off')
    note.set_title('표본 수 및 집계 기준', loc='left', fontsize=12, fontweight='bold', pad=15)
    lines = []
    for domain in domains:
        lines.append(domain + '   성공 N / 전체 N')
        for method, label in METHODS.items():
            g = groups[domain, method]
            lines.append(f"  {label}: {g['success']} / {g['total']}  (scene {g['scenes']})")
        lines.append('')
    lines.extend(['모든 scene의 개별 시행을 합산했습니다.', '각 방법은 scene 1–4별 10회씩, 총 40회입니다.',
                  '질의확률 = 질의 발생 step / 전체 step', 'planLength = 로그에 기록된 실행 step 수',
                  '오차막대는 신뢰구간이 아닌 표준편차입니다.'])
    note.text(0, 1, '\n'.join(lines), transform=note.transAxes, va='top',
              fontsize=10, linespacing=1.65, color='#455B6D')
    fig.subplots_adjust(left=.055, right=.965, top=.835, bottom=.075, wspace=.35, hspace=.42)
    destination.mkdir(parents=True, exist_ok=True)
    for extension in ('png', 'pdf', 'svg'):
        fig.savefig(destination / f'{stem}.{extension}', dpi=200, facecolor=fig.get_facecolor())
    plt.close(fig)


def draw_individual(groups, output=OUT, all_runs=False):
    from matplotlib.backends.backend_pdf import PdfPages

    destination = output / 'individual'
    destination.mkdir(exist_ok=True)
    domains = list(DOMAINS)
    with PdfPages(destination / 'all_metrics_7_pages.pdf') as document:
        for index, (key, title, unit, scale) in enumerate(METRICS, 1):
            fig, ax = plt.subplots(figsize=(9, 7))
            fig.patch.set_facecolor('#FAFBFD')
            ax.set_facecolor('#FAFBFD')
            fig.suptitle(title.replace('성공 시 ', '') if all_runs else title, x=.1, y=.965, ha='left', fontsize=18, fontweight='bold')
            subtitle = ('전체 시행 기준' if key == 'success_rate'
                        else ('성공·실패 전체 평균 ± 표본 표준편차' if all_runs else '성공 시행 평균 ± 표본 표준편차'))
            fig.text(.1, .905, 'Tomato / Waste · 전체 scene 통합 · ' + subtitle,
                     fontsize=11, color='#556575')
            upper = 0
            for j, method in enumerate(METHODS):
                stats = [groups[d, method][key] for d in domains]
                values = [s[0] * scale for s in stats]
                errors = [s[1] * scale if s[1] is not None else 0 for s in stats]
                positions = np.arange(len(domains)) + (j - .5) * .32
                ax.bar(positions, values, width=.28, color=COLORS[method],
                       label=METHODS[method], zorder=3,
                       yerr=None if key == 'success_rate' else errors,
                       error_kw={'ecolor': '#374553', 'capsize': 5, 'elinewidth': 1.2})
                upper = max(upper, max(v + e for v, e in zip(values, errors)))
                for p, v, e in zip(positions, values, errors):
                    label = f'{v:.1f}%' if scale == 100 else f'{v:.2f}'
                    ax.annotate(label, (p, v + e), xytext=(0, 8), textcoords='offset points',
                                ha='center', fontsize=13, fontweight='bold', color='#263B4D')
            ax.set_xticks(np.arange(len(domains)), domains, fontsize=13)
            ax.set_ylabel(unit, fontsize=12)
            ax.set_xlim(-.65, len(domains) - .35)
            ax.set_ylim(0, 120 if key == 'success_rate' else max(upper * 1.3, 1))
            if key == 'success_rate':
                ax.set_yticks([0, 25, 50, 75, 100])
            ax.yaxis.grid(True, color='#DEE4EB', linewidth=.8, zorder=0)
            ax.tick_params(axis='both', length=0, pad=8)
            ax.legend(loc='upper right', frameon=False, ncol=2)
            samples = []
            for domain in domains:
                counts = '  /  '.join(
                    f"{label} {groups[domain, method]['success']}/{groups[domain, method]['total']}"
                    for method, label in METHODS.items())
                samples.append(f'{domain}: {counts}')
            fig.text(.96, .105, '표본 수 (성공 N / 전체 N)\n' + '\n'.join(samples),
                     ha='right', va='top', fontsize=9, color='#556575', linespacing=1.5)
            notes = {
                'query_probability': '질의확률 = 시행별 질의 발생 step 수 / 전체 step 수',
                'steps': 'planLength = 로그에 기록된 실행 step 수',
                'planning_residual_time_s': '시행별 전체 시간 − (상호작용 시간 + 실행 시간)',
            }
            fig.text(.1, .16, notes.get(key, '각 방법은 scene 1–4별 10회씩, 총 40회'),
                     fontsize=9, color='#556575')
            fig.subplots_adjust(left=.1, right=.96, top=.85, bottom=.24)
            stem = f'{index:02d}_{key}'
            for extension in ('png', 'pdf', 'svg'):
                fig.savefig(destination / f'{stem}.{extension}', dpi=200,
                            facecolor=fig.get_facecolor())
            document.savefig(fig, facecolor=fig.get_facecolor())
            plt.close(fig)


if __name__ == '__main__':
    groups = load_data()
    save_statistics(groups)
    draw(groups, list(DOMAINS), OUT, 'all_scenes_comparison')
    for domain, folder in DOMAINS.items():
        draw(groups, [domain], folder / 'graphs', 'all_scenes_comparison')
    draw_individual(groups)
    for (domain, method), group in groups.items():
        print(domain, METHODS[method], f"success={group['success']}/{group['total']}",
              f"planLength={group['steps'][0]:.3f}")
    print('Saved figures and pooled statistics:', OUT)
