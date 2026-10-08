"""Export exactly one plot per PDF for LaTeX subfigure composition."""
from pathlib import Path
import csv
import json
import math
import os

os.environ.setdefault('MPLCONFIGDIR', '/tmp/brl-paper-matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

MANUSCRIPT = Path(__file__).resolve().parents[1]
SUMMARY = MANUSCRIPT.parents[1] / 'experiments_final/04_Total_Summary/data'
FIGURES = MANUSCRIPT / 'figures'
DOMAINS = ['WasteSorting', 'TomatoHarvesting']
SLUGS = {'WasteSorting': 'waste', 'TomatoHarvesting': 'tomato'}
METHODS = ['KnowNo', 'IntroPlan', 'Query-Action', 'Ours']
COLORS = dict(zip(METHODS, ['#E59448', '#8B77B5', '#65A48B', '#287DA8']))
HATCHES = dict(zip(METHODS, ['', '//', '..', 'xx']))
METRICS = {
    'success_rate': ('success', 'Success rate (%)'),
    'questions': ('queries', 'Query count'),
    'query_probability': ('query_probability', 'Query probability (%)'),
    'steps': ('plan_length', 'Plan length'),
    'plan_time_s': ('plan_time', 'Plan time (s)'),
    'total_time_s': ('total_time', 'Total time (s)'),
    'interaction_time_s': ('interaction_time', 'Interaction time (s)'),
    'planning_residual_time_s': ('planning_time', 'Planning time (s)'),
}
MANIFEST = []


def read(path):
    with path.open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def style():
    plt.rcParams.update({
        'font.family': 'DejaVu Sans', 'font.size': 10,
        'axes.spines.top': False, 'axes.spines.right': False,
        'pdf.fonttype': 42,
    })


def upper_limit(metric, rows):
    if metric in ['success_rate', 'query_probability']:
        return 100
    if metric == 'questions':
        return 30
    if metric == 'steps':
        return 20
    top = max(float(r['mean']) + float(r.get('sample_sd') or 0) for r in rows)
    spacing = 20 if top > 100 else 10 if top > 40 else 5
    return max(spacing, spacing * math.ceil(top * 1.18 / spacing))


def save_figure(fig, filename, metric, records):
    assert len(fig.axes) == 1, filename
    fig.savefig(FIGURES / filename, bbox_inches='tight',
                metadata={'Title': filename.removesuffix('.pdf')})
    plt.close(fig)
    MANIFEST.append({'file': filename, 'axes': 1, 'metric': metric, 'records': records})


def bars(filename, metric, records, labels, colors, hatches, ymax, groups=None, compact=False, narrow=False):
    figsize = (3.25, 2.8) if compact else (6.2 if len(records) > 4 else 4.8, 2.75)
    if narrow:
        figsize = (3.25 if len(records) == 4 else 4.4, 2.8)
    fig, ax = plt.subplots(figsize=figsize)
    positions = [0, .245, .65, .895] if compact else list(range(len(records)))
    if narrow:
        positions = [i * .245 + (.10 if groups and i >= 3 else 0) for i in range(len(records))]
    bar_width = .22 if compact or narrow else .65
    value_labels = []
    for i, row in enumerate(records):
        value = float(row['mean'])
        if metric == 'success_rate':
            lo, hi = float(row['ci95_low']), float(row['ci95_high'])
            error = [[max(0, value - lo)], [max(0, hi - value)]]
            top = hi
            suffix = ''
        else:
            sd = row.get('sample_sd')
            error = float(sd) if sd not in ['', None] else None
            top = value + (error or 0)
            suffix = '*' if error is None else ''
        x = positions[i]
        ax.bar(x, value, width=bar_width, color=colors[i], edgecolor='#333333',
               linewidth=.5, hatch=hatches[i], zorder=3)
        if error is not None:
            ax.errorbar(x, value, yerr=error, fmt='none', ecolor='#333333',
                        capsize=2 if compact else 3, linewidth=.8 if compact else 1, zorder=4)
        label_offset = 4
        value_labels.append(ax.annotate(f'{value:.2f}{suffix}', (x, top), xytext=(0, label_offset),
                    textcoords='offset points', ha='center', fontsize=13 if compact or narrow else 9,
                    annotation_clip=False))
    ax.set_xticks(positions, labels)
    if not compact and not narrow:
        ax.set_ylabel(METRICS[metric][1], fontsize=10)
    if compact:
        ax.set_xlim(-.16, 1.10)
        ax.tick_params(axis='both', labelsize=14)
        ax.set_xticklabels([])
        for index, (x, label) in enumerate(zip(positions, labels)):
            ax.annotate(label, (x, -.045), xycoords=ax.get_xaxis_transform(),
                        xytext=(13 if index % 2 == 0 else -13, 0),
                        textcoords='offset points',
                        ha='right' if index % 2 == 0 else 'left', va='top',
                        fontsize=14, fontstretch='condensed', annotation_clip=False)
    if narrow:
        ax.set_xlim(-.16, positions[-1] + .16)
        ax.tick_params(axis="both", labelsize=14)
        for tick in ax.get_xticklabels():
            tick.set_fontsize(12)
            tick.set_fontstretch("condensed")
            if not groups:
                tick.set_rotation(30)
                tick.set_ha("right")
    ax.set_ylim(0, ymax)
    if metric in ['success_rate', 'query_probability']:
        ax.set_yticks(range(0, 101, 20))
    elif metric in ['questions', 'steps']:
        ax.set_yticks(range(0, ymax + 1, 5))
    ax.yaxis.grid(True, color='#dddddd', linewidth=.6)
    ax.set_axisbelow(True)
    if groups:
        for position, label in groups:
            ax.text(position, -.26, label, transform=ax.get_xaxis_transform(),
                    ha='center', va='top', fontsize=13 if compact or narrow else 10,
                    fontstretch='condensed' if compact or narrow else 'normal')
        fig.subplots_adjust(bottom=.30, left=.10 if compact or narrow else .16, right=.98, top=.90)
    else:
        fig.subplots_adjust(bottom=.30 if narrow else .19, left=.10 if narrow else .16, right=.98, top=.90)
    if compact or narrow:
        # Keep larger value labels distinct even if bar heights are similar.
        fig.canvas.draw()
        for index, label in enumerate(value_labels):
            for _ in range(12):
                renderer = fig.canvas.get_renderer()
                bounds = label.get_window_extent(renderer).expanded(1.05, 1.15)
                if not any(bounds.overlaps(previous.get_window_extent(renderer))
                           for previous in value_labels[:index]):
                    break
                dx, dy = label.get_position()
                label.set_position((dx, dy + 16))
                fig.canvas.draw()
    save_figure(fig, filename, metric, records)


def render_oracle(rows):
    style()
    lookup = {(r['domain'], r['method'], r['metric']): r for r in rows}
    for metric in ['success_rate', 'questions', 'query_probability', 'plan_time_s']:
        records = [lookup[d, m, metric] for d in DOMAINS for m in METHODS]
        ymax = upper_limit(metric, records)
        for domain in DOMAINS:
            rows = [lookup[domain, m, metric] for m in METHODS]
            filename = f'exp_oracle_{SLUGS[domain]}_{METRICS[metric][0]}.pdf'
            bars(filename, metric, rows, ['KnowNo', 'IntroPlan', 'Query-\nAction', 'Ours'],
                 [COLORS[m] for m in METHODS], [HATCHES[m] for m in METHODS], ymax, narrow=True)


def render_physical():
    rows = [r for r in read(SUMMARY / 'vlm_human_statistics.csv') if r['mode'] == 'success']
    lookup = {(r['environment'], r['domain'], r['method'], r['metric']): r for r in rows}
    methods = ['KnowNo', 'Ours']
    for environment in ['VLM', 'Human']:
        for metric in ['success_rate', 'questions', 'query_probability', 'steps',
                       'total_time_s', 'interaction_time_s', 'planning_residual_time_s']:
            records = [lookup[environment, d, m, metric] for d in DOMAINS for m in methods]
            filename = f'exp_{environment.lower()}_{METRICS[metric][0]}.pdf'
            bars(filename, metric, records, methods * 2,
                 [COLORS[m] for m in methods] * 2, [HATCHES[m] for m in methods] * 2,
                 upper_limit(metric, records), groups=[(.1225, DOMAINS[0]), (.7725, DOMAINS[1])],
                 compact=True)


def render_response_comparison():
    rows = read(SUMMARY / 'manuscript_oracle_vlm_human_statistics.csv')
    lookup = {(r['domain'], r['method'], r['environment'], r['metric']): r for r in rows}
    environments = ['Oracle', 'VLM', 'Human']
    for domain in DOMAINS:
        for metric in ['success_rate', 'questions']:
            records = [lookup[domain, m, e, metric]
                       for m in ['KnowNo', 'Ours'] for e in environments]
            filename = f'exp_response_{SLUGS[domain]}_{METRICS[metric][0]}.pdf'
            bars(filename, metric, records, environments * 2,
                 [COLORS[m] for m in ['KnowNo', 'Ours'] for _ in environments],
                 ['', '//', '..'] * 2, upper_limit(metric, records),
                 groups=[(.245, 'KnowNo'), (1.08, 'Ours')], narrow=True)


def render_ablation():
    rows = read(SUMMARY / 'ablation_unified_by_domain.csv')
    conditions = ['random', 'ours_what_only', 'cp_when', 'value_when',
                  'ours_when_only', 'value_what', 'ours']
    lookup = {(r['domain'], r['condition']): r for r in rows}
    for metric in ['success_rate', 'questions']:
        for domain, source_domain in zip(DOMAINS, ['wastesorting', 'tomato']):
            records = []
            for condition in conditions:
                source = lookup[source_domain, condition]
                row = {'domain': domain, 'method': condition,
                       'total_n': source['n_attempted'], 'success_n': source['successes']}
                if metric == 'success_rate':
                    row.update(mean=source['success_percent_errors_as_failure'],
                               ci95_low=source['success_ci_low'], ci95_high=source['success_ci_high'])
                else:
                    row.update(mean=source['questions_success_only_mean'],
                               sample_sd=source['questions_success_only_sd'])
                records.append(row)
            filename = f'exp_ablation_{SLUGS[domain]}_{METRICS[metric][0]}.pdf'
            bars(filename, metric, records, ['Fully-\nRandom', 'W1', 'W2', 'W3', 'Q1', 'Q2', 'Ours'],
                 ['#A4ADB5'] + ['#65A48B'] * 3 + ['#8B77B5'] * 2 + [COLORS['Ours']],
                 ['', '//', '..', 'xx', '//', '..', 'xx'], upper_limit(metric, records), narrow=True)


def render_threshold():
    rows = read(SUMMARY / 'manuscript_threshold_by_domain.csv')
    for metric in ['success_rate', 'questions']:
        fig, ax = plt.subplots(figsize=(4.8, 2.75))
        for domain, color, marker in zip(DOMAINS, ['#65A48B', '#287DA8'], ['o', 's']):
            records = sorted([r for r in rows if r['domain'] == domain], key=lambda r: float(r['threshold']))
            x = [float(r['threshold']) for r in records]
            y = [float(r['success_percent'] if metric == 'success_rate' else r['questions_mean']) for r in records]
            ax.plot(x, y, color=color, marker=marker, markersize=4,
                    linestyle='-' if domain == DOMAINS[0] else '--', label=domain)
        ax.axvline(.8, color='#777777', linestyle=':', linewidth=1)
        ax.set_xticks([i / 10 for i in range(11)])
        ax.set_xlabel('Confidence threshold')
        ax.set_ylabel(METRICS[metric][1])
        ax.set_ylim(0, 100 if metric == 'success_rate' else 30)
        ax.set_yticks(range(0, 101, 20) if metric == 'success_rate' else range(0, 31, 5))
        ax.grid(axis='y', color='#dddddd', linewidth=.6)
        ax.set_axisbelow(True)
        ax.legend(frameon=False, fontsize=9, loc='lower right' if metric == 'success_rate' else 'upper left')
        fig.subplots_adjust(bottom=.20, left=.16, right=.98, top=.96)
        filename = 'exp_threshold_success.pdf' if metric == 'success_rate' else 'exp_threshold_queries.pdf'
        save_figure(fig, filename, metric, rows)


def main():
    style()
    render_oracle(read(FIGURES / 'exp_oracle_baselines_data.csv'))
    render_physical()
    render_response_comparison()
    render_ablation()
    render_threshold()
    (FIGURES / 'experiment_panels_manifest.json').write_text(
        json.dumps(MANIFEST, indent=2, ensure_ascii=False) + '\n')
    print(f'Exported {len(MANIFEST)} PDFs, each containing exactly one plot.')


if __name__ == '__main__':
    main()
