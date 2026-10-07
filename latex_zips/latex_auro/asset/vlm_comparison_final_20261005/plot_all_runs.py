"""Generate all-run tables and graphs separately from success-only outputs."""

import csv
from html import escape

import plot_comparison as plots


def save_table(groups, destination):
    headers = ['지표', '단위'] + [f'{d} / {m}' for d in plots.DOMAINS for m in plots.METHODS.values()]
    rows = [['전체 시행 수', 'N'] + [str(g['total']) for g in groups.values()],
            ['성공 / 실패', '회'] + [f"{g['success']} / {g['total'] - g['success']}" for g in groups.values()]]
    for key, title, unit, scale in plots.METRICS:
        values = []
        for group in groups.values():
            avg, sd = group[key]
            values.append(f'{avg * scale:.2f}' if sd is None else f'{avg * scale:.2f} ± {sd * scale:.2f}')
        rows.append([title.split('. ', 1)[1].replace('성공 시 ', ''), unit] + values)
    with (destination / 'summary_table.csv').open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(rows)
    caption = ('각 도메인·방법의 모든 scene을 합산. 성공률 외에는 성공·실패 전체 시행의 평균 ± 표본 표준편차(ddof=1). '
               '질의확률은 시행별 질의 발생 step 수 / 전체 step 수의 평균. '
               '실패 시행의 시간과 planLength는 종료 시점까지 기록된 값. '
               '표본 수 및 scene 구성 차이는 그대로 유지.')
    html = '<!doctype html><html lang="ko"><meta charset="utf-8"><title>전체 시행 집계표</title>'
    html += '<style>body{font:16px sans-serif;margin:40px;color:#263b4d}table{border-collapse:collapse}th,td{padding:14px;border:1px solid #dce2e8;text-align:right}th,td:first-child{text-align:left}th{background:#edf3f8}p{max-width:1100px;line-height:1.8}</style>'
    html += '<h1>Tomato / Waste — 성공·실패 전체 시행 집계</h1><p>' + escape(caption) + '</p><table><thead><tr>'
    html += ''.join('<th>' + escape(c) + '</th>' for c in headers) + '</tr></thead><tbody>'
    html += ''.join('<tr>' + ''.join('<td>' + escape(c) + '</td>' for c in row) + '</tr>' for row in rows)
    html += '</tbody></table></html>'
    (destination / 'summary_table.html').write_text(html, encoding='utf-8')
    fig, ax = plots.plt.subplots(figsize=(18, 6.5))
    ax.axis('off')
    fig.suptitle('Tomato / Waste | 성공·실패 전체 시행 집계', fontsize=20, y=.96)
    table = ax.table(cellText=rows, colLabels=headers, loc='center', cellLoc='center',
                     colWidths=[.30, .05, .1625, .1625, .1625, .1625])
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2.1)
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor('#DCE2E8')
        if row == 0:
            cell.set_facecolor('#EDF3F8')
            cell.set_text_props(weight='bold')
    fig.text(.04, .07, '성공률 외: 전체 시행 평균 ± 표본 표준편차. 질의확률의 표준편차 단위: %p.', fontsize=11)
    fig.text(.04, .03, '실패 시행은 종료 시점까지의 기록값을 포함하며, 각 방법은 scene 1–4별 10회씩, 총 40회입니다.', fontsize=11)
    fig.subplots_adjust(left=.025, right=.975, top=.88, bottom=.14)
    for extension in ('png', 'pdf'):
        fig.savefig(destination / f'summary_table.{extension}', dpi=200)
    plots.plt.close(fig)


if __name__ == '__main__':
    destination = plots.OUT / 'all_runs'
    groups = plots.load_data(all_runs=True)
    plots.save_statistics(groups, destination, all_runs=True)
    plots.draw(groups, list(plots.DOMAINS), destination, 'all_scenes_comparison', all_runs=True)
    plots.draw_individual(groups, output=destination, all_runs=True)
    save_table(groups, destination)
    print('Saved all-run tables and figures:', destination)
