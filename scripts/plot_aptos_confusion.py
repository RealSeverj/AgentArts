"""Plot four fixed-prediction DR confusion matrices with priority cells marked."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle
import numpy as np

BASE = Path(__file__).resolve().parents[1]


def stage_label(model, fallback):
    stage = model.get('actual_stage')
    if stage == 'head_only':
        return 'Head only'
    if isinstance(stage, str) and stage.startswith('last_') and stage.endswith('_blocks'):
        return 'Last ' + stage[5:-7] + ' blocks fine-tuned'
    if stage == 'full':
        return 'Full fine-tuning'
    if 'actual_stage' in model:
        return 'Stage not recorded'
    return fallback


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path, default=BASE / 'cloud_upload/aptos_vitl_20261113/critical_errors_and_bootstrap.json')
    parser.add_argument('--output-prefix', type=Path)
    parser.add_argument('--test-source', choices=['APTOS2019', 'DDR'], default='APTOS2019',
                        help='Explicit internal test source; DDR expects 3898, APTOS2019 expects 354')
    args = parser.parse_args()
    report = json.loads(args.report.read_text(encoding='utf-8'))
    if report.get('all_checks_passed') is not True:
        raise ValueError('Plot requires the successfully audited fixed-prediction report')
    expected = {'APTOS2019': 354, 'DDR': 3898}[args.test_source]
    if (report['cohorts']['test'].get('source') != args.test_source or report['cohorts']['test'].get('n') != expected
            or report['cohorts']['external_test'].get('source') != 'IDRiD'
            or report['cohorts']['external_test'].get('n') != 506):
        raise ValueError('Plot source/count does not match the explicitly selected fixed test cohort')
    prefix = args.output_prefix or args.report.parent / 'confusion_matrices'
    prefix.parent.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11,
                         'axes.titlesize': 12, 'axes.labelsize': 11.5,
                         'svg.fonttype': 'none', 'figure.facecolor': 'white'})
    fig, axes = plt.subplots(2, 2, figsize=(13.4, 12.4))
    fig.subplots_adjust(left=.095, right=.875, bottom=.22, top=.865, wspace=.27, hspace=.46)
    red = '#c51b28'
    for row, (split, cohort_label) in enumerate([('test', args.test_source + ' test'), ('external_test', 'IDRiD external test')]):
        cohort = report['cohorts'][split]
        for col, (key, model_label) in enumerate([('new_head', 'Head only'), ('new_partial', 'Partial fine-tuning')]):
            model, ax = cohort['models'][key], axes[row, col]
            model_label = stage_label(model, model_label)
            counts = np.asarray(model['fixed_prediction_metrics']['confusion_matrix'], dtype=np.float64)
            if counts.shape != (5, 5) or np.any(counts != counts.astype(np.int64)) or int(counts.sum()) != cohort['n']:
                raise ValueError('Confusion matrix is not a complete five-grade integer count matrix')
            counts = counts.astype(np.int64)
            support = counts.sum(axis=1)
            percentages = np.divide(counts * 100., support[:, None], out=np.zeros((5, 5)), where=support[:, None] > 0)
            panel = ax.imshow(percentages, cmap='Blues', vmin=0, vmax=100, interpolation='nearest')
            critical = model['prioritized_errors']['true_3_or_4_to_0_or_1']
            if int(counts[3:, :2].sum()) != critical['count'] or int(support[3:].sum()) != critical['true_grade_denominator']:
                raise ValueError('Highlighted cells do not match reported priority error count/denominator')
            score, ci = model['qwk_bootstrap']['point_qwk'], model['qwk_bootstrap']['ci95_percentile']
            rate = critical['rate']
            rate_text = f'{rate:.2%}' if rate is not None else 'undefined'
            title = (f'{cohort_label} | {model_label}, epoch {model["checkpoint_completed_epoch"]}\n'
                     f'QWK {score:.4f} (95% CI {ci[0]:.4f}-{ci[1]:.4f})\n'
                     f'Critical: {critical["count"]}/{critical["true_grade_denominator"]} ({rate_text}) | N={cohort["n"]}')
            ax.set_title(title, pad=12, linespacing=1.45)
            ax.set_xlabel('Predicted grade', labelpad=8)
            ax.set_ylabel('True grade (row support)', labelpad=8)
            ax.set_xticks(range(5), labels=range(5))
            ax.set_yticks(range(5), labels=[f'{i}  (n={int(support[i])})' for i in range(5)])
            ax.set_xticks(np.arange(-.5, 5, 1), minor=True)
            ax.set_yticks(np.arange(-.5, 5, 1), minor=True)
            ax.grid(which='minor', color='white', linewidth=1.5)
            ax.tick_params(which='minor', bottom=False, left=False)
            ax.tick_params(which='major', length=0, pad=6)
            for true in range(5):
                for predicted in range(5):
                    priority = true in (3, 4) and predicted in (0, 1)
                    color = 'white' if percentages[true, predicted] > 55 else red if priority else '#152235'
                    ax.text(predicted, true, str(counts[true, predicted]), ha='center', va='center',
                            fontsize=14, fontweight='bold' if priority else 'medium', color=color, zorder=5)
                    if priority:
                        ax.add_patch(Rectangle((predicted - .465, true - .465), .93, .93,
                                               fill=False, edgecolor=red, linewidth=2.7, zorder=4))
            for spine in ax.spines.values():
                spine.set_color('#536273')
                spine.set_linewidth(.7)
    cbar_ax = fig.add_axes([.908, .285, .017, .465])
    colorbar = fig.colorbar(panel, cax=cbar_ax, ticks=[0, 20, 40, 60, 80, 100])
    colorbar.set_label('Within true-grade row (%)', labelpad=12)
    colorbar.outline.set_linewidth(.6)
    fig.suptitle('DR grading: fixed held-out confusion matrices', fontsize=19, fontweight='semibold', y=.972)
    fig.text(.5, .932, 'All 25 cells are retained in each panel; integer labels are actual sample counts.',
             ha='center', fontsize=11.5, color='#334155')
    fig.legend(handles=[Patch(facecolor='none', edgecolor=red, linewidth=2.5,
                              label='Red boxes: true grade 3/4 -> predicted grade 0/1 (predefined priority pattern)')],
               loc='lower center', bbox_to_anchor=(.5, .125), frameon=False, fontsize=11.5)
    source_note = ('APTOS uses the current original split; near-duplicate review has not established absolute independence.'
                   if args.test_source == 'APTOS2019' else
                   'Current training source: DDR. Supervision history: APTOS2019 -> DDR; this is not a DDR-only model.')
    notes = [
        source_note,
        '95% CIs: 5,000 whole-group bootstrap resamples. Declared groups are not established independent patients.',
        'Matrices use fixed exported predictions. Other underestimation and overestimation remain visible; no test-based tuning.'
    ]
    for y, text in zip([.096, .074, .052], notes):
        fig.text(.5, y, text, ha='center', va='center', fontsize=10.2, color='#334155')
    source_hash = hashlib.sha256(args.report.read_bytes()).hexdigest()
    description = 'Complete five-class fixed-prediction confusion matrices; red boxes mark true 3/4 to predicted 0/1. Source report SHA256: ' + source_hash
    png, svg = prefix.with_suffix('.png'), prefix.with_suffix('.svg')
    fig.savefig(png, dpi=230, bbox_inches='tight', pad_inches=.25,
                metadata={'Title': 'DR grading fixed held-out confusion matrices', 'Description': description})
    fig.savefig(svg, bbox_inches='tight', pad_inches=.25,
                metadata={'Title': 'DR grading fixed held-out confusion matrices', 'Description': description})
    plt.close(fig)
    print(json.dumps({'png': str(png.resolve()), 'svg': str(svg.resolve()), 'source_report_sha256': source_hash}))


if __name__ == '__main__':
    main()
