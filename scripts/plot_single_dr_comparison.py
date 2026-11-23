"""Plot an audited single DDR report and its optional fixed-checkpoint reference.

Input is single_model_report.json from summarize_single_dr_experiment.py.
This renderer reads statistics only: no model, images, training, or extra cohort.
Colors are percentages within each true-grade row; annotations are integer counts.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle
import numpy as np


GRADES = list(range(5))
EXPECTED_COUNT = 3898
PRIORITY_KEY = 'true_3_or_4_to_0_or_1'
RED = '#be2134'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def finite_number(value, name):
    require(isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value), f'{name} must be a finite number')
    return float(value)


def ci_text(value, percent=False):
    """Do not invent an interval if every bootstrap replicate was undefined."""
    interval = value.get('ci95_percentile') if isinstance(value, dict) else None
    if interval is None:
        return '95% CI undefined'
    require(isinstance(interval, list) and len(interval) == 2, 'Malformed percentile interval')
    low, high = (finite_number(x, 'CI endpoint') for x in interval)
    require(low <= high, 'Reversed percentile interval')
    if percent:
        return f'95% CI {low:.2%}-{high:.2%}'
    return f'95% CI {low:.4f}-{high:.4f}'


def matrix_qwk(counts):
    expected = np.outer(counts.sum(axis=1), counts.sum(axis=0)) / counts.sum()
    distance = (np.arange(5)[:, None] - np.arange(5)[None, :]) ** 2
    denominator = float((expected * distance).sum())
    return 1.0 - float((counts * distance).sum()) / denominator if denominator else None


def validate_model(record, label):
    metrics = record.get('metrics', {})
    require(metrics.get('n') == EXPECTED_COUNT, f'{label} does not cover DDR N=3898')
    counts = np.asarray(metrics.get('confusion_matrix'), dtype=np.float64)
    require(counts.shape == (5, 5) and np.isfinite(counts).all()
            and np.all(counts >= 0) and np.all(counts == np.floor(counts))
            and int(counts.sum()) == EXPECTED_COUNT,
            f'{label} needs a complete 5x5 nonnegative integer count matrix')
    counts = counts.astype(np.int64)
    support = counts.sum(axis=1)
    accuracy = finite_number(metrics.get('accuracy'), f'{label} accuracy')
    score = finite_number(metrics.get('qwk'), f'{label} QWK')
    calculated_qwk = matrix_qwk(counts)
    require(math.isclose(accuracy, float(np.trace(counts)) / EXPECTED_COUNT, rel_tol=0, abs_tol=1e-10)
            and calculated_qwk is not None
            and math.isclose(score, calculated_qwk, rel_tol=0, abs_tol=1e-10),
            f'{label} metrics disagree with its confusion matrix')
    qwk_bootstrap = record.get('qwk_bootstrap', {})
    require(math.isclose(finite_number(qwk_bootstrap.get('point_qwk'), 'Bootstrap point QWK'),
                         score, rel_tol=0, abs_tol=1e-10),
            f'{label} bootstrap point QWK differs from its metrics')
    critical = record.get('prioritized_errors', {}).get(PRIORITY_KEY, {})
    numerator, denominator = int(counts[3:, :2].sum()), int(support[3:].sum())
    require(critical.get('count') == numerator and critical.get('true_grade_denominator') == denominator,
            f'{label} highlighted errors disagree with its matrix')
    rate = critical.get('rate')
    if denominator:
        require(math.isclose(finite_number(rate, 'Prioritized error rate'), numerator / denominator,
                             rel_tol=0, abs_tol=1e-10), 'Prioritized error rate has the wrong denominator')
    else:
        require(rate is None, 'A zero denominator requires an undefined error rate')
    require(type(record.get('checkpoint_completed_epoch')) is int
            and record['checkpoint_completed_epoch'] > 0, f'{label} checkpoint epoch is missing')
    # Validate displayed intervals before rendering any output.
    ci_text(qwk_bootstrap)
    ci_text(record.get('uncertainty', {}).get('accuracy'), percent=True)
    return counts, support, accuracy, score, critical


def geometry_label(record):
    mode = record.get('preprocessing', {}).get('mode', 'stretch')
    if mode == 'fundus_crop_letterbox_v1':
        return 'Crop + aspect-preserving resize + pad'
    if mode == 'stretch':
        return 'Stretch resize'
    return 'Configured image preprocessing'


def plot_report(report, report_path, prefix):
    require(report.get('format_version') == 1 and report.get('all_checks_passed') is True,
            'Plot requires a successfully audited single-model report')
    require(report.get('source') == 'DDR' and report.get('split') == 'test'
            and report.get('n') == EXPECTED_COUNT and report.get('grade_order') == GRADES,
            'Plot requires the fixed DDR five-grade test cohort N=3898')
    require(report.get('confusion_matrix_axes') == {'rows': 'true grade', 'columns': 'predicted grade'},
            'Report confusion-matrix axes differ from the expected true rows / predicted columns')
    records = [('Current experiment', report)]
    current_manifest = report.get('verification', {}).get('manifest_sha256')
    require(isinstance(current_manifest, str) and len(current_manifest) == 64
            and all(char in '0123456789abcdef' for char in current_manifest),
            'Current report lacks its audited test manifest SHA256')
    if 'reference' in report:
        reference = report['reference']
        require(isinstance(reference, dict)
                and reference.get('verification', {}).get('manifest_sha256') == current_manifest,
                'Reference must use exactly the same audited DDR test manifest')
        records.append(('Previous reference', reference))
    checked = [validate_model(record, label) for label, record in records]
    if len(checked) == 2:
        require(np.array_equal(checked[0][1], checked[1][1]), 'Reference true-grade supports differ')
    report_path, prefix = Path(report_path).resolve(), Path(prefix).resolve()
    png, svg = prefix.with_suffix('.png'), prefix.with_suffix('.svg')
    protected = {report_path}
    protected.update(Path(item['path']).resolve() for item in report.get('input_files', []) if 'path' in item)
    require(not ({png, svg} & protected), 'Plot output cannot overwrite a report input')
    if png.exists() or svg.exists():
        raise FileExistsError('Plot output exists; choose a new --output-prefix to preserve existing results')
    png.parent.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11,
                         'axes.titlesize': 11.6, 'axes.labelsize': 11.5,
                         'svg.fonttype': 'none', 'figure.facecolor': 'white'})
    panels = len(records)
    fig, axes = plt.subplots(1, panels, figsize=(14.4 if panels == 2 else 9.2, 9.5), squeeze=False)
    fig.subplots_adjust(left=.09 if panels == 2 else .15, right=.895 if panels == 2 else .82,
                        top=.73, bottom=.29, wspace=.32)
    for index, ((label, record), values) in enumerate(zip(records, checked)):
        ax = axes[0, index]
        counts, support, accuracy, score, critical = values
        percentages = np.divide(counts * 100., support[:, None], out=np.zeros((5, 5)), where=support[:, None] > 0)
        panel = ax.imshow(percentages, cmap='Blues', vmin=0, vmax=100, interpolation='nearest')
        rate = critical['rate']
        rate_text = f'{rate:.2%}' if rate is not None else 'undefined'
        title = (f'{label} | epoch {record["checkpoint_completed_epoch"]}\n'
                 f'{geometry_label(record)}\n'
                 f'QWK {score:.4f} ({ci_text(record["qwk_bootstrap"])})\n'
                 f'Accuracy {accuracy:.2%} ({ci_text(record.get("uncertainty", {}).get("accuracy"), percent=True)})\n'
                 f'True 3/4 -> predicted 0/1: {critical["count"]}/{critical["true_grade_denominator"]} ({rate_text})')
        ax.set_title(title, pad=14, linespacing=1.5)
        ax.set_xlabel('Predicted grade (columns)', labelpad=10)
        ax.set_ylabel('True grade (rows; row support)', labelpad=10)
        ax.set_xticks(GRADES, labels=GRADES)
        ax.set_yticks(GRADES, labels=[f'{grade}  (n={int(support[grade])})' for grade in GRADES])
        ax.set_xticks(np.arange(-.5, 5, 1), minor=True)
        ax.set_yticks(np.arange(-.5, 5, 1), minor=True)
        ax.grid(which='minor', color='white', linewidth=1.5)
        ax.tick_params(which='minor', left=False, bottom=False)
        ax.tick_params(which='major', length=0, pad=7)
        for true in GRADES:
            for predicted in GRADES:
                priority = true in (3, 4) and predicted in (0, 1)
                color = 'white' if percentages[true, predicted] > 55 else RED if priority else '#18243a'
                ax.text(predicted, true, str(counts[true, predicted]), ha='center', va='center',
                        fontsize=15, fontweight='bold' if priority else 'medium', color=color, zorder=5)
                if priority:
                    ax.add_patch(Rectangle((predicted - .465, true - .465), .93, .93,
                                           fill=False, edgecolor=RED, linewidth=2.7, zorder=4))
        for spine in ax.spines.values():
            spine.set_color('#536273')
            spine.set_linewidth(.7)
    cbar_ax = fig.add_axes([.928 if panels == 2 else .87, .325, .015 if panels == 2 else .022, .35])
    colorbar = fig.colorbar(panel, cax=cbar_ax, ticks=[0, 20, 40, 60, 80, 100])
    colorbar.set_label('Within true-grade row (%)', labelpad=11)
    colorbar.outline.set_linewidth(.6)
    fig.suptitle('DR grading: fixed DDR test confusion matrices', fontsize=19, fontweight='semibold', y=.965)
    fig.text(.5, .918, 'DDR held-out test | N=3,898 | all 25 cells retained | integer labels = actual sample counts',
             ha='center', fontsize=11.5, color='#334155')
    fig.legend(handles=[Patch(facecolor='none', edgecolor=RED, linewidth=2.5,
                              label='Red boxes: true grade 3/4 -> predicted grade 0/1 (predefined priority error)')],
               loc='lower center', bbox_to_anchor=(.5, .155), frameon=False, fontsize=10.8)
    design = report.get('bootstrap_design', {})
    replicates = design.get('replicates')
    bootstrap_note = (f'95% CIs: {replicates:,} whole-connected-group percentile bootstrap resamples.'
                      if type(replicates) is int and replicates > 0 else
                      '95% CIs: whole-connected-group percentile bootstrap; see the source report for resampling details.')
    notes = [bootstrap_note,
             'Fixed exported predictions; no test-based tuning. Groups are not established independent clinical patients.',
             'Intervals exclude retraining/checkpoint-selection uncertainty. Other underestimation and overestimation remain visible.']
    if 'paired_qwk_delta' in report:
        delta = report['paired_qwk_delta']
        point = finite_number(delta.get('point_delta_current_minus_reference'), 'Paired QWK delta')
        notes.append(f'Paired QWK difference (current - reference): {point:+.4f} ({ci_text(delta)}).')
    for y, note in zip([.135, .112, .089, .066], notes):
        fig.text(.5, y, note, ha='center', va='center', fontsize=9.8 if panels == 2 else 8.3, color='#334155')
    digest = hashlib.sha256(report_path.read_bytes()).hexdigest()
    fig.text(.5, .035, 'Source report SHA256: ' + digest, ha='center', fontsize=7.5, color='#64748b')
    description = ('Five-grade fixed DDR test confusion matrices. Colors are normalized within true-grade rows; '
                   'integer annotations are sample counts. Red boxes mark true 3/4 to predicted 0/1. '
                   'Source report SHA256: ' + digest)
    try:
        fig.savefig(png, dpi=230, bbox_inches='tight', pad_inches=.24,
                    metadata={'Title': 'DR grading fixed DDR test confusion matrices', 'Description': description})
        fig.savefig(svg, bbox_inches='tight', pad_inches=.24,
                    metadata={'Title': 'DR grading fixed DDR test confusion matrices', 'Description': description})
    finally:
        plt.close(fig)
    return {'png': str(png), 'svg': str(svg), 'panels': panels, 'n': EXPECTED_COUNT,
            'source_report_sha256': digest, 'all_checks_passed': True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--output-prefix', type=Path, required=True)
    args = parser.parse_args()
    report = json.loads(args.report.read_text(encoding='utf-8-sig'))
    result = plot_report(report, args.report, args.output_prefix)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    main()
