"""Compute the selected glaucoma challenger's 1,000 paired draws on the training host."""
import argparse
from pathlib import Path

from scripts import screening_evaluation as ev
from scripts import summarize_glaucoma_optimization as stats


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('candidate','baseline','release','output'):p.add_argument('--'+name,required=True,type=Path)
    args=p.parse_args()
    choice=ev.read_json(args.candidate/'final_selection.json');winner=choice['selected']
    if winner!='glaucoma_vitl392_letterbox':raise ValueError('Only a validation-selected challenger requires new paired regression')
    for key,place in [(winner,args.candidate),('historical',args.baseline)]:
        r=choice['validation'][key]
        if r['val_predictions_sha256']!=stats.sha(place/'val_predictions.jsonl') or r['thresholds_sha256']!=stats.sha(place/'thresholds.json'):
            raise ValueError('Validation prediction or threshold file differs')
    pairs={s:stats.paired(s,args.baseline,args.candidate,args.release,1000,20261114) for s in stats.SPLITS}
    ev.write_json(args.output,dict(selected=winner,selection_sha256=stats.sha(args.candidate/'final_selection.json'),draws=1000,seed=20261114,splits=pairs))


if __name__=='__main__':main()
