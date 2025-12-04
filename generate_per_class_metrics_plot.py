import re
import os
import datetime
import matplotlib.pyplot as plt
from collections import OrderedDict

REPORT_PATH = os.path.join('evaluation_desktop_test', 'classification_report_desktop_test.txt')
OUTPUT_DIR = 'evaluation_reports'
TARGET_LINE = 0.85

CLASS_ORDER = [
    'battery','biological','brown-glass','cardboard','clothes',
    'green-glass','metal','paper','plastic','shoes','trash','white-glass'
]

row_re = re.compile(r"^\s*(?P<class>[a-zA-Z\-]+)\s+(?P<precision>\d+\.\d{4})\s+(?P<recall>\d+\.\d{4})\s+(?P<f1>\d+\.\d{4})\s+(?P<support>\d+)")


def parse_classification_report(path: str):
    precision = {}
    recall = {}
    f1 = {}
    support = {}

    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            m = row_re.match(line)
            if not m:
                continue
            cls = m.group('class')
            # Normalize class names to match expected keys
            cls = cls.strip().lower()
            p = float(m.group('precision'))
            r = float(m.group('recall'))
            f = float(m.group('f1'))
            s = int(m.group('support'))
            precision[cls] = p
            recall[cls] = r
            f1[cls] = f
            support[cls] = s

    # Order consistently
    ordered = OrderedDict()
    for cls in CLASS_ORDER:
        if cls in precision:
            ordered[cls] = {
                'precision': precision[cls],
                'recall': recall[cls],
                'f1': f1[cls],
                'support': support[cls],
            }
    return ordered


def plot_metrics(metrics: OrderedDict, target: float = TARGET_LINE):
    classes = list(metrics.keys())
    prec = [metrics[c]['precision'] for c in classes]
    rec = [metrics[c]['recall'] for c in classes]
    f1s = [metrics[c]['f1'] for c in classes]
    supp = [metrics[c]['support'] for c in classes]

    fig, axs = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Per-Class Performance Metrics', fontsize=16, fontweight='bold')

    # Precision
    axs[0,0].bar(classes, prec, color='#76b5c5')
    axs[0,0].axhline(target, color='red', linestyle='--', linewidth=1.5, label=f'{int(target*100)}% Target')
    axs[0,0].set_title('Precision by Class')
    axs[0,0].set_ylabel('Precision')
    axs[0,0].set_ylim(0,1.05)
    axs[0,0].tick_params(axis='x', labelrotation=45)
    plt.setp(axs[0,0].get_xticklabels(), ha='right')
    axs[0,0].legend(loc='lower right')

    # Recall
    axs[0,1].bar(classes, rec, color='#e07a7a')
    axs[0,1].axhline(target, color='red', linestyle='--', linewidth=1.5, label=f'{int(target*100)}% Target')
    axs[0,1].set_title('Recall by Class')
    axs[0,1].set_ylabel('Recall')
    axs[0,1].set_ylim(0,1.05)
    axs[0,1].tick_params(axis='x', labelrotation=45)
    plt.setp(axs[0,1].get_xticklabels(), ha='right')
    axs[0,1].legend(loc='lower right')

    # F1
    axs[1,0].bar(classes, f1s, color='#74c69d')
    axs[1,0].axhline(target, color='red', linestyle='--', linewidth=1.5, label=f'{int(target*100)}% Target')
    axs[1,0].set_title('F1-Score by Class')
    axs[1,0].set_ylabel('F1-Score')
    axs[1,0].set_ylim(0,1.05)
    axs[1,0].tick_params(axis='x', labelrotation=45)
    plt.setp(axs[1,0].get_xticklabels(), ha='right')
    axs[1,0].legend(loc='lower right')

    # Support (sample count)
    axs[1,1].bar(classes, supp, color='#c8a2c8')
    axs[1,1].set_title('Test Samples by Class')
    axs[1,1].set_ylabel('Sample Count')
    axs[1,1].tick_params(axis='x', labelrotation=45)
    plt.setp(axs[1,1].get_xticklabels(), ha='right')

    fig.tight_layout(rect=[0, 0.03, 1, 0.95])

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    out_path = os.path.join(OUTPUT_DIR, f'per_class_metrics_{ts}.png')
    plt.savefig(out_path, dpi=200)
    return out_path


if __name__ == '__main__':
    metrics = parse_classification_report(REPORT_PATH)
    out = plot_metrics(metrics)
    print(out)
