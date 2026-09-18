"""Benchmark de detección reproducible por clase, a varias resoluciones.

Uso:
  python benchmark.py                                  # último best.pt, imgsz 1024+1280
  python benchmark.py --weights piu_ia/yolo26_v2/weights/best.pt --sizes 1024,1280
  python benchmark.py --compare benchmarks/baseline_yolo26_v1.json

Guarda el detalle en benchmarks/<run>.json y, si hay baseline, imprime el delta.
"""

import argparse
import json
import sys
from pathlib import Path

from gpu import detect_device, find_best_weights, setup_device_env

setup_device_env()

from ultralytics import YOLO

BASE = Path(__file__).parent


def run(weights: str, sizes, data: str, device):
    model = YOLO(str(weights))
    names = model.names
    result = {'weights': str(weights), 'data': data, 'sizes': {}}
    for imgsz in sizes:
        m = model.val(data=str(BASE / data), imgsz=imgsz, device=device, plots=False, verbose=False)
        per = {}
        for idx, ci in enumerate(m.box.ap_class_index):
            per[names[int(ci)]] = {
                'mAP50': float(m.box.ap50[idx]),
                'mAP50-95': float(m.box.ap[idx].mean()),
                'P': float(m.box.p[idx]),
                'R': float(m.box.r[idx]),
            }
        result['sizes'][str(imgsz)] = {
            'mAP50': float(m.box.map50),
            'mAP50-95': float(m.box.map),
            'P': float(m.box.mp),
            'R': float(m.box.mr),
            'per_class': per,
        }
    return result


def print_result(r, baseline=None):
    for sz, v in r['sizes'].items():
        print(f"\n=== imgsz={sz}  mAP50={v['mAP50']:.4f}  mAP50-95={v['mAP50-95']:.4f}  "
              f"P={v['P']:.3f}  R={v['R']:.3f}")
        bl = baseline.get('sizes', {}).get(sz, {}).get('per_class', {}) if baseline else {}
        for c, x in v['per_class'].items():
            line = (f"  {c:<10} mAP50={x['mAP50']:.3f}  mAP50-95={x['mAP50-95']:.3f}  "
                    f"P={x['P']:.3f}  R={x['R']:.3f}")
            if c in bl:
                d = x['mAP50'] - bl[c]['mAP50']
                line += f"   (mAP50 {'+' if d >= 0 else ''}{d:.3f} vs baseline)"
            print(line)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--weights', default=None)
    ap.add_argument('--data', default='data_merged.yml')
    ap.add_argument('--sizes', default='1024,1280')
    ap.add_argument('--out', default=None, help='JSON de salida (default benchmarks/<run>.json)')
    ap.add_argument('--compare', default=None, help='JSON baseline para comparar')
    args = ap.parse_args()

    weights = args.weights or find_best_weights()
    if not weights:
        sys.exit('No hay weights: entrená primero o pasá --weights')
    sizes = [int(s) for s in args.sizes.split(',')]
    device = detect_device()

    print(f"Benchmark: {weights} | data={args.data} | device={device} | sizes={sizes}")
    result = run(weights, sizes, args.data, device)

    baseline = None
    if args.compare and Path(args.compare).exists():
        baseline = json.loads(Path(args.compare).read_text())

    print_result(result, baseline)

    name = Path(weights).parent.parent.name
    out = Path(args.out) if args.out else BASE / 'benchmarks' / f'{name}.json'
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    print(f"\nGuardado: {out}")


if __name__ == '__main__':
    main()
