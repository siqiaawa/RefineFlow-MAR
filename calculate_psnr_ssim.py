"""Saved-PNG PSNR/SSIM, preserving the original SwinIR/Y-channel formulas.

No-argument defaults evaluate the included example results. CLI parameters
select other GT and output directories without changing the metric protocol.
"""
import argparse
import csv
import json
from pathlib import Path

import cv2
from utils.metrics import calculate_psnr, calculate_ssim

ROOT = Path(__file__).resolve().parent
DEFAULT_GT = ROOT/'test_data/gtdata'
DEFAULT_RESULTS = ROOT/'test_results_dps_rf'
IMAGE_SUFFIXES = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}


def image_files(folder):
    folder = Path(folder)
    if not folder.is_dir():
        raise FileNotFoundError(folder)
    return sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)


def evaluate_directories(gt_path, results_path):
    results, gts = image_files(results_path), image_files(gt_path)
    print(f'Found {len(results)} result images and {len(gts)} GT images')
    if not results or not gts:
        raise ValueError('Both directories must contain readable image files.')
    if len(gts) == len(results):
        pairs = list(zip(results, gts))
        matching = 'one-to-one, sorted file order'
    elif len(gts) == 1:
        pairs = [(result, gts[0]) for result in results]
        matching = 'many-to-one, single GT'
    else:
        raise ValueError(f'Cannot match {len(results)} results to {len(gts)} GTs: use equal counts or one GT.')
    print('Mode:', matching)
    rows = []
    for result_path, gt_file in pairs:
        # The legacy read mode: even 16-bit PNGs become 8-bit BGR.
        result = cv2.imread(str(result_path), cv2.IMREAD_COLOR)
        gt = cv2.imread(str(gt_file), cv2.IMREAD_COLOR)
        if result is None or gt is None:
            raise ValueError(f'Unreadable comparison: {result_path} / {gt_file}')
        if result.shape != gt.shape:
            raise ValueError(f'Shape mismatch (no resizing): {result.shape} / {gt.shape}')
        if min(result.shape[:2]) < 11:
            raise ValueError('SSIM needs images at least 11x11 pixels.')
        psnr = float(calculate_psnr(result, gt, test_y_channel=True))
        ssim = float(calculate_ssim(result, gt, test_y_channel=True))
        rows.append({'result': result_path.name, 'gt': gt_file.name, 'psnr_db': psnr, 'ssim': ssim})
        print(f'{result_path.name} <-> {gt_file.name}: PSNR = {psnr:.4f}, SSIM = {ssim:.4f}')
    report = {'gt_path': str(Path(gt_path).resolve()), 'results_path': str(Path(results_path).resolve()),
              'matching': matching, 'comparisons': rows, 'total_comparisons': len(rows),
              'average_psnr_db': sum(r['psnr_db'] for r in rows)/len(rows),
              'average_ssim': sum(r['ssim'] for r in rows)/len(rows),
              'protocol': {'implementation': 'utils.metrics.calculate_psnr / calculate_ssim (unchanged)',
                           'read_mode': 'cv2.IMREAD_COLOR, 8-bit BGR', 'opencv_version': cv2.__version__, 'test_y_channel': True,
                           'data_range': 255, 'metal_masking': False,
                           'region': 'whole image; SSIM uses original 11x11 valid-window support',
                           'resizing': False}}
    print('\nOverall Results:')
    print(f'  Total comparisons: {len(rows)}')
    print(f'  Average PSNR: {report["average_psnr_db"]:.4f}')
    print(f'  Average SSIM: {report["average_ssim"]:.4f}')
    print('  Results path:', results_path)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gt-path', default=DEFAULT_GT, type=Path)
    parser.add_argument('--results-path', default=DEFAULT_RESULTS, type=Path)
    parser.add_argument('--output-csv', type=Path)
    parser.add_argument('--output-json', type=Path)
    args = parser.parse_args(argv)
    report = evaluate_directories(args.gt_path, args.results_path)
    if args.output_csv:
        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        with args.output_csv.open('w', encoding='utf-8-sig', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=['result', 'gt', 'psnr_db', 'ssim'])
            writer.writeheader()
            writer.writerows(report['comparisons'])
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
