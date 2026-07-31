#python calculate_psnr_ssim.py

import os
import cv2
from utils.metrics import calculate_psnr, calculate_ssim

# Sample script to calculate PSNR and SSIM metrics from saved images in two directories
# using calculate_psnr and calculate_ssim functions from: https://github.com/JingyunLiang/SwinIR
# Supports:
# - If same number of GT and results: one-to-one matching by order
# - If only 1 GT and multiple results: all results match to the single GT

gt_path = '/Data2/huxiaowan/ysq/RefineFlow-MAR/test_data/gtdata'
results_path = '/Data2/huxiaowan/ysq/RefineFlow-MAR/test_results_dps_rf'

imgsName = sorted(os.listdir(results_path))
gtsName = sorted(os.listdir(gt_path))

num_results = len(imgsName)
num_gts = len(gtsName)

print(f"Found {num_results} result images and {num_gts} GT images")

# Determine matching strategy
if num_gts == num_results:
    # One-to-one matching by order
    print("Mode: One-to-one matching (same number of images)\n")
    pairs = [(imgsName[i], gtsName[i]) for i in range(num_results)]
elif num_gts == 1:
    # All results match to single GT
    print(f"Mode: Many-to-one matching ({num_results} results to 1 GT)\n")
    pairs = [(imgsName[i], gtsName[0]) for i in range(num_results)]
else:
    print(f"Error: Cannot match {num_results} results to {num_gts} GTs")
    print("Only supports:")
    print("  - Same number of results and GTs (one-to-one)")
    print("  - Multiple results to 1 GT (many-to-one)")
    exit(1)

cumulative_psnr, cumulative_ssim = 0, 0
total_comparisons = 0

# Group by GT for better output organization
gt_groups = {}
for result_name, gt_name in pairs:
    if gt_name not in gt_groups:
        gt_groups[gt_name] = []
    gt_groups[gt_name].append(result_name)

for gt_name, result_names in sorted(gt_groups.items()):
    gt = cv2.imread(os.path.join(gt_path, gt_name), cv2.IMREAD_COLOR)
    
    if gt is None:
        print(f"Error: Could not read GT image {gt_name}")
        continue
    
    if len(result_names) > 1:
        print(f'\n=== Ground Truth: {gt_name} ({len(result_names)} results) ===')
    
    gt_psnr_sum, gt_ssim_sum = 0, 0
    
    for result_name in result_names:
        res = cv2.imread(os.path.join(results_path, result_name), cv2.IMREAD_COLOR)
        
        if res is None:
            print(f"Error: Could not read result image {result_name}")
            continue
        
        cur_psnr = calculate_psnr(res, gt, test_y_channel=True)
        cur_ssim = calculate_ssim(res, gt, test_y_channel=True)
        
        if len(result_names) > 1:
            print(f'  {result_name}: PSNR = {cur_psnr:.4f}, SSIM = {cur_ssim:.4f}')
        else:
            print(f'{result_name} <-> {gt_name}: PSNR = {cur_psnr:.4f}, SSIM = {cur_ssim:.4f}')
        
        cumulative_psnr += cur_psnr
        cumulative_ssim += cur_ssim
        gt_psnr_sum += cur_psnr
        gt_ssim_sum += cur_ssim
        total_comparisons += 1
    
    if len(result_names) > 1:
        print(f'  Average for this GT: PSNR = {gt_psnr_sum/len(result_names):.4f}, SSIM = {gt_ssim_sum/len(result_names):.4f}')

if total_comparisons > 0:
    print('\n' + '='*60)
    print('Overall Results:')
    print(f'  Total comparisons: {total_comparisons}')
    print(f'  Average PSNR: {cumulative_psnr / total_comparisons:.4f}')
    print(f'  Average SSIM: {cumulative_ssim / total_comparisons:.4f}')
    print(f'  Results path: {results_path}')
    print('='*60)
else:
    print("\nNo valid comparisons were made!")
print(results_path)
