"""
H5 inspection and visualization tool for the DuDoDp-MAR data format.

Supported formats (based on generate_dudodp_data.py):
  - gt.h5: {'image': [512,512], float32, [0,1]}
  - N.h5:  {'ma_CT': [512,512], 'ma_sinogram': [720,721], 
            'LI_CT': [512,512], 'LI_sinogram': [720,721],
            'metal_trace': [720,721]}
  
  Image data: normalized attenuation coefficients in [0, 1]
  Sinogram: line-integral values in [0, ~4]
  Metal trace: binary mask in {0, 1}

Features:
1. Inspect H5 structure and dataset metadata.
2. Convert H5 datasets to 16-bit or 8-bit PNG with adaptive normalization.
3. Display statistics such as minimum, maximum, and mean.
4. Detect CT images, sinograms, and masks automatically.

Usage:
    python inspect_h5_data.py /Data2/huxiaowan/ysq/RefineFlow-MAR/test_data/oral_ct/test_720geo/patient_0001/000/gt.h5 --save-png
"""

import argparse
import h5py
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import cv2


def print_h5_structure(h5_file, indent=0):
    """Print the H5 file structure recursively."""
    def print_attrs(obj):
        """Print an object's attributes."""
        if obj.attrs:
            for key, val in obj.attrs.items():
                print(f"{'  ' * (indent + 1)}@{key}: {val}")
    
    print(f"{'  ' * indent}📁 File: {h5_file.filename}")
    print_attrs(h5_file)
    
    def recursive_print(name, obj):
        if isinstance(obj, h5py.Dataset):
            print(f"{'  ' * (indent + 1)}📊 Dataset: {name}")
            print(f"{'  ' * (indent + 2)}Shape: {obj.shape}")
            print(f"{'  ' * (indent + 2)}Dtype: {obj.dtype}")
            print(f"{'  ' * (indent + 2)}Size: {obj.size} elements ({obj.nbytes / 1024:.2f} KB)")
            print_attrs(obj)
        elif isinstance(obj, h5py.Group):
            print(f"{'  ' * (indent + 1)}📂 Group: {name}")
            print_attrs(obj)
    
    h5_file.visititems(recursive_print)


def analyze_dataset(data, name):
    """Analyze dataset statistics."""
    print(f"\n{'='*60}")
    print(f"📊 Dataset: {name}")
    print(f"{'='*60}")
    print(f"Shape:        {data.shape}")
    print(f"Dtype:        {data.dtype}")
    print(f"Min value:    {np.min(data):.6f}")
    print(f"Max value:    {np.max(data):.6f}")
    print(f"Mean value:   {np.mean(data):.6f}")
    print(f"Std value:    {np.std(data):.6f}")
    print(f"Median value: {np.median(data):.6f}")
    
    # Identify the data type.
    min_val = np.min(data)
    max_val = np.max(data)
    
    print(f"\n💡 Data Type Inference:")
    if 'image' in name.lower() and min_val >= -0.01 and max_val <= 1.01:
        print(f"  → CT Image (normalized attenuation): [0, 1]")
    elif 'sinogram' in name.lower() or 'sino' in name.lower():
        print(f"  → Sinogram (line integral): expected [0, ~4.0]")
    elif 'trace' in name.lower() or 'mask' in name.lower():
        unique_vals = len(np.unique(data))
        print(f"  → Binary Mask/Trace: {unique_vals} unique values")
    elif 'ct' in name.lower() and min_val >= -0.01 and max_val <= 1.01:
        print(f"  → CT Image (normalized): [0, 1]")
    
    # Print the value distribution.
    print(f"\nValue distribution:")
    percentiles = [0, 1, 5, 25, 50, 75, 95, 99, 100]
    for p in percentiles:
        val = np.percentile(data, p)
        print(f"  {p:3d}%: {val:.6f}")
    
    # Check special values.
    num_zeros = np.sum(data == 0)
    num_ones = np.sum(data == 1)
    num_nan = np.sum(np.isnan(data))
    num_inf = np.sum(np.isinf(data))
    
    if num_zeros > 0:
        print(f"\nZeros:    {num_zeros} ({num_zeros/data.size*100:.2f}%)")
    if num_ones > 0:
        print(f"Ones:     {num_ones} ({num_ones/data.size*100:.2f}%)")
        # A large number of 1.0 values may indicate clipping.
        if ('image' in name.lower() or 'ct' in name.lower()) and num_ones / data.size > 0.05:
            print(f"  ⚠️  Warning: >5% pixels are 1.0, data may be clipped (lost contrast)")
    if num_nan > 0:
        print(f"NaN:      {num_nan}")
    if num_inf > 0:
        print(f"Inf:      {num_inf}")


def normalize_for_display(data, name, bit_depth=16, keep_original_range=False):
    """Normalize data to the requested bit depth for display and saving.
    
    For medical CT data in DuDoDp format:
      - Images (image, ma_CT, LI_CT): float32 in [0, 1], preserved or normalized.
      - Sinograms: float32 in [0, ~4], normalized.
      - Metal traces: binary values in {0, 1}, scaled directly to the maximum.
    
    Args:
        data: Input float32/float64 data.
        name: Dataset name.
        bit_depth: Output bit depth, either 8 or 16.
        keep_original_range: Preserve [0,1] mapping by multiplying by max_value.
    
    Returns:
        Normalized uint8 or uint16 data.
    """
    data = data.astype(np.float64)
    
    # Remove NaN and infinity values.
    if np.any(np.isnan(data)) or np.any(np.isinf(data)):
        print(f"  ⚠️  Warning: {name} contains NaN or Inf values, replacing with 0")
        data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
    
    min_val = np.min(data)
    max_val = np.max(data)
    
    # Handle constant-valued data.
    if max_val - min_val < 1e-10:
        print(f"  ⚠️  Warning: {name} has constant values")
        if bit_depth == 16:
            return np.ones_like(data, dtype=np.uint16) * 32768
        else:
            return np.ones_like(data, dtype=np.uint8) * 128
    
    # Determine the maximum output value.
    if bit_depth == 16:
        max_output = 65535
        dtype = np.uint16
    else:
        max_output = 255
        dtype = np.uint8
    
    # Adaptive normalization strategy
    # Preserve the original mapping for data in [0, 1], such as CT images.
    if keep_original_range and min_val >= -0.01 and max_val <= 1.01:
        # Map [0, 1] directly to [0, max_output].
        normalized = np.clip(data, 0, 1) * max_output
        print(f"  📊 Mapping [0, 1] -> [0, {max_output}] (preserving original scale)")
    # Map binary data such as metal_trace directly.
    elif max_val <= 1.01 and len(np.unique(data)) <= 3:
        normalized = data * max_output
        print(f"  📊 Binary data mapping: {{0, 1}} -> {{0, {max_output}}}")
    # Use min-max normalization for other data, such as sinograms.
    else:
        normalized = (data - min_val) / (max_val - min_val) * max_output
        print(f"  📊 Min-Max normalization: [{min_val:.4f}, {max_val:.4f}] -> [0, {max_output}]")
    
    return np.clip(normalized, 0, max_output).astype(dtype)


def save_image_as_png(data, output_path, name, bit_depth=16, colormap='gray'):
    """Save data as a PNG image.
    
    Select the normalization strategy from the dataset name:
      - Images (image, CT, ct): preserve the original [0,1] range.
      - Sinograms (sinogram, sino): use min-max normalization.
      - Traces (trace, mask): use binary mapping.
    
    Args:
        data: Input data.
        output_path: Output path.
        name: Dataset name.
        bit_depth: Output bit depth, either 8 or 16.
        colormap: Colormap used by matplotlib.
    """
    # Preserve the original range for CT images.
    name_lower = name.lower()
    keep_range = ('image' in name_lower or 'ct' in name_lower or 
                  name_lower.endswith('_ct'))
    
    # Handle each supported dimensionality.
    if len(data.shape) == 2:
        # Save 2D data directly.
        img = normalize_for_display(data, name, bit_depth, keep_original_range=keep_range)
        cv2.imwrite(str(output_path), img)
        print(f"  ✅ Saved 2D image ({bit_depth}bit): {output_path}")
        
    elif len(data.shape) == 3:
        # Save the first channel or middle slice of 3D data.
        if data.shape[0] <= 3:
            # Treat the leading dimension as channels and select the first.
            img = normalize_for_display(data[0], name, bit_depth, keep_original_range=keep_range)
            cv2.imwrite(str(output_path), img)
            print(f"  ✅ Saved 3D image ({bit_depth}bit, channel 0): {output_path}")
        else:
            # Treat the leading dimension as slices and select the middle one.
            mid_slice = data.shape[0] // 2
            img = normalize_for_display(data[mid_slice], name, bit_depth, keep_original_range=keep_range)
            cv2.imwrite(str(output_path), img)
            print(f"  ✅ Saved 3D image ({bit_depth}bit, slice {mid_slice}): {output_path}")
    
    elif len(data.shape) == 4:
        # Select the first batch item and first channel from 4D data.
        img = normalize_for_display(data[0, 0], name, bit_depth, keep_original_range=keep_range)
        cv2.imwrite(str(output_path), img)
        print(f"  ✅ Saved 4D image ({bit_depth}bit, batch 0, channel 0): {output_path}")
    
    else:
        print(f"  ⚠️  Warning: Cannot save {len(data.shape)}D data as image")


def create_comparison_plot(datasets_dict, output_path):
    """Create a comparison plot for multiple datasets."""
    num_datasets = len(datasets_dict)
    if num_datasets == 0:
        return
    
    # Calculate the subplot layout.
    cols = min(3, num_datasets)
    rows = (num_datasets + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(6*cols, 5*rows))
    if num_datasets == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    for idx, (name, data) in enumerate(datasets_dict.items()):
        ax = axes[idx]
        
        # Select a 2D slice from multidimensional data.
        if len(data.shape) == 2:
            display_data = data
        elif len(data.shape) == 3:
            display_data = data[0] if data.shape[0] <= 3 else data[data.shape[0]//2]
        elif len(data.shape) == 4:
            display_data = data[0, 0]
        else:
            display_data = data.reshape(data.shape[-2:])
        
        # Display the image.
        im = ax.imshow(display_data, cmap='gray')
        ax.set_title(f"{name}\n[{np.min(data):.3f}, {np.max(data):.3f}]", fontsize=10)
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    # Hide unused subplots.
    for idx in range(num_datasets, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✅ Saved comparison plot: {output_path}")
    plt.close()


def inspect_h5_file(h5_path, save_png=False, output_dir=None, show_plot=False, bit_depth=16):
    """Inspect an H5 file and optionally save its datasets as PNG images.
    
    Args:
        h5_path: H5 file path.
        save_png: Whether to save PNG files.
        output_dir: Output directory.
        show_plot: Whether to display images.
        bit_depth: PNG bit depth, either 8 or 16.
    """
    h5_path = Path(h5_path)
    
    if not h5_path.exists():
        print(f"❌ Error: File not found: {h5_path}")
        return
    
    print(f"\n{'='*60}")
    print(f"🔍 Inspecting H5 File: {h5_path.name}")
    print(f"{'='*60}")
    
    # Open the H5 file.
    with h5py.File(h5_path, 'r') as f:
        # Print the file structure.
        print("\n📋 File Structure:")
        print_h5_structure(f)
        
        # Collect all datasets.
        datasets_dict = {}
        
        def collect_datasets(name, obj):
            if isinstance(obj, h5py.Dataset):
                datasets_dict[name] = obj[()]
        
        f.visititems(collect_datasets)
        
        # Analyze each dataset.
        for name, data in datasets_dict.items():
            analyze_dataset(data, name)
        
        # Save PNG images.
        if save_png and output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            h5_name = h5_path.stem
            
            print(f"\n{'='*60}")
            print(f"💾 Saving images to: {output_dir}")
            print(f"{'='*60}")
            
            # Save each dataset.
            for name, data in datasets_dict.items():
                safe_name = name.replace('/', '_').replace('\\', '_')
                png_path = output_dir / f"{h5_name}_{safe_name}.png"
                save_image_as_png(data, png_path, name, bit_depth=bit_depth)
            
            # Create a comparison plot.
            comparison_path = output_dir / f"{h5_name}_comparison.png"
            create_comparison_plot(datasets_dict, comparison_path)
    
    print(f"\n{'='*60}")
    print(f"✅ Inspection complete!")
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(
        description='Inspect and visualize H5 files',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Inspect H5 metadata only
  python inspect_h5_data.py data/test.h5
  
  # Inspect and save as 16-bit PNG (default, preserves precision)
  python inspect_h5_data.py data/test.h5 --save-png
  
  # Save as 8-bit PNG for preview only
  python inspect_h5_data.py data/test.h5 --save-png --bit-depth 8
  
  # Specify an output directory
  python inspect_h5_data.py data/test.h5 --save-png --output-dir ./output
  
  # Process every H5 file in a directory
  python inspect_h5_data.py data/*.h5 --save-png --output-dir ./output
        """
    )
    
    parser.add_argument('h5_files', nargs='+', help='H5 file path(s) to inspect')
    parser.add_argument('--save-png', action='store_true', 
                        help='Save datasets as PNG images')
    parser.add_argument('--output-dir', type=str, default='./h5_inspection_output',
                        help='Output directory for PNG images (default: ./h5_inspection_output)')
    parser.add_argument('--bit-depth', type=int, choices=[8, 16], default=16,
                        help='PNG bit depth: 8 or 16 (default: 16)')
    parser.add_argument('--show-plot', action='store_true',
                        help='Show matplotlib plots (in addition to saving)')
    
    args = parser.parse_args()
    
    # Process each H5 input.
    h5_files = []
    for pattern in args.h5_files:
        pattern_path = Path(pattern)
        
        # Add an existing file directly.
        if pattern_path.exists() and pattern_path.is_file():
            h5_files.append(pattern_path)
        # Resolve wildcard patterns with glob.
        elif '*' in pattern or '?' in pattern:
            # Distinguish absolute and relative paths.
            if pattern_path.is_absolute():
                # Resolve an absolute pattern from its parent directory.
                parent = pattern_path.parent
                pattern_name = pattern_path.name
                if parent.exists():
                    h5_files.extend(parent.glob(pattern_name))
            else:
                # Resolve a relative pattern from the current directory.
                h5_files.extend(Path('.').glob(pattern))
        # Find all H5 files when the input is a directory.
        elif pattern_path.exists() and pattern_path.is_dir():
            h5_files.extend(pattern_path.glob('*.h5'))
            h5_files.extend(pattern_path.glob('**/*.h5'))
        else:
            print(f"⚠️  Warning: Pattern not found or invalid: {pattern}")
    
    if not h5_files:
        print("❌ No H5 files found!")
        return
    
    print(f"\n🔍 Found {len(h5_files)} H5 file(s) to process\n")
    
    for h5_file in h5_files:
        inspect_h5_file(
            h5_file, 
            save_png=args.save_png, 
            output_dir=args.output_dir,
            show_plot=args.show_plot,
            bit_depth=args.bit_depth
        )


if __name__ == '__main__':
    main()
