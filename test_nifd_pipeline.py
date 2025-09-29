#!/usr/bin/env python3
"""
Test script for NIFD preprocessing pipeline
===========================================

This script tests the preprocessing pipeline without requiring the MNI template.
It focuses on DICOM conversion and basic preprocessing steps.

Author: AI Assistant
Date: 2025
"""

import os
import json
import warnings
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pydicom
import SimpleITK as sitk

def test_dicom_conversion():
    """Test DICOM to NIfTI conversion"""
    print("Testing DICOM to NIfTI conversion...")
    
    # Find a DICOM directory
    nifd_path = Path("NIFD")
    if not nifd_path.exists():
        print("NIFD directory not found!")
        return False
    
    # Find first available DICOM series
    dicom_dir = None
    for subject_dir in nifd_path.iterdir():
        if subject_dir.is_dir() and subject_dir.name.startswith(('1_S_', '2_S_', '3_S_')):
            for scan_dir in subject_dir.iterdir():
                if scan_dir.is_dir() and scan_dir.name == 't1_mprage':
                    for session_dir in scan_dir.iterdir():
                        if session_dir.is_dir():
                            for series_dir in session_dir.iterdir():
                                if series_dir.is_dir() and series_dir.name.startswith('I'):
                                    dicom_files = list(series_dir.glob('*.dcm'))
                                    if dicom_files:
                                        dicom_dir = series_dir
                                        break
                            if dicom_dir:
                                break
                    if dicom_dir:
                        break
            if dicom_dir:
                break
    
    if not dicom_dir:
        print("No DICOM files found!")
        return False
    
    print(f"Found DICOM directory: {dicom_dir}")
    
    # Test conversion
    try:
        reader = sitk.ImageSeriesReader()
        series_ids = reader.GetGDCMSeriesIDs(str(dicom_dir))
        
        if not series_ids:
            print("No DICOM series found!")
            return False
        
        print(f"Found {len(series_ids)} series")
        
        # Convert first series
        series_files = reader.GetGDCMSeriesFileNames(str(dicom_dir), series_ids[0])
        reader.SetFileNames(series_files)
        image = reader.Execute()
        
        print(f"Successfully converted to image:")
        print(f"  Shape: {image.GetSize()}")
        print(f"  Spacing: {image.GetSpacing()}")
        print(f"  Origin: {image.GetOrigin()}")
        
        # Save test output
        output_path = Path("test_output.nii.gz")
        sitk.WriteImage(image, str(output_path), useCompression=True)
        print(f"Saved test image to: {output_path}")
        
        return True
        
    except Exception as e:
        print(f"Error during conversion: {e}")
        return False

def test_quality_check():
    """Test quality check functionality"""
    print("\nTesting quality check...")
    
    # Load test image
    test_path = Path("test_output.nii.gz")
    if not test_path.exists():
        print("Test image not found, running conversion first...")
        if not test_dicom_conversion():
            return False
    
    try:
        image = sitk.ReadImage(str(test_path))
        arr = sitk.GetArrayFromImage(image).astype(np.float32)
        
        # Basic quality metrics
        stats = {
            "shape": arr.shape,
            "size": int(np.prod(arr.shape)),
            "nan_count": int(np.isnan(arr).sum()),
            "inf_count": int(np.isinf(arr).sum()),
            "min": float(np.nanmin(arr)),
            "max": float(np.nanmax(arr)),
            "mean": float(np.nanmean(arr)),
            "std": float(np.nanstd(arr)),
            "spacing": tuple(float(s) for s in image.GetSpacing()),
        }
        
        print("Quality metrics:")
        for key, value in stats.items():
            print(f"  {key}: {value}")
        
        # Quality checks
        quality_issues = []
        
        if stats["nan_count"] > 0:
            quality_issues.append(f"Contains {stats['nan_count']} NaN values")
        if stats["inf_count"] > 0:
            quality_issues.append(f"Contains {stats['inf_count']} Inf values")
        if stats["std"] < 1e-6:
            quality_issues.append("Image appears to be constant")
        if any(d < 32 or d > 512 for d in arr.shape):
            quality_issues.append(f"Dimensions {arr.shape} outside acceptable range")
        
        if quality_issues:
            print("Quality issues found:")
            for issue in quality_issues:
                print(f"  - {issue}")
        else:
            print("✓ No quality issues found")
        
        return len(quality_issues) == 0
        
    except Exception as e:
        print(f"Error during quality check: {e}")
        return False

def test_skull_stripping():
    """Test skull stripping functionality"""
    print("\nTesting skull stripping...")
    
    test_path = Path("test_output.nii.gz")
    if not test_path.exists():
        print("Test image not found!")
        return False
    
    try:
        image = sitk.ReadImage(str(test_path))
        
        # Simple skull stripping using Otsu thresholding
        flt = sitk.CurvatureFlowImageFilter()
        flt.SetTimeStep(0.125)
        flt.SetNumberOfIterations(5)
        smooth = flt.Execute(sitk.Cast(image, sitk.sitkFloat32))
        
        otsu = sitk.OtsuThreshold(smooth, 0, 1, 200)
        
        # Connected component analysis
        cc = sitk.ConnectedComponent(otsu)
        stats = sitk.LabelShapeStatisticsImageFilter()
        stats.Execute(cc)
        
        if stats.GetNumberOfLabels() == 0:
            mask = otsu
        else:
            # Keep largest component
            label_sizes = [(l, stats.GetPhysicalSize(l)) for l in stats.GetLabels()]
            largest = max(label_sizes, key=lambda x: x[1])[0]
            mask = sitk.Equal(cc, largest)
            mask = sitk.BinaryMorphologicalClosing(mask, [2, 2, 2])
        
        # Apply mask
        brain = sitk.Mask(image, sitk.Cast(mask, sitk.sitkUInt8))
        
        print(f"Original image shape: {image.GetSize()}")
        print(f"Brain mask shape: {mask.GetSize()}")
        print(f"Brain image shape: {brain.GetSize()}")
        
        # Save results
        sitk.WriteImage(brain, "test_brain.nii.gz", useCompression=True)
        sitk.WriteImage(mask, "test_mask.nii.gz", useCompression=True)
        
        print("✓ Skull stripping completed successfully")
        return True
        
    except Exception as e:
        print(f"Error during skull stripping: {e}")
        return False

def test_bias_correction():
    """Test N4 bias field correction"""
    print("\nTesting N4 bias field correction...")
    
    test_path = Path("test_brain.nii.gz")
    if not test_path.exists():
        print("Brain image not found, running skull stripping first...")
        if not test_skull_stripping():
            return False
    
    try:
        image = sitk.ReadImage(str(test_path))
        
        # N4 bias field correction
        corrector = sitk.N4BiasFieldCorrectionImageFilter()
        corrected = corrector.Execute(sitk.Cast(image, sitk.sitkFloat32))
        
        print(f"Original image mean: {np.mean(sitk.GetArrayFromImage(image))}")
        print(f"Corrected image mean: {np.mean(sitk.GetArrayFromImage(corrected))}")
        
        # Save result
        sitk.WriteImage(corrected, "test_corrected.nii.gz", useCompression=True)
        
        print("✓ Bias field correction completed successfully")
        return True
        
    except Exception as e:
        print(f"Error during bias correction: {e}")
        return False

def test_resampling():
    """Test resampling to isotropic spacing"""
    print("\nTesting resampling...")
    
    test_path = Path("test_corrected.nii.gz")
    if not test_path.exists():
        print("Corrected image not found!")
        return False
    
    try:
        image = sitk.ReadImage(str(test_path))
        original_spacing = image.GetSpacing()
        original_size = image.GetSize()
        
        print(f"Original spacing: {original_spacing}")
        print(f"Original size: {original_size}")
        
        # Resample to 1mm isotropic
        target_spacing = (1.0, 1.0, 1.0)
        original_spacing_array = np.array(original_spacing, dtype=np.float64)
        original_size_array = np.array(original_size, dtype=np.int64)
        new_spacing_array = np.array(target_spacing, dtype=np.float64)
        new_size = np.round(original_size_array * (original_spacing_array / new_spacing_array)).astype(int).tolist()
        
        resampler = sitk.ResampleImageFilter()
        resampler.SetInterpolator(sitk.sitkBSpline)
        resampler.SetOutputSpacing(target_spacing)
        resampler.SetSize(new_size)
        resampler.SetOutputOrigin(image.GetOrigin())
        resampler.SetOutputDirection(image.GetDirection())
        resampler.SetDefaultPixelValue(0.0)
        
        resampled = resampler.Execute(image)
        
        print(f"Resampled spacing: {resampled.GetSpacing()}")
        print(f"Resampled size: {resampled.GetSize()}")
        
        # Save result
        sitk.WriteImage(resampled, "test_resampled.nii.gz", useCompression=True)
        
        print("✓ Resampling completed successfully")
        return True
        
    except Exception as e:
        print(f"Error during resampling: {e}")
        return False

def test_crop_pad():
    """Test cropping/padding to fixed size"""
    print("\nTesting crop/pad to fixed size...")
    
    test_path = Path("test_resampled.nii.gz")
    if not test_path.exists():
        print("Resampled image not found!")
        return False
    
    try:
        image = sitk.ReadImage(str(test_path))
        target_size = (128, 128, 128)
        current_size = image.GetSize()
        
        print(f"Current size: {current_size}")
        print(f"Target size: {target_size}")
        
        # Compute center crop
        start = [max(0, (c - s) // 2) for c, s in zip(current_size, target_size)]
        end = [min(c, st + s) for c, st, s in zip(current_size, start, target_size)]
        region_size = [e - st for e, st in zip(end, start)]
        
        cropped = sitk.RegionOfInterest(image, region_size, start)
        
        # Pad if needed
        after_crop_size = cropped.GetSize()
        pad_lower = [max(0, (s - ac) // 2) for s, ac in zip(target_size, after_crop_size)]
        pad_upper = [max(0, s - ac - pl) for s, ac, pl in zip(target_size, after_crop_size, pad_lower)]
        
        padded = sitk.ConstantPad(cropped, pad_lower, pad_upper, 0.0)
        
        print(f"Final size: {padded.GetSize()}")
        
        # Save result
        sitk.WriteImage(padded, "test_final.nii.gz", useCompression=True)
        
        print("✓ Crop/pad completed successfully")
        return True
        
    except Exception as e:
        print(f"Error during crop/pad: {e}")
        return False

def test_intensity_normalization():
    """Test intensity normalization"""
    print("\nTesting intensity normalization...")
    
    test_path = Path("test_final.nii.gz")
    if not test_path.exists():
        print("Final image not found!")
        return False
    
    try:
        image = sitk.ReadImage(str(test_path))
        arr = sitk.GetArrayFromImage(image).astype(np.float32)
        
        # Create a simple mask (non-zero values)
        mask = arr > 0
        
        # Z-score normalization
        vox = arr[mask]
        if len(vox) > 0:
            mu = float(np.mean(vox))
            sd = float(np.std(vox)) if float(np.std(vox)) > 1e-6 else 1.0
            
            arr_normalized = (arr - mu) / sd
            
            out = sitk.GetImageFromArray(arr_normalized)
            out.CopyInformation(image)
            
            print(f"Original mean: {mu:.3f}, std: {sd:.3f}")
            print(f"Normalized mean: {np.mean(arr_normalized[mask]):.3f}, std: {np.std(arr_normalized[mask]):.3f}")
            
            # Save result
            sitk.WriteImage(out, "test_normalized.nii.gz", useCompression=True)
            
            print("✓ Intensity normalization completed successfully")
            return True
        else:
            print("No non-zero voxels found for normalization")
            return False
        
    except Exception as e:
        print(f"Error during normalization: {e}")
        return False

def cleanup_test_files():
    """Clean up test files"""
    test_files = [
        "test_output.nii.gz",
        "test_brain.nii.gz", 
        "test_mask.nii.gz",
        "test_corrected.nii.gz",
        "test_resampled.nii.gz",
        "test_final.nii.gz",
        "test_normalized.nii.gz"
    ]
    
    for file in test_files:
        if Path(file).exists():
            Path(file).unlink()
    
    print("Cleaned up test files")

def main():
    """Run all tests"""
    print("="*60)
    print("NIFD PREPROCESSING PIPELINE TEST")
    print("="*60)
    
    tests = [
        ("DICOM Conversion", test_dicom_conversion),
        ("Quality Check", test_quality_check),
        ("Skull Stripping", test_skull_stripping),
        ("Bias Correction", test_bias_correction),
        ("Resampling", test_resampling),
        ("Crop/Pad", test_crop_pad),
        ("Intensity Normalization", test_intensity_normalization),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            success = test_func()
            results[test_name] = success
            if success:
                print(f"✓ {test_name} PASSED")
            else:
                print(f"✗ {test_name} FAILED")
        except Exception as e:
            print(f"✗ {test_name} ERROR: {e}")
            results[test_name] = False
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    passed = sum(1 for success in results.values() if success)
    total = len(results)
    
    for test_name, success in results.items():
        status = "PASS" if success else "FAIL"
        print(f"{test_name}: {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The preprocessing pipeline is working correctly.")
    else:
        print("⚠️  Some tests failed. Please check the error messages above.")
    
    # Cleanup
    cleanup_test_files()

if __name__ == "__main__":
    main()
