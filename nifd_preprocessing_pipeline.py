#!/usr/bin/env python3
"""
NIFD Dataset Preprocessing Pipeline for Alzheimer's Classification
================================================================

This pipeline processes the NIFD dataset containing:
- T1 MPRAGE scans (primary for analysis)
- T2 FLAIR scans (for white matter lesions)
- T2 SPC scans (PD-weighted alternative)

The pipeline performs:
1. DICOM to NIfTI conversion
2. Quality assessment and filtering
3. Skull stripping
4. Bias field correction (N4ITK)
5. Registration to MNI space
6. Resampling to 1mm³
7. Cropping/padding to 128×128×128
8. Intensity normalization
9. Optional denoising
10. Data augmentation (training time)
11. Group-wise train/val/test split

Author: AI Assistant
Date: 2025
"""

import os
import re
import json
import shutil
import warnings
import subprocess
from pathlib import Path
from typing import Dict, Tuple, Optional, List, Union
from dataclasses import dataclass

import numpy as np
import pydicom
import SimpleITK as sitk
from sklearn.model_selection import GroupShuffleSplit

# Optional dependencies
try:
    from monai.transforms import (
        Compose, RandFlipd, RandAffined, RandGaussianNoised, RandElasticd
    )
    MONAI_AVAILABLE = True
except ImportError:
    MONAI_AVAILABLE = False
    warnings.warn("MONAI not available; data augmentations will be skipped.")

@dataclass
class ScanInfo:
    """Information about a scan"""
    subject_id: str
    scan_type: str  # 't1', 't2_flair', 't2_spc'
    session_date: str
    series_path: Path
    dicom_count: int
    series_description: str
    sequence_name: str
    echo_time: float
    repetition_time: float
    inversion_time: Optional[float]

class NIFDPreprocessor:
    """Main preprocessing class for NIFD dataset"""
    
    def __init__(self, 
                 input_root: Union[str, Path],
                 output_root: Union[str, Path],
                 mni_template: Union[str, Path],
                 target_spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
                 target_size: Tuple[int, int, int] = (128, 128, 128),
                 do_denoise: bool = False,
                 denoise_method: str = "gaussian",
                 prefer_hdbet: bool = True):
        
        self.input_root = Path(input_root)
        self.output_root = Path(output_root)
        self.mni_template = Path(mni_template)
        self.target_spacing = target_spacing
        self.target_size = target_size
        self.do_denoise = do_denoise
        self.denoise_method = denoise_method
        self.prefer_hdbet = prefer_hdbet
        
        # Create output directories
        self.ensure_dir(self.output_root)
        self.ensure_dir(self.output_root / "01_nifti")
        self.ensure_dir(self.output_root / "02_processed")
        self.ensure_dir(self.output_root / "03_final")
        
        # Load MNI template
        self.template = self.read_image(self.mni_template)
        
        # Scan type priorities (for subjects with multiple scans)
        self.scan_priorities = {
            't1_mprage': 1,      # Highest priority
            't2_flair': 2,       # Second priority
            't2_spc_ns_sag_p2_iso': 3,  # Third priority
        }
        
        # Results tracking
        self.manifest = {
            "kept": [],
            "dropped": [],
            "qc_stats": {},
            "scan_info": {},
            "processing_log": []
        }
    
    def ensure_dir(self, path: Path):
        """Create directory if it doesn't exist"""
        path.mkdir(parents=True, exist_ok=True)
    
    def read_image(self, path: Path) -> sitk.Image:
        """Read image using SimpleITK"""
        return sitk.ReadImage(str(path))
    
    def write_image(self, img: sitk.Image, path: Path):
        """Write image using SimpleITK"""
        self.ensure_dir(path.parent)
        sitk.WriteImage(img, str(path), useCompression=True)
    
    def log(self, message: str):
        """Log processing message"""
        print(f"[NIFD] {message}")
        self.manifest["processing_log"].append(message)
    
    def analyze_dataset_structure(self) -> List[ScanInfo]:
        """Analyze the NIFD dataset structure and extract scan information"""
        self.log("Analyzing dataset structure...")
        scan_info_list = []
        
        for subject_dir in self.input_root.iterdir():
            if not subject_dir.is_dir() or not subject_dir.name.startswith(('1_S_', '2_S_', '3_S_')):
                continue
                
            subject_id = subject_dir.name
            self.log(f"Processing subject: {subject_id}")
            
            # Find all scan types for this subject
            for scan_dir in subject_dir.iterdir():
                if not scan_dir.is_dir():
                    continue
                    
                scan_type = scan_dir.name
                if scan_type not in self.scan_priorities:
                    continue
                
                # Find the most recent session (highest priority)
                sessions = []
                for session_dir in scan_dir.iterdir():
                    if session_dir.is_dir() and session_dir.name.count('_') >= 2:
                        sessions.append(session_dir)
                
                if not sessions:
                    continue
                
                # Sort by date (assuming YYYY-MM-DD format)
                sessions.sort(key=lambda x: x.name.split('_')[0], reverse=True)
                latest_session = sessions[0]
                session_date = latest_session.name.split('_')[0]
                
                # Find DICOM series
                for series_dir in latest_session.iterdir():
                    if series_dir.is_dir() and series_dir.name.startswith('I'):
                        dicom_files = list(series_dir.glob('*.dcm'))
                        if dicom_files:
                            # Read DICOM header
                            try:
                                ds = pydicom.dcmread(str(dicom_files[0]))
                                scan_info = ScanInfo(
                                    subject_id=subject_id,
                                    scan_type=scan_type,
                                    session_date=session_date,
                                    series_path=series_dir,
                                    dicom_count=len(dicom_files),
                                    series_description=getattr(ds, 'SeriesDescription', 'Unknown'),
                                    sequence_name=getattr(ds, 'SequenceName', 'Unknown'),
                                    echo_time=float(getattr(ds, 'EchoTime', 0)),
                                    repetition_time=float(getattr(ds, 'RepetitionTime', 0)),
                                    inversion_time=float(getattr(ds, 'InversionTime', 0)) if hasattr(ds, 'InversionTime') else None
                                )
                                scan_info_list.append(scan_info)
                                self.log(f"  Found {scan_type}: {len(dicom_files)} slices, TR={scan_info.repetition_time:.1f}ms, TE={scan_info.echo_time:.1f}ms")
                                break
                            except Exception as e:
                                self.log(f"  Error reading DICOM header for {series_dir}: {e}")
                                continue
        
        self.log(f"Found {len(scan_info_list)} scans across {len(set(s.subject_id for s in scan_info_list))} subjects")
        return scan_info_list
    
    def convert_dicom_to_nifti(self, scan_info: ScanInfo) -> Optional[Path]:
        """Convert DICOM series to NIfTI format"""
        try:
            reader = sitk.ImageSeriesReader()
            series_ids = reader.GetGDCMSeriesIDs(str(scan_info.series_path))
            
            if not series_ids:
                self.log(f"  No DICOM series found in {scan_info.series_path}")
                return None
            
            # Use first series
            series_files = reader.GetGDCMSeriesFileNames(str(scan_info.series_path), series_ids[0])
            reader.SetFileNames(series_files)
            image = reader.Execute()
            
            # Save as NIfTI
            output_path = self.output_root / "01_nifti" / f"{scan_info.subject_id}_{scan_info.scan_type}_{scan_info.session_date}.nii.gz"
            self.write_image(image, output_path)
            
            self.log(f"  Converted {scan_info.scan_type} to {output_path.name}")
            return output_path
            
        except Exception as e:
            self.log(f"  Error converting {scan_info.scan_type}: {e}")
            return None
    
    def quality_check(self, image: sitk.Image, scan_info: ScanInfo) -> Tuple[bool, Dict]:
        """Perform quality assessment on the image"""
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
            "origin": tuple(float(o) for o in image.GetOrigin()),
            "direction": tuple(float(d) for d in image.GetDirection()),
        }
        
        # Quality checks
        quality_issues = []
        
        # Check for NaN or Inf values
        if stats["nan_count"] > 0:
            quality_issues.append(f"Contains {stats['nan_count']} NaN values")
        if stats["inf_count"] > 0:
            quality_issues.append(f"Contains {stats['inf_count']} Inf values")
        
        # Check dimensions
        min_dim, max_dim = 32, 512
        if any(d < min_dim or d > max_dim for d in arr.shape):
            quality_issues.append(f"Dimensions {arr.shape} outside acceptable range [{min_dim}, {max_dim}]")
        
        # Check for constant image
        if stats["std"] < 1e-6:
            quality_issues.append("Image appears to be constant (std < 1e-6)")
        
        # Check for reasonable intensity range
        if not np.isfinite(stats["min"]) or not np.isfinite(stats["max"]):
            quality_issues.append("Non-finite intensity values")
        
        # Check spacing (should be reasonable for brain imaging)
        spacing = np.array(stats["spacing"])
        if np.any(spacing < 0.1) or np.any(spacing > 10.0):
            quality_issues.append(f"Unusual spacing: {stats['spacing']}")
        
        is_good = len(quality_issues) == 0
        
        if not is_good:
            self.log(f"  Quality issues for {scan_info.scan_type}: {', '.join(quality_issues)}")
        
        return is_good, stats
    
    def skull_strip_hdbet(self, nifti_path: Path, out_path: Path) -> Optional[Path]:
        """Skull stripping using HD-BET if available"""
        try:
            cmd = [
                "hd-bet",
                "-i", str(nifti_path),
                "-o", str(out_path.parent / out_path.stem),
                "-device", "cpu",
                "-mode", "fast"
            ]
            result = subprocess.run(" ".join(cmd), check=True, shell=True, 
                                  capture_output=True, text=True)
            
            bet_path = out_path.parent / f"{out_path.stem}_bet.nii.gz"
            if bet_path.exists():
                bet_path.rename(out_path)
                return out_path
                
        except Exception as e:
            self.log(f"  HD-BET failed: {e}")
        
        return None
    
    def skull_strip_simple(self, image: sitk.Image) -> Tuple[sitk.Image, sitk.Image]:
        """Simple skull stripping using Otsu thresholding"""
        # Smooth the image first
        flt = sitk.CurvatureFlowImageFilter()
        flt.SetTimeStep(0.125)
        flt.SetNumberOfIterations(5)
        smooth = flt.Execute(sitk.Cast(image, sitk.sitkFloat32))
        
        # Otsu thresholding
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
            # Morphological closing to fill holes
            mask = sitk.BinaryMorphologicalClosing(mask, [2, 2, 2])
        
        # Apply mask
        brain = sitk.Mask(image, sitk.Cast(mask, sitk.sitkUInt8))
        
        return brain, mask
    
    def skull_strip(self, image_path: Path, out_path: Path) -> Tuple[sitk.Image, sitk.Image]:
        """Perform skull stripping"""
        if self.prefer_hdbet:
            bet_result = self.skull_strip_hdbet(image_path, out_path)
            if bet_result is not None:
                brain = self.read_image(out_path)
                # Create mask from brain > 0
                mask = sitk.BinaryThreshold(brain, lowerThreshold=1e-6, upperThreshold=1e12, 
                                          insideValue=1, outsideValue=0)
                return brain, mask
        
        # Fallback to simple method
        image = self.read_image(image_path)
        brain, mask = self.skull_strip_simple(image)
        self.write_image(brain, out_path)
        return brain, mask
    
    def n4_bias_correction(self, image: sitk.Image, mask: Optional[sitk.Image] = None) -> sitk.Image:
        """Apply N4 bias field correction"""
        img = sitk.Cast(image, sitk.sitkFloat32)
        
        if mask is None:
            # Create a simple mask
            mask = sitk.OtsuThreshold(img, 0, 1, 200)
        
        corrector = sitk.N4BiasFieldCorrectionImageFilter()
        corrected = corrector.Execute(img, mask)
        
        return corrected
    
    def register_to_template(self, moving: sitk.Image, fixed: sitk.Image) -> sitk.Image:
        """Register image to MNI template"""
        moving = sitk.Cast(moving, sitk.sitkFloat32)
        fixed = sitk.Cast(fixed, sitk.sitkFloat32)
        
        # Initial alignment
        initial_tx = sitk.CenteredTransformInitializer(
            fixed, moving, sitk.Euler3DTransform(), 
            sitk.CenteredTransformInitializerFilter.GEOMETRY
        )
        
        # Registration method
        reg = sitk.ImageRegistrationMethod()
        reg.SetMetricAsMattesMutualInformation(numberOfHistogramBins=32)
        reg.SetMetricSamplingStrategy(reg.RANDOM)
        reg.SetMetricSamplingPercentage(0.2)
        reg.SetInterpolator(sitk.sitkLinear)
        reg.SetOptimizerAsRegularStepGradientDescent(
            learningRate=2.0, minStep=1e-4, numberOfIterations=200, relaxationFactor=0.5
        )
        reg.SetOptimizerScalesFromPhysicalShift()
        
        reg.SetInitialTransform(initial_tx, inPlace=False)
        reg.SetShrinkFactorsPerLevel([4, 2, 1])
        reg.SetSmoothingSigmasPerLevel([2, 1, 0])
        
        # Rigid registration
        rigid_tx = reg.Execute(fixed, moving)
        
        # Affine refinement
        affine = sitk.AffineTransform(3)
        affine.SetMatrix(sitk.AffineTransform(rigid_tx).GetMatrix())
        affine.SetCenter(sitk.AffineTransform(rigid_tx).GetCenter())
        
        reg.SetInitialTransform(affine, inPlace=False)
        reg.SetOptimizerAsRegularStepGradientDescent(
            learningRate=1.0, minStep=1e-4, numberOfIterations=200, relaxationFactor=0.5
        )
        
        affine_tx = reg.Execute(fixed, moving)
        
        # Resample
        resampled = sitk.Resample(moving, fixed, affine_tx, sitk.sitkLinear, 0.0, moving.GetPixelID())
        
        return resampled
    
    def resample_iso(self, image: sitk.Image, spacing: Tuple[float, float, float]) -> sitk.Image:
        """Resample to isotropic spacing"""
        original_spacing = np.array(list(image.GetSpacing()), dtype=np.float64)
        original_size = np.array(list(image.GetSize()), dtype=np.int64)
        new_spacing = np.array(spacing, dtype=np.float64)
        new_size = np.round(original_size * (original_spacing / new_spacing)).astype(int).tolist()
        
        resampler = sitk.ResampleImageFilter()
        resampler.SetInterpolator(sitk.sitkBSpline)
        resampler.SetOutputSpacing(tuple(new_spacing.tolist()))
        resampler.SetSize(new_size)
        resampler.SetOutputOrigin(image.GetOrigin())
        resampler.SetOutputDirection(image.GetDirection())
        resampler.SetDefaultPixelValue(0.0)
        
        return resampler.Execute(image)
    
    def crop_or_pad(self, image: sitk.Image, size: Tuple[int, int, int]) -> sitk.Image:
        """Crop or pad image to target size"""
        current_size = image.GetSize()
        
        # Compute center crop
        start = [max(0, (c - s) // 2) for c, s in zip(current_size, size)]
        end = [min(c, st + s) for c, st, s in zip(current_size, start, size)]
        region_size = [e - st for e, st in zip(end, start)]
        
        cropped = sitk.RegionOfInterest(image, region_size, start)
        
        # Pad if needed
        after_crop_size = cropped.GetSize()
        pad_lower = [max(0, (s - ac) // 2) for s, ac in zip(size, after_crop_size)]
        pad_upper = [max(0, s - ac - pl) for s, ac, pl in zip(size, after_crop_size, pad_lower)]
        
        padded = sitk.ConstantPad(cropped, pad_lower, pad_upper, 0.0)
        
        return padded
    
    def zscore_normalize(self, image: sitk.Image, mask: Optional[sitk.Image] = None) -> sitk.Image:
        """Z-score normalization within brain mask"""
        arr = sitk.GetArrayFromImage(image).astype(np.float32)
        
        if mask is not None:
            m = sitk.GetArrayFromImage(mask).astype(bool)
        else:
            m = np.ones_like(arr, dtype=bool)
        
        vox = arr[m]
        if len(vox) == 0:
            return image
        
        mu = float(np.mean(vox))
        sd = float(np.std(vox)) if float(np.std(vox)) > 1e-6 else 1.0
        
        arr = (arr - mu) / sd
        
        out = sitk.GetImageFromArray(arr)
        out.CopyInformation(image)
        
        return out
    
    def denoise(self, image: sitk.Image, method: str = "gaussian", sigma: float = 1.0) -> sitk.Image:
        """Apply denoising"""
        method = method.lower()
        
        if method == "gaussian":
            gf = sitk.DiscreteGaussianImageFilter()
            gf.SetVariance(sigma**2)
            return gf.Execute(image)
        elif method == "curvature":
            cf = sitk.CurvatureFlowImageFilter()
            cf.SetTimeStep(0.125)
            cf.SetNumberOfIterations(5)
            return cf.Execute(sitk.Cast(image, sitk.sitkFloat32))
        else:
            self.log(f"Unknown denoise method {method}, skipping.")
            return image
    
    def process_scan(self, scan_info: ScanInfo) -> Optional[Path]:
        """Process a single scan through the entire pipeline"""
        self.log(f"Processing {scan_info.subject_id} - {scan_info.scan_type}")
        
        # 1. Convert DICOM to NIfTI
        nifti_path = self.convert_dicom_to_nifti(scan_info)
        if nifti_path is None:
            return None
        
        # 2. Quality check
        image = self.read_image(nifti_path)
        is_good, qc_stats = self.quality_check(image, scan_info)
        
        if not is_good:
            self.manifest["dropped"].append({
                "subject": scan_info.subject_id,
                "scan_type": scan_info.scan_type,
                "reason": "Quality check failed",
                "qc_stats": qc_stats
            })
            return None
        
        # Store QC stats
        self.manifest["qc_stats"][f"{scan_info.subject_id}_{scan_info.scan_type}"] = qc_stats
        
        # 3. Skull stripping
        stripped_path = self.output_root / "02_processed" / f"{scan_info.subject_id}_{scan_info.scan_type}_stripped.nii.gz"
        brain, brain_mask = self.skull_strip(nifti_path, stripped_path)
        
        # 4. N4 bias correction
        corrected = self.n4_bias_correction(brain, brain_mask)
        
        # 5. Register to MNI template
        registered = self.register_to_template(corrected, self.template)
        
        # 6. Resample to target spacing
        resampled = self.resample_iso(registered, self.target_spacing)
        
        # 7. Crop or pad to target size
        sized = self.crop_or_pad(resampled, self.target_size)
        
        # 8. Intensity normalization
        # Recompute mask in registered space
        mask_reg = sitk.BinaryThreshold(sized, 1e-6, 1e12, 1, 0)
        normalized = self.zscore_normalize(sized, mask_reg)
        
        # 9. Optional denoising
        if self.do_denoise:
            normalized = self.denoise(normalized, method=self.denoise_method, sigma=1.0)
        
        # 10. Save final result
        final_path = self.output_root / "03_final" / f"{scan_info.subject_id}_{scan_info.scan_type}_preprocessed.nii.gz"
        self.write_image(normalized, final_path)
        
        self.log(f"  Completed: {final_path.name}")
        return final_path
    
    def group_splits(self, file_paths: List[Path], test_size: float = 0.2, val_size: float = 0.1) -> Dict[str, List[str]]:
        """Create train/val/test splits ensuring no patient leakage"""
        # Extract patient IDs from file paths
        groups = []
        for path in file_paths:
            # Extract subject ID from filename (e.g., "1_S_0001_t1_mprage_2010-05-21_preprocessed.nii.gz")
            match = re.search(r'(\d+_S_\d+)_', path.name)
            if match:
                groups.append(match.group(1))
            else:
                groups.append(path.stem.split('_')[0])
        
        idx = np.arange(len(file_paths))
        
        # Train+Val vs Test split
        gss1 = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=42)
        trainval_idx, test_idx = next(gss1.split(idx, groups=groups))
        
        # Train vs Val split on remaining data
        groups_trainval = [groups[i] for i in trainval_idx]
        gss2 = GroupShuffleSplit(n_splits=1, test_size=val_size/(1-test_size), random_state=42)
        train_idx, val_idx = next(gss2.split(trainval_idx, groups=groups_trainval))
        
        paths = np.array([str(p) for p in file_paths])
        
        return {
            "train": paths[trainval_idx[train_idx]].tolist(),
            "val": paths[trainval_idx[val_idx]].tolist(),
            "test": paths[test_idx].tolist(),
        }
    
    def create_augmentations(self, keys: List[str] = ["image"]) -> Optional[object]:
        """Create data augmentation transforms"""
        if not MONAI_AVAILABLE:
            return None
        
        return Compose([
            RandFlipd(keys=keys, prob=0.5, spatial_axis=0),
            RandFlipd(keys=keys, prob=0.5, spatial_axis=1),
            RandFlipd(keys=keys, prob=0.5, spatial_axis=2),
            RandAffined(
                keys=keys, prob=0.7,
                rotate_range=(np.pi/18, np.pi/18, np.pi/18),  # ~10 degrees
                scale_range=(0.1, 0.1, 0.1),
                mode=("bilinear",)
            ),
            RandElasticd(
                keys=keys, prob=0.2, 
                sigma_range=(2, 4), 
                magnitude_range=(1, 2), 
                mode=("bilinear",)
            ),
            RandGaussianNoised(keys=keys, prob=0.15, mean=0.0, std=0.05),
        ])
    
    def run_preprocessing(self) -> Dict:
        """Run the complete preprocessing pipeline"""
        self.log("Starting NIFD preprocessing pipeline...")
        
        # 1. Analyze dataset structure
        scan_info_list = self.analyze_dataset_structure()
        
        if not scan_info_list:
            self.log("No valid scans found!")
            return self.manifest
        
        # 2. Process each scan
        processed_paths = []
        
        for scan_info in scan_info_list:
            result_path = self.process_scan(scan_info)
            if result_path:
                processed_paths.append(result_path)
                self.manifest["kept"].append({
                    "subject": scan_info.subject_id,
                    "scan_type": scan_info.scan_type,
                    "path": str(result_path),
                    "session_date": scan_info.session_date
                })
        
        self.log(f"Successfully processed {len(processed_paths)} scans")
        
        # 3. Create train/val/test splits
        if processed_paths:
            splits = self.group_splits(processed_paths)
            
            # Save splits
            splits_path = self.output_root / "splits.json"
            with open(splits_path, 'w') as f:
                json.dump(splits, f, indent=2)
            
            self.log(f"Created splits: {len(splits['train'])} train, {len(splits['val'])} val, {len(splits['test'])} test")
        
        # 4. Save manifest
        manifest_path = self.output_root / "manifest.json"
        with open(manifest_path, 'w') as f:
            json.dump(self.manifest, f, indent=2)
        
        self.log(f"Preprocessing complete! Results saved to {self.output_root}")
        
        return self.manifest

def main():
    """Main function to run the preprocessing pipeline"""
    
    # Configuration
    INPUT_ROOT = "NIFD"
    OUTPUT_ROOT = "preprocessed_nifd"
    MNI_TEMPLATE = "MNI152_T1_1mm.nii.gz"
    
    # Check if MNI template exists
    if not Path(MNI_TEMPLATE).exists():
        print(f"Error: MNI template not found at {MNI_TEMPLATE}")
        print("Please download MNI152_T1_1mm.nii.gz and place it in the current directory")
        return
    
    # Create preprocessor
    preprocessor = NIFDPreprocessor(
        input_root=INPUT_ROOT,
        output_root=OUTPUT_ROOT,
        mni_template=MNI_TEMPLATE,
        target_spacing=(1.0, 1.0, 1.0),
        target_size=(128, 128, 128),
        do_denoise=False,
        denoise_method="gaussian",
        prefer_hdbet=True
    )
    
    # Run preprocessing
    results = preprocessor.run_preprocessing()
    
    # Print summary
    print("\n" + "="*60)
    print("PREPROCESSING SUMMARY")
    print("="*60)
    print(f"Total scans processed: {len(results['kept'])}")
    print(f"Scans dropped: {len(results['dropped'])}")
    print(f"Output directory: {OUTPUT_ROOT}")
    print(f"Manifest saved to: {OUTPUT_ROOT}/manifest.json")
    print(f"Splits saved to: {OUTPUT_ROOT}/splits.json")
    
    if results['dropped']:
        print("\nDropped scans:")
        for dropped in results['dropped']:
            print(f"  - {dropped['subject']} ({dropped['scan_type']}): {dropped['reason']}")

if __name__ == "__main__":
    main()
