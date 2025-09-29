#!/usr/bin/env python3
"""
Simplified NIFD Preprocessing Pipeline
=====================================

This is a simplified version that works without MNI template registration.
It performs all other preprocessing steps and can be extended later.

Author: AI Assistant
Date: 2025
"""

import os
import json
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

import numpy as np
import pydicom
import SimpleITK as sitk
from sklearn.model_selection import GroupShuffleSplit

@dataclass
class ScanInfo:
    """Information about a scan"""
    subject_id: str
    scan_type: str
    session_date: str
    series_path: Path
    dicom_count: int
    series_description: str
    sequence_name: str
    echo_time: float
    repetition_time: float
    inversion_time: Optional[float]

class SimpleNIFDPreprocessor:
    """Simplified NIFD preprocessor without MNI registration"""
    
    def __init__(self, 
                 input_root: str,
                 output_root: str,
                 target_spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
                 target_size: Tuple[int, int, int] = (128, 128, 128),
                 do_denoise: bool = False):
        
        self.input_root = Path(input_root)
        self.output_root = Path(output_root)
        self.target_spacing = target_spacing
        self.target_size = target_size
        self.do_denoise = do_denoise
        
        # Create output directories
        self.ensure_dir(self.output_root)
        self.ensure_dir(self.output_root / "01_nifti")
        self.ensure_dir(self.output_root / "02_processed")
        self.ensure_dir(self.output_root / "03_final")
        
        # Scan type priorities
        self.scan_priorities = {
            't1_mprage': 1,
            't2_flair': 2,
            't2_spc_ns_sag_p2_iso': 3,
        }
        
        # Results tracking
        self.manifest = {
            "kept": [],
            "dropped": [],
            "qc_stats": {},
            "processing_log": []
        }
    
    def ensure_dir(self, path: Path):
        """Create directory if it doesn't exist"""
        path.mkdir(parents=True, exist_ok=True)
    
    def log(self, message: str):
        """Log processing message"""
        print(f"[NIFD] {message}")
        self.manifest["processing_log"].append(message)
    
    def analyze_dataset_structure(self) -> List[ScanInfo]:
        """Analyze the NIFD dataset structure"""
        self.log("Analyzing dataset structure...")
        scan_info_list = []
        
        for subject_dir in self.input_root.iterdir():
            if not subject_dir.is_dir() or not subject_dir.name.startswith(('1_S_', '2_S_', '3_S_')):
                continue
                
            subject_id = subject_dir.name
            self.log(f"Processing subject: {subject_id}")
            
            for scan_dir in subject_dir.iterdir():
                if not scan_dir.is_dir():
                    continue
                    
                scan_type = scan_dir.name
                if scan_type not in self.scan_priorities:
                    continue
                
                # Find the most recent session
                sessions = []
                for session_dir in scan_dir.iterdir():
                    if session_dir.is_dir() and session_dir.name.count('_') >= 2:
                        sessions.append(session_dir)
                
                if not sessions:
                    continue
                
                sessions.sort(key=lambda x: x.name.split('_')[0], reverse=True)
                latest_session = sessions[0]
                session_date = latest_session.name.split('_')[0]
                
                # Find DICOM series
                for series_dir in latest_session.iterdir():
                    if series_dir.is_dir() and series_dir.name.startswith('I'):
                        dicom_files = list(series_dir.glob('*.dcm'))
                        if dicom_files:
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
                                self.log(f"  Found {scan_type}: {len(dicom_files)} slices")
                                break
                            except Exception as e:
                                self.log(f"  Error reading DICOM header: {e}")
                                continue
        
        self.log(f"Found {len(scan_info_list)} scans")
        return scan_info_list
    
    def convert_dicom_to_nifti(self, scan_info: ScanInfo) -> Optional[Path]:
        """Convert DICOM series to NIfTI"""
        try:
            reader = sitk.ImageSeriesReader()
            series_ids = reader.GetGDCMSeriesIDs(str(scan_info.series_path))
            
            if not series_ids:
                return None
            
            series_files = reader.GetGDCMSeriesFileNames(str(scan_info.series_path), series_ids[0])
            reader.SetFileNames(series_files)
            image = reader.Execute()
            
            output_path = self.output_root / "01_nifti" / f"{scan_info.subject_id}_{scan_info.scan_type}_{scan_info.session_date}.nii.gz"
            self.ensure_dir(output_path.parent)
            sitk.WriteImage(image, str(output_path), useCompression=True)
            
            return output_path
            
        except Exception as e:
            self.log(f"  Error converting {scan_info.scan_type}: {e}")
            return None
    
    def quality_check(self, image: sitk.Image, scan_info: ScanInfo) -> Tuple[bool, Dict]:
        """Perform quality assessment"""
        arr = sitk.GetArrayFromImage(image).astype(np.float32)
        
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
        
        is_good = len(quality_issues) == 0
        
        if not is_good:
            self.log(f"  Quality issues: {', '.join(quality_issues)}")
        
        return is_good, stats
    
    def skull_strip_simple(self, image: sitk.Image) -> Tuple[sitk.Image, sitk.Image]:
        """Simple skull stripping using Otsu thresholding"""
        # Smooth the image
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
            mask = sitk.BinaryMorphologicalClosing(mask, [2, 2, 2])
        
        # Apply mask
        brain = sitk.Mask(image, sitk.Cast(mask, sitk.sitkUInt8))
        
        return brain, mask
    
    def n4_bias_correction(self, image: sitk.Image, mask: Optional[sitk.Image] = None) -> sitk.Image:
        """Apply N4 bias field correction"""
        img = sitk.Cast(image, sitk.sitkFloat32)
        
        if mask is None:
            mask = sitk.OtsuThreshold(img, 0, 1, 200)
        
        corrector = sitk.N4BiasFieldCorrectionImageFilter()
        corrected = corrector.Execute(img, mask)
        
        return corrected
    
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
            return image
    
    def process_scan(self, scan_info: ScanInfo) -> Optional[Path]:
        """Process a single scan through the pipeline"""
        self.log(f"Processing {scan_info.subject_id} - {scan_info.scan_type}")
        
        # 1. Convert DICOM to NIfTI
        nifti_path = self.convert_dicom_to_nifti(scan_info)
        if nifti_path is None:
            return None
        
        # 2. Quality check
        image = sitk.ReadImage(str(nifti_path))
        is_good, qc_stats = self.quality_check(image, scan_info)
        
        if not is_good:
            self.manifest["dropped"].append({
                "subject": scan_info.subject_id,
                "scan_type": scan_info.scan_type,
                "reason": "Quality check failed",
                "qc_stats": qc_stats
            })
            return None
        
        self.manifest["qc_stats"][f"{scan_info.subject_id}_{scan_info.scan_type}"] = qc_stats
        
        # 3. Skull stripping
        brain, brain_mask = self.skull_strip_simple(image)
        
        # 4. N4 bias correction
        corrected = self.n4_bias_correction(brain, brain_mask)
        
        # 5. Resample to target spacing
        resampled = self.resample_iso(corrected, self.target_spacing)
        
        # 6. Crop or pad to target size
        sized = self.crop_or_pad(resampled, self.target_size)
        
        # 7. Intensity normalization
        mask_reg = sitk.BinaryThreshold(sized, 1e-6, 1e12, 1, 0)
        normalized = self.zscore_normalize(sized, mask_reg)
        
        # 8. Optional denoising
        if self.do_denoise:
            normalized = self.denoise(normalized, method="gaussian", sigma=1.0)
        
        # 9. Save final result
        final_path = self.output_root / "03_final" / f"{scan_info.subject_id}_{scan_info.scan_type}_preprocessed.nii.gz"
        self.ensure_dir(final_path.parent)
        sitk.WriteImage(normalized, str(final_path), useCompression=True)
        
        self.log(f"  Completed: {final_path.name}")
        return final_path
    
    def group_splits(self, file_paths: List[Path], test_size: float = 0.2, val_size: float = 0.1) -> Dict[str, List[str]]:
        """Create train/val/test splits ensuring no patient leakage"""
        import re
        
        # Extract patient IDs from file paths
        groups = []
        for path in file_paths:
            match = re.search(r'(\d+_S_\d+)_', path.name)
            if match:
                groups.append(match.group(1))
            else:
                groups.append(path.stem.split('_')[0])
        
        unique_groups = list(set(groups))
        n_groups = len(unique_groups)
        
        # Handle single subject case
        if n_groups == 1:
            self.log("Only one subject found - creating single split for all scans")
            paths = [str(p) for p in file_paths]
            return {
                "train": paths,  # All scans go to training
                "val": [],       # No validation set
                "test": []       # No test set
            }
        
        # Handle two subjects case
        if n_groups == 2:
            self.log("Only two subjects found - using one for train, one for test")
            group_to_subjects = {}
            for i, group in enumerate(groups):
                if group not in group_to_subjects:
                    group_to_subjects[group] = []
                group_to_subjects[group].append(i)
            
            group_list = list(group_to_subjects.keys())
            train_indices = group_to_subjects[group_list[0]]
            test_indices = group_to_subjects[group_list[1]]
            
            paths = [str(p) for p in file_paths]
            return {
                "train": [paths[i] for i in train_indices],
                "val": [],
                "test": [paths[i] for i in test_indices]
            }
        
        # Handle three or more subjects - use proper splitting
        idx = np.arange(len(file_paths))
        
        # Train+Val vs Test split
        gss1 = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=42)
        trainval_idx, test_idx = next(gss1.split(idx, groups=groups))
        
        # Train vs Val split
        groups_trainval = [groups[i] for i in trainval_idx]
        gss2 = GroupShuffleSplit(n_splits=1, test_size=val_size/(1-test_size), random_state=42)
        train_idx, val_idx = next(gss2.split(trainval_idx, groups=groups_trainval))
        
        paths = np.array([str(p) for p in file_paths])
        
        return {
            "train": paths[trainval_idx[train_idx]].tolist(),
            "val": paths[trainval_idx[val_idx]].tolist(),
            "test": paths[test_idx].tolist(),
        }
    
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
    OUTPUT_ROOT = "preprocessed_nifd_simple"
    
    # Create preprocessor
    preprocessor = SimpleNIFDPreprocessor(
        input_root=INPUT_ROOT,
        output_root=OUTPUT_ROOT,
        target_spacing=(1.0, 1.0, 1.0),
        target_size=(128, 128, 128),
        do_denoise=False
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
    
    # Show final processed files
    final_dir = Path(OUTPUT_ROOT) / "03_final"
    if final_dir.exists():
        final_files = list(final_dir.glob("*.nii.gz"))
        print(f"\nFinal processed files ({len(final_files)}):")
        for file in final_files:
            print(f"  - {file.name}")

if __name__ == "__main__":
    main()
