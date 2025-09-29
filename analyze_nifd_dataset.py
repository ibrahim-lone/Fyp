#!/usr/bin/env python3
"""
NIFD Dataset Analysis Script
============================

This script analyzes the NIFD dataset structure, scan types, and quality metrics.
It provides detailed information about the dataset before preprocessing.

Author: AI Assistant
Date: 2025
"""

import os
import json
import warnings
from pathlib import Path
from typing import Dict, List, Tuple
from dataclasses import dataclass, asdict

import numpy as np
import pydicom
import SimpleITK as sitk
import matplotlib.pyplot as plt
from collections import defaultdict, Counter

@dataclass
class ScanAnalysis:
    """Analysis results for a single scan"""
    subject_id: str
    scan_type: str
    session_date: str
    dicom_count: int
    series_description: str
    sequence_name: str
    echo_time: float
    repetition_time: float
    inversion_time: float
    image_shape: Tuple[int, int, int]
    image_spacing: Tuple[float, float, float]
    image_origin: Tuple[float, float, float]
    intensity_min: float
    intensity_max: float
    intensity_mean: float
    intensity_std: float
    has_nan: bool
    has_inf: bool
    quality_score: float  # 0-1, higher is better

class NIFDAnalyzer:
    """Analyzer for NIFD dataset"""
    
    def __init__(self, input_root: str):
        self.input_root = Path(input_root)
        self.analyses = []
        self.summary_stats = {}
    
    def analyze_dicom_header(self, dicom_path: Path) -> Dict:
        """Extract information from DICOM header"""
        try:
            ds = pydicom.dcmread(str(dicom_path))
            
            return {
                'series_description': getattr(ds, 'SeriesDescription', 'Unknown'),
                'sequence_name': getattr(ds, 'SequenceName', 'Unknown'),
                'echo_time': float(getattr(ds, 'EchoTime', 0)),
                'repetition_time': float(getattr(ds, 'RepetitionTime', 0)),
                'inversion_time': float(getattr(ds, 'InversionTime', 0)) if hasattr(ds, 'InversionTime') else None,
                'slice_thickness': float(getattr(ds, 'SliceThickness', 0)),
                'pixel_spacing': getattr(ds, 'PixelSpacing', [0, 0]),
                'rows': int(getattr(ds, 'Rows', 0)),
                'columns': int(getattr(ds, 'Columns', 0)),
                'manufacturer': getattr(ds, 'Manufacturer', 'Unknown'),
                'model': getattr(ds, 'ManufacturerModelName', 'Unknown'),
                'magnetic_field_strength': float(getattr(ds, 'MagneticFieldStrength', 0)),
            }
        except Exception as e:
            warnings.warn(f"Error reading DICOM header {dicom_path}: {e}")
            return {}
    
    def analyze_image_quality(self, image: sitk.Image) -> Dict:
        """Analyze image quality metrics"""
        arr = sitk.GetArrayFromImage(image).astype(np.float32)
        
        # Basic statistics
        stats = {
            'shape': arr.shape,
            'spacing': image.GetSpacing(),
            'origin': image.GetOrigin(),
            'direction': image.GetDirection(),
            'min': float(np.nanmin(arr)),
            'max': float(np.nanmax(arr)),
            'mean': float(np.nanmean(arr)),
            'std': float(np.nanstd(arr)),
            'has_nan': bool(np.isnan(arr).any()),
            'has_inf': bool(np.isinf(arr).any()),
            'nan_count': int(np.isnan(arr).sum()),
            'inf_count': int(np.isinf(arr).sum()),
        }
        
        # Quality score calculation
        quality_issues = 0
        max_issues = 5
        
        # Check for NaN/Inf
        if stats['has_nan']:
            quality_issues += 1
        if stats['has_inf']:
            quality_issues += 1
        
        # Check for constant image
        if stats['std'] < 1e-6:
            quality_issues += 1
        
        # Check dimensions (should be reasonable for brain imaging)
        if any(d < 32 or d > 512 for d in arr.shape):
            quality_issues += 1
        
        # Check spacing (should be reasonable)
        spacing = np.array(stats['spacing'])
        if np.any(spacing < 0.1) or np.any(spacing > 10.0):
            quality_issues += 1
        
        stats['quality_score'] = 1.0 - (quality_issues / max_issues)
        
        return stats
    
    def convert_and_analyze_scan(self, scan_dir: Path, subject_id: str, scan_type: str, session_date: str) -> ScanAnalysis:
        """Convert DICOM to image and analyze"""
        # Find DICOM files
        dicom_files = list(scan_dir.glob('*.dcm'))
        if not dicom_files:
            raise ValueError(f"No DICOM files found in {scan_dir}")
        
        # Analyze first DICOM header
        header_info = self.analyze_dicom_header(dicom_files[0])
        
        # Convert to image
        try:
            reader = sitk.ImageSeriesReader()
            series_ids = reader.GetGDCMSeriesIDs(str(scan_dir))
            
            if not series_ids:
                raise ValueError(f"No DICOM series found in {scan_dir}")
            
            series_files = reader.GetGDCMSeriesFileNames(str(scan_dir), series_ids[0])
            reader.SetFileNames(series_files)
            image = reader.Execute()
            
        except Exception as e:
            raise ValueError(f"Error converting DICOM to image: {e}")
        
        # Analyze image quality
        quality_stats = self.analyze_image_quality(image)
        
        # Create analysis result
        analysis = ScanAnalysis(
            subject_id=subject_id,
            scan_type=scan_type,
            session_date=session_date,
            dicom_count=len(dicom_files),
            series_description=header_info.get('series_description', 'Unknown'),
            sequence_name=header_info.get('sequence_name', 'Unknown'),
            echo_time=header_info.get('echo_time', 0),
            repetition_time=header_info.get('repetition_time', 0),
            inversion_time=header_info.get('inversion_time', 0),
            image_shape=quality_stats['shape'],
            image_spacing=quality_stats['spacing'],
            image_origin=quality_stats['origin'],
            intensity_min=quality_stats['min'],
            intensity_max=quality_stats['max'],
            intensity_mean=quality_stats['mean'],
            intensity_std=quality_stats['std'],
            has_nan=quality_stats['has_nan'],
            has_inf=quality_stats['has_inf'],
            quality_score=quality_stats['quality_score']
        )
        
        return analysis
    
    def analyze_dataset(self) -> List[ScanAnalysis]:
        """Analyze the entire dataset"""
        print("Analyzing NIFD dataset...")
        
        analyses = []
        scan_priorities = {
            't1_mprage': 1,
            't2_flair': 2,
            't2_spc_ns_sag_p2_iso': 3,
        }
        
        for subject_dir in self.input_root.iterdir():
            if not subject_dir.is_dir() or not subject_dir.name.startswith(('1_S_', '2_S_', '3_S_')):
                continue
            
            subject_id = subject_dir.name
            print(f"  Analyzing subject: {subject_id}")
            
            # Find all scan types for this subject
            for scan_dir in subject_dir.iterdir():
                if not scan_dir.is_dir():
                    continue
                
                scan_type = scan_dir.name
                if scan_type not in scan_priorities:
                    continue
                
                # Find the most recent session
                sessions = []
                for session_dir in scan_dir.iterdir():
                    if session_dir.is_dir() and session_dir.name.count('_') >= 2:
                        sessions.append(session_dir)
                
                if not sessions:
                    continue
                
                # Sort by date and take the latest
                sessions.sort(key=lambda x: x.name.split('_')[0], reverse=True)
                latest_session = sessions[0]
                session_date = latest_session.name.split('_')[0]
                
                # Find DICOM series
                for series_dir in latest_session.iterdir():
                    if series_dir.is_dir() and series_dir.name.startswith('I'):
                        try:
                            analysis = self.convert_and_analyze_scan(
                                series_dir, subject_id, scan_type, session_date
                            )
                            analyses.append(analysis)
                            print(f"    ✓ {scan_type}: {analysis.dicom_count} slices, "
                                  f"shape={analysis.image_shape}, "
                                  f"quality={analysis.quality_score:.2f}")
                            break
                        except Exception as e:
                            print(f"    ✗ {scan_type}: Error - {e}")
                            continue
        
        self.analyses = analyses
        return analyses
    
    def generate_summary_statistics(self) -> Dict:
        """Generate summary statistics"""
        if not self.analyses:
            return {}
        
        # Basic counts
        total_scans = len(self.analyses)
        subjects = list(set(a.subject_id for a in self.analyses))
        scan_types = [a.scan_type for a in self.analyses]
        
        # Quality statistics
        quality_scores = [a.quality_score for a in self.analyses]
        good_quality = sum(1 for q in quality_scores if q >= 0.8)
        poor_quality = sum(1 for q in quality_scores if q < 0.6)
        
        # Scan type distribution
        scan_type_counts = Counter(scan_types)
        
        # Image dimension statistics
        shapes = [a.image_shape for a in self.analyses]
        spacings = [a.image_spacing for a in self.analyses]
        
        # Intensity statistics by scan type
        intensity_stats = defaultdict(list)
        for analysis in self.analyses:
            intensity_stats[analysis.scan_type].append({
                'mean': analysis.intensity_mean,
                'std': analysis.intensity_std,
                'min': analysis.intensity_min,
                'max': analysis.intensity_max
            })
        
        summary = {
            'total_scans': total_scans,
            'total_subjects': len(subjects),
            'scan_type_distribution': dict(scan_type_counts),
            'quality_distribution': {
                'excellent': good_quality,
                'poor': poor_quality,
                'average_quality': np.mean(quality_scores),
                'quality_std': np.std(quality_scores)
            },
            'image_dimensions': {
                'shapes': shapes,
                'spacings': spacings,
                'unique_shapes': list(set(shapes)),
                'unique_spacings': list(set(spacings))
            },
            'intensity_statistics': dict(intensity_stats),
            'subjects': subjects
        }
        
        self.summary_stats = summary
        return summary
    
    def create_visualizations(self, output_dir: str = "analysis_plots"):
        """Create visualization plots"""
        if not self.analyses:
            print("No analyses available for visualization")
            return
        
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # 1. Quality score distribution
        plt.figure(figsize=(10, 6))
        quality_scores = [a.quality_score for a in self.analyses]
        plt.hist(quality_scores, bins=20, alpha=0.7, edgecolor='black')
        plt.xlabel('Quality Score')
        plt.ylabel('Number of Scans')
        plt.title('Distribution of Scan Quality Scores')
        plt.axvline(x=0.8, color='green', linestyle='--', label='Good Quality Threshold')
        plt.axvline(x=0.6, color='red', linestyle='--', label='Poor Quality Threshold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(output_path / 'quality_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Scan type distribution
        plt.figure(figsize=(8, 6))
        scan_types = [a.scan_type for a in self.analyses]
        type_counts = Counter(scan_types)
        plt.bar(type_counts.keys(), type_counts.values(), alpha=0.7, edgecolor='black')
        plt.xlabel('Scan Type')
        plt.ylabel('Number of Scans')
        plt.title('Distribution of Scan Types')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        plt.savefig(output_path / 'scan_type_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Image dimensions
        plt.figure(figsize=(12, 8))
        shapes = [a.image_shape for a in self.analyses]
        unique_shapes = list(set(shapes))
        
        # Create subplot for each dimension
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Z dimension (slices)
        z_dims = [s[0] for s in shapes]
        axes[0, 0].hist(z_dims, bins=20, alpha=0.7, edgecolor='black')
        axes[0, 0].set_xlabel('Z Dimension (Slices)')
        axes[0, 0].set_ylabel('Count')
        axes[0, 0].set_title('Distribution of Z Dimensions')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Y dimension (height)
        y_dims = [s[1] for s in shapes]
        axes[0, 1].hist(y_dims, bins=20, alpha=0.7, edgecolor='black')
        axes[0, 1].set_xlabel('Y Dimension (Height)')
        axes[0, 1].set_ylabel('Count')
        axes[0, 1].set_title('Distribution of Y Dimensions')
        axes[0, 1].grid(True, alpha=0.3)
        
        # X dimension (width)
        x_dims = [s[2] for s in shapes]
        axes[1, 0].hist(x_dims, bins=20, alpha=0.7, edgecolor='black')
        axes[1, 0].set_xlabel('X Dimension (Width)')
        axes[1, 0].set_ylabel('Count')
        axes[1, 0].set_title('Distribution of X Dimensions')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Spacing
        spacings = [a.image_spacing for a in self.analyses]
        spacing_means = [np.mean(s) for s in spacings]
        axes[1, 1].hist(spacing_means, bins=20, alpha=0.7, edgecolor='black')
        axes[1, 1].set_xlabel('Average Spacing (mm)')
        axes[1, 1].set_ylabel('Count')
        axes[1, 1].set_title('Distribution of Average Spacing')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path / 'image_dimensions.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 4. Intensity statistics by scan type
        plt.figure(figsize=(12, 8))
        scan_types = list(set(a.scan_type for a in self.analyses))
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        for i, scan_type in enumerate(scan_types):
            type_analyses = [a for a in self.analyses if a.scan_type == scan_type]
            means = [a.intensity_mean for a in type_analyses]
            stds = [a.intensity_std for a in type_analyses]
            
            row, col = i // 2, i % 2
            if i < 4:  # Only plot if we have space
                axes[row, col].scatter(means, stds, alpha=0.7)
                axes[row, col].set_xlabel('Mean Intensity')
                axes[row, col].set_ylabel('Intensity Std')
                axes[row, col].set_title(f'{scan_type} - Intensity Statistics')
                axes[row, col].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path / 'intensity_statistics.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Visualizations saved to {output_path}")
    
    def save_analysis_report(self, output_file: str = "nifd_analysis_report.json"):
        """Save detailed analysis report"""
        if not self.analyses:
            print("No analyses available for report")
            return
        
        # Convert analyses to dictionaries
        analyses_dict = [asdict(analysis) for analysis in self.analyses]
        
        report = {
            'summary_statistics': self.summary_stats,
            'detailed_analyses': analyses_dict,
            'scan_priorities': {
                't1_mprage': 1,
                't2_flair': 2,
                't2_spc_ns_sag_p2_iso': 3,
            },
            'quality_thresholds': {
                'excellent': 0.8,
                'good': 0.6,
                'poor': 0.4
            }
        }
        
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"Analysis report saved to {output_file}")
    
    def print_summary(self):
        """Print summary to console"""
        if not self.summary_stats:
            print("No summary statistics available")
            return
        
        print("\n" + "="*60)
        print("NIFD DATASET ANALYSIS SUMMARY")
        print("="*60)
        
        print(f"Total scans analyzed: {self.summary_stats['total_scans']}")
        print(f"Total subjects: {self.summary_stats['total_subjects']}")
        
        print(f"\nScan type distribution:")
        for scan_type, count in self.summary_stats['scan_type_distribution'].items():
            print(f"  {scan_type}: {count}")
        
        print(f"\nQuality distribution:")
        qd = self.summary_stats['quality_distribution']
        print(f"  Excellent quality (≥0.8): {qd['excellent']}")
        print(f"  Poor quality (<0.6): {qd['poor']}")
        print(f"  Average quality: {qd['average_quality']:.3f}")
        print(f"  Quality std: {qd['quality_std']:.3f}")
        
        print(f"\nImage dimensions:")
        unique_shapes = self.summary_stats['image_dimensions']['unique_shapes']
        print(f"  Unique shapes: {len(unique_shapes)}")
        for shape in unique_shapes[:5]:  # Show first 5
            print(f"    {shape}")
        if len(unique_shapes) > 5:
            print(f"    ... and {len(unique_shapes) - 5} more")
        
        print(f"\nSubjects: {', '.join(self.summary_stats['subjects'])}")

def main():
    """Main function to run the analysis"""
    input_root = "NIFD"
    
    if not Path(input_root).exists():
        print(f"Error: Input directory {input_root} not found")
        return
    
    # Create analyzer
    analyzer = NIFDAnalyzer(input_root)
    
    # Run analysis
    analyses = analyzer.analyze_dataset()
    
    if not analyses:
        print("No valid scans found for analysis")
        return
    
    # Generate summary statistics
    summary = analyzer.generate_summary_statistics()
    
    # Create visualizations
    analyzer.create_visualizations()
    
    # Save report
    analyzer.save_analysis_report()
    
    # Print summary
    analyzer.print_summary()

if __name__ == "__main__":
    main()
