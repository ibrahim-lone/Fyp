#!/usr/bin/env python3
"""
MRI Preprocessing Pipeline - Phase 2: Core Image Preprocessing
============================================================

This module implements Phase 2 of the MRI preprocessing pipeline:
4. Skull Stripping (Brain Extraction)
5. Bias Field Correction
6. Spatial Normalization to MNI Space
7. Intensity Normalization
8. Spatial Resampling & Resizing
9. Tissue Segmentation

Author: FYP Team
Date: 2025
Project: Dementia Detection using MRI from PPMI, ADNI, and OASIS datasets
"""
import time
import nibabel as nib
import os
import json
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
import nibabel as nib
from scipy import ndimage
from skimage import morphology, filters, segmentation, measure
from sklearn.mixture import GaussianMixture
import warnings
warnings.filterwarnings('ignore')

# Optional: SimpleITK for high-quality native registration
try:
    import SimpleITK as sitk
    SITK_AVAILABLE = True
    print("✅ SimpleITK available - using high-quality native registration")
except Exception:
    SITK_AVAILABLE = False
    print("⚠️ SimpleITK not available - falling back to simple resize for registration")

class MRIPreprocessorPhase2:
    def __init__(self, input_dir, output_dir, dataset_name="Unknown"):
        """
        Initialize Phase 2 preprocessing pipeline.
        
        Args:
            input_dir: Path to Phase 1 output (BIDS structure)
            output_dir: Path to Phase 2 output directory
            dataset_name: Name of the dataset (PPMI, ADNI, OASIS)
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.dataset_name = dataset_name
        
        # Create Phase 2 output directories
        self.skull_stripped_dir = self.output_dir / "04_skull_stripped"
        self.bias_corrected_dir = self.output_dir / "05_bias_corrected"
        self.normalized_dir = self.output_dir / "06_spatially_normalized"
        self.intensity_norm_dir = self.output_dir / "07_intensity_normalized"
        self.resampled_dir = self.output_dir / "08_resampled"
        self.segmented_dir = self.output_dir / "09_tissue_segmented"
        self.qa_dir = self.output_dir / "phase2_quality_assessment"
        
        for d in [self.skull_stripped_dir, self.bias_corrected_dir, self.normalized_dir,
                  self.intensity_norm_dir, self.resampled_dir, self.segmented_dir, self.qa_dir]:
            d.mkdir(parents=True, exist_ok=True)
        
        print(f"Phase 2 initialized for {dataset_name} dataset")
    
    def find_phase1_files(self):
        """Find NIfTI files from Phase 1 BIDS structure."""
        nifti_files = list(self.input_dir.rglob("*.nii.gz"))
        if not nifti_files:
            raise ValueError("No NIfTI files found from Phase 1. Run Phase 1 first.")
        return nifti_files
    
    def skull_stripping(self, nifti_files):
        """
        Step 4: Skull Stripping (Brain Extraction)
        
        Removes skull, scalp, and non-brain tissue using intensity-based methods
        and morphological operations.
        """
        print("Step 4: Skull Stripping (Brain Extraction)...")
        
        import time
        step_start = time.time()
        processed_files = []
        
        for nifti_file in tqdm(nifti_files, desc="Skull stripping"):
            try:
                # Load image
                img = nib.load(nifti_file)
                data = img.get_fdata()
                
                brain_mask = None
                # Try SynthStrip first (CPU)
                skull_path = self._try_synthstrip(nifti_file)
                if skull_path is not None and Path(skull_path).exists():
                    skull_img = nib.load(skull_path)
                    skull_data = skull_img.get_fdata()
                    brain_mask = (skull_data > 0).astype(np.float32)
                
                # Then try HD-BET (CPU enforced)
                if brain_mask is None:
                    skull_path = self._try_hdbet(nifti_file)
                    if skull_path is not None and Path(skull_path).exists():
                        skull_img = nib.load(skull_path)
                        skull_data = skull_img.get_fdata()
                        brain_mask = (skull_data > 0).astype(np.float32)
                
                # Next: template-propagated brain mask via fast affine (2 mm)
                if brain_mask is None and SITK_AVAILABLE:
                    try:
                        brain_mask = self._template_mask_skull_strip(nifti_file, data, img.affine)
                    except Exception:
                        brain_mask = None
                
                if brain_mask is None:
                    # Fallback: enhanced classical mask
                    brain_mask = self._extract_brain_mask_enhanced(data)
                
                skull_stripped_data = data * brain_mask
                
                # Create output filename
                output_filename = f"brain_{nifti_file.name}"
                output_path = self.skull_stripped_dir / output_filename
                
                # Save skull-stripped image
                skull_stripped_img = nib.Nifti1Image(skull_stripped_data, img.affine, img.header)
                nib.save(skull_stripped_img, output_path)
                
                # Save brain mask
                mask_filename = f"mask_{nifti_file.name}"
                mask_path = self.skull_stripped_dir / mask_filename
                mask_img = nib.Nifti1Image(brain_mask.astype(np.uint8), img.affine, img.header)
                nib.save(mask_img, mask_path)
                
                processed_files.append(output_path)
                # Inline quality: IoU vs fast affine-warped template mask (proxy)
                try:
                    tpl_mask = self._template_mask_skull_strip(nifti_file, data, img.affine) if SITK_AVAILABLE else None
                    if tpl_mask is not None and tpl_mask.shape == brain_mask.shape:
                        pred = brain_mask > 0
                        ref = tpl_mask > 0
                        inter = np.logical_and(pred, ref).sum()
                        union = np.logical_or(pred, ref).sum()
                        iou = float(inter) / float(union) if union > 0 else 0.0
                        skull_q = int(round(max(0.0, min(1.0, iou)) * 100))
                        print(f"✅ Skull stripped: {output_filename} | Quality: {skull_q}% (IoU={iou:.3f})")
                    else:
                        ratio = float(np.sum(brain_mask > 0) / brain_mask.size)
                        skull_q = 20 if ratio <= 0 or ratio >= 0.8 else int(max(20, min(100, 100 - abs(ratio - 0.25) * 400)))
                        print(f"✅ Skull stripped: {output_filename} | Quality: {skull_q}% (brain ratio={ratio:.3f})")
                except Exception:
                    print(f"✅ Skull stripped: {output_filename}")
                
            except Exception as e:
                print(f"❌ Error skull stripping {nifti_file}: {e}")
        
        step_dur = time.time() - step_start
        print(f"⏱️ Skull stripping time: {step_dur:.2f}s ({len(processed_files)} files)")
        return processed_files
    
    def _try_hdbet(self, nifti_file):
        try:
            import shutil, subprocess, tempfile, os, torch
            hdbet = shutil.which('hd-bet') or shutil.which('hd-bet.exe')
            if not hdbet:
                print("⚠️ HD-BET not found in PATH.")
                return None

            # Prepare temp directory and output path
            tmpdir = Path(tempfile.mkdtemp(prefix='hdbet_'))
            out_path = tmpdir / f"{nifti_file.stem}_brain.nii.gz"

            # Detect GPU availability
            use_gpu = torch.cuda.is_available()
            device_flag = []
            env = os.environ.copy()
            if use_gpu:
                print("🟢 HD-BET: GPU available — auto-detecting CUDA")
                env["CUDA_VISIBLE_DEVICES"] = "0"
            else:
                print("🟡 HD-BET: No GPU found — running on CPU")
                env["CUDA_VISIBLE_DEVICES"] = "-1"

            # Build command (no --device flag)
            cmd = [
                hdbet,
                "-i", str(nifti_file),
                "-o", str(out_path),
            ]
            print(f"🚀 Running HD-BET: {' '.join(cmd)}")
            res = subprocess.run(cmd, env=env, capture_output=True, text=True)

            # Log output
            print("HD-BET return code:", res.returncode)
            if res.stdout.strip():
                print("HD-BET stdout:", res.stdout.strip())
            if res.stderr.strip():
                print("HD-BET stderr:", res.stderr.strip())

            # Check success
            if res.returncode == 0 and out_path.exists():
                print(f"✅ HD-BET success: {out_path}")
                return str(out_path)
            else:
                # Fallback to CPU mode if GPU failed
                if use_gpu:
                    print("⚠️ GPU mode failed — retrying in CPU mode...")
                    env["CUDA_VISIBLE_DEVICES"] = "-1"
                    cmd = [hdbet, "-i", str(nifti_file), "-o", str(out_path)]
                    res = subprocess.run(cmd, env=env, capture_output=True, text=True)
                    if res.returncode == 0 and out_path.exists():
                        print(f"✅ HD-BET CPU fallback success: {out_path}")
                        return str(out_path)
                print("❌ HD-BET failed.")
                return None

        except Exception as e:
            print(f"❌ Exception in _try_hdbet: {e}")
            return None


    def _try_synthstrip(self, nifti_file):
        """Attempt skull stripping with SynthStrip (CPU via FreeSurfer binary). Returns output path or None."""
        try:
            import shutil, subprocess, tempfile
            # Common names: synthstrip or mri_synthstrip (FreeSurfer)
            ss = shutil.which('synthstrip') or shutil.which('mri_synthstrip.exe') or shutil.which('mri_synthstrip')
            if not ss:
                return None
            tmpdir = Path(tempfile.mkdtemp(prefix='synthstrip_'))
            out_path = str((tmpdir / 'synthstrip_out.nii.gz').resolve())
            cmd = [ss, '--i', str(nifti_file), '--o', out_path, '--no-cuda']
            res = subprocess.run(cmd, capture_output=True, text=True)
            if res.returncode != 0:
                return None
            return out_path if Path(out_path).exists() else None
        except Exception:
            return None

    def _extract_brain_mask_enhanced(self, data):
        """Enhanced classical brain mask: robust thresholding + morphology."""
        mask_nz = data > 0
        if not np.any(mask_nz):
            return np.zeros_like(data, dtype=np.float32)
        p1, p99 = np.percentile(data[mask_nz], [1, 99])
        clipped = np.clip(data, p1, p99)
        sm = ndimage.gaussian_filter(clipped.astype(np.float32), sigma=1.5)
        sm_min, sm_max = sm[mask_nz].min(), sm[mask_nz].max()
        sm_norm = np.zeros_like(sm, dtype=np.float32)
        if sm_max > sm_min:
            sm_norm = (sm - sm_min) / (sm_max - sm_min)
        try:
            thr = filters.threshold_otsu(sm_norm[mask_nz])
        except Exception:
            thr = 0.2
        init = sm_norm > max(0.2, min(0.6, thr * 0.9))
        init = morphology.binary_opening(init, morphology.ball(2))
        init = morphology.binary_closing(init, morphology.ball(4))
        init = morphology.remove_small_objects(init, min_size=15000)
        init = ndimage.binary_fill_holes(init)
        lbl = measure.label(init)
        props = measure.regionprops(lbl)
        if props:
            largest = max(props, key=lambda x: x.area).label
            init = (lbl == largest)
        init = morphology.binary_dilation(init, morphology.ball(1))
        return init.astype(np.float32)

    def _template_mask_skull_strip(self, nifti_file, data, affine):
        """Generate brain mask by warping MNI brain mask to subject with fast affine."""
        # Paths
        tpl_img_path = Path("mni152_templates/mni152_T1w.nii.gz")
        tpl_mask_path = Path("mni152_templates/mni152_brain_mask.nii.gz")
        if not tpl_img_path.exists() or not tpl_mask_path.exists():
            return None
        # Read images
        fixed = sitk.ReadImage(str(nifti_file)) if isinstance(nifti_file, (str, Path)) else sitk.GetImageFromArray(np.transpose(data.astype(np.float32),(2,1,0)))
        moving_tpl = sitk.ReadImage(str(tpl_img_path))
        mask_tpl = sitk.ReadImage(str(tpl_mask_path))
        # Downsample to 2 mm for speed
        def to_2mm(img):
            spacing = img.GetSpacing()
            new_spacing = [2.0,2.0,2.0]
            size = img.GetSize()
            new_size = [int(round(size[i]*spacing[i]/new_spacing[i])) for i in range(3)]
            return sitk.Resample(img, new_size, sitk.Transform(), sitk.sitkLinear, img.GetOrigin(), new_spacing, img.GetDirection(), 0.0, img.GetPixelID())
        fixed_d = to_2mm(fixed)
        moving_d = to_2mm(moving_tpl)
        # Affine registration (fast)
        reg = sitk.ImageRegistrationMethod()
        reg.SetMetricAsMattesMutualInformation(32)
        reg.SetMetricSamplingStrategy(reg.RANDOM)
        reg.SetMetricSamplingPercentage(0.1)
        reg.SetInterpolator(sitk.sitkLinear)
        reg.SetOptimizerAsRegularStepGradientDescent(2.0, 1e-3, 100, relaxationFactor=0.5)
        reg.SetShrinkFactorsPerLevel([4,2])
        reg.SetSmoothingSigmasPerLevel([2,1])
        reg.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
        tx0 = sitk.CenteredTransformInitializer(fixed_d, moving_d, sitk.Euler3DTransform(), sitk.CenteredTransformInitializerFilter.GEOMETRY)
        reg.SetInitialTransform(sitk.AffineTransform(tx0), inPlace=False)
        try:
            reg.SetNumberOfThreads(2)
        except Exception:
            pass
        aff = reg.Execute(fixed_d, moving_d)
        # Warp template mask to fixed space at full res
        mask_res = sitk.Resample(mask_tpl, fixed, aff, sitk.sitkNearestNeighbor, 0, sitk.sitkUInt8)
        mask_arr = sitk.GetArrayFromImage(mask_res).astype(np.uint8)
        mask_arr = np.transpose(mask_arr, (2,1,0))
        # Morphological refine
        mask_arr = morphology.binary_closing(mask_arr.astype(bool), morphology.ball(2))
        mask_arr = ndimage.binary_fill_holes(mask_arr)
        lbl = measure.label(mask_arr)
        props = measure.regionprops(lbl)
        if props:
            largest = max(props, key=lambda x: x.area).label
            mask_arr = (lbl == largest)
        return mask_arr.astype(np.float32)
    
    def bias_field_correction(self, skull_stripped_files):
        """
        Step 5: Bias Field Correction
        Corrects intensity non-uniformity caused by MRI scanner variations.
        """
        print("Step 5: Bias Field Correction...")
        processed_files = []

        for nifti_file in tqdm(skull_stripped_files, desc="Bias correction"):
            try:
                start_time = time.time()

                # Load skull-stripped image
                img = nib.load(nifti_file)
                data = img.get_fdata()

                # Create brain mask (everything > 0 is brain)
                brain_mask = (data > 0).astype(np.uint8)

                # ✅ Robust normalization (z-score → clip → min-max)
                vals = data[brain_mask > 0]
                mean, std = vals.mean(), vals.std()
                norm_data = (data - mean) / (std + 1e-8)
                norm_data = np.clip(norm_data, -3, 3)
                norm_data = (norm_data - norm_data.min()) / (norm_data.max() - norm_data.min())

                # Perform bias field correction
                corrected_data = self._correct_bias_field(norm_data, brain_mask)

                # Undo normalization scale
                corrected_data = corrected_data * std + mean

                # Create output filename
                output_filename = f"bc_{nifti_file.name}"
                output_path = self.bias_corrected_dir / output_filename

                # Save bias-corrected image
                corrected_img = nib.Nifti1Image(corrected_data, img.affine, img.header)
                nib.save(corrected_img, output_path)
                processed_files.append(output_path)

                # Inline quality metric — std reduction
                try:
                    mask = brain_mask > 0
                    std0 = float(np.std(data[mask])) if np.any(mask) else 0
                    std1 = float(np.std(corrected_data[mask])) if np.any(mask) else 0
                    improv = (std0 - std1) / std0 if std0 > 0 else 0
                    bias_q = int(round(50 + 50 * max(0.0, min(1.0, improv))))
                    print(f"✅ Bias corrected: {output_filename} | Quality: {bias_q}% | Time: {time.time() - start_time:.2f}s | σ↓ {improv*100:.1f}%")
                except Exception:
                    print(f"✅ Bias corrected: {output_filename}")

            except Exception as e:
                print(f"❌ Error in bias correction {nifti_file}: {e}")

        return processed_files


    def _correct_bias_field(self, data, mask=None):
        """
        Bias field correction using SimpleITK N4 with refined mask and normalization.
        """
        if mask is None:
            mask = (data > 0).astype(np.uint8)
        if not np.any(mask):
            return data

        if 'sitk' in globals() and SITK_AVAILABLE:
            try:
                t0 = time.time()
                print(f"🧠 Input shape: {data.shape}, mean intensity: {data.mean():.3f}")

                # Convert to SimpleITK image (z,y,x)
                img = sitk.GetImageFromArray(np.transpose(data.astype(np.float32), (2, 1, 0)))
                img = sitk.Cast(img, sitk.sitkFloat32)

                # ✅ Improved mask: combine otsu + provided mask
                otsu_mask = sitk.OtsuThreshold(img, 0, 1, 200)
                itk_mask = sitk.GetImageFromArray(np.transpose(mask.astype(np.uint8), (2, 1, 0)))
                combined_mask = sitk.And(otsu_mask, itk_mask)

                # ✅ Morphological cleanup (closing + erosion)
                combined_mask = sitk.BinaryMorphologicalClosing(combined_mask, [2]*3)
                combined_mask = sitk.BinaryErode(combined_mask, [1]*3)

                # ✅ Use shrink factor = 1 for best quality
                shrink = 2
                print(f"⚙️ Using shrink factor: {shrink}")

                img_shr = sitk.Shrink(img, [shrink, shrink, shrink])
                mask_shr = sitk.Shrink(combined_mask, [shrink, shrink, shrink])

                # N4 Bias Field Correction
                n4 = sitk.N4BiasFieldCorrectionImageFilter()
                n4.SetBiasFieldFullWidthAtHalfMaximum(0.15)
                n4.SetMaximumNumberOfIterations([100, 50, 30, 20])
                n4.SetConvergenceThreshold(1e-7)
                n4.SetSplineOrder(3)
                n4.SetWienerFilterNoise(0.11)

                print("⏳ Running N4 bias field correction...")
                corrected_shr = n4.Execute(img_shr, mask_shr)
                print(f"✅ N4 done in {time.time() - t0:.2f}s")

                # Apply estimated bias field on full-res image
                log_field = n4.GetLogBiasFieldAsImage(img)
                corrected = sitk.Exp(-log_field) * img

                # Convert back to numpy (x,y,z)
                arr = sitk.GetArrayFromImage(corrected).astype(np.float32)
                arr = np.transpose(arr, (2, 1, 0))
                print(f"✅ Bias correction total time: {time.time() - t0:.2f}s")
                return arr

            except Exception as e:
                print(f"⚠️ SimpleITK N4 correction failed, using fallback. Error: {e}")

                # Fallback: smoothing-based correction
                smoothed = ndimage.gaussian_filter(data.astype(np.float64), sigma=50)
                bias_field = np.where(smoothed > 0, smoothed, 1)
                corrected = np.where(mask > 0, data / bias_field * np.mean(bias_field[mask > 0]), data)
                return corrected.astype(np.float32)

        # If SITK unavailable
        smoothed = ndimage.gaussian_filter(data.astype(np.float64), sigma=50)
        bias_field = np.where(smoothed > 0, smoothed, 1)
        corrected = np.where(mask > 0, data / bias_field * np.mean(bias_field[mask > 0]), data)
        return corrected.astype(np.float32)


    
    def spatial_normalization(self, bias_corrected_files):
        """
        Step 6: Spatial Normalization to MNI Space
        
        Uses SimpleITK (if available) for affine + BSpline deformable registration to MNI152.
        Falls back to simple resizing if SimpleITK is unavailable or fails.
        """
        print("Step 6: RESEARCH-GRADE MNI152 Registration (SimpleITK)...")
        
        processed_files = []
        
        # Define standard MNI dimensions
        mni_shape = (182, 218, 182)  # Standard MNI152 dimensions
        
        for nifti_file in tqdm(bias_corrected_files, desc="Spatial normalization"):
            try:
                # Load bias-corrected image
                img = nib.load(nifti_file)
                data = img.get_fdata()
                
                # Prefer SimpleITK registration if available
                if SITK_AVAILABLE:
                    registered = self._sitk_register_to_mni(nifti_file, data, img.affine)
                    if registered is not None:
                        normalized_data = registered
                    else:
                        normalized_data = self._normalize_to_mni(data, mni_shape)
                else:
                    normalized_data = self._normalize_to_mni(data, mni_shape)

                # Create MNI-like affine matrix
                mni_affine = np.array([
                    [-1., 0., 0., 90.],
                    [0., 1., 0., -126.],
                    [0., 0., 1., -72.],
                    [0., 0., 0., 1.]
                ])
                
                # Create output filename
                output_filename = f"mni_{nifti_file.name}"
                output_path = self.normalized_dir / output_filename
                
                # Save normalized image
                normalized_img = nib.Nifti1Image(normalized_data, mni_affine)
                nib.save(normalized_img, output_path)
                
                processed_files.append(output_path)
                # Inline quality: NCC vs MNI template
                try:
                    tpl = nib.load("mni152_templates/mni152_T1w.nii.gz")
                    tdat = tpl.get_fdata()
                    if tdat.shape == normalized_data.shape:
                        a = normalized_data - normalized_data.mean()
                        b = tdat - tdat.mean()
                        ncc = float(np.mean((a/(a.std()+1e-6)) * (b/(b.std()+1e-6))))
                        reg_q = int(round(max(0.0, min(1.0, (ncc + 1) / 2.0)) * 100))
                        print(f"✅ Spatially normalized: {output_filename} | Quality: {reg_q}% (NCC={ncc:.3f})")
                    else:
                        print(f"✅ Spatially normalized: {output_filename}")
                except Exception:
                    print(f"✅ Spatially normalized: {output_filename}")
                
            except Exception as e:
                print(f"❌ Error in spatial normalization {nifti_file}: {e}")
        
        return processed_files
    
    def _normalize_to_mni(self, data, target_shape):
        """Normalize image to MNI space using simple resizing."""
        # Calculate zoom factors for each dimension
        zoom_factors = [target_shape[i] / data.shape[i] for i in range(3)]
        
        # Resize using scipy zoom
        normalized_data = ndimage.zoom(data, zoom_factors, order=1)
        
        return normalized_data

    def _sitk_register_to_mni(self, nifti_file, data, affine):
        """Register image to MNI using SimpleITK (affine + BSpline)."""
        try:
            # Keep threading conservative for stability on Windows
            try:
                sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(2)
            except Exception:
                pass
            # Path to MNI template bundled in repo
            template_path = Path("mni152_templates/mni152_T1w.nii.gz")
            if not template_path.exists():
                return None

            fixed = sitk.ReadImage(str(template_path))
            if isinstance(nifti_file, (str, Path)) and Path(nifti_file).exists():
                moving = sitk.ReadImage(str(nifti_file))
            else:
                # Write temp moving
                tmp_path = Path(self.output_dir) / "_sitk_tmp_input.nii.gz"
                nib.save(nib.Nifti1Image(data.astype(np.float32), affine), str(tmp_path))
                moving = sitk.ReadImage(str(tmp_path))

            fixed = sitk.Cast(fixed, sitk.sitkFloat32)
            moving = sitk.Cast(moving, sitk.sitkFloat32)

            # Histogram matching for robustness
            hm = sitk.HistogramMatchingImageFilter()
            hm.SetNumberOfHistogramLevels(256)
            hm.SetNumberOfMatchPoints(12)
            hm.ThresholdAtMeanIntensityOn()
            moving_hm = hm.Execute(moving, fixed)

            # Affine stage
            init = sitk.CenteredTransformInitializer(
                fixed, moving_hm, sitk.Euler3DTransform(), sitk.CenteredTransformInitializerFilter.GEOMETRY
            )
            affine_tx = sitk.AffineTransform(3)
            affine_tx.SetMatrix(sitk.Euler3DTransform(init).GetMatrix())
            affine_tx.SetTranslation(sitk.Euler3DTransform(init).GetTranslation())

            reg1 = sitk.ImageRegistrationMethod()
            reg1.SetMetricAsMattesMutualInformation(32)
            reg1.SetMetricSamplingStrategy(reg1.RANDOM)
            reg1.SetMetricSamplingPercentage(0.15)
            reg1.SetInterpolator(sitk.sitkLinear)
            reg1.SetOptimizerAsRegularStepGradientDescent(2.0, 1e-4, 200, relaxationFactor=0.5)
            reg1.SetOptimizerScalesFromPhysicalShift()
            reg1.SetShrinkFactorsPerLevel([8,4,2])
            reg1.SetSmoothingSigmasPerLevel([4,2,1])
            reg1.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
            reg1.SetInitialTransform(affine_tx, inPlace=False)
            try:
                reg1.SetNumberOfThreads(2)
            except Exception:
                pass
            affine_tx = reg1.Execute(fixed, moving_hm)

            # Early quality check after affine: skip BSpline if alignment is good
            moving_affine_full = sitk.Resample(moving, fixed, affine_tx, sitk.sitkLinear, 0.0, moving.GetPixelID())
            a_arr = sitk.GetArrayFromImage(moving_affine_full).astype(np.float32)
            f_arr = sitk.GetArrayFromImage(fixed).astype(np.float32)
            # Compute NCC quickly on a central crop to speed up
            try:
                def _central_crop(x, frac=0.5):
                    z,y,xs = x.shape
                    cz, cy, cx = int(z*(1-frac)/2), int(y*(1-frac)/2), int(xs*(1-frac)/2)
                    ez, ey, ex = z-cz, y-cy, xs-cx
                    return x[cz:ez, cy:ey, cx:ex]
                ca = _central_crop(a_arr)
                cf = _central_crop(f_arr)
                ca = (ca - ca.mean()) / (ca.std()+1e-6)
                cf = (cf - cf.mean()) / (cf.std()+1e-6)
                ncc_affine = float((ca*cf).mean())
            except Exception:
                ncc_affine = 0.0
            # If NCC >= 0.75, accept affine-only (FAST PATH)
            if ncc_affine >= 0.75:
                registered = moving_affine_full
                arr = sitk.GetArrayFromImage(registered).astype(np.float32)
                arr = np.transpose(arr, (2,1,0))
                return arr

            # BSpline deformable stage (SyN-like). Coarse grid & few iterations for speed.
            grid_spacing = [120.0, 120.0, 120.0]
            phys_size = [sz*sp for sz, sp in zip(fixed.GetSize(), fixed.GetSpacing())]
            mesh = [max(1, int(round(ps/gs))) for ps, gs in zip(phys_size, grid_spacing)]
            bspline_tx = sitk.BSplineTransformInitializer(fixed, mesh, order=3)

            reg2 = sitk.ImageRegistrationMethod()
            reg2.SetMetricAsMattesMutualInformation(32)
            reg2.SetMetricSamplingStrategy(reg2.RANDOM)
            reg2.SetMetricSamplingPercentage(0.10)
            reg2.SetInterpolator(sitk.sitkLinear)
            reg2.SetOptimizerAsLBFGSB(gradientConvergenceTolerance=1e-5, numberOfIterations=10, maximumNumberOfCorrections=3)
            reg2.SetShrinkFactorsPerLevel([4,2,1])
            reg2.SetSmoothingSigmasPerLevel([2,1,0])
            reg2.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
            reg2.SetInitialTransform(bspline_tx, inPlace=False)
            try:
                reg2.SetNumberOfThreads(2)
            except Exception:
                pass

            composite = sitk.Transform(affine_tx)
            moving_affine = sitk.Resample(moving_hm, fixed, composite, sitk.sitkLinear, 0.0, moving_hm.GetPixelID())
            try:
                bspline_tx = reg2.Execute(fixed, moving_affine)
            except Exception:
                # Fall back to affine-only if BSpline fails
                registered_affine = sitk.Resample(moving, fixed, affine_tx, sitk.sitkLinear, 0.0, moving.GetPixelID())
                arr = sitk.GetArrayFromImage(registered_affine).astype(np.float32)
                return np.transpose(arr, (2,1,0))

            final_tx = sitk.Transform(affine_tx)
            final_tx.AddTransform(bspline_tx)

            registered = sitk.Resample(moving, fixed, final_tx, sitk.sitkLinear, 0.0, moving.GetPixelID())
            arr = sitk.GetArrayFromImage(registered).astype(np.float32)
            arr = np.transpose(arr, (2,1,0))
            return arr
        except Exception:
            return None
    
    def intensity_normalization(self, normalized_files):
        """
        Step 7: Intensity Normalization
        
        Normalizes intensity values to standard ranges for consistent analysis.
        """
        print("Step 7: Intensity Normalization...")
        
        processed_files = []
        
        for nifti_file in tqdm(normalized_files, desc="Intensity normalization"):
            try:
                # Load spatially normalized image
                img = nib.load(nifti_file)
                data = img.get_fdata()
                
                # Perform intensity normalization
                normalized_data = self._normalize_intensity(data)
                
                # Create output filename
                output_filename = f"inorm_{nifti_file.name}"
                output_path = self.intensity_norm_dir / output_filename
                
                # Save intensity-normalized image
                normalized_img = nib.Nifti1Image(normalized_data, img.affine, img.header)
                nib.save(normalized_img, output_path)
                
                processed_files.append(output_path)
                # Inline quality: range and mean closeness to 0.5
                try:
                    mask = normalized_data > 0
                    rng = float(np.percentile(normalized_data[mask], 99) - np.percentile(normalized_data[mask], 1)) if np.any(mask) else 0
                    meanv = float(np.mean(normalized_data[mask])) if np.any(mask) else 0
                    score_range = max(0.0, min(1.0, rng))
                    score_mean = max(0.0, min(1.0, 1.0 - abs(meanv - 0.5)))
                    inorm_q = int(round(50 * score_range + 50 * score_mean))
                    print(f"✅ Intensity normalized: {output_filename} | Quality: {inorm_q}% (range≈{rng:.2f}, mean={meanv:.2f})")
                except Exception:
                    print(f"✅ Intensity normalized: {output_filename}")
                
            except Exception as e:
                print(f"❌ Error in intensity normalization {nifti_file}: {e}")
        
        return processed_files
    
    def _normalize_intensity(self, data):
        """Normalize intensity to [0, 1] range with percentile clipping."""
        # Create mask for non-zero voxels
        mask = data > 0
        
        if not np.any(mask):
            return data
        
        # Calculate percentiles to clip outliers
        p1, p99 = np.percentile(data[mask], [1, 99])
        
        # Clip outliers
        clipped_data = np.clip(data, p1, p99)
        
        # Normalize to [0, 1]
        normalized_data = (clipped_data - p1) / (p99 - p1)
        
        # Keep original zeros
        normalized_data = np.where(mask, normalized_data, 0)
        
        return normalized_data.astype(np.float32)
    
    def spatial_resampling(self, intensity_normalized_files):
        """
        Step 8: Spatial Resampling & Resizing
        
        Resample to uniform voxel size and standard dimensions.
        """
        print("Step 8: Spatial Resampling & Resizing...")
        
        processed_files = []
        target_shape = (128, 128, 128)  # Standard size for deep learning
        
        for nifti_file in tqdm(intensity_normalized_files, desc="Resampling"):
            try:
                # Load intensity-normalized image
                img = nib.load(nifti_file)
                data = img.get_fdata()
                
                # Resample to target shape
                resampled_data = self._resample_to_shape(data, target_shape)
                
                # Create isotropic affine matrix (1mm³ voxels)
                iso_affine = np.array([
                    [1., 0., 0., -64.],
                    [0., 1., 0., -64.],
                    [0., 0., 1., -64.],
                    [0., 0., 0., 1.]
                ])
                
                # Create output filename
                output_filename = f"resampled_{nifti_file.name}"
                output_path = self.resampled_dir / output_filename
                
                # Save resampled image
                resampled_img = nib.Nifti1Image(resampled_data, iso_affine)
                nib.save(resampled_img, output_path)
                
                processed_files.append(output_path)
                res_q = 100 if tuple(resampled_data.shape) == (128,128,128) else 70
                print(f"✅ Resampled: {output_filename} | Quality: {res_q}% (shape={resampled_data.shape})")
                
            except Exception as e:
                print(f"❌ Error in resampling {nifti_file}: {e}")
        
        return processed_files
    
    def _resample_to_shape(self, data, target_shape):
        """Resample data to target shape."""
        zoom_factors = [target_shape[i] / data.shape[i] for i in range(3)]
        resampled_data = ndimage.zoom(data, zoom_factors, order=1)
        return resampled_data
    
    def tissue_segmentation(self, resampled_files):
        """
        Step 9: Tissue Segmentation
        
        Segment brain tissue into Gray Matter (GM), White Matter (WM), and CSF.
        """
        print("Step 9: Tissue Segmentation...")
        
        processed_files = []
        
        for nifti_file in tqdm(resampled_files, desc="Tissue segmentation"):
            try:
                # Load resampled image
                img = nib.load(nifti_file)
                data = img.get_fdata()
                
                # Perform tissue segmentation
                gm, wm, csf = self._segment_tissues(data)
                
                # Save segmentation maps
                base_filename = nifti_file.stem.replace('.nii', '')
                
                # Gray matter
                gm_filename = f"{base_filename}_GM.nii.gz"
                gm_path = self.segmented_dir / gm_filename
                gm_img = nib.Nifti1Image(gm, img.affine, img.header)
                nib.save(gm_img, gm_path)
                
                # White matter
                wm_filename = f"{base_filename}_WM.nii.gz"
                wm_path = self.segmented_dir / wm_filename
                wm_img = nib.Nifti1Image(wm, img.affine, img.header)
                nib.save(wm_img, wm_path)
                
                # CSF
                csf_filename = f"{base_filename}_CSF.nii.gz"
                csf_path = self.segmented_dir / csf_filename
                csf_img = nib.Nifti1Image(csf, img.affine, img.header)
                nib.save(csf_img, csf_path)
                
                processed_files.extend([gm_path, wm_path, csf_path])
                print(f"✅ Segmented: {base_filename}")
                
            except Exception as e:
                print(f"❌ Error in segmentation {nifti_file}: {e}")
        
        return processed_files
    
    def _segment_tissues(self, data):
        """Segment brain tissue using Gaussian Mixture Model."""
        # Create mask for brain tissue
        mask = data > 0.1
        
        if not np.any(mask):
            return np.zeros_like(data), np.zeros_like(data), np.zeros_like(data)
        
        # Extract brain voxels
        brain_voxels = data[mask].reshape(-1, 1)
        
        # Fit Gaussian Mixture Model (3 components: CSF, GM, WM)
        gmm = GaussianMixture(n_components=3, random_state=42)
        labels = gmm.fit_predict(brain_voxels)
        
        # Sort components by mean intensity (CSF < GM < WM)
        means = gmm.means_.flatten()
        sorted_indices = np.argsort(means)
        
        # Create tissue maps
        csf_map = np.zeros_like(data)
        gm_map = np.zeros_like(data)
        wm_map = np.zeros_like(data)
        
        # Assign labels to tissue types
        csf_label = sorted_indices[0]  # Lowest intensity
        gm_label = sorted_indices[1]   # Middle intensity
        wm_label = sorted_indices[2]   # Highest intensity
        
        # Fill tissue maps
        csf_mask = mask.copy()
        csf_mask[mask] = labels == csf_label
        csf_map[csf_mask] = 1
        
        gm_mask = mask.copy()
        gm_mask[mask] = labels == gm_label
        gm_map[gm_mask] = 1
        
        wm_mask = mask.copy()
        wm_mask[mask] = labels == wm_label
        wm_map[wm_mask] = 1
        
        return gm_map.astype(np.float32), wm_map.astype(np.float32), csf_map.astype(np.float32)
    
    def phase2_quality_assessment(self, final_files):
        """Generate quality assessment for Phase 2 results."""
        print("Generating Phase 2 Quality Assessment...")
        
        qa_data = []
        
        # Find the final resampled files (not segmentation maps)
        resampled_files = [f for f in final_files if 'resampled_' in f.name and not any(tissue in f.name for tissue in ['_GM', '_WM', '_CSF'])]
        
        for nifti_file in tqdm(resampled_files, desc="QA assessment"):
            try:
                img = nib.load(nifti_file)
                data = img.get_fdata()
                
                metrics = {
                    'filename': str(nifti_file.name),
                    'final_shape': list(data.shape),
                    'file_size_mb': nifti_file.stat().st_size / (1024*1024),
                    'intensity_range': [float(data.min()), float(data.max())],
                    'mean_intensity': float(np.mean(data[data > 0])) if np.any(data > 0) else 0,
                    'std_intensity': float(np.std(data[data > 0])) if np.any(data > 0) else 0,
                    'brain_volume_ratio': float(np.sum(data > 0) / data.size)
                }
                qa_data.append(metrics)
                
            except Exception as e:
                print(f"Error assessing {nifti_file}: {e}")
        
        # Save QA report
        qa_report_path = self.qa_dir / "phase2_qa_report.json"
        with open(qa_report_path, 'w') as f:
            json.dump(qa_data, f, indent=2)
        
        print(f"✅ Phase 2 QA report saved to: {qa_report_path}")
        return qa_data
    
    def run_complete_phase2(self):
        """Run the complete Phase 2 preprocessing pipeline."""
        print(f"\n=== MRI Preprocessing Phase 2 - {self.dataset_name} ===")
        print(f"Input: {self.input_dir}")
        print(f"Output: {self.output_dir}")
        
        try:
            import time
            perf = []

            # Find Phase 1 output files
            t0 = time.time()
            phase1_files = self.find_phase1_files()
            perf.append({"step":"discover_phase1","files":len(phase1_files),"duration_sec":round(time.time()-t0,3)})
            print(f"Found {len(phase1_files)} files from Phase 1")
            
            # Step 4: Skull Stripping
            t0 = time.time()
            skull_stripped_files = self.skull_stripping(phase1_files)
            perf.append({"step":"skull_stripping","files":len(skull_stripped_files),"duration_sec":round(time.time()-t0,3)})
            
            # Step 5: Bias Field Correction
            t0 = time.time()
            bias_corrected_files = self.bias_field_correction(skull_stripped_files)
            perf.append({"step":"bias_correction","files":len(bias_corrected_files),"duration_sec":round(time.time()-t0,3)})
            
            # Step 6: Spatial Normalization
            t0 = time.time()
            normalized_files = self.spatial_normalization(bias_corrected_files)
            perf.append({"step":"registration","files":len(normalized_files),"duration_sec":round(time.time()-t0,3)})
            
            # Step 7: Intensity Normalization
            t0 = time.time()
            intensity_normalized_files = self.intensity_normalization(normalized_files)
            perf.append({"step":"intensity_normalization","files":len(intensity_normalized_files),"duration_sec":round(time.time()-t0,3)})
            
            # Step 8: Spatial Resampling
            t0 = time.time()
            resampled_files = self.spatial_resampling(intensity_normalized_files)
            perf.append({"step":"resampling","files":len(resampled_files),"duration_sec":round(time.time()-t0,3)})
            
            # Step 9: Tissue Segmentation
            t0 = time.time()
            segmented_files = self.tissue_segmentation(resampled_files)
            perf.append({"step":"segmentation","files":len(segmented_files),"duration_sec":round(time.time()-t0,3)})
            
            # Quality Assessment
            t0 = time.time()
            qa_data = self.phase2_quality_assessment(resampled_files + segmented_files)

            # Quality percentages per step (coarse but informative)
            quality = self._compute_quality_percentages(
                phase1_files=phase1_files,
                skull_stripped=skull_stripped_files,
                bias_corrected=bias_corrected_files,
                normalized=normalized_files,
                intensity_norm=intensity_normalized_files,
                resampled=resampled_files,
                segmented=segmented_files,
            )
            perf.append({"step":"qa","files":len(resampled_files)+len(segmented_files),"duration_sec":round(time.time()-t0,3)})
            
            # Compute throughput (voxels/sec) per step when applicable
            # For a single subject, approximate using sizes from produced NIfTIs
            def voxels_of(paths):
                total = 0
                for p in paths:
                    try:
                        total += int(np.prod(nib.load(p).shape))
                    except Exception:
                        pass
                return total
            total_sec = sum(s["duration_sec"] for s in perf)
            # Add percent of total time per step
            for s in perf:
                s["percent_time"] = round((s["duration_sec"] / total_sec) * 100.0, 1) if total_sec > 0 else 0.0

            report = {
                "dataset": self.dataset_name,
                "total_duration_sec": round(total_sec, 3),
                "steps": perf,
                "quality_percent": quality,
            }
            # Map outputs to estimate voxels per step
            report["throughput"] = {
                "registration_voxels_per_sec": round(voxels_of(normalized_files)/max(1e-6, [s for s in perf if s["step"]=="registration"][0]["duration_sec"]), 2) if normalized_files else 0,
                "intensity_voxels_per_sec": round(voxels_of(intensity_normalized_files)/max(1e-6, [s for s in perf if s["step"]=="intensity_normalization"][0]["duration_sec"]), 2) if intensity_normalized_files else 0,
                "resampling_voxels_per_sec": round(voxels_of(resampled_files)/max(1e-6, [s for s in perf if s["step"]=="resampling"][0]["duration_sec"]), 2) if resampled_files else 0,
            }

            # Save performance summary
            perf_path = self.output_dir / "performance_summary.json"
            with open(perf_path, 'w') as f:
                json.dump(report, f, indent=2)
            print(f"\n📊 Performance summary saved to: {perf_path}")
            # Print percent breakdown
            print("\n📊 Time distribution by step (% of total):")
            for s in perf:
                print(f" - {s['step']}: {s['percent_time']}% ({s['duration_sec']}s)")
            print("\n🏁 Quality (percent scores):")
            for k, v in quality.items():
                print(f" - {k}: {v}%")
            
            # Summary
            print(f"\n=== Phase 2 Complete ===")
            print(f"Skull stripped: {len(skull_stripped_files)} files")
            print(f"Bias corrected: {len(bias_corrected_files)} files")
            print(f"Spatially normalized: {len(normalized_files)} files")
            print(f"Intensity normalized: {len(intensity_normalized_files)} files")
            print(f"Resampled: {len(resampled_files)} files")
            print(f"Tissue segmented: {len(segmented_files)} files")
            print(f"Results saved to: {self.output_dir}")
            
            return True
            
        except Exception as e:
            print(f"Error in Phase 2 preprocessing: {e}")
            return False

    def _compute_quality_percentages(self, phase1_files, skull_stripped, bias_corrected, normalized, intensity_norm, resampled, segmented):
        """Compute simple quality percentages per step using available outputs."""
        def safe_load(path):
            try:
                img = nib.load(path)
                return img.get_fdata(), img.affine
            except Exception:
                return None, None
        # Skull stripping: evaluate brain volume ratio from saved mask if available
        skull_q = 0
        if skull_stripped:
            brain_path = skull_stripped[0]
            mask_path = (Path(brain_path).parent / ("mask_" + Path(brain_path).name))
            mdat, _ = safe_load(str(mask_path))
            if mdat is None:
                # Derive mask from skull-stripped image if explicit mask missing
                bdat, _ = safe_load(str(brain_path))
                if bdat is not None:
                    mdat = (bdat > 0).astype(np.uint8)
            if mdat is not None and mdat.size > 0:
                ratio = float(np.sum(mdat > 0) / mdat.size)
                # Ideal brain ratio ~0.12–0.4 → map to [0,100]
                if ratio <= 0 or ratio >= 0.8:
                    skull_q = 20
                else:
                    # center at 0.25, penalize distance
                    skull_q = int(max(20, min(100, 100 - abs(ratio - 0.25) * 400)))
        # Bias correction: reduction in intensity std within brain mask
        bias_q = 0
        if skull_stripped and bias_corrected:
            bdat0, _ = safe_load(str(skull_stripped[0]))
            bdat1, _ = safe_load(str(bias_corrected[0]))
            if bdat0 is not None and bdat1 is not None:
                mask = bdat0 > 0
                std0 = float(np.std(bdat0[mask])) if np.any(mask) else 0
                std1 = float(np.std(bdat1[mask])) if np.any(mask) else 0
                if std0 > 0 and std1 > 0:
                    improv = max(0.0, min(1.0, (std0 - std1) / std0))
                    bias_q = int(round(50 + 50 * improv))
        # Registration: NCC between normalized image and MNI template
        reg_q = 0
        if normalized:
            ndat, _ = safe_load(str(normalized[0]))
            tdat, _ = safe_load("mni152_templates/mni152_T1w.nii.gz")
            if ndat is not None and tdat is not None and ndat.shape == tdat.shape:
                a = (ndat - np.mean(ndat))
                b = (tdat - np.mean(tdat))
                denom = (np.std(a) * np.std(b))
                if denom > 0:
                    ncc = float(np.mean((a/np.std(a)) * (b/np.std(b))))
                    reg_q = int(round(max(0.0, min(1.0, (ncc + 1) / 2.0)) * 100))
        # Intensity normalization: check range and mean
        inorm_q = 0
        if intensity_norm:
            idat, _ = safe_load(str(intensity_norm[0]))
            if idat is not None:
                mask = idat > 0
                rng = float(np.percentile(idat[mask], 99) - np.percentile(idat[mask], 1)) if np.any(mask) else 0
                meanv = float(np.mean(idat[mask])) if np.any(mask) else 0
                score_range = max(0.0, min(1.0, rng))
                score_mean = max(0.0, min(1.0, 1.0 - abs(meanv - 0.5)))
                inorm_q = int(round(50 * score_range + 50 * score_mean))
        # Resampling: shape match to target
        res_q = 0
        if resampled:
            rdat, _ = safe_load(str(resampled[0]))
            if rdat is not None:
                res_q = 100 if tuple(rdat.shape) == (128,128,128) else 70
        # Segmentation: presence of 3 classes
        seg_q = 0
        if segmented:
            gm_path = [p for p in segmented if str(p).endswith("_GM.nii.gz")]
            wm_path = [p for p in segmented if str(p).endswith("_WM.nii.gz")]
            csf_path = [p for p in segmented if str(p).endswith("_CSF.nii.gz")]
            if gm_path and wm_path and csf_path:
                gm, _ = safe_load(str(gm_path[0]))
                wm, _ = safe_load(str(wm_path[0]))
                csf, _ = safe_load(str(csf_path[0]))
                if gm is not None and wm is not None and csf is not None:
                    # require non-empty masks
                    nz = sum([int(np.any(gm)), int(np.any(wm)), int(np.any(csf))])
                    seg_q = 100 if nz == 3 else 60
        return {
            "skull_stripping": skull_q,
            "bias_correction": bias_q,
            "registration": reg_q,
            "intensity_normalization": inorm_q,
            "resampling": res_q,
            "segmentation": seg_q,
        }


def main():
    """Main function for command line usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description="MRI Preprocessing Phase 2")
    parser.add_argument("--input", required=True, help="Input directory (Phase 1 BIDS output)")
    parser.add_argument("--output", required=True, help="Output directory for Phase 2")
    parser.add_argument("--dataset", default="Unknown", help="Dataset name (PPMI/ADNI/OASIS)")
    parser.add_argument("--only-skull", action="store_true", help="Run only skull stripping and exit")
    
    args = parser.parse_args()
    
    # Run preprocessing
    processor = MRIPreprocessorPhase2(args.input, args.output, args.dataset)
    if args.only_skull:
        try:
            files = processor.find_phase1_files()
            skull_files = processor.skull_stripping(files)
            print("\n=== Skull Stripping Only ===")
            print(f"Skull stripped: {len(skull_files)} files")
            if skull_files:
                print(f"First output: {skull_files[0]}")
            print(f"Results saved to: {processor.skull_stripped_dir}")
            print("\n✅ Skull stripping completed successfully!")
            return 0
        except Exception as e:
            print(f"\n❌ Skull stripping failed: {e}")
            return 1
    
    success = processor.run_complete_phase2()
    
    if success:
        print("\n✅ Phase 2 preprocessing completed successfully!")
    else:
        print("\n❌ Phase 2 preprocessing failed!")
        return 1

if __name__ == "__main__":
    main()
