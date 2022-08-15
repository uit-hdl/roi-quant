# ROI TILs quantification project

[TBD] Requirements and Docker container

## Repository structure

### Folders:

`roi_segmentation`: [TBD] models and supplementary files for training and running region of interest (ROI) segmentation on downsampled whole-slide images (WSIs)

`qc_classificator`:  models and supplementary files for training and running classifier for quality control (QC) procedure

`til_quantification`: [TBD] scripts for patch inference, based on [project](https://github.com/uit-hdl/hover_serving)

### Scripts:

`1_extract_patches.py` - patch extraction scripts from WSI, based on provided tumor contours

`2_prepare_patches.py` - rename and copy files for next step

`3_qc.py` - quality control procedure to filter out nectoric and lung tissue patches

`4_infer.py` - [TBD] quantify cells in selected patches

`5_stats.py` - summarize TILs quantification results

