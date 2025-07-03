### Extracting conventional IVD features and radiomics from T2-weighted MRI
This is an example of extracting:
- A. Intervertebral disc (IVD) peak signal intensity difference with two normalisation methods.
- B. IVD and vertebral body (VB) conventional geometric indices (e.g. IVD height index).
- C. Full set of Radiomic features from the IVD (shape and texture features) using pyradiomics across a range of binwidths and resamplings. 

### Running the example for feature extraction
Set up the environment using pip and requirements.txt (**NB** pyradiomics may not work on versions of python >3.9) and use run.py to use the script on the sample masks and image. 

### Paper
This code was used in the following study: [McSweeney, T. et al. Robust Radiomic Signatures of Intervertebral Disc Degeneration from MRI. Spine (2025)](https://doi.org/10.1097/BRS.0000000000005435). Additional scripts for image mask perturbations and statistical analysis used in the paper are in the notebooks folder. Deep learning segmentations were carried out using models available here: [https://imedslab.github.io/spineslicer/#](https://imedslab.github.io/spineslicer/#).

![](figure.png)
