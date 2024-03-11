### Extractiong conventional IVD features and radiomics from T2-weighted MRI
This is an example of extracting 
- A: Intervertebral disc (IVD) delta signal intensity
- B: IVD and vertebral body (VB) conventional geometric indices (e.g. IVD height index) 
- C: Full set of Radiomic features from the IVD (shape and texture features)

The script calculates A and B given an input of 3 masks in nrrd format (upper VB, lower VB, IVD between them) and a 2D slice also in nrrd format. It is sensitive to issues in the mask so might fail if there are large errors in the mask. The pyradiomics package is used to calculate C using the IVD mask and image across a range of binwidths and resamplings, and this is much more robust to mask issues.

### Note:
- For A and B pixel spacing is not necessary since the important outputs are normalized to the individual IVD. For C, the pixel spacing (in the nrrd header) is necessary to calculate some radiomic features.

- Each set of features is calculated for a single sagittal plane image, with up to 5 images numerically closest to the mid-sagittal slice used per participant.

- The only thing in the params files that would need to be adjusted is the label setting. This should be set to match the label of the mask.

![image](https://github.com/TerMcs/quantspine/assets/66838178/875371b9-d8c4-4a95-a99d-b07827cfd871)

![image](https://github.com/TerMcs/quantspine/assets/66838178/ceb2dfcf-d494-45bd-9182-228fe37a5d6a)

![image](https://github.com/TerMcs/quantspine/assets/66838178/25f89d89-51ba-4e35-882c-a86ce55a1792)



