# 3D-Unet

**Data Types (DICOM .dcm):**
- 3D CT Scan
- 3D Structure Set that references CT scans by slice number (source of Primary Target Volume(PTV) contours and spinal cord contours)
- 3D dose array
- Dose Plan
**Data Inconsistencies**: 
- corrupted data
- not all patients had a spinal contour 

**Preprocessing Steps:**
**3D Interpolation:** The x, y, z spacings for the CT scans and the dose arrays were different. In all cases, the dose array spacing was larger than the CT scan (which is why the number of z-slices in the original dose array was always lower than the amount of CT slices). Therefore, we perform a 3D interpolation of the dose 


**Trouble **
- Aligning slices


## Additional Preprocessing**:
 
**Spinal Cord Mask within the PTV mask**: When the cord mask is within the PTV mask, the prescribed dose for that particular slice should be a donut shape (where the donut is where the dose is being beamed, and the hole is the where the dose is not being beamed) This is extremely important as subjecting the spinal cord to dose can cause fractures, immense pain, and paralysis. 
- Solution: Subtract the cord mask from the PTV mask.  
**Multiple PTV masks on the same slice:**. In the "View-Dose(Save Training) notebook, we can see an example of where a patient has 2 separate PTV volumes both indexed to the same CT scan. Rather than placing 2 contours for the same CT scan, the medical imaging software makes a duplicate CT scan so there is only 1 contour per scan.  We must concatenate all PTV masks that reference the same CT scan; otherwise, our loss function will not operate correctly. 
- Solution: Find all referenced PTVs for an indexed CT scan and concatenate the masks to obtain the true mask for the CT scan
- **Neck and Sacrum:** Remove the patients that have cancer in the neck and sacrum areas. Due to our small training data set, we wanted to 

## 3D-Unet Model for predicting Dose
**Inputs:** CT Scan, PTV mask, spinal cord mask
**Model Attributes:**.  
- 3x3 kernel size (stride 2), 
- 4 downsamples/upsamples (3, 32, 64, 128, 256, 512 channels)
- Batch Normalization Layers
- 2x2x2 Maxpool 
- ReLU activation functions (play with MISH Activation Function to see if it makes any difference)
- 
## Custom Loss Function (Weighted by Mask)**
**Method:** Arbitrarily iterate through different combinations of loss functions (L1, Huber, MSE) and penalization constants to see which produces the best results
- **PTV Mask:** Heavily penalize the model for adding dose outside of the PTV mask
- **Cord Mask:** Heavily penalize the model for adding dose to the spine
- **Note:** Unfortunately, not all the patients had a cord mask (placed a 0 matrix for patient's with no cord mask); therefore, many of the model's predictions have dose in the spine mask because 
- 
