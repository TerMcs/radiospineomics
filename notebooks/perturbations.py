import pandas as pd
import os
import random
import cv2
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import SimpleITK as sitk


def random_translate(image, mask, max_translation):

    negative_max_translation = random.randint(-max_translation, -2)
    positive_max_translation = random.randint(3, max_translation + 1)
    translations = [negative_max_translation, positive_max_translation]
    tx = random.choice(translations)
    ty = random.choice(translations)

    image = image.astype(np.uint8)
    mask = mask.astype(np.uint8)

    translation_matrix = np.float32([[1, 0, tx], [0, 1, ty]])
    translated_image = cv2.warpAffine(image, translation_matrix, (image.shape[1], image.shape[0]), flags=cv2.INTER_LINEAR)
    translated_mask = cv2.warpAffine(mask, translation_matrix, (image.shape[1], image.shape[0]), flags=cv2.INTER_NEAREST)

    return translated_image, translated_mask


def random_rotate(image, mask, max_angle):
    negative_angle = random.uniform(-max_angle, -2)
    positive_angle = random.uniform(3, (max_angle + 1))
    angles = [negative_angle, positive_angle]
    angle = random.choice(angles)

    image = image.astype(np.uint8)
    mask = mask.astype(np.uint8)

    center = (image.shape[1] // 2, image.shape[0] // 2, )
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1)
    rotated_image = cv2.warpAffine(image, rotation_matrix, (image.shape[1], image.shape[0]), flags=cv2.INTER_LINEAR)
    rotated_mask = cv2.warpAffine(mask, rotation_matrix, (image.shape[1], image.shape[0]), flags=cv2.INTER_NEAREST)

    return rotated_image, rotated_mask


def apply_translation_rotation(image, mask, max_translation, max_angle):
    image, mask = random_translate(image, mask, max_translation)
    image, mask = random_rotate(image, mask, max_angle)

    return image, mask


def add_gaussian_noise(image, mean, sigma):
    row, col = image.shape
    gauss = np.random.normal(mean, sigma, (row, col))
    gauss = gauss.reshape(row, col)

    noisy = image + gauss

    noisy = np.clip(noisy, 0, 255).astype('uint8')
    
    return noisy


def dilate_mask(mask, size):
    kernel = np.ones((size, size), np.uint8)
    dilated_mask = cv2.dilate(mask, kernel, iterations=2)
    return dilated_mask


def erode_mask(mask, size):
    kernel = np.ones((size, size), np.uint8)
    eroded_mask = cv2.erode(mask, kernel, iterations=2)
    return eroded_mask


def contour_randomisation(mask, probability=0.1, kernel_size=3):
    edges = cv2.Canny(mask, 100, 200)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))

    modified_mask = np.copy(mask)

    for y in range(edges.shape[0]):
        for x in range(edges.shape[1]):
            if edges[y, x] != 0:  
                if random.random() < probability: 
                    operation = random.choice(['dilate', 'erode'])
                    if operation == 'dilate':
                        dilation_result = cv2.dilate(mask[y:y+kernel_size, x:x+kernel_size], kernel, iterations=1)
                        modified_mask[y:y+kernel_size, x:x+kernel_size] = dilation_result
                    elif operation == 'erode':
                        erosion_result = cv2.erode(mask[y:y+kernel_size, x:x+kernel_size], kernel, iterations=1)
                        modified_mask[y:y+kernel_size, x:x+kernel_size] = erosion_result

    return modified_mask


##################################################################################################
mask_path = "data/nrrd/perturbations/masks/dl/"
all_files = os.listdir(mask_path)
mask_files = [file for file in all_files if file.endswith('.nrrd')]

image_files = []
for mask in mask_files:
    image = mask.split('-')[1]
    image_files.append(image)
##################################################################################################


im_root = "data/nrrd/perturbations/images/"
mask_root = "data/nrrd/perturbations/masks/"

image_directory                                                     = im_root + "original/" 
noisy_image_directory                                               = im_root + "noisy/" 
rot_translation_image_directory                                     = im_root + "rotated_translated/" 
rot_translation_noisy_image_directory                               = im_root + "rotated_translated_noisy/" 

juhani_mask_directory                                               = mask_root + "juhani/" 
terence_mask_directory                                              = mask_root + "terence/" 
dl_mask_directory                                                   = mask_root + "dl/" 
dl_rotation_translation_mask_directory                              = mask_root + "dl_rotation_translation/" 
dl_erosion_mask_directory                                           = mask_root + "dl_erosion/" 
dl_dilation_mask_directory                                          = mask_root + "dl_dilation/" 
dl_contour_randomisation_mask_directory                             = mask_root + "dl_contour_randomisation/" 
dl_dilation_rotation_translation_mask_directory                     = mask_root + "dl_dilation_rotation_translation/"
dl_erosion_rotation_translation_mask_directory                      = mask_root + "dl_erosion_rotation_translation/"
dl_contour_randomisation_rotation_translation_mask_directory        = mask_root + "dl_contour_randomisation_rotation_translation/"

noise = [0.1, 0.2, 0.4, 0.8, 1.6, 3.2, 6.4, 12.8, 25.6, 51.2]
max_translation = 10
max_angle = 45

mask_selection = mask_files
image_selection = image_files

def main():

    print("Generating noisy images")
    for sigma in noise:
        for image_id in tqdm(image_selection):
            image_path = os.path.join(image_directory, image_id)
            
            image = sitk.ReadImage(image_path)
            spacing = image.GetSpacing()
            image = sitk.GetArrayFromImage(image)
        
            
            noisy_image = add_gaussian_noise(image, 0, sigma)
            sigma_str = str(sigma).replace('.', '')
            noisy_image_path = os.path.join(noisy_image_directory, sigma_str, image_id)

            noisy_image = sitk.GetImageFromArray(noisy_image)
            noisy_image.SetSpacing(spacing)
            sitk.WriteImage(noisy_image, noisy_image_path)

    print("Generating translated images and masks")
    for mask_id in tqdm(mask_selection):

        mask_path = os.path.join(dl_mask_directory, mask_id)
        mask = sitk.ReadImage(mask_path)
        mask = sitk.GetArrayFromImage(mask)

        image_id = mask_id.split('-')[1]
        image_path = os.path.join(image_directory, image_id)
        image = sitk.ReadImage(image_path)
        spacing = image.GetSpacing()
        image = sitk.GetArrayFromImage(image)

        rotated_translated_image, rotated_translated_mask = apply_translation_rotation(image, mask, max_translation, max_angle)
        rotated_translated_image_path = os.path.join(rot_translation_image_directory, mask_id)
        rotated_translated_mask_path = os.path.join(dl_rotation_translation_mask_directory, mask_id)
        
        rotated_translated_image = sitk.GetImageFromArray(rotated_translated_image)
        rotated_translated_image.SetSpacing(spacing)
        sitk.WriteImage(rotated_translated_image, rotated_translated_image_path)

        rotated_translated_mask = sitk.GetImageFromArray(rotated_translated_mask)
        rotated_translated_mask.SetSpacing(spacing)
        sitk.WriteImage(rotated_translated_mask, rotated_translated_mask_path)


    print("Generating rotated and translated noisy images and masks")
    for mask_id in tqdm(mask_selection):

        rotated_translated_image_path = os.path.join(rot_translation_image_directory, mask_id)

        for sigma in noise:
            rotated_translated_image = sitk.ReadImage(rotated_translated_image_path)
            spacing = rotated_translated_image.GetSpacing()
            rotated_translated_image = sitk.GetArrayFromImage(rotated_translated_image)

            rotated_translated_noisy_image = add_gaussian_noise(rotated_translated_image, 0, sigma)
            sigma_str = str(sigma).replace('.', '')
            rotated_translated_noisy_image_path = os.path.join(rot_translation_noisy_image_directory, sigma_str, mask_id)
            
            rotated_translated_noisy_image = sitk.GetImageFromArray(rotated_translated_noisy_image)
            rotated_translated_noisy_image.SetSpacing(spacing)
            sitk.WriteImage(rotated_translated_noisy_image, rotated_translated_noisy_image_path)

    print("Adding dilation and erosion to the masks")
    for mask_id in tqdm(mask_selection):
        mask_path = os.path.join(dl_mask_directory, mask_id)
        mask = sitk.ReadImage(mask_path)
        spacing = mask.GetSpacing()
        mask = sitk.GetArrayFromImage(mask)

        dilated_mask = dilate_mask(mask, 2)
        dilated_mask_path = os.path.join(dl_dilation_mask_directory, mask_id)
        dilated_mask = sitk.GetImageFromArray(dilated_mask)
        dilated_mask.SetSpacing(spacing)
        sitk.WriteImage(dilated_mask, dilated_mask_path)

        eroded_mask = erode_mask(mask, 2)
        eroded_mask_path = os.path.join(dl_erosion_mask_directory, mask_id)
        eroded_mask = sitk.GetImageFromArray(eroded_mask)
        eroded_mask.SetSpacing(spacing)
        sitk.WriteImage(eroded_mask, eroded_mask_path)

    print("Adding dilation and erosion to the rotated translated masks")
    for mask_id in tqdm(mask_selection):
        mask_path = os.path.join(dl_rotation_translation_mask_directory, mask_id)
        mask = sitk.ReadImage(mask_path)
        spacing = mask.GetSpacing()
        mask = sitk.GetArrayFromImage(mask)

        dilated_mask = dilate_mask(mask, 2)
        dilated_mask_path = os.path.join(dl_dilation_rotation_translation_mask_directory, mask_id)
        dilated_mask = sitk.GetImageFromArray(dilated_mask)
        dilated_mask.SetSpacing(spacing)
        sitk.WriteImage(dilated_mask, dilated_mask_path)

        eroded_mask = erode_mask(mask, 2)
        eroded_mask_path = os.path.join(dl_erosion_rotation_translation_mask_directory, mask_id)
        eroded_mask = sitk.GetImageFromArray(eroded_mask)
        eroded_mask.SetSpacing(spacing)
        sitk.WriteImage(eroded_mask, eroded_mask_path)

    print("Adding contour randomisation to the masks")
    for mask_id in tqdm(mask_selection):
        mask_path = os.path.join(dl_mask_directory, mask_id)
        mask = sitk.ReadImage(mask_path)
        spacing = mask.GetSpacing()
        mask = sitk.GetArrayFromImage(mask)

        mask = mask.astype(np.uint8)

        contour_randomised_mask = contour_randomisation(mask, 0.05, 3)
        contour_randomised_mask_path = os.path.join(dl_contour_randomisation_mask_directory, mask_id)
        contour_randomised_mask = sitk.GetImageFromArray(contour_randomised_mask)
        contour_randomised_mask.SetSpacing(spacing)
        sitk.WriteImage(contour_randomised_mask, contour_randomised_mask_path)


    print("Adding contour randomisation to the rotated translated masks")
    for mask_id in tqdm(mask_selection):
        mask_path = os.path.join(dl_rotation_translation_mask_directory, mask_id)
        mask = sitk.ReadImage(mask_path)
        spacing = mask.GetSpacing()
        mask = sitk.GetArrayFromImage(mask)

        mask = mask.astype(np.uint8)

        contour_randomised_mask = contour_randomisation(mask, 0.05, 3)
        contour_randomised_mask_path = os.path.join(dl_contour_randomisation_rotation_translation_mask_directory, mask_id)
        contour_randomised_mask = sitk.GetImageFromArray(contour_randomised_mask)
        contour_randomised_mask.SetSpacing(spacing)
        sitk.WriteImage(contour_randomised_mask, contour_randomised_mask_path)

if __name__ == "__main__":
    main()


