import glob
import os
import pandas as pd
import radiomics
import yaml

from functions import calculate_indices


mask_directory = 'sample_data/masks/'
image_directory = 'sample_data/images/'
ivd_mask_list = [os.path.basename(x) for x in glob.glob(mask_directory + '*IVD*.nrrd')]

# Settings:
shape_params_file = 'shape_params.yaml'
texture_params_file = 'texture_params.yaml'
resamplings = [[0,0], [0.27, 0.27], [0.135, 0.135]]
binwidths = [1, 2, 4, 8, 16, 32, 64]

conventional_features = []
texture_features = []
shape_features = []

for ivd in ivd_mask_list:

    slice_id = ivd.split('-')[1]
    upper_vb = ivd.split('_')[0] + '_VB-' + slice_id
    lower_vb = ivd.split('_')[1] + '_VB-' + slice_id
    
    image_filepath = image_directory + slice_id
    fsu_filepaths = [mask_directory + upper_vb, mask_directory + ivd, mask_directory + lower_vb]
    

    ###################################################################
    ###### Calculate conventional features ############################
    ###################################################################

    try:
        geometry_results = calculate_indices(fsu_filepaths, image_filepath)
        geometry_results = pd.DataFrame(geometry_results, index=[0])
        geometry_results['id'] = ivd
    
    # If necessary, identify masks/images that fail and either skip or record them in the same dataframe:
    except Exception as e:
        print(f'Error extracting conventional features for {ivd}: {e}')
        # continue
        geometry_results = pd.DataFrame({'id': [ivd]})

    conventional_features.append(geometry_results)


    ###################################################################
    ###### Calculate texture features #################################
    ###################################################################

    results = []

    for resampling in resamplings:
        for binwidth in binwidths:

            texture_extractor = radiomics.featureextractor.RadiomicsFeatureExtractor(texture_params_file)

            with open(texture_params_file, 'r') as f:
                params = yaml.safe_load(f)
                params['setting']['binWidth'] = binwidth
                params['setting']['resampledPixelSpacing'] = resampling

            with open(texture_params_file, 'w') as f:
                yaml.safe_dump(params, f)

            try:
                texture_featureVector = texture_extractor.execute(image_filepath, fsu_filepaths[1])
                df = pd.DataFrame.from_dict(texture_featureVector, orient='index')
                df = df.T
                df['id'] = ivd
                df['resampling'] = resampling[0]
                df['binwidth'] = binwidth
            
            except Exception as e:
                print(f'Error extracting texture features for {ivd} at resampling {resampling} and binwidth {binwidth}: {e}')
                # continue
                df =  pd.DataFrame({'id': [ivd], 'resampling': [resampling[0]], 'binwidth': [binwidth]})

            results.append(df)

    texture_results = pd.concat(results, ignore_index=True)

    texture_features.append(texture_results)


    ###################################################################
    ###### Calculate shape features ###################################
    ###################################################################

    shape_extractor = radiomics.featureextractor.RadiomicsFeatureExtractor(shape_params_file)

    try:
        shape_featureVector = shape_extractor.execute(image_filepath, fsu_filepaths[1])
        shape_results = pd.DataFrame.from_dict(shape_featureVector, orient= 'index')
        shape_results = shape_results.T
        shape_results['id'] = ivd
        
    except Exception as e:
        print(f'Error extracting shape features for {ivd}: {e}')
        # continue
        shape_results = pd.DataFrame({'id': [ivd]})
        
    shape_features.append(shape_results)


conventional_features = pd.concat(conventional_features, ignore_index=True)
conventional_features.to_csv('sample_results/conventional_features.csv', index=False)

texture_features = pd.concat(texture_features, ignore_index=True)
texture_features.to_csv('sample_results/texture_features.csv', index=False)

shape_features = pd.concat(shape_features, ignore_index=True)
shape_features.to_csv('sample_results/shape_features.csv', index=False)
