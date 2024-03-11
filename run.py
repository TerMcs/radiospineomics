import os
import pandas as pd
import radiomics
import yaml

from functions import calculate_indices


# Functional spinal unit (FSU) masks and the whole spine image are selected.
# Mask list should be in the order cranial VB, IVD, caudal VB.
fsu = ['sample_data/masks/L1_VB-P0741_756_IMG00004.nrrd',
       'sample_data/masks/L1_L2_IVD-P0741_756_IMG00004.nrrd', 
       'sample_data/masks/L2_VB-P0741_756_IMG00004.nrrd']

image = 'sample_data/images/P0741_756_IMG00004.nrrd'


###################################################################
###### Calculate conventional features ############################
###################################################################

try:
    geometry_results = calculate_indices(fsu, image)

# If necessary, identify masks that fail:
except Exception as e:
    print(f'Error extracting features for {fsu[1]}: {e}')

geometry_results = pd.DataFrame(geometry_results, index=[0])
geometry_results['id'] = fsu[1].split('/')[-1]
geometry_results.to_csv('sample_results/conventional_features.csv', index=False)


###################################################################
###### Calculate texture features #################################
###################################################################

texture_params_file = 'texture_params.yaml'

# Define resampling and binwidths
resamplings = [[0,0], [0.27, 0.27], [0.135, 0.135]]
binwidths = [1, 2, 4, 8, 16, 32, 64]

results = []

# Run the radiomics feature extractor looping through chosen resampling and binwidths:
for resampling in resamplings:
    for binwidth in binwidths:

        extractor = radiomics.featureextractor.RadiomicsFeatureExtractor(texture_params_file)

        with open(texture_params_file, 'r') as f:
            params = yaml.safe_load(f)
            params['setting']['binWidth'] = binwidth
            params['setting']['resampledPixelSpacing'] = resampling

        with open(texture_params_file, 'w') as f:
            yaml.safe_dump(params, f)

        try:
            featureVector = extractor.execute(image, fsu[1])
        
        except Exception as e:
            print(f'Error extracting features for {fsu[1]}: {e}')
            continue

        df = pd.DataFrame.from_dict(featureVector, orient= 'index')
        df = df.T
        df['id'] = fsu[1].split('/')[-1]
        df['resampling'] = resampling[0]
        df['binwidth'] = binwidth
        
        results.append(df)

texture_features = pd.concat(results, ignore_index=True)

texture_features.to_csv('sample_results/texture_features.csv', index=False)


###################################################################
###### Calculate shape features ###################################
###################################################################

# The shape features don't need to be calculated for each resampling and binwidth.

shape_params_file = 'shape_params.yaml'

extractor = radiomics.featureextractor.RadiomicsFeatureExtractor(shape_params_file)

try:
    featureVector = extractor.execute(image, fsu[1])

except Exception as e:
    print(f'Error extracting features for {fsu[1]}: {e}')

shape_features = pd.DataFrame.from_dict(featureVector, orient= 'index')
shape_features = shape_features.T
shape_features['id'] = fsu[1].split('/')[-1]

shape_features.to_csv('sample_results/shape_features.csv', index=False)

