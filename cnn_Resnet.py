import numpy as np
import collections
import SimpleITK as sitk
from scipy.ndimage.interpolation import zoom
import os,sys
import pandas as pd
from keras.preprocessing import image
import pandas
import logging

## Set GPU
#os.environ["CUDA_VISIBLE_DEVICES"]="0"
from keras.models import Model
# Load model
from keras.applications.resnet50 import ResNet50
from keras.applications.imagenet_utils import preprocess_input, decode_predictions

w='vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5'
base_model = ResNet50(weights=w, include_top=False,pooling='max')
model = Model(inputs=base_model.input, outputs=base_model.layers[38].output)


# read images in Nifti format 
def loadSegArraywithID(fold,iden):
    
    path = fold
    pathList = os.listdir(path)
    
    segPath = [os.path.join(path,i) for i in pathList if ('label' in i.lower()) & (iden in i.lower())][0]
    seg = sitk.ReadImage(segPath)
    return seg
# read regions of interest (ROI) in Nifti format 
def loadImgArraywithID(fold,iden):
    
    path = fold
    pathList = os.listdir(path)
    
    imgPath = [os.path.join(path,i) for i in pathList if ('im' in i.lower()) & (iden in i.lower())][0]
    img = sitk.ReadImage(imgPath)    
    return img

# Feature Extraction
#Cropping box
def maskcroppingbox(images_array, use2D=False):
    images_array_2 = np.argwhere(images_array)
    (zstart, ystart, xstart), (zstop, ystop, xstop) = images_array_2.min(axis=0), images_array_2.max(axis=0) + 1
    return (zstart, ystart, xstart), (zstop, ystop, xstop)
        
def featureextraction(imageFilepath,maskFilepath):
    image_array = sitk.GetArrayFromImage(imageFilepath) 
    mask_array = sitk.GetArrayFromImage(maskFilepath)
    (zstart, ystart, xstart), (zstop, ystop, xstop) = maskcroppingbox(mask_array, use2D=False)
    roi_images = image_array[zstart-1:zstop+1,ystart:ystop,xstart:xstop].transpose((2,1,0))
    roi_images1 = zoom(roi_images, zoom=[224/roi_images.shape[0], 224/roi_images.shape[1],1], order=3)
    roi_images2 = np.array(roi_images1,dtype=np.float)    
    
    x = image.img_to_array(roi_images2)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    base_model_pool_features = model.predict(x)
    
    feature_map = base_model_pool_features[0]
    feature_map = feature_map.transpose((2,1,0))
    features = np.max(feature_map,-1)
    features = np.max(features,-1)
    deeplearningfeatures = collections.OrderedDict()
    for ind_,f_ in enumerate(features):
    	deeplearningfeatures[str(ind_)] = f_
    return deeplearningfeatures

def main():
  outPath = '/Resnet'

  inputCSV = os.path.join('DL.csv')
  outputFilepath = os.path.join('Output_SAG_Resnet_original_notop.csv')

  # Configure logging
  rLogger = logging.getLogger('DLradiomics')

  # Initialize logging for batch log messages
  logger = rLogger.getChild('batch')

  # logger.info('pyradiomics version: %s', radiomics.__version__)
  logger.info('Loading CSV')

  # ####### Up to this point, this script is equal to the 'regular' batchprocessing script ########

  try:
    # Use pandas to read and transpose ('.T') the input data
    # The transposition is needed so that each column represents one test case. This is easier for iteration over
    # the input cases
    flists = pandas.read_csv(inputCSV).T
  except Exception:
    logger.error('CSV READ FAILED', exc_info=True)
    exit(-1)

  logger.info('Loading Done')
  logger.info('Patients: %d', len(flists.columns))

  # Instantiate a pandas data frame to hold the results of all patients
  results = pandas.DataFrame()

  for entry in flists:  # Loop over all columns (i.e. the test cases)
    logger.info("(%d/%d) Processing Patient (Image: %s, Mask: %s)",
                entry + 1,
                len(flists),
                flists[entry]['Image'],
                flists[entry]['Mask'])

    imageFilepath = flists[entry]['Image']
    maskFilepath = flists[entry]['Mask']
    label = flists[entry].get('Label', None)

    if str(label).isdigit():
      label = int(label)
    else:
      label = None

    if (imageFilepath is not None) and (maskFilepath is not None):
      featureVector = flists[entry]  # This is a pandas Series
      featureVector['Image'] = os.path.basename(imageFilepath)
      featureVector['Mask'] = os.path.basename(maskFilepath)

      try:
        # PyRadiomics returns the result as an ordered dictionary, which can be easily converted to a pandas Series
        # The keys in the dictionary will be used as the index (labels for the rows), with the values of the features
        # as the values in the rows.
        # color_channel = 0
        im=sitk.ReadImage(imageFilepath)
        mask=sitk.ReadImage(maskFilepath)

        result = pandas.Series(featureextraction(im, mask))
        featureVector = featureVector.append(result)

      except Exception:
        logger.error('FEATURE EXTRACTION FAILED:', exc_info=True)

      # To add the calculated features for this case to our data frame, the series must have a name (which will be the
      # name of the column.
      featureVector.name = entry
      # By specifying an 'outer' join, all calculated features are added to the data frame, including those not
      # calculated for previous cases. This also ensures we don't end up with an empty frame, as for the first patient
      # it is 'joined' with the empty data frame.
      results = results.join(featureVector, how='outer')  # If feature extraction failed, results will be all NaN

  logger.info('Extraction complete, writing CSV')
  # .T transposes the data frame, so that each line will represent one patient, with the extracted features as columns
  results.T.to_csv(outputFilepath, index=False, na_rep='NaN')
  logger.info('CSV writing complete')


if __name__ == '__main__':
  main()
