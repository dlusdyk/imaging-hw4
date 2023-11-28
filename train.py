import os
import numpy as np
import mahotas
from sklearn.metrics import confusion_matrix
from scipy.spatial.distance import cdist
from skimage.measure import label, regionprops, moments, moments_central, moments_normalized, moments_hu
from skimage.morphology import closing, opening, dilation, erosion
from skimage import io, exposure
from skimage.filters import threshold_otsu, gaussian
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

def extract_features(img_path, plot=False):
  ###################################
  #----------- VISUALIZE -----------#
  ###################################
  img = io.imread(os.getcwd() + img_path)
  if plot:
    print('Img shape:', img.shape)
    io.imshow(img)
    plt.title('Original Image')
    io.show()

  ###################################
  #----------- HISTOGRAM -----------#
  ###################################
  hist = exposure.histogram(img)
  if plot:
    plt.bar(hist[1], hist[0])
    plt.title('Histogram')
    plt.show()

  ###################################################
  #----------- GAUSSIAN BLUR ENHANCEMENT -----------#
  ###################################################
  img = gaussian(img, sigma=1.0)

  #######################################################
  #----------- HISTOGRAM AFTER GAUSSIAN BLUR -----------#
  #######################################################
  hist = exposure.histogram(img)
  if plot:
    plt.bar(hist[1], hist[0])
    plt.title('Histogram After Gaussian Blur')
    plt.show()

  #################################################
  #----------- BINARIZATION (ENHANCED) -----------#
  #################################################
  #th = 200  # th can be tuned
  th = threshold_otsu(img)
  print('Otsu threshold:', th)
  img_binary = (img < th).astype(np.double)
  if plot:
    io.imshow(img_binary)
    plt.title('Binary Image')
    io.show()

  #############################################
  #----------- CLOSING ENHANCEMENT -----------#
  #############################################
  img_binary = closing(img_binary)
  #img_binary = dilation(img_binary)
  #img_binary = erosion(img_binary)
  #img_binary = closing(img_binary)
  #img_binary = opening(img_binary)

  #####################################
  #----------- CC ANALYSIS -----------#
  #####################################
  img_label = label(img_binary, background=0)
  if plot:
    io.imshow(img_label)
    plt.title('Labeled Image')
    io.show()
    print('Number of connected components before thresholding bounding boxes:', np.amax(img_label))

  ###########################################################
  #----------- FEATURE EXTRACTION (WITH ZERNIKE) -----------#
  ###########################################################
  regions = regionprops(img_label)
  if plot:
    io.imshow(img_binary)
  ax = plt.gca()

  Features = []
  props_list = []
  component_count = 0
  for props in regions:
    # Compute corners of bounding box
    minr, minc, maxr, maxc = props.bbox

    # Compute component height and width
    height = maxr - minr
    width = maxc - minc

    # Threshold to filter out smaller components
    if (height > 7 and width > 7):  # 7 7
      props_list.append(props)
      component_count += 1

      # Compute Hu moments
      roi = img_binary[minr:maxr, minc:maxc]
      m = moments(roi)
      cc = m[0, 1] / m[0, 0]
      cr = m[1, 0] / m[0, 0]
      mu = moments_central(roi, center=(cr, cc))
      nu = moments_normalized(mu)
      hu = moments_hu(nu)

      # Compute Zernike moments
      zernike = mahotas.features.zernike_moments(roi, 17) # 17

      # Append combined features array
      combined_moments = np.concatenate([hu, zernike])
      Features.append(combined_moments)

      # Add bounding boxes to image
      ax.add_patch(Rectangle((minc, minr), maxc - minc, maxr - minr, fill=False, edgecolor='red', linewidth=1))
  if plot:
    ax.set_title('Bounding Boxes')
    io.show()
    print('Number of components after thresholding bounding boxes:', component_count)
  return Features, img_binary, props_list

def full_dataset_extraction(paths, classes, plot=False):
  AllFeatures = []
  CharacterClass = []
  classes_index = 0
  img_binary_list = []
  props_list_list = []
  for path in paths:
    Features, img_binary, props_list = extract_features(path, plot=plot)
    img_binary_list.append(img_binary)
    props_list_list.append(props_list)
    #print('Features length:', len(Features))
    if plot:
      print('==============================================================================================================')
    #print(Features)
    AllFeatures.append(Features)
    for i in range(len(Features)):
      CharacterClass.append(classes[classes_index])
    classes_index += 1

  return AllFeatures, CharacterClass, img_binary_list, props_list_list

def normalize_features(Features):
  # Compute mean and SD of all features
  concatenated_features = np.concatenate(Features)
  mean = np.mean(concatenated_features)
  std = np.std(concatenated_features)

  # Write these stats to file system
  write_stats_to_files(mean, std)

  print('Mean:', mean)
  print('Std:', std)

  # Normalize each feature
  Features_Normalized = []
  for arr in Features:
      normalized_arr = (arr - mean) / std
      Features_Normalized.append(normalized_arr)

  #print('Features_Normalized:\n', Features_Normalized)
  print('Features_Normalized Length:', len(Features_Normalized))

  return Features_Normalized, mean, std

def predict(Features_Normalized, CharacterClass):
  D = cdist(Features_Normalized, Features_Normalized)
  #print(type(D))
  #print(D.ndim)
  #print(D)
  #print(len(D))
  plt.close()
  io.imshow(D)
  plt.title('Distance Matrix')
  io.show()

  D_index = np.argsort(D, axis=1)
  io.imshow(D_index)
  plt.title('Distance Index Matrix')
  io.show()
  print('D_index:\n', D_index)

  #print('CharacterClass:', CharacterClass[D_index[79, 1]])
  D_index_rows, _ = D_index.shape
  predictions = []
  for i in range(D_index_rows):
    predictions.append(CharacterClass[D_index[i, 1]])
  #print(predictions)

  confM = confusion_matrix(CharacterClass, predictions)
  io.imshow(confM)
  plt.title('Confusion Matrix')
  io.show()

  return D, D_index, predictions, confM

def predict_knn(Features_Normalized, CharacterClass, k=3, plot=True):
  knn = KNeighborsClassifier(n_neighbors=k)
  knn.fit(Features_Normalized, CharacterClass)

  # Predict on the training data for demonstration, should use a test set
  predictions = knn.predict(Features_Normalized)

  confM = confusion_matrix(CharacterClass, predictions)
  if plot:
    io.imshow(confM)
    plt.title('Confusion Matrix')
    io.show()

  return predictions, confM

def plot_labeled_result(img_binary, props_list, predictions):
  io.imshow(img_binary)
  ax = plt.gca()

  for idx, props in enumerate(props_list):
    # Compute corners of bounding box
    minr, minc, maxr, maxc = props.bbox

    # Add bounding boxes to image
    ax.add_patch(Rectangle((minc, minr), maxc - minc, maxr - minr, fill=False, edgecolor='red', linewidth=1))
    # Add prediction of bounding box
    ax.text(minc, minr - 5, f'{predictions[idx]}', color='green', fontsize=10)

  ax.set_title('Bounding Boxes With Predictions')
  io.show()

def percent_predicted_successfully(predictions, true_values):
  match = 0
  for idx, prediction in enumerate(predictions):
    if prediction == true_values[idx]: match += 1
  return round((match / len(true_values)) * 100, 2)

def write_stats_to_files(mean, std):
  with open(os.getcwd() + '/stats/mean.txt', 'w') as file:
    file.write(str(mean))
  with open(os.getcwd() + '/stats/std.txt', 'w') as file:
    file.write(str(std))

def main(plot=True):
  paths = ('/images/a.bmp', '/images/d.bmp', '/images/m.bmp', '/images/n.bmp', '/images/o.bmp', '/images/p.bmp', '/images/q.bmp', '/images/r.bmp', '/images/u.bmp', '/images/w.bmp')
  classes = ('a', 'd', 'm', 'n', 'o', 'p', 'q', 'r', 'u', 'w')

  Features, CharacterClass, img_binary_list, props_list_list = full_dataset_extraction(paths, classes, plot=plot)
  Features = [item for sublist in Features for item in sublist]
  #print('Features:\n', Features)
  print('Features Length:', len(Features))
  print('CharacterClass Length:', len(CharacterClass))

  Features_Normalized, mean, std = normalize_features(Features)

  #D, D_index, predictions, confM = predict(Features_Normalized, CharacterClass)
  predictions, confM = predict_knn(Features_Normalized, CharacterClass, k=3, plot=plot)

  print('Percent predicted successfully:', percent_predicted_successfully(predictions, CharacterClass))

  # Store arrays
  np.save(os.getcwd() + '/np/Features_Normalized.npy', Features_Normalized)
  np.save(os.getcwd() + '/np/CharacterClass.npy', CharacterClass)

  if plot:
    # Plot bounding boxed images with their labeled predictions
    for idx, img_binary in enumerate(img_binary_list):
      plot_labeled_result(img_binary, props_list_list[idx], predictions[:len(props_list_list[idx])])
      predictions = predictions[len(props_list_list[idx]):]

#main(plot=False)