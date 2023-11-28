import os
import numpy as np
import mahotas
from sklearn.metrics import confusion_matrix
from scipy.spatial.distance import cdist
from skimage.measure import label, regionprops, moments, moments_central, moments_normalized, moments_hu
from skimage.morphology import closing, opening, dilation
from skimage import io, exposure
from skimage.filters import threshold_otsu, gaussian
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import pickle

def extract_test_image_features(img_path, plot=False):
  ###################################
  #----------- VISUALIZE -----------#
  ###################################
  img = io.imread(os.getcwd() + img_path)
  if plot:
    print(img.shape)
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
  img_binary = (img < th).astype(np.double)
  if plot:
    io.imshow(img_binary)
    plt.title('Binary Image')
    io.show()

  #############################################
  #----------- CLOSING ENHANCEMENT -----------#
  #############################################
  ##img_binary = closing(img_binary)
  #img_bianry = dilation(img_binary)
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
    if (height > 10 and width > 10):  # 10 10
      props_list.append(props)
      component_count += 1

      # Compute moments
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

def normalize_features(Features, training_mean, training_std):
  concatenated_features = np.concatenate(Features)
  Features_Normalized = []
  for arr in Features:
    normalized_arr = (arr - training_mean) / training_std
    Features_Normalized.append(normalized_arr)
  return Features_Normalized

def predict(Test_Features_Normalized, Training_Features_Normalized, CharacterClass):
  D = cdist(Test_Features_Normalized, Training_Features_Normalized)
  #print(type(D))
  #print(D.ndim)
  #print(D)
  #print(len(D))
  plt.close()
  io.imshow(D)
  plt.title('Distance Matrix For Single Nearest Neighbor (Not KNN)', loc='right')
  io.show()

  '''
  D_index = np.argsort(D, axis=1)
  io.imshow(D_index)
  plt.title('Distance Index Matrix')
  io.show()
  #print(D_index)

  #print('CharacterClass:', CharacterClass[D_index[79, 1]])
  D_index_rows, _ = D_index.shape
  predictions = []
  for i in range(D_index_rows):
    predictions.append(CharacterClass[D_index[i, 0]])
  #print(prediction)

  return D, D_index, predictions
  '''

def predict_knn(Test_Features_Normalized, Training_Features_Normalized, CharacterClass, k=3):
  knn = KNeighborsClassifier(n_neighbors=k)
  knn.fit(Training_Features_Normalized, CharacterClass)

  predictions = knn.predict(Test_Features_Normalized)

  return predictions

def percent_predicted_successfully(predictions, true_values):
  correct = 0
  for idx, prediction in enumerate(predictions):
    if idx == 70:
      break
    if prediction == true_values[idx]:
      correct += 1
  print('Number of characters correctly identified:', correct)
  return round((correct / 70) * 100, 2)

def plot_labeled_result(img_binary, props_list, predictions, plot=True):
  if plot:
    io.imshow(img_binary)
  ax = plt.gca()

  for idx, props in enumerate(props_list):
    # Compute corners of bounding box
    minr, minc, maxr, maxc = props.bbox

    # Add bounding boxes to image
    ax.add_patch(Rectangle((minc, minr), maxc - minc, maxr - minr, fill=False, edgecolor='red', linewidth=1))
    # Add prediction of bounding box
    ax.text(minc, minr - 5, f'{predictions[idx]}', color='green', fontsize=10)

  if plot:
    ax.set_title('Bounding Boxes With Predictions')
    io.show()

def compare_to_ground_truth(pkl_file_name, props_list, predictions):
  pkl_file = open(os.getcwd() + '/pkl/' + pkl_file_name, 'rb')
  mydict = pickle.load(pkl_file)
  pkl_file.close()
  classes = mydict['classes'.encode()]
  locations = mydict['locations'.encode()]

  classes = np.char.lower(classes)

  #print(classes)
  #print(locations)

  match_count = 0
  for i, location in enumerate(locations):
    col, row = location
    for j, props in enumerate(props_list):
      # Compute corners of bounding box
      minr, minc, maxr, maxc = props.bbox

      if (row >= minr and row <= maxr and col >= minc and col <= maxc):
        if (classes[i] == predictions[j]): match_count += 1
        break

  print('Number of predictions matching the ground truth:', match_count)
  print('Percent of ground truth characters predicted:', round((match_count / len(locations)) * 100, 2))

def main(test_image_name, pkl_file_name, plot=True):
  Training_Features_Normalized = np.load(os.getcwd() + '/np/Features_Normalized.npy')
  CharacterClass = np.load(os.getcwd() + '/np/CharacterClass.npy')

  with open(os.getcwd() + '/stats/mean.txt', 'r') as file:
    training_mean = float(file.read())
  with open(os.getcwd() + '/stats/std.txt', 'r') as file:
    training_std = float(file.read())

  print('Training mean:', training_mean)
  print('Training std:', training_std)

  Features, img_binary, props_list = extract_test_image_features('/images/' + test_image_name, plot=plot)
  Test_Features_Normalized = normalize_features(Features, training_mean, training_std)
  
  #D, D_index, predictions = predict(Test_Features_Normalized, Training_Features_Normalized, CharacterClass)
  if plot:
    predict(Test_Features_Normalized, Training_Features_Normalized, CharacterClass)
  
  predictions = predict_knn(Test_Features_Normalized, Training_Features_Normalized, CharacterClass, k=6)  # 6

  '''
  confM = confusion_matrix(CharacterClass, predictions)
  io.imshow(confM)
  plt.title('Confusion Matrix')
  io.show()
  '''
  '''
  print('Predictions: [', end='')
  for idx, pred in enumerate(predictions):
    if idx % 7 == 0:
      print('\n')
    print(str(pred) + ', ', end='')
  print(']')
  '''

  characters = ('a', 'd', 'm', 'n', 'o', 'p', 'q', 'r', 'u', 'w')
  true_values = []
  idx = 0
  for i in range(1, 71):
    true_values.append(characters[idx])
    if (i % 7 == 0): idx += 1

  #percent_successfully_predicted = percent_predicted_successfully(predictions, true_values)
  #print('Percent successfully predicted:', percent_successfully_predicted)

  plot_labeled_result(img_binary, props_list, predictions, plot=plot)

  compare_to_ground_truth(pkl_file_name, props_list, predictions)

#main(plot=True)