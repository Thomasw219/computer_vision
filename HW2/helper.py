import nbimporter
import numpy as np
import scipy.ndimage
from skimage import io
import skimage.transform
import os,time
import util
import multiprocessing
import threading
import queue
import torch
import torchvision
import torchvision.transforms

def preprocess_image(image):
    '''
    Preprocesses the image to load into the prebuilt network.

    [input]
    * image: numpy.ndarray of shape (H,W,3)

    [output]
    * image_processed: torch.array of shape (3,H,W)
    '''

    # ----- TODO -----
    
    if(len(image.shape) == 2):
        image = np.stack((image, image, image), axis=-1)

    if(image.shape == 3 and image.shape[2] == 1):
        image = np.concatenate((image, image, image), axis=-1)

    if(image.shape[2] == 4):
        image = image[:, :, 0:3]
    '''
    HINTS:
    1.> Resize the image (look into skimage.transform.resize)
    2.> normalize the image
    3.> convert the image from numpy to torch
    '''
    # YOUR CODE HERE
    image = skimage.transform.resize(image, (224, 224))
    output = []
    dim = 3
    means = [0.485, 0.456, 0.406]
    stds = [0.229, 0.224, 0.225]
    for d in range(dim):
        channel_image = image[:, :, d]
        normalized_image = channel_image / np.std(channel_image) * stds[d]
        normalized_image = normalized_image - np.mean(normalized_image) + means[d]
        output.append(normalized_image)
        
    image_processed = torch.tensor(np.stack(output, axis=0))
    return image_processed

path_img = "./data/aquarium/sun_aztvjgubyrgvirup.jpg"
image = io.imread(path_img)

vgg16 = torchvision.models.vgg16(pretrained=True).double()
vgg16.eval()

def get_image_feature(args):
    '''
    Extracts deep features from the prebuilt VGG-16 network.
    This is a function run by a subprocess.
    [input]
    * i: index of training image
    * image_path: path of image file
    * vgg16: prebuilt VGG-16 network.
    
    [output]
    * feat: evaluated deep feature
    '''
    i, image_path, vgg16 = args
    img_torch = io.imread(image_path)
    print(i)
    
    '''
    HINTS:
    1.> Think along the lines of evaluate_deep_extractor
    '''
    # ----- TODO -----
    # YOUR CODE HERE
    img_torch = preprocess_image(image)
    
    with torch.no_grad():
        vgg_classifier = torch.nn.Sequential(*list(vgg16.classifier.children())[:-3])
        vgg_feat_feat = vgg16.features(img_torch[None, ])
        vgg_feat_feat = vgg_classifier(vgg_feat_feat.flatten())
    feat = vgg_feat_feat.numpy()
    
    return [i,feat]

def build_recognition_system(vgg16, num_workers=2):
    '''
    Creates a trained recognition system by generating training features from all training images.

    [input]
    * vgg16: prebuilt VGG-16 network.
    * num_workers: number of workers to process in parallel

    [saved]
    * features: numpy.ndarray of shape (N,K)
    * labels: numpy.ndarray of shape (N)
    '''

    train_data = np.load("./data/train_data.npz", allow_pickle=True)
    '''
    HINTS:
    1.> Similar approach as Q1.2.2 and Q3.1.1 (create an argument list and use multiprocessing)
    2.> Keep track of the order in which input is given to multiprocessing
    '''
    # YOUR CODE HERE
    start = 500
    end = 1000
    image_names = train_data['files'][start:end]
    image_labels = train_data['labels'][start:end]
    
    # I tried multiprocessing but it would hang and I couldn't find a solution
    ordered_features = []
    for i in range(image_names.shape[0]):
        full_image_name = './data/' + image_names[i]
        i, feat = get_image_feature([i, full_image_name, vgg16])
        ordered_features.append(feat.flatten())
    

    '''
    HINTS:
    1.> reorder the features to their correct place as input
    '''
    # YOUR CODE HERE
    ordered_features = np.array(ordered_features)
    labels = np.array(image_labels)
        
    print("done", ordered_features.shape)
    
    np.savez('trained_system_deep{}-{}.npz'.format(start, end - 1), features=ordered_features, labels=labels)

#build_recognition_system(vgg16, num_workers=1)

def helper_func(args):
    # YOUR CODE HERE
    i, file_path, vgg16, trained_features, train_labels = args
    feature = get_image_feature([i, file_path, vgg16])[1]
    distances = scipy.spatial.distance.cdist(feature.reshape((1, -1)), trained_features).ravel()
    nearest_image_idx = np.argmin(distances)
    pred_label = train_labels[nearest_image_idx]
    return [i, pred_label]


def evaluate_recognition_system(vgg16, num_workers=2):
    '''
    Evaluates the recognition system for all test images and returns the confusion matrix.

    [input]
    * vgg16: prebuilt VGG-16 network.
    * num_workers: number of workers to process in parallel

    [output]
    * conf: numpy.ndarray of shape (8,8)
    * accuracy: accuracy of the evaluated system
    '''
    '''
    HINTS:
    (1) Students can write helper functions (in this cell) to use multi-processing
    '''
    test_data = np.load("./data/test_data.npz", allow_pickle=True)

    # ----- TODO -----
    trained_system = np.load("trained_system_deep.npz", allow_pickle=True)
    image_names = test_data['files']
    test_labels = test_data['labels']

    trained_features = trained_system['features']
    train_labels = trained_system['labels']

    print("Trained features shape: ", trained_features.shape)
    
    '''
    HINTS:
    1.> [Important] Can write a helper function in this cell of jupyter notebook for multiprocessing
    
    2.> Helper function will compute the vgg features for test image (get_image_feature) and find closest
        matching feature from trained_features.
    
    3.> Since trained feature is of shape (N,K) -> smartly repeat the test image feature N times (bring it to
        same shape as (N,K)). Then we can simply compute distance in a vectorized way.
    
    4.> Distance here can be sum over (a-b)**2
    
    5.> np.argmin over distance can give the closest point
    '''
    # YOUR CODE HERE
    ordered_labels = []
    for i in range(image_names.shape[0]):
        full_image_name = './data/' + image_names[i]
        i, pred_label = helper_func([i, full_image_name, vgg16, trained_features, train_labels])
        ordered_labels.append(pred_label)
    ordered_labels = np.array(ordered_labels)
        
    print("Predicted labels shape: ", ordered_labels.shape)
    
    '''
    HINTS:
    1.> Compute the confusion matrix (8x8)
    '''
    # YOUR CODE HERE
    conf_matrix = np.zeros((8, 8))
    for n in range(image_names.shape[0]):
        i = test_labels[n]
        j = ordered_labels[n]
        conf_matrix[i, j] += 1
    accuracy = np.trace(conf_matrix) / np.sum(conf_matrix)
    
    np.save("./trained_conf_matrix.npy",conf_matrix)
    return conf_matrix, accuracy
    # pass
    
evaluate_recognition_system(vgg16, num_workers=7)
