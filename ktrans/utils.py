
import numpy as np
import pylab as plt
from glob import glob

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.activations import linear
from tensorflow.keras.layers.experimental import preprocessing

from tf_keras_vis.utils import normalize

def loss_maker(i):
    def loss(output):
        return output[:,i]
    return loss

def model_modifier(m):
    m.layers[-1].activation = linear
    return m

def vanilla_saliency(img,model,class_id=None):
    """ Vanilla Saliency """
    
    from tf_keras_vis.saliency import Saliency
    
    if img.ndim==3:
        if class_id is None:
            pred = model.predict(img[None,:,:,:])
            class_id = np.argmax(pred)

        saliency = Saliency(model,
                            model_modifier=model_modifier,
                            clone=True)
        loss = loss_maker(class_id)
        saliency_map = saliency(loss, img)
        saliency_map = normalize(saliency_map)
        return saliency_map[0]
    
    elif img.ndim==4:
        saliency_map = []
        nd = img.shape[0]
        if class_id is None:
            class_id = nd*[None]
        
        for i in range(nd):
            ss = vanilla_saliency(img[i],model,class_id=class_id[i])
            saliency_map.append(ss)
        saliency_map = np.array(saliency_map)#np.concatenate(saliency_map,axis=0)
        return saliency_map
    else:
        assert 0,'Shape error'
        
def smoothgrad(img,model,class_id=None):
    """ SmoothGrad """
    
    from tf_keras_vis.saliency import Saliency
    
    if img.ndim==3:
        if class_id is None:
            pred = model.predict(img[None,:,:,:])
            class_id = np.argmax(pred)

        saliency = Saliency(model,
                            model_modifier=model_modifier,
                            clone=True)
        loss = loss_maker(class_id)
        saliency_map = saliency(loss, img,
                                smooth_samples=20, # The number of calculating gradients iterations.
                                smooth_noise=0.20)
        saliency_map = normalize(saliency_map)
        return saliency_map[0]

    elif img.ndim==4:
        saliency_map = []
        nd = img.shape[0]
        if class_id is None:
            class_id = nd*[None]
        
        for i in range(nd):
            ss = smoothgrad(img[i],model,class_id=class_id[i])
            saliency_map.append(ss)
        saliency_map = np.array(saliency_map)#np.concatenate(saliency_map,axis=0)
        return saliency_map
    else:
        assert 0,'Shape error'

def gradcam(img,model,class_id=None):
    """ GradCAM """
    
    from tf_keras_vis.gradcam import Gradcam
    
    if img.ndim==3:    
        if class_id is None:
            pred = model.predict(img[None,:,:,:])
            class_id = np.argmax(pred)

        gcam = Gradcam(model,
                       model_modifier=model_modifier,
                       clone=True)
        loss = loss_maker(class_id)
        saliency_map = gcam(loss,img,
                             penultimate_layer=-1 # model.layers number
                             )
        saliency_map = normalize(saliency_map)
        return saliency_map[0]

    elif img.ndim==4:
        saliency_map = []
        nd = img.shape[0]
        if class_id is None:
            class_id = nd*[None]
        
        for i in range(nd):
            ss = gradcam(img[i],model,class_id=class_id[i])
            saliency_map.append(ss)
        saliency_map = np.array(saliency_map)#np.concatenate(saliency_map,axis=0)
        return saliency_map
    else:
        assert 0,'Shape error'


def gradcampp(img,model,class_id=None):
    """ GradCAM++ """
    
    from tf_keras_vis.gradcam import GradcamPlusPlus

    if img.ndim==3:    
        if class_id is None:
            pred = model.predict(img[None,:,:,:])
            class_id = np.argmax(pred)

        gcam = GradcamPlusPlus(model,
                               model_modifier=model_modifier,
                               clone=True)

        loss = loss_maker(class_id)
        saliency_map = gcam(loss,img,penultimate_layer=-1)
        saliency_map = normalize(saliency_map)

        return saliency_map[0]

    elif img.ndim==4:
        saliency_map = []
        nd = img.shape[0]
        if class_id is None:
            class_id = nd*[None]
        
        for i in range(nd):
            ss = gradcampp(img[i],model,class_id=class_id[i])
            saliency_map.append(ss)
        saliency_map = np.array(saliency_map)#np.concatenate(saliency_map,axis=0)
        return saliency_map
    else:
        assert 0,'Shape error'


def scorecam(img,model,class_id=None):
    """ ScoreCAM """
    
    from tf_keras_vis.scorecam import ScoreCAM

    if img.ndim==3:    
        if class_id is None:
            pred = model.predict(img[None,:,:,:])
            class_id = np.argmax(pred)

        scam = ScoreCAM(model, model_modifier, clone=True)
        loss = loss_maker(class_id)
        saliency_map = scam(loss,img,penultimate_layer=-1)
        saliency_map = normalize(saliency_map)
        return saliency_map[0]

    elif img.ndim==4:
        tf.get_logger().setLevel('ERROR')
        saliency_map = []
        nd = img.shape[0]
        if class_id is None:
            class_id = nd*[None]
        
        for i in range(nd):
            ss = scorecam(img[i],model,class_id=class_id[i])
            saliency_map.append(ss)
        saliency_map = np.array(saliency_map)#np.concatenate(saliency_map,axis=0)
        return saliency_map
    else:
        assert 0,'Shape error'

def activation_maximization(model,class_id=None):
    """ ActivationMaximization """
    
    from tf_keras_vis.activation_maximization import ActivationMaximization
    from tf_keras_vis.utils.callbacks import Print
    
    if class_id is None:
        pred = model.predict(img[None,:,:,:])
        class_id = np.argmax(y_us[i])

    activation_maximization = ActivationMaximization(model,
                                                    model_modifier,
                                                    clone=True)
    loss = loss_maker(class_id)
    shape = [1]+list(model.input_shape[1:])
    seed_input = tf.random.uniform(shape, 0, 1)
    activation = activation_maximization(loss,
                                         seed_input=seed_input, # To generate multiple images
                                         steps=500,
    #                                           callbacks=[Print(interval=50)],
                                        )
    activation = normalize(activation)
    return activation



def describe_labels(y0,verbose=0):
    y = y0+0
    if y.ndim==2:
        y = np.argmax(y,axis=1)
    class_labels, nums = np.unique(y,return_counts=True)
    n_class = len(class_labels)
    if verbose:
        print('labels/numbers are:\n',*['{:5s}/{:6d}\n'.format(str(i),j) for i,j in zip(class_labels,nums)])
    return n_class,class_labels, nums

def augment(aug,x):
    aug.fit(x)
    out = []
    for i in x:
        out.append(aug.random_transform(i))
    return np.array(out)

def balance_aug(x0,y0,aug=None,nmax=None,mixup=False):
    x = x0+0
    y = y0+0
    n_class,class_labels, nums = describe_labels(y,verbose=0)
    if nmax is None:
        nmax = max(nums)
    for i,(lbl,n0) in enumerate(zip(class_labels,nums)):
        if y.ndim==1:
            filt = y==lbl
        elif y.ndim==2:
            filt = y[:,i].astype(bool)
        else:
            assert 0,'Unknown label shape!'
        if nmax<=n0:
            inds = np.argwhere(filt).reshape(-1)
            x = np.delete(x,inds[nmax:],axis=0)
            y = np.delete(y,inds[nmax:],axis=0)
            continue
        delta = nmax-n0
        x_sub = x[filt]
        y_sub = y[filt]
        inds = np.arange(n0)
        nrep = (nmax//len(inds))+1
        inds = np.repeat(inds, nrep)
        np.random.shuffle(inds)
        inds = inds[:delta]
        x_sub = x_sub[inds]
        y_sub = y_sub[inds]
        if not aug is None:
            x_sub = augment(aug,x_sub)
        x = np.concatenate([x,x_sub],axis=0)
        y = np.concatenate([y,y_sub],axis=0)
    return x,y

def shuffle_data(x,y):
    ndata = x.shape[0]
    inds = np.arange(ndata)
    np.random.shuffle(inds)
    return x[inds],y[inds]


# Distorts the color distibutions of images
class RandomColorAffine(layers.Layer):
    def __init__(self, brightness=0, jitter=0, **kwargs):
        super().__init__(**kwargs)

        self.brightness = brightness
        self.jitter = jitter

    def call(self, images, training=True):
        if training:
            batch_size = tf.shape(images)[0]

            # Same for all colors
            brightness_scales = 1 + tf.random.uniform(
                (batch_size, 1, 1, 1), minval=-self.brightness, maxval=self.brightness
            )
            # Different for all colors
            jitter_matrices = tf.random.uniform(
                (batch_size, 1, 3, 3), minval=-self.jitter, maxval=self.jitter
            )

            color_transforms = (
                tf.eye(3, batch_shape=[batch_size, 1]) * brightness_scales
                + jitter_matrices
            )
            images = tf.clip_by_value(tf.matmul(images, color_transforms), 0, 1)
        return images

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'brightness': self.brightness,
            'jitter': self.jitter,
        })
        return config

# Image augmentation module
def get_augmenter(input_shape,min_area, brightness, jitter):
    zoom_factor = 1.0 - tf.sqrt(min_area)
    return keras.Sequential(
        [
            keras.layers.InputLayer(input_shape=input_shape),
#             preprocessing.Rescaling(1 / 255),
            preprocessing.RandomFlip("horizontal"),
            preprocessing.RandomTranslation(zoom_factor / 2, zoom_factor / 2),
            preprocessing.RandomZoom((-zoom_factor, 0.0), (-zoom_factor, 0.0)),
            RandomColorAffine(brightness, jitter),
        ]
    )


def visualize_augmentations(dataset,num_images,classification_augmentation,contrastive_augmentation):
    # Sample a batch from a dataset
    images = next(iter(dataset))[0][0][:num_images]
    # Apply augmentations
    augmented_images = zip(
        images,
        get_augmenter(**classification_augmentation)(images),
        get_augmenter(**contrastive_augmentation)(images),
        get_augmenter(**contrastive_augmentation)(images),
    )
    row_titles = [
        "Original:",
        "Weakly augmented:",
        "Strongly augmented:",
        "Strongly augmented:",
    ]
    plt.figure(figsize=(num_images * 2.2, 4 * 2.2), dpi=100)
    for column, image_row in enumerate(augmented_images):
        for row, image in enumerate(image_row):
            plt.subplot(4, num_images, row * num_images + column + 1)
            plt.imshow(image)
            if column == 0:
                plt.title(row_titles[row], loc="left")
            plt.axis("off")
    plt.tight_layout()

# Define the encoder architecture
def get_encoder(input_shape,nfilter,n_project=None):
    if n_project is None:
        n_project = nfilter
#     input_shape=(image_size, image_size, image_channels)
    return keras.Sequential(
        [
            keras.layers.InputLayer(input_shape=input_shape),
            layers.Conv2D(nfilter, kernel_size=3, strides=2, activation="relu"),
            layers.Conv2D(nfilter, kernel_size=3, strides=2, activation="relu"),
            layers.Conv2D(nfilter, kernel_size=3, strides=2, activation="relu"),
            layers.Conv2D(nfilter, kernel_size=3, strides=2, activation="relu"),
            layers.Flatten(),
            layers.Dense(n_project, activation="relu"),
        ],
        name="encoder",
    )

def model_modifier(m):
    m.layers[-1].activation = linear
    return m


