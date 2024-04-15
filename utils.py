#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 26 18:13:05 2021

@author: ifeanyi.ezukwoke
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

seed = 124
import math
import os
from tqdm import tqdm
import tensorflow as tf
from tensorflow import keras
from sklearn.mixture import GaussianMixture
import torch; torch.manual_seed(seed)
import torch.utils
import torch.distributions
import numpy as np
import torch.nn.functional as F
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt; plt.rcParams['figure.dpi'] = 100
device = 'cuda' if torch.cuda.is_available() else 'cpu'

#word embeddings use..
from gensim.corpora import Dictionary
from gensim.models import Word2Vec
from gensim.similarities import WordEmbeddingSimilarityIndex
from gensim.similarities import SoftCosineSimilarity, SparseTermSimilarityMatrix

#import dependencies for evaluation metric
from sklearn.preprocessing import minmax_scale
from sklearn.metrics import mean_squared_error
from pyitlib import discrete_random_variable as drv

#dependencies for validation Eastword disentaglement metric
from sklearn.linear_model import Lasso
from sklearn.ensemble.forest import RandomForestRegressor

# training utilities...
     
def MSE(x_original, x_predicted):
    ''' 
    Parameters
    ----------
    x_original : np.array
        original data.
    x_predicted : np.array
        reconstructed data or decoder output.

    Returns
    -------
    int
        Mean Square Error loss.

    '''
    return F.mse_loss(x_original, x_predicted)
    

def binary_cross_entropy_loss(x_original, x_predicted, withsigmoid = False):
    '''
    Parameters
    ----------
    x_original : np.array
        original data.
    x_predicted : np.array
        reconstructed data or decoder output.

    Returns
    -------
    int
        Binary CrossEntropy Loss.

    '''
    if not withsigmoid:
        return F.binary_cross_entropy(x_predicted, x_original)
    else:
        return F.binary_cross_entropy_with_logits(x_predicted, x_original)
    

def plot_latent(model, test, input_dim, num_batches = 100, dt = 'MNIST_cnn'):
    plt.figure(figsize=(12, 10))
    for i, x in enumerate(test):  #(x, y) if MNIST dataset
        if dt == 'MNIST_ff':
            x = x.view(-1, 1, input_dim).to(device)
            z = model.encoder(x.to(device))
            z = z.to(device).detach().numpy()
            z = z.reshape(-1, z.shape[2])
        elif dt == 'MNIST_cnn':
            x = x.view(-1, 1, input_dim).to(device)
            z, _, _ = model.encoder(x.to(device))
            z = z.to(device).detach().numpy()
        elif dt == 'normal':
            x = x.view(-1, 1, input_dim).to(device)
            z, _, _ = model.encoder(x.to(device))
            z = z.to(device).detach().numpy()
        plt.scatter(z[:, 0], z[:, 1], cmap='tab10', edgecolor="black", s = 20) #, c=y for MNISt datastet
    plt.xlabel('z[0]-latent')
    plt.ylabel('z[1]-latent')
        
        
def plot_losses(epochs, ELBO, RECON_LOSS, KL_DIV):
    '''
    Parameters
    ----------
    epochs : int
        Number of epochs to run training.
    ELBO : int
        Evidence Lower Bound or Variational Loss.
    RECON_LOSS : int
        Reconstruction loss.
    KL_DIV : TYPE
        KL divergence between two Gaussian distribution.

    Returns
    -------
    matplotlib object.
    '''
    #vae mean and std
    epochs = np.arange(1, epochs+1)
    #check index of when ELBO is +ve and -ve
    bc_pos = lambda x: [(i,j) for (i,j) in enumerate(x) if j>0]
    #bc_neg = lambda x: [(i,j) for (i,j) in enumerate(x) if j<0]
    ind_kl_elbo_pos = bc_pos(ELBO)[-1][0] +1 #index of last postive value. shift of 1 since epoch is shifted by +1
    ind_kl_elbo_neg = ind_kl_elbo_pos + 1 #index of negative value
    #---
    try:
        fig, (ax1, ax2) = plt.subplots(2, 1)
        ax1.plot(epochs, ELBO, lw = 1.5, c = 'r', label = 'ELBO',
                path_effects=[pe.SimpleLineShadow(), pe.Normal()])
        ax1.plot(epochs, RECON_LOSS, path_effects=[pe.SimpleLineShadow(), pe.Normal()],
                lw = 1.5, c = 'g', label = 'Reconstruction loss')
        ax1.axhline(y = 0, color='r', ls = '--')
        ax1.axvline(x = ind_kl_elbo_pos, ymin = 0.0, ymax = epochs[-1], color='b')
        ax1.axvline(x = ind_kl_elbo_neg, ymin = 0.0, ymax = epochs[-1], color='r')
        ax1.legend()
        ax1.grid()
        ax1.set_xlabel('Numbers of epochs')
        ax1.set_ylabel('Loss')
        ax2.plot(epochs, KL_DIV, path_effects=[pe.SimpleLineShadow(), pe.Normal()],
                 lw = 1.5, c = 'b', label = 'KL divergence')
        ax2.axvline(x = ind_kl_elbo_pos, ymin = 0.0, ymax = epochs[-1], color='b')
        ax2.axvline(x = ind_kl_elbo_neg, ymin = 0.0, ymax = epochs[-1], color='r')
        ax2.legend()
        ax2.grid()
        ax2.set_xlabel('Numbers of epochs')
        ax2.set_ylabel('KL divegrence')
        plt.show()
    except:
        pass
    
    
def plot_losses_with_latent(epochs, ELBO, RECON_LOSS, KL_DIV, z_latent_pos, z_latent_neg):
    '''
    Parameters
    ----------
    epochs : int
        Number of epochs to run training.
    ELBO : int
        Evidence Lower Bound or Variational Loss.
    RECON_LOSS : int
        Reconstruction loss.
    KL_DIV : TYPE
        KL divergence between two Gaussian distribution.

    Returns
    -------
    matplotlib object.

    '''
    #vae mean and std
    epochs = np.arange(1, epochs+1)
    #check index of when ELBO is +ve and -ve
    bc_pos = lambda x: [(i,j) for (i,j) in enumerate(x) if j>0]
    #bc_neg = lambda x: [(i,j) for (i,j) in enumerate(x) if j<0]
    ind_kl_elbo_pos = bc_pos(ELBO)[-1][0] +1 #index of last postive value. shift of 1 since epoch is shifted by +1
    ind_kl_elbo_neg = ind_kl_elbo_pos + 1 #index of negative value
    #---

    fig, ax = plt.subplots(2, 2)
    if isinstance(z_latent_pos, np.ndarray):
        ax[0, 0].scatter(z_latent_pos[:, 0], z_latent_pos[:, 1], cmap='tab10', edgecolor="blue", s = 20) 
        ax[0, 0].set_xlabel('z[0]-latent: Before')
        ax[0, 0].set_ylabel('z[1]-latent')
    else:
        pass
    ax[0, 1].plot(epochs, ELBO, lw = 1.5, c = 'r', label = 'ELBO',
            path_effects=[pe.SimpleLineShadow(), pe.Normal()])
    ax[0, 1].plot(epochs, RECON_LOSS, path_effects=[pe.SimpleLineShadow(), pe.Normal()],
            lw = 1.5, c = 'g', label = 'Reconstruction loss')
    ax[0, 1].axhline(y = 0, color='r', ls = '--')
    ax[0, 1].axvline(x = ind_kl_elbo_pos, ymin = 0.0, ymax = epochs[-1], color='b')
    ax[0, 1].axvline(x = ind_kl_elbo_neg, ymin = 0.0, ymax = epochs[-1], color='r')
    ax[0, 1].legend()
    ax[0, 1].grid()
    ax[0, 1].set_xlabel('Numbers of epochs')
    ax[0, 1].set_ylabel('Loss')
    if isinstance(z_latent_neg, np.ndarray):
        ax[1, 0].scatter(z_latent_neg[:, 0], z_latent_neg[:, 1], cmap='tab10', edgecolor="red", s = 20) #, c=y for MNISt datastet
        ax[1, 0].set_xlabel('z[0]-latent: After')
        ax[1, 0].set_ylabel('z[1]-latent')
    else:
        pass
    ax[1, 1].plot(epochs, KL_DIV, path_effects=[pe.SimpleLineShadow(), pe.Normal()],
             lw = 1.5, c = 'b', label = 'KL divergence')
    ax[1, 1].axvline(x = ind_kl_elbo_pos, ymin = 0.0, ymax = epochs[-1], color='b')
    ax[1, 1].axvline(x = ind_kl_elbo_neg, ymin = 0.0, ymax = epochs[-1], color='r')
    ax[1, 1].legend()
    ax[1, 1].grid()
    ax[1, 1].set_xlabel('Numbers of epochs')
    ax[1, 1].set_ylabel('KL divegrence')
    plt.show()
    
def best_gmm_model(z, n_components = 50):
    np.random.seed(50)
    lowest_bic = np.infty
    bic = []
    n_components_range = range(1, n_components)
    cv_types = ['spherical', 'tied', 'diag', 'full']
    ls = z
    for cv_type in cv_types:
        for n_components in n_components_range:
            # Fit a Gaussian mixture with EM
            gmm = GaussianMixture(n_components=n_components, covariance_type=cv_type)
            gmm.fit(ls)
            bic.append(gmm.bic(ls))
            if bic[-1] < lowest_bic:
                lowest_bic = bic[-1]
                best_gmm = gmm
    return best_gmm
     
def compute_kernel(x, y):
    x_size = tf.shape(x)[0]
    y_size = tf.shape(y)[0]
    dim = tf.shape(x)[1]
    tiled_x = tf.tile(tf.reshape(x, tf.stack([x_size, 1, dim])), tf.stack([1, y_size, 1]))
    tiled_y = tf.tile(tf.reshape(y, tf.stack([1, y_size, dim])), tf.stack([x_size, 1, 1]))
    return tf.exp(-tf.reduce_mean(tf.square(tiled_x - tiled_y), axis=-1) / tf.cast(dim, tf.float32))

        
def z_mahalanobis(z, diag:bool = False, psd = False)->float:
    '''
    Parameters
    ----------
    z : numpy array
        latent array/code.
    diag : bool, optional
        Diagonal of the covariance matrix. The default is False.

    Returns
    -------
    float
        mahalanobis mean of the latent vector.

    '''
    z = z.numpy()
    m = lambda z: z - z.mean(axis = 0) #mean of vectors
    z_m = m(z) #mean centered data
    
    cov = 1/(len(z)-1)*z_m.T.dot(z_m)
    diag_cov = np.diag(np.diag(cov))
    #check if matrix entries are 
    if not psd:
        cov = 1/(len(z)-1)*z_m.T.dot(z_m)
        diag_cov = np.diag(np.diag(cov))
    else:
        cov = 1/(len(z)-1)*z_m.T.dot(z_m)
        cov = np.where(cov < 0, 0, cov)
        diag_cov = np.diag(np.diag(cov))
        diag_cov = np.where(diag_cov < 0, 0, diag_cov)
    if not diag:
        inv_cov = np.linalg.inv(cov) #inverse of a full covariance matrix
    else:
        inv_cov = np.linalg.inv(diag_cov) #inverse of diagonal covariance matrix
    trans_x = z_m.dot(inv_cov).dot(z_m.T)
    mah_mat_mean = np.mean(trans_x.diagonal())
    return tf.Variable(mah_mat_mean, dtype=tf.float32)


def z_mahalanobis_v2(z_1, z_2, diag:bool = False, psd = False)->float:
    '''
    Parameters
    ----------
    z : numpy array
        latent array/code.
    diag : bool, optional
        Diagonal of the covariance matrix. The default is False.

    Returns
    -------
    float
        mahalanobis mean of the latent vector.

    '''
    z_1 = z_1.numpy()
    z_2 = z_2.numpy()
    m = lambda z: z - z.mean(axis = 0) #mean of vectors
    z_m_1 = m(z_1) #mean centered data matrix 1
    z_m_2 = m(z_2) #mean centered data matrix 2
    
    #check if matrix entries are 
    if not psd:
        cov = 1/(len(z_1)-1)*z_m_1.T.dot(z_m_2)
        diag_cov = np.diag(np.diag(cov))
    else:
        cov = 1/(len(z_1)-1)*z_m_1.T.dot(z_m_2)
        cov = np.where(cov < 0, 0, cov)
        diag_cov = np.diag(np.diag(cov))
        diag_cov = np.where(diag_cov < 0, 0, diag_cov)
    if not diag:
        inv_cov = np.linalg.inv(cov) #inverse of a full covariance matrix
    else:
        inv_cov = np.linalg.inv(diag_cov) #inverse of diagonal covariance matrix
    trans_x = z_m_1.dot(inv_cov).dot(z_m_2.T)
    mah_mat_mean = np.mean(trans_x.diagonal())
    return tf.Variable(mah_mat_mean, dtype=tf.float32)


def z_mahalanobis_rkhs(z, diag:bool = False, psd = False)->float:
    '''Reproducing Kernel Hilbert Space (RKHS)
       Mahalanobis distance
    

    Parameters
    ----------
    z : numpy array
        latent array/code.
    diag : bool, optional
        Diagonal of the covariance matrix. The default is False.
    
    psd: bool, optional
        is matrix is not positive semi definite
        
    Returns
    -------
    float
        mahalanobis mean of the latent vector.

    '''
    #z_sample = tf.keras.backend.random_normal(tf.shape(z), dtype = tf.float32, mean = 0., stddev = 1.0)
    z = compute_kernel(z, z)
    z = z.numpy()
    m = lambda z: z - z.mean(axis = 0) #mean of vectors
    z_m = m(z) #mean centered data
    #check if matrix entries are 
    if not psd:
        cov = 1/(len(z)-1)*z_m.T.dot(z_m)
        diag_cov = np.diag(np.diag(cov))
    else:
        cov = 1/(len(z)-1)*z_m.T.dot(z_m)
        cov = np.where(cov < 0, 0, cov)
        diag_cov = np.diag(np.diag(cov))
        diag_cov = np.where(diag_cov < 0, 0, diag_cov)
    if not diag:
        inv_cov = np.linalg.inv(cov) #inverse of a full covariance matrix
    else:
        inv_cov = np.linalg.inv(diag_cov) #inverse of diagonal covariance matrix
    trans_x = z_m.dot(inv_cov).dot(z_m.T)
    mah_mat_mean = np.mean(trans_x.diagonal())
    return tf.Variable(mah_mat_mean, dtype=tf.float32)

#prepare FA data
def keywordReturn(text:list, keyword:str)->bool:
    '''
    Parameters
    ----------
    text : list
        list of the tokens to search.
    keyword : str
        keyword.

    Returns
    -------
    bool
        True or False.

    '''
    return True if keyword in text else False

class Mergefeatures():
    def __init__(self, string):
        super(Mergefeatures, self).__init__()
        self.string = string
        return
    
    def concat(self):
        '''Concatenate along the horizontal axis
        '''
        z = ','.join(y.strip('[]') for y in self.string)
        z = [x.strip().strip("''") for x in z.split(',')]
        z = ' '.join(x for x in z if not x == 'nan' if not x == ' ' if not x == '')
        z = [x for x in z.split(' ')]
        return z
    

class SimilarityMat():
    def __init__(self, model, text):
        '''Similarity Matrix
        
        Parameters
        ----------
        model:
        text: textual document data set
        
        Return
        ------
        None
        '''
        self.model = model
        self.text = text
        #self.SimilarityMatrix()
        return
    
    def SimilarityMatrix(self):
        '''
        
        Return
        -------
        Returns Similarity matrix
        '''
        termsim_index = WordEmbeddingSimilarityIndex(self.model.wv)
        self.dictionary = Dictionary(self.text)
        bow_corpus = [self.dictionary.doc2bow(document) for document in self.text]
        similarity_matrix = SparseTermSimilarityMatrix(termsim_index, self.dictionary)
        self.docsim_index = SoftCosineSimilarity(bow_corpus, similarity_matrix)
        return self

    def QuerySim(self, query):
        '''
        Parameters
        ----------
        query: Text to vectorize
        
        
        Return
        ------
        document similarity matrix
        '''
        return self.docsim_index[self.dictionary.doc2bow(query)]
    
    def wordEmbed(self):
        """Word Embedding vector

        Parameters
        ----------
        None
        

        Return
        -------
        word embedding vector        
        """
        self.features = []

        for tokens in self.text:
            self.zero_vector = np.zeros(self.model.vector_size)
            self.vectors = []
            for token in tokens:
                if token in self.model.wv:
                    try:
                        self.vectors.append(self.model.wv[token])
                    except KeyError:
                        continue
            if self.vectors:
                self.vectors = np.asarray(self.vectors)
                self.mean_vect = self.vectors.mean(axis = 0)
                self.features.append(self.mean_vect)
            else:
                self.features.append(self.zero_vector)
        self.feat_vect = np.array([x for x in self.features]) #reshape vectors
        return self.feat_vect


               


#%% PID controllers

class PIDControl_v1(object):
    def __init__(self):
        self.I_k1 = 0.0
        self.W_k1 = 1.0
        self.e_k1 = 0.0
    
    def _Kp_func(self, Err, scale=1):
        return 1.0/(1.0 + float(scale)*np.exp(Err))
        
    def pid(self, exp_KL, kl_divergence, Kp=0.1, Ki=-0.0001, Kd=0.01):
        """
        position PID algorithm
        Input: KL_loss
        return: weight for KL loss, beta
        """
        error_k = exp_KL - kl_divergence
        ## comput U as the control factor
        Pk = Kp * self._Kp_func(error_k)+1
        Ik = self.I_k1 + Ki * error_k
        
        ## window up for integrator
        if self.W_k1 < 1:
            Ik = self.I_k1
            
        Wk = Pk + Ik
        self.W_k1 = Wk
        self.I_k1 = Ik
        
        ## min and max value
        if Wk < 1:
            Wk = 1
        
        # return Wk, error_k
        return Wk

class PIDControl_v2(object):
    def __init__(self):
        self.I_k1 = tf.Variable(0.0,trainable=False)
        ## W_k1 record the previous time weight W value
        self.W_k1 = tf.Variable(0.0,trainable=False)
           
    def _Kp_func(self, Err, scale=1.0):
        return 1.0/(1.0+tf.exp(scale*Err))
    
    def pid(self, exp_KL, KL_loss, Kp=0.1, Ki=-0.0001):
        """ increment PID algorithm
   		Input: KL_loss
   		return: weight for KL loss, WL
           
           $\beta$ only --> K_p = 0.1, K_i = -0.0001
           $\beta$ only --> K_p = 0.1, K_i = -0.0001
   		"""
        self.exp_KL = exp_KL
        error_k = tf.stop_gradient(self.exp_KL - KL_loss)
        ## comput P control
        Pk = Kp * self._Kp_func(error_k)
        ## I control accumulate error from time 0 to T
        Ik = self.I_k1 + Ki * error_k
        ## when time = k-1
        Ik = tf.cond(self.W_k1 < 0, lambda:self.I_k1, lambda:tf.cond(self.W_k1 > 1, lambda:self.I_k1, lambda:Ik))
        # Ik = tf.cond(self.W_k1 > 1, lambda:self.I_k1, lambda:Ik)
        ## update k-1 accumulated error
        op1 = tf.compat.v1.assign(self.I_k1,Ik)  ## I_k1 = Ik
        ## update weight WL
        Wk = Pk + Ik
        op2 = tf.compat.v1.assign(self.W_k1,Wk)   ## self.W_k1 = Wk
        ## min and max value --> 0 and 1
        ## if Wk > 1, Wk = 1; if Wk<0, Wk = 0
        with tf.control_dependencies([op1,op2]):
            Wk = tf.cond(Wk > 1, lambda: 1.0, lambda: tf.cond(Wk < 0, lambda: 0.0, lambda: Wk))
            
        return Wk


#%% Plotting utils

def plot_latent_space(vae, n = 20, figsize = 15):
    '''Source: https://keras.io/examples/generative/vae/
    

    Parameters
    ----------
    vae : tensorflow model
        DESCRIPTION.
    n : int, optional
        DESCRIPTION. The default is 20.
    figsize : matplotlib.pyplot object, optional
        matplot graph. The default is 15.

    Returns
    -------
    matplot object.

    '''
    # display a n*n 2D manifold of digits
    digit_size = 28
    scale = 1.0
    figure = np.zeros((digit_size * n, digit_size * n))
    # linearly spaced coordinates corresponding to the 2D plot
    # of digit classes in the latent space
    grid_x = np.linspace(-scale, scale, n)
    grid_y = np.linspace(-scale, scale, n)[::-1]

    for i, yi in enumerate(grid_y):
        for j, xi in enumerate(grid_x):
            z_sample = np.array([[xi, yi]])
            x_decoded = vae.decoder.predict(z_sample)
            digit = x_decoded[0].reshape(digit_size, digit_size)
            figure[
                i * digit_size : (i + 1) * digit_size,
                j * digit_size : (j + 1) * digit_size,
            ] = digit

    plt.figure(figsize=(figsize, figsize))
    plt.axis('off')
    #start_range = digit_size // 2
    #end_range = n * digit_size + start_range
    #pixel_range = np.arange(start_range, end_range, digit_size)
    #sample_range_x = np.round(grid_x, 1)
    #sample_range_y = np.round(grid_y, 1)
    #plt.xticks(pixel_range, sample_range_x)
    #plt.yticks(pixel_range, sample_range_y)
    #plt.xlabel("z[0]")
    #plt.ylabel("z[1]")
    plt.imshow(figure, cmap="Greys_r")
    plt.show()
    

#%% Evaluation metric

class Metric:
    def __init__(self, x, y, nb_bins = 1000):
        '''
        Parameters
        ----------
        x : R^{NxM} np.array (M: dimension of data)
            True factors of the data.
        y : R^{NxK} np.array (K: lower dimensional latent code)
            Latent factor/code obtanined from Inference after N-epoch training.
        nb_bins : int, optional
            Number of bins to use for discretization. 
            The default is 1000.
            
        Returns
        -------
        None.

        '''
        self.x  = x
        self.y  = y
        self.nb_bins = nb_bins
        return 
    
    
    def get_mutual_information(self, x, y, normalize = True):
        '''
        Get mutual information

        Parameters
        ----------
        x : R^{NxM} np.array
            True label.
        y : R^{NxK} np.array
            Predicted label. Note that N: number of observation; K: size of latent space
        normalize : bool, optional
            normalize mutual information score. The default is True.

        Returns
        -------
        TYPE
            DESCRIPTION.

        '''
        self.x, self.y = x, y
        if normalize:
            return drv.information_mutual_normalised(self.x, self.y, norm_factor='Y', cartesian_product = True)
        else:
            return drv.information_mutual(self.x, self.y, cartesian_product = True)
    

    def get_bin_index(self, factor, nb_bins):
        '''Get bin index
    
        Parameters
        ----------
        x : np.array
            data.
        nb_bins : int
            number of bins to use for discretization.
    
        Returns
        -------
        TYPE
            DESCRIPTION.
    
        '''
        self.nb_bins = nb_bins
        # get bins limits
        bins = np.linspace(0, 1, self.nb_bins + 1)
    
        # discretize input variable
        return np.digitize(factor, bins[:-1], right = False).astype(int)


    def compute_sparsity_(self, norm = True):
        '''Compute sparsity score of the latent space obtained
            from inference.
            
        Parameters
        ----------
        norm : bool, optional
            normalize z latent by standard deviation. The default is True.

        Returns
        -------
        float
            Sparsity score.

        '''
        zs = self.y #latent
        l_dim = zs.shape[-1]
        if norm:
            zs = zs / tf.math.reduce_std(zs)
        numr_ = tf.math.reduce_sum(tf.math.abs(zs), axis = -1)
        denm_ = tf.math.sqrt(tf.math.reduce_sum(tf.math.pow(zs, 2), -1))
        l1_l2 = tf.reduce_mean(numr_/denm_)
        return (math.sqrt(l_dim) - l1_l2) / (math.sqrt(l_dim) - 1)


    @staticmethod
    def mse(predicted, target):
        '''Mean Squre Error
    
        Parameters
        ----------
        predicted : array
            prediction or outcome.
        target : array
            expectated prediction outcome.

        Returns
        -------
        float
            Mean Square Error.

        '''
        # mean square error
        predicted = predicted[:, None] if len(predicted.shape) == 1 else predicted  # (n,)->(n,1)
        target = target[:, None] if len(target.shape) == 1 else target  # (n,)->(n,1)
        err = predicted - target
        err = err.T.dot(err) / len(err)
        return err[0, 0]
    
    
    @staticmethod
    def rmse(predicted, target):
        '''Root Mean Squre Error
    
        Parameters
        ----------
        predicted : array
            prediction or outcome.
        target : array
            expectated prediction outcome.

        Returns
        -------
        float
            Root Mean Square Error.

        '''
        # root mean square error
        return np.sqrt(Metric.mse(predicted, target))
    
    
    @staticmethod
    def nmse(predicted, target, eps = 1e-8):
        '''Normalized Mean Squre Error
    
        Parameters
        ----------
        predicted : array
            prediction or outcome.
        target : array
            expectated prediction outcome.

        Returns
        -------
        float
            Normalized Mean Squre Error.
        '''
        # normalized mean square error
        return Metric.mse(predicted, target) / np.maximum(np.var(target), eps)
    
    @staticmethod
    def nrmse(predicted, target, eps = 1e-8):
        '''Normalized Root Mean Squre Error
    
        Parameters
        ----------
        predicted : array
            prediction or outcome.
        target : array
            expectated prediction outcome.

        Returns
        -------
        float
            Normalized Root Mean Squre Error.

        '''
        return Metric.rmse(predicted, target) / np.maximum(np.std(target), eps)
    
    
    @staticmethod
    def entropic_scores(R, eps = 1e-8):
        ''' Entropy scores 
        
        Parameters
        ----------
        R : np.array
            importance matrix: (num_latents, num_factors).
        eps : np.float, optional
            DESCRIPTION. The default is 1e-8.

        Returns
        -------
        float
            Entropy score.
        '''
        R = np.abs(R)
        P = R / np.maximum(np.sum(R, axis=0), eps)
        # H_norm: (num_factors,)
        H_norm = -np.sum(P * np.log(np.maximum(P, eps)), axis=0)
        if P.shape[0] > 1:
            H_norm = H_norm / np.log(P.shape[0])
        return 1 - H_norm


    def kming(self, L = 1000, M = 10000):
        '''Factor VAE disentanglement metric
        
        @misc{modularity,
                      title={Disentangling by Factorising}, 
                      author={Hyunjik Kim, Andriy Mnih},
                      year={2019},
                }
            paper: https://arxiv.org/pdf/1802.05983.pdf

        Parameters
        ----------
        L : TYPE, optional
            DESCRIPTION. The default is 25.
        M : TYPE, optional
            DESCRIPTION. The default is 1000.

        Returns
        -------
        float
            Z-min score proposed by Kim \& Minh.

        '''
        N, D = self.x.shape
        _, K = self.y.shape #image data conats (NxLxM) dimension
        zs_std = tf.math.reduce_std(self.x, axis = 0)
        ys_uniq = [tf.unique(tf.reshape(c, [-1, ]))[0] for c in tf.split(self.y, K, 1)]  #extract unique values from each y
        V = tf.zeros([D, K], tf.float32).numpy()
        ks = np.random.randint(0, K, M)  #K: is the range of value; M: is the dimension # sample fixed-factor idxs ahead of time
    
        for m in range(M):
            k = ks[m]
            fk_vals = ys_uniq[k]
            # fix fk
            fk = fk_vals[np.random.choice(len(fk_vals))]
            # choose L random x that have this fk at factor k
            zsh = self.x[self.y[:, k] == fk]
            zsh = tf.random.shuffle(zsh)[:L]
            d_star = tf.argmin(tf.math.reduce_variance(zsh/zs_std, axis=0)) #note that biased variance is computed in tensorflow
            V[d_star, k] += 1
        return tf.reduce_sum(tf.math.reduce_max(V, axis=1)) #/M

    def mig(self, continuous_factors = True):
        '''
            MIG (Mutual Information Gap) metric from R. T. Q. Chen, X. Li, R. B. Grosse, and D. K. Duvenaud,
            “Isolating sources of disentanglement in variationalautoencoders,”
            in NeurIPS, 2018.
            paper: https://arxiv.org/pdf/1802.04942.pdf
            
    
        Parameters
        ----------
        continuous_factors : bool, optional
            True:   factors are described as continuous variables
            False:  factors are described as discrete variables. 
            The default is True.
    
        Returns
        -------
        mig_score : TYPE
            DESCRIPTION.
    
        '''
        # count the number of factors and latent codes
        nb_factors = self.x.shape[1] #original factors of data variable
        nb_codes = self.y.shape[1] #latent variable
        
        # quantize factors if they are continuous
        if continuous_factors:
            factors = minmax_scale(np.nan_to_num(self.x))  # normalize in [0, 1] all columns
            factors = self.get_bin_index(self.x, self.nb_bins)  # quantize values and get indexes
        else:
            factors = self.x
            
        # quantize latent codes
        if continuous_factors:
            codes = minmax_scale(np.nan_to_num(self.y))  # normalize in [0, 1] all columns
            codes = self.get_bin_index(self.y, self.nb_bins)  # quantize values and get indexes
        else:
            codes = self.y
    
        # compute mutual information matrix
        mi_matrix = np.zeros((nb_factors, nb_codes))
        for f in range(nb_factors):
            for c in range(nb_codes):
                mi_matrix[f, c] = self.get_mutual_information(factors[:, f], codes[:, c], normalize = False)
        
        #compute discrete entropies
        # num_factors = self.x.shape[0]
        # h = np.zeros(num_factors)
        # for e in range(num_factors):
        #     h[e] = self.get_mutual_information(factors[e, :], factors[e, :], normalize = False)
            
        # compute the mean gap for all factors
        sum_gap = 0
        for f in range(nb_factors):
            mi_f = np.sort(mi_matrix[f, :])
            # get diff between highest and second highest term and add it to total gap
            sum_gap += mi_f[-1] - mi_f[-2]
            
        # compute the mean gap
        mig_score = sum_gap / nb_factors
        
        return mig_score


    def modularity(self, continuous_factors=True):
        '''
            @misc{modularity,
                      title={Learning deep disentangled embeddings with the f-statistic loss}, 
                      author={K. Ridgeway and M. C. Mozer},
                      year={2018},
                }
            paper: https://arxiv.org/pdf/1802.05312.pdf
    
        Parameters
        ----------
        continuous_factors : bool, optional
            True:   factors are described as continuous variables
            False:  factors are described as discrete variables. 
            The default is True.
    
        Returns
        -------
        mig_score : TYPE
            DESCRIPTION.
    
        '''
        # count the number of factors and latent codes
        nb_factors = self.x.shape[1]
        nb_codes = self.y.shape[1]
        
        # quantize factors if they are continuous
        if continuous_factors:
            factors = minmax_scale(np.nan_to_num(self.x))  # normalize in [0, 1] all columns
            factors = self.get_bin_index(self.x, self.nb_bins)  # quantize values and get indexes
        else:
            factors = self.x
            
        # quantize latent codes
        if continuous_factors:
            codes = minmax_scale(np.nan_to_num(self.y))  # normalize in [0, 1] all columns
            codes = self.get_bin_index(self.y, self.nb_bins)  # quantize values and get indexes
        else:
            codes = self.y
        
        # compute mutual information matrix
        mi_matrix = np.zeros((nb_factors, nb_codes))
        for f in range(nb_factors):
            for c in range(nb_codes):
                mi_matrix[f, c] = self.get_mutual_information(factors[:, f], codes[:, c], normalize = False)
    
        # compute the score for all codes
        sum_score = 0
        for c in range(nb_codes):
            # find the index of the factor with the maximum MI
            max_mi_idx = np.argmax(mi_matrix[:, c])
    
            # compute numerator
            numerator = 0
            for f, mi_f in enumerate(mi_matrix[:, c]):
                if f != max_mi_idx:
                    numerator += mi_f ** 2
            
            # get the score for this code
            s = 1 - numerator / (mi_matrix[max_mi_idx, c] ** 2 * (nb_factors - 1))
            sum_score += s
        
        # compute the mean gap
        modularity_score = sum_score / nb_codes
        
        return modularity_score


    def jemmig(self, continuous_factors = True):
        '''
            @misc{jemmig,
                      title={Theory and Evaluation Metrics for Learning Disentangled Representations}, 
                      author={Kien Do and Truyen Tran},
                      year={2021},
                }
            paper: https://arxiv.org/pdf/1908.09961.pdf
    
        Parameters
        ----------
        continuous_factors : bool, optional
            True:   factors are described as continuous variables
            False:  factors are described as discrete variables. 
            The default is True.
    
        Returns
        -------
        mig_score : float
            Mutual Information Gap score.
        '''
        # count the number of factors and latent codes
        nb_factors = self.x.shape[1]
        nb_codes = self.y.shape[1]
        
        # quantize factors if they are continuous
        if continuous_factors:
            factors = minmax_scale(np.nan_to_num(self.x))  # normalize in [0, 1] all columns
            factors = self.get_bin_index(self.x, self.nb_bins)  # quantize values and get indexes
        else:
            factors = self.x
            
        # quantize latent codes
        if continuous_factors:
            codes = minmax_scale(np.nan_to_num(self.y))  # normalize in [0, 1] all columns
            codes = self.get_bin_index(self.y, self.nb_bins)  # quantize values and get indexes
        else:
            codes = self.y
    
        # compute mutual information matrix
        mi_matrix = np.zeros((nb_factors, nb_codes))
        for f in range(nb_factors):
            for c in range(nb_codes):
                mi_matrix[f, c] = self.get_mutual_information(factors[:, f], codes[:, c], normalize = False)
    
        # compute joint entropy matrix 
        je_matrix = np.zeros((nb_factors, nb_codes))
        for f in range(nb_factors):
            for c in range(nb_codes):
                X = np.stack((factors[:, f], codes[:, c]), 0)
                je_matrix[f, c] = drv.entropy_joint(X)
    
        # compute the mean gap for all factors
        sum_gap = 0
        for f in range(nb_factors):
            mi_f = np.sort(mi_matrix[f, :])
            je_idx = np.argsort(mi_matrix[f, :])[-1]
    
            # Compute unormalized JEMMIG
            jemmig_not_normalized = je_matrix[f, je_idx] - mi_f[-1] + mi_f[-2]
    
            # normalize by H(f) + log(#bins)
            jemmig_f = jemmig_not_normalized / (drv.entropy_joint(factors[:, f]) + np.log2(self.nb_bins))
            jemmig_f = 1 - jemmig_f
            sum_gap += jemmig_f
        
        # compute the mean gap
        jemmig_score = sum_gap / nb_factors
        
        return jemmig_score


    def mig_sup(self, continuous_factors = True):
        '''
            @misc{do2021theory,
                      title={Progressive learning and disentanglement of hierarchicalrepresentations}, 
                      author={MIG-SUP metric from Z. Li, J. V. Murkute, P. K. Gyawali, and L. Wang,},
                      year={2020},
                }
            paper: https://arxiv.org/pdf/2002.10549.pdf
    
        Parameters
        ----------
        continuous_factors : bool, optional
            True:   factors are described as continuous variables
            False:  factors are described as discrete variables. 
            The default is True.
    
        Returns
        -------
        mig_sup : float
            mig_sup score.
    
        '''
        # count the number of factors and latent codes
        nb_factors = self.x.shape[1]
        nb_codes = self.y.shape[1]
        
        # quantize factors if they are continuous
        if continuous_factors:
            factors = minmax_scale(np.nan_to_num(self.x))  # normalize in [0, 1] all columns
            factors = self.get_bin_index(self.x, self.nb_bins)  # quantize values and get indexes
        else:
            factors = self.x
            
        # quantize latent codes
        if continuous_factors:
            codes = minmax_scale(np.nan_to_num(self.y))  # normalize in [0, 1] all columns
            codes = self.get_bin_index(self.y, self.nb_bins)  # quantize values and get indexes
        else:
            codes = self.y
    
        # compute mutual information matrix
        mi_matrix = np.zeros((nb_factors, nb_codes))
        for f in range(nb_factors):
            for c in range(nb_codes):
                mi_matrix[f, c] = self.get_mutual_information(factors[:, f], codes[:, c])
    
        # compute the mean gap for all codes
        sum_gap = 0
        for c in range(nb_codes):
            mi_c = np.sort(mi_matrix[:, c])
            # get diff between highest and second highest term and add it to total gap
            sum_gap += mi_c[-1] - mi_c[-2]
        
        # compute the mean gap
        mig_sup_score = sum_gap / nb_codes
        
        return mig_sup_score


    def dcimig(self, continuous_factors = True):
        '''
            @misc{DCIMIG_Sepliarskaia,
                      title={Evaluating disentangled representations}, 
                      author={ulian Zaidi, onathan Boilard, Ghyslain Gagnon, Marc-André Carbonneau},
                      year={2021},
                }
            paper: https://arxiv.org/pdf/2012.09276.pdf
    
        Parameters
        ----------
        continuous_factors : bool, optional
            True:   factors are described as continuous variables
            False:  factors are described as discrete variables. 
            The default is True.
    
        Returns
        -------
        dcimig : float
            DCI MIG score.
    
        '''
        # count the number of factors and latent codes
        nb_factors = self.x.shape[1]
        nb_codes = self.y.shape[1]
        
        # quantize factors if they are continuous
        if continuous_factors:
            factors = minmax_scale(np.nan_to_num(self.x))  # normalize in [0, 1] all columns
            factors = self.get_bin_index(self.x, self.nb_bins)  # quantize values and get indexes
        else:
            factors = self.x
            
        # quantize latent codes
        if continuous_factors:
            codes = minmax_scale(np.nan_to_num(self.y))  # normalize in [0, 1] all columns
            codes = self.get_bin_index(self.y, self.nb_bins)  # quantize values and get indexes
        else:
            codes = self.y
        
        # compute mutual information matrix
        mi_matrix = np.zeros((nb_factors, nb_codes))
        for f in range(nb_factors):
            for c in range(nb_codes):
                mi_matrix[f, c] = self.get_mutual_information(factors[:, f], codes[:, c], normalize=False)
    
        # compute the gap for all codes
        for c in range(nb_codes):
            mi_c = np.sort(mi_matrix[:, c])
            max_idx = np.argmax(mi_matrix[:, c])
    
            # get diff between highest and second highest term gap
            gap = mi_c[-1] - mi_c[-2]
    
            # replace the best by the gap and the rest by 0
            mi_matrix[:, c] = mi_matrix[:, c] * 0
            mi_matrix[max_idx, c] = gap
    
        # find the best gap for each factor
        gap_sum = 0
        for f in range(nb_factors):
            gap_sum += np.max(mi_matrix[f, :])
    
        # sum the entropy for each factors
        factor_entropy = 0
        for f in range(nb_factors):
            factor_entropy += drv.entropy(factors[:, f])
    
        # compute the mean gap
        dcimig_score = gap_sum / factor_entropy
        
        return dcimig_score


    def lasso_dmetric_(self, params={"alpha": 0.02}, cont_mask = None, prnt = False):
        '''
            LASSO REGRESSION metric for measuring disentanglement, Completeness
            and Informativeness.
            
            @misc{eastwood-iclr_2018,
                          title={A FRAMEWORK FOR THE QUANTITATIVE EVALUATION OF DISENTANGLED REPRESENTATIONS}, 
                          author={Cian Eastwood \& Christopher K. I. Williams},
                          year={2018},
                    }
                paper: https://openreview.net/pdf?id=By-7dz-AZ
                sc: https://github.com/clarken92/DisentanglementMetrics/blob/main/utils/metrics/metrics_eastwood.py

        Parameters
        ----------
        params : dictionary, optional
            parameters for training LASSO regression model. The default is {"alpha": 0.02}.
        cont_mask : TYPE, optional
            Continuous mask. The default is None.
        prnt: bool, optional
            print output of this function. The default is False

        Returns
        -------
        results : dictionary
            Disentanglement
            Completeness
            Informativeness       
        '''
        err_fn = Metric.nrmse
        assert len(self.y.shape) == len(self.x.shape) == 2, "'latents' and 'factors' must be 2D arrays!"
        assert len(self.y) == len(self.x), "'latents' and 'factors' must have the same length!"
        num_factors = self.x.shape[1]
        if not cont_mask:
            cont_mask = [True] * num_factors
        else:
            assert len(cont_mask) == num_factors, "len(cont_mask) = {len(cont_mask)}"
        R = []
        train_errors = []
        for k in tqdm(range(num_factors)):
            if cont_mask[k]:
                # (N, )
                factors_k = self.x[:, k]
                model = Lasso(**params)
                model.fit(np.nan_to_num(self.y), factors_k)
                # (N, )
                factors_k_pred = model.predict(np.nan_to_num(self.y))
                train_errors.append(err_fn(factors_k_pred, factors_k))
                # Get the weight of the linear regressor, whose shape is (num_latents, 1)
                R.append(np.abs(model.coef_[:, None]))
            else:
                pass
        # (num_latents, num_factors)
        R = np.concatenate(R, axis=1)
        assert R.shape[1] == np.sum(np.asarray(cont_mask, dtype=np.int32)), \
            f"R.shape = {R.shape[1]} while #cont = {np.sum(np.asarray(cont_mask, dtype=np.int32))}"
        # Disentanglement: (num_latents,)
        disentanglement_scores = Metric.entropic_scores(R.T)
        c_rel_importance = np.sum(R, axis=1) / np.sum(R)  # relative importance of each code variable
        assert 1 - 1e-4 < np.sum(c_rel_importance) < 1 + 1e-4, f"c_rel_importance: {c_rel_importance}"
        disentanglement = np.sum(disentanglement_scores * c_rel_importance)
        # Completeness
        completeness_scores = Metric.entropic_scores(R)
        #print("completeness_scores: {}".format(completeness_scores))
        completeness = np.mean(completeness_scores)
        # Informativeness
        train_avg_error = np.mean(train_errors)    
        results = {
                'importance_matrix': R,
                'disentanglement': disentanglement,
                'completeness': completeness,
                'informativeness': train_avg_error,
                }
        if prnt:
            print(f"Disentanglement: {results['disentanglement']}\n\
                  Completeness: {results['completeness']}\n\
                      Informativeness: {results['train_avg_error']}")
        else:
            pass
        return results
    
    
    def randomforest_dmetric_(self, params = {"n_estimators": 10, "max_depth": 8},
                              cont_mask = None, prnt = False):
        '''
            RandomForest REGRESSION metric for measuring disentanglement, Completeness
            and Informativeness.
            
            @misc{eastwood-iclr_2018,
                          title={A FRAMEWORK FOR THE QUANTITATIVE EVALUATION OF DISENTANGLED REPRESENTATIONS}, 
                          author={Cian Eastwood \& Christopher K. I. Williams},
                          year={2018},
                    }
                paper: https://openreview.net/pdf?id=By-7dz-AZ
                sc: https://github.com/clarken92/DisentanglementMetrics/blob/main/utils/metrics/metrics_eastwood.py
            
        Parameters
        ----------
        params : dictionary, optional
            parameters for training Random Forest regression model. The default is {"n_estimators": 10, "max_depth": 8}.
        cont_mask : TYPE, optional
            Continuous mask. The default is None.
        prnt: bool, optional
            print output of this function. The default is False
            
        Returns
        -------
        results : dictionary
            Disentanglement
            Completeness
            Informativeness
        '''
        err_fn = Metric.nrmse
        assert len(self.y.shape) == len(self.x.shape) == 2, "'latents' and 'factors' must be 2D arrays!"
        assert len(self.y) == len(self.x), "'latents' and 'factors' must have the same length!"
        num_factors = self.x.shape[1]
        R = []
        train_errors = []
        if not cont_mask:
            cont_mask = [True] * num_factors
        else:
            assert len(cont_mask) == num_factors, "len(cont_mask) = {len(cont_mask)}"
        for k in tqdm(range(num_factors)):
            if cont_mask:
                # (N, )
                factors_k = self.x[:, k]
                model = RandomForestRegressor(**params)
                model.fit(np.nan_to_num(self.y), factors_k)
                # (N, )
                factors_k_pred = model.predict(np.nan_to_num(self.y))
                train_errors.append(err_fn(factors_k_pred, factors_k))
                # Get the weight of the linear regressor, whose shape is (num_latents, 1)
                R.append(np.abs(model.feature_importances_[:, None]))
            else:
                pass
        # (num_latents, num_factors)
        R = np.concatenate(R, axis=1)
        assert R.shape[1] == np.sum(np.asarray(cont_mask, dtype=np.int32)), \
            f"R.shape = {R.shape[1]} while #cont = {np.sum(np.asarray(cont_mask, dtype=np.int32))}"
        # Disentanglement: (num_latents,)
        disentanglement_scores = Metric.entropic_scores(R.T)
        c_rel_importance = np.sum(R, axis=1) / np.sum(R)  #relative importance of each code variable
        assert 1 - 1e-4 < np.sum(c_rel_importance) < 1 + 1e-4, f"c_rel_importance: {c_rel_importance}"
        disentanglement = np.sum(disentanglement_scores * c_rel_importance)
        # Completeness
        completeness_scores = Metric.entropic_scores(R)
        completeness = np.mean(completeness_scores)
        # Informativeness
        train_avg_error = np.mean(train_errors)
        results = {
                'importance_matrix': R,
                'disentanglement': disentanglement,
                'completeness': completeness,
                'informativeness': train_avg_error,
            }
    
        if prnt:
            print(f"Disentanglement: {results['disentanglement']}\n\
                  Completeness: {results['completeness']}\n\
                      Informativeness: {results['train_avg_error']}")
        else:
            pass
        return results
    
class compute_metric:
    def __init__(self, x, y,  continuous_factors:bool = True, bins:int = 1000, printit = False):
        '''Compute evaluation metric for disentanglement

        Parameters
        ----------
        continuous_factors : bool, optional
            True:   factors are described as continuous variables
            False:  factors are described as discrete variables. 
            The default is True.
        bins : int, optional
            Number of bins to use for discretization. 
            The default is 1000.
    
        Returns
        -------
        dict : 
            disentanglement metrics
            
        
        '''
        self.x = x
        self.y = y
        self.continuous_factors = continuous_factors
        self.bins = bins
        self.printit = printit
        return
    
    def run(self, nb_runs_fvae = 20):
        fvae_metric = []
        mig_metric = []
        for ii in range(nb_runs_fvae):
            fvae_metric.append(Metric(self.x, self.y).kming())
        factor_vae_metric_mu, factor_vae_metric_sigma = np.mean(fvae_metric), np.std(fvae_metric)
        for ii in range(nb_runs_fvae):
            mig_metric.append( Metric(self.x, self.y, self.bins).mig(self.continuous_factors))
        mig_metric_mu, mig_metric_sigma = np.mean(mig_metric), np.std(mig_metric)
        modularity = Metric(self.x, self.y, self.bins).modularity(self.continuous_factors)
        jemmig = Metric(self.x, self.y, self.bins).jemmig(self.continuous_factors)
        mig_sup = Metric(self.x, self.y, self.bins).mig_sup(self.continuous_factors)
        dcimig = Metric(self.x, self.y).dcimig(self.continuous_factors)
        # lasso_dmetric = Metric(self.x, self.y).lasso_dmetric_()
        # randomforest_dmetric = Metric(self.x, self.y).randomforest_dmetric_()
        self.metrics =  {
                    'factorvae_score_mu': factor_vae_metric_mu,
                    'factorvae_score_sigma': factor_vae_metric_sigma,
                    'mig_score_mu': mig_metric_mu,
                    'mig_score_sigma': mig_metric_sigma,
                    'modularity': modularity,
                    'jemmig': jemmig,
                    'mig_sup': mig_sup,
                    'dcimig': dcimig,
                    # 'lasso_dmetric': lasso_dmetric,
                    # 'randomforest_dmetric': randomforest_dmetric
                    }
        
        if not self.printit:
            return self.metrics
        else:
            print('-'*70)
            print("|\t Factor-VAE \t|\t MIG \t|\t Modularity \t|\t Jemmig \t|")
            print('-'*70)
            print(f"|\t {self.metrics['factorvae_score_mu']} +/- {self.metrics['factorvae_score_sigma']} \t|"+
                  f"\t {self.metrics['mig_score_mu']} +/- {self.metrics['mig_score_sigma']} \t|\t {self.metrics['modularity']} \t|\t {self.metrics['jemmig']} \t|")
            return self.metrics
            

#%%% saving model...

def model_saver(model, 
                x_test, 
                hidden_dim, 
                latent_dims, 
                batch_size, 
                beta,
                gamma,
                distrib_type,
                loss_type,
                lr, 
                epochs,
                archi_type,
                mmd_typ,
                datatype):
    """
    Model saver: save model with attributes (loggers)

    Parameters
    ----------
    model : tf.keras.model
        trained model.
    x_test : np.array
        test data.
    hidden_dim : int
        hidden number of neurons.
    latent_dims : int
        latent dimension.
    batch_size : int
        batch size for training mini-batch algorithm.
    beta : int
        parameter for training beta VAE.
    gamma : int
        parameter for training GCVAE.
    distrib_type : str
        distribution type: Gaussian or Bernoulli.
    loss_type : loss type
        loss type is the vea loss model in use.
    lr : float
        learning rate.
    epochs : int
        Numbers of epochs to run algorithm.
    archi_type : str
        deep learning architecture used.
    mmd_typ : str
        type of KL(q||p) used. options [mmd, mah, mah_gcvae].
    datatype : str
        Type of data model is training. ex, mnist, celebA etc.


    Returns
    -------
    None.

    """
    path= os.getcwd() #get working directory
    if not loss_type == 'gcvae':
        if not os.path.exists(os.path.join(path, f"{distrib_type}/{loss_type}/{datatype}/latent_{latent_dims}/{epochs}")):
            try:
                os.makedirs(os.path.join(path, f"{distrib_type}/{loss_type}/{datatype}/latent_{latent_dims}/{epochs}"), exist_ok = False)
                if not os.path.exists(os.path.join(path, f"{distrib_type}/{loss_type}/{datatype}/latent_{latent_dims}/{epochs}/model.h5")):
                    tf.saved_model.save(model.model, f"{path}/{distrib_type}/{loss_type}/{datatype}/latent_{latent_dims}/{epochs}/model.h5")
                    np.save( f"{path}/{distrib_type}/{loss_type}/{datatype}/latent_{latent_dims}/{epochs}/loggers.npy", model.loggers)
                else:
                    tf.saved_model.save(model.model, f"{path}/{distrib_type}/{loss_type}/{datatype}/latent_{latent_dims}/{epochs}/model.h5")
                    np.save( f"{path}/{distrib_type}/{loss_type}/{datatype}/latent_{latent_dims}/{epochs}/loggers.npy", model.loggers)
            except OSError:
                if not os.path.exists(os.path.join(path, f"{distrib_type}/{loss_type}/{datatype}/latent_{latent_dims}/{epochs}/model.h5")):
                    tf.saved_model.save(model.model, f"{path}/{distrib_type}/{loss_type}/{datatype}/latent_{latent_dims}/{epochs}/model.h5")
                    np.save( f"{path}/{distrib_type}/{loss_type}/{datatype}/latent_{latent_dims}/{epochs}/loggers.npy", model.loggers)
                else:
                    tf.saved_model.save(model.model, f"{path}/{distrib_type}/{loss_type}/{datatype}/latent_{latent_dims}/{epochs}/model.h5")
                    np.save( f"{path}/{distrib_type}/{loss_type}/{datatype}/latent_{latent_dims}/{epochs}/loggers.npy", model.loggers)
        else:
            if not os.path.exists(os.path.join(path, f"{distrib_type}/{loss_type}/{datatype}/latent_{latent_dims}/{epochs}/model.h5")):
                tf.saved_model.save(model.model, f"{path}/{distrib_type}/{loss_type}/{datatype}/latent_{latent_dims}/{epochs}/model.h5")
                np.save( f"{path}/{distrib_type}/{loss_type}/{datatype}/latent_{latent_dims}/{epochs}/loggers.npy", model.loggers)
            else:
                tf.saved_model.save(model.model, f"{path}/{distrib_type}/{loss_type}/{datatype}/latent_{latent_dims}/{epochs}/model.h5")
                np.save( f"{path}/{distrib_type}/{loss_type}/{datatype}/latent_{latent_dims}/{epochs}/loggers.npy", model.loggers)
    else:
        if not os.path.exists(os.path.join(path, f"{distrib_type}/{loss_type}/{datatype}/latent_{latent_dims}/{epochs}/{mmd_typ}")):
            try:
                os.makedirs(os.path.join(path, f"{distrib_type}/{loss_type}/{datatype}/latent_{latent_dims}/{epochs}/{mmd_typ}"), exist_ok = False)
                if not os.path.exists(os.path.join(path, f"{distrib_type}/{loss_type}/{datatype}/latent_{latent_dims}/{epochs}/{mmd_typ}/model.h5")):
                    tf.saved_model.save(model.model, f"{path}/{distrib_type}/{loss_type}/{datatype}/latent_{latent_dims}/{epochs}/{mmd_typ}/model.h5")
                    np.save( f"{path}/{distrib_type}/{loss_type}/{datatype}/latent_{latent_dims}/{epochs}/{mmd_typ}/loggers.npy", model.loggers)
                else:
                    tf.saved_model.save(model.model, f"{path}/{distrib_type}/{loss_type}/{datatype}/latent_{latent_dims}/{epochs}/{mmd_typ}/model.h5")
                    np.save( f"{path}/{distrib_type}/{loss_type}/{datatype}/latent_{latent_dims}/{epochs}/{mmd_typ}/loggers.npy", model.loggers)
            except OSError:
                if not os.path.exists(os.path.join(path, f"{distrib_type}/{loss_type}/{datatype}/latent_{latent_dims}/{epochs}/{mmd_typ}/model.h5")):
                    tf.saved_model.save(model.model, f"{path}/{distrib_type}/{loss_type}/{datatype}/latent_{latent_dims}/{epochs}/{mmd_typ}/model.h5")
                    np.save( f"{path}/{distrib_type}/{loss_type}/{datatype}/latent_{latent_dims}/{epochs}/{mmd_typ}/loggers.npy", model.loggers)
                else:
                    tf.saved_model.save(model.model, f"{path}/{distrib_type}/{loss_type}/{datatype}/latent_{latent_dims}/{epochs}/{mmd_typ}/model.h5")
                    np.save( f"{path}/{distrib_type}/{loss_type}/{datatype}/latent_{latent_dims}/{epochs}/{mmd_typ}/loggers.npy", model.loggers)
        else:
            if not os.path.exists(os.path.join(path, f"{distrib_type}/{loss_type}/{datatype}/latent_{latent_dims}/{epochs}/{mmd_typ}/model.h5")):
                tf.saved_model.save(model.model, f"{path}/{distrib_type}/{loss_type}/{datatype}/latent_{latent_dims}/{epochs}/{mmd_typ}/model.h5")
                np.save( f"{path}/{distrib_type}/{loss_type}/{datatype}/latent_{latent_dims}/{epochs}/{mmd_typ}/loggers.npy", model.loggers)
            else:
                tf.saved_model.save(model.model, f"{path}/{distrib_type}/{loss_type}/{datatype}/latent_{latent_dims}/{epochs}/{mmd_typ}/model.h5")
                np.save( f"{path}/{distrib_type}/{loss_type}/{datatype}/latent_{latent_dims}/{epochs}/{mmd_typ}/loggers.npy", model.loggers)
                
    #dsentanglement metrics...compute and save
    z_mean, z_std, z = model.model.encoder.predict(x_test, batch_size = batch_size) #\mu, stdev, reparam
    results = compute_metric(x_test.reshape(x_test.shape[0], x_test.shape[1]), z, True, 10).run()
    results['z'] = z
    
    if not loss_type == 'gcvae':
        if not os.path.exists(os.path.join(path, f"{distrib_type}/{loss_type}/{datatype}/latent_{latent_dims}/{epochs}")):
            try:
                os.makedirs(os.path.join(path, f"{distrib_type}/{loss_type}/{datatype}/latent_{latent_dims}/{epochs}"), exist_ok = False)
                if not os.path.exists(os.path.join(path, f"{distrib_type}/{loss_type}/{datatype}/latent_{latent_dims}/{epochs}/results.npy")):
                    np.save(os.path.join(path, f"{distrib_type}/{loss_type}/{datatype}/latent_{latent_dims}/{epochs}/results.npy"), results)
                else:
                    np.save(os.path.join(path, f"{distrib_type}/{loss_type}/{datatype}/latent_{latent_dims}/{epochs}/results.npy"), results)
            except OSError:
                if not os.path.exists(os.path.join(path, f"{distrib_type}/{loss_type}/{datatype}/latent_{latent_dims}/{epochs}/results.npy")):
                    np.save(os.path.join(path, f"{distrib_type}/{loss_type}/{datatype}/latent_{latent_dims}/{epochs}/results.npy"), results)
                else:
                    np.save(os.path.join(path, f"{distrib_type}/{loss_type}/{datatype}/latent_{latent_dims}/{epochs}/results.npy"), results)
        else:
            if not os.path.exists(os.path.join(path, f"{distrib_type}/{loss_type}/{datatype}/latent_{latent_dims}/{epochs}/results.npy")):
                np.save(os.path.join(path, f"{distrib_type}/{loss_type}/{datatype}/latent_{latent_dims}/{epochs}/results.npy"), results)
            else:
                np.save(os.path.join(path, f"{distrib_type}/{loss_type}/{datatype}/latent_{latent_dims}/{epochs}/results.npy"), results)
    else:
        if not os.path.exists(os.path.join(path, f"{distrib_type}/{loss_type}/{datatype}/latent_{latent_dims}/{epochs}/{mmd_typ}")):
            try:
                os.makedirs(os.path.join(path, f"{distrib_type}/{loss_type}/{datatype}/latent_{latent_dims}/{epochs}/{mmd_typ}"), exist_ok = False)
                if not os.path.exists(os.path.join(path, f"{distrib_type}/{loss_type}/{datatype}/latent_{latent_dims}/{epochs}/{mmd_typ}/results.npy")):
                    np.save(os.path.join(path, f"{distrib_type}/{loss_type}/{datatype}/latent_{latent_dims}/{epochs}/{mmd_typ}/results.npy"), results)
                else:
                    np.save(os.path.join(path, f"{distrib_type}/{loss_type}/{datatype}/latent_{latent_dims}/{epochs}/{mmd_typ}/results.npy"), results)
            except OSError:
                if not os.path.exists(os.path.join(path, f"{distrib_type}/{loss_type}/{datatype}/latent_{latent_dims}/{epochs}/{mmd_typ}/results.npy")):
                    np.save(os.path.join(path, f"{distrib_type}/{loss_type}/{datatype}/latent_{latent_dims}/{epochs}/{mmd_typ}/results.npy"), results)
                else:
                    np.save(os.path.join(path, f"{distrib_type}/{loss_type}/{datatype}/latent_{latent_dims}/{epochs}/{mmd_typ}/results.npy"), results)
        else:
            if not os.path.exists(os.path.join(path, f"{distrib_type}/{loss_type}/{datatype}/latent_{latent_dims}/{epochs}/{mmd_typ}/results.npy")):
                np.save(os.path.join(path, f"{distrib_type}/{loss_type}/{datatype}/latent_{latent_dims}/{epochs}/{mmd_typ}/results.npy"), results)
            else:
                np.save(os.path.join(path, f"{distrib_type}/{loss_type}/{datatype}/latent_{latent_dims}/{epochs}/{mmd_typ}/results.npy"), results)
                
 
def model_saver_scriterion(model, 
                x_test, 
                hidden_dim, 
                latent_dims, 
                batch_size, 
                beta,
                gamma,
                distrib_type,
                loss_type,
                lr, 
                epochs,
                archi_type,
                mmd_typ,
                datatype,
                scriterion):
    """
    Model saver: save model with attributes (loggers)if stopping criterions is in use.

    Parameters
    ----------
    model : tf.keras.model
        trained model.
    x_test : np.array
        test data.
    hidden_dim : int
        hidden number of neurons.
    latent_dims : int
        latent dimension.
    batch_size : int
        batch size for training mini-batch algorithm.
    beta : int
        parameter for training beta VAE.
    gamma : int
        parameter for training GCVAE.
    distrib_type : str
        distribution type: Gaussian or Bernoulli.
    loss_type : loss type
        loss type is the vea loss model in use.
    lr : float
        learning rate.
    epochs : int
        Numbers of epochs to run algorithm.
    archi_type : str
        deep learning architecture used.
    mmd_typ : str
        type of KL(q||p) used. options [mmd, mah, mah_gcvae].
    datatype : str
        Type of data model is training. ex, mnist, celebA etc.


    Returns
    -------
    None.

    """
    path= os.getcwd() #get working directory
    if not loss_type == 'gcvae':
        if not os.path.exists(os.path.join(path, f"{distrib_type}/{loss_type}/{datatype}/latent_{latent_dims}/{epochs}")):
            try:
                os.makedirs(os.path.join(path, f"{distrib_type}/{loss_type}/{datatype}/latent_{latent_dims}/{epochs}"), exist_ok = False)
                if not os.path.exists(os.path.join(path, f"{distrib_type}/{loss_type}/{datatype}/latent_{latent_dims}/{epochs}/model.h5")):
                    tf.saved_model.save(model.model, f"{path}/{distrib_type}/{loss_type}/{datatype}/latent_{latent_dims}/{epochs}/model.h5")
                    np.save( f"{path}/{distrib_type}/{loss_type}/{datatype}/latent_{latent_dims}/{epochs}/loggers.npy", model.loggers)
                else:
                    tf.saved_model.save(model.model, f"{path}/{distrib_type}/{loss_type}/{datatype}/latent_{latent_dims}/{epochs}/model.h5")
                    np.save( f"{path}/{distrib_type}/{loss_type}/{datatype}/latent_{latent_dims}/{epochs}/loggers.npy", model.loggers)
            except OSError:
                if not os.path.exists(os.path.join(path, f"{distrib_type}/{loss_type}/{datatype}/latent_{latent_dims}/{epochs}/model.h5")):
                    tf.saved_model.save(model.model, f"{path}/{distrib_type}/{loss_type}/{datatype}/latent_{latent_dims}/{epochs}/model.h5")
                    np.save( f"{path}/{distrib_type}/{loss_type}/{datatype}/latent_{latent_dims}/{epochs}/loggers.npy", model.loggers)
                else:
                    tf.saved_model.save(model.model, f"{path}/{distrib_type}/{loss_type}/{datatype}/latent_{latent_dims}/{epochs}/model.h5")
                    np.save( f"{path}/{distrib_type}/{loss_type}/{datatype}/latent_{latent_dims}/{epochs}/loggers.npy", model.loggers)
        else:
            if not os.path.exists(os.path.join(path, f"{distrib_type}/{loss_type}/{datatype}/latent_{latent_dims}/{epochs}/model.h5")):
                tf.saved_model.save(model.model, f"{path}/{distrib_type}/{loss_type}/{datatype}/latent_{latent_dims}/{epochs}/model.h5")
                np.save( f"{path}/{distrib_type}/{loss_type}/{datatype}/latent_{latent_dims}/{epochs}/loggers.npy", model.loggers)
            else:
                tf.saved_model.save(model.model, f"{path}/{distrib_type}/{loss_type}/{datatype}/latent_{latent_dims}/{epochs}/model.h5")
                np.save( f"{path}/{distrib_type}/{loss_type}/{datatype}/latent_{latent_dims}/{epochs}/loggers.npy", model.loggers)
    else:
        if not scriterion == 'useStop':
            if not os.path.exists(os.path.join(path, f"{distrib_type}/{loss_type}/{datatype}/latent_{latent_dims}/{epochs}/{mmd_typ}")):
                try:
                    os.makedirs(os.path.join(path, f"{distrib_type}/{loss_type}/{datatype}/latent_{latent_dims}/{epochs}/{mmd_typ}"), exist_ok = False)
                    if not os.path.exists(os.path.join(path, f"{distrib_type}/{loss_type}/{datatype}/latent_{latent_dims}/{epochs}/{mmd_typ}/model.h5")):
                        tf.saved_model.save(model.model, f"{path}/{distrib_type}/{loss_type}/{datatype}/latent_{latent_dims}/{epochs}/{mmd_typ}/model.h5")
                        np.save( f"{path}/{distrib_type}/{loss_type}/{datatype}/latent_{latent_dims}/{epochs}/{mmd_typ}/loggers.npy", model.loggers)
                    else:
                        tf.saved_model.save(model.model, f"{path}/{distrib_type}/{loss_type}/{datatype}/latent_{latent_dims}/{epochs}/{mmd_typ}/model.h5")
                        np.save( f"{path}/{distrib_type}/{loss_type}/{datatype}/latent_{latent_dims}/{epochs}/{mmd_typ}/loggers.npy", model.loggers)
                except OSError:
                    if not os.path.exists(os.path.join(path, f"{distrib_type}/{loss_type}/{datatype}/latent_{latent_dims}/{epochs}/{mmd_typ}/model.h5")):
                        tf.saved_model.save(model.model, f"{path}/{distrib_type}/{loss_type}/{datatype}/latent_{latent_dims}/{epochs}/{mmd_typ}/model.h5")
                        np.save( f"{path}/{distrib_type}/{loss_type}/{datatype}/latent_{latent_dims}/{epochs}/{mmd_typ}/loggers.npy", model.loggers)
                    else:
                        tf.saved_model.save(model.model, f"{path}/{distrib_type}/{loss_type}/{datatype}/latent_{latent_dims}/{epochs}/{mmd_typ}/model.h5")
                        np.save( f"{path}/{distrib_type}/{loss_type}/{datatype}/latent_{latent_dims}/{epochs}/{mmd_typ}/loggers.npy", model.loggers)
            else:
                if not os.path.exists(os.path.join(path, f"{distrib_type}/{loss_type}/{datatype}/latent_{latent_dims}/{epochs}/{mmd_typ}/model.h5")):
                    tf.saved_model.save(model.model, f"{path}/{distrib_type}/{loss_type}/{datatype}/latent_{latent_dims}/{epochs}/{mmd_typ}/model.h5")
                    np.save( f"{path}/{distrib_type}/{loss_type}/{datatype}/latent_{latent_dims}/{epochs}/{mmd_typ}/loggers.npy", model.loggers)
                else:
                    tf.saved_model.save(model.model, f"{path}/{distrib_type}/{loss_type}/{datatype}/latent_{latent_dims}/{epochs}/{mmd_typ}/model.h5")
                    np.save( f"{path}/{distrib_type}/{loss_type}/{datatype}/latent_{latent_dims}/{epochs}/{mmd_typ}/loggers.npy", model.loggers)
        else:
            if not os.path.exists(os.path.join(path, f"{distrib_type}/{loss_type}/{datatype}/{scriterion}/latent_{latent_dims}/{epochs}/{mmd_typ}")):
                try:
                    os.makedirs(os.path.join(path, f"{distrib_type}/{loss_type}/{datatype}/{scriterion}/latent_{latent_dims}/{epochs}/{mmd_typ}"), exist_ok = False)
                    if not os.path.exists(os.path.join(path, f"{distrib_type}/{loss_type}/{datatype}/{scriterion}/latent_{latent_dims}/{epochs}/{mmd_typ}/model.h5")):
                        tf.saved_model.save(model.model, f"{path}/{distrib_type}/{loss_type}/{datatype}/{scriterion}/latent_{latent_dims}/{epochs}/{mmd_typ}/model.h5")
                        np.save( f"{path}/{distrib_type}/{loss_type}/{datatype}/{scriterion}/latent_{latent_dims}/{epochs}/{mmd_typ}/loggers.npy", model.loggers)
                    else:
                        tf.saved_model.save(model.model, f"{path}/{distrib_type}/{loss_type}/{datatype}/{scriterion}/latent_{latent_dims}/{epochs}/{mmd_typ}/model.h5")
                        np.save( f"{path}/{distrib_type}/{loss_type}/{datatype}/{scriterion}/latent_{latent_dims}/{epochs}/{mmd_typ}/loggers.npy", model.loggers)
                except OSError:
                    if not os.path.exists(os.path.join(path, f"{distrib_type}/{loss_type}/{datatype}/{scriterion}/latent_{latent_dims}/{epochs}/{mmd_typ}/model.h5")):
                        tf.saved_model.save(model.model, f"{path}/{distrib_type}/{loss_type}/{datatype}/{scriterion}/latent_{latent_dims}/{epochs}/{mmd_typ}/model.h5")
                        np.save( f"{path}/{distrib_type}/{loss_type}/{datatype}/{scriterion}/latent_{latent_dims}/{epochs}/{mmd_typ}/loggers.npy", model.loggers)
                    else:
                        tf.saved_model.save(model.model, f"{path}/{distrib_type}/{loss_type}/{datatype}/{scriterion}/latent_{latent_dims}/{epochs}/{mmd_typ}/model.h5")
                        np.save( f"{path}/{distrib_type}/{loss_type}/{datatype}/{scriterion}/latent_{latent_dims}/{epochs}/{mmd_typ}/loggers.npy", model.loggers)
            else:
                if not os.path.exists(os.path.join(path, f"{distrib_type}/{loss_type}/{datatype}/{scriterion}/latent_{latent_dims}/{epochs}/{mmd_typ}/model.h5")):
                    tf.saved_model.save(model.model, f"{path}/{distrib_type}/{loss_type}/{datatype}/{scriterion}/latent_{latent_dims}/{epochs}/{mmd_typ}/model.h5")
                    np.save( f"{path}/{distrib_type}/{loss_type}/{datatype}/{scriterion}/latent_{latent_dims}/{epochs}/{mmd_typ}/loggers.npy", model.loggers)
                else:
                    tf.saved_model.save(model.model, f"{path}/{distrib_type}/{loss_type}/{datatype}/{scriterion}/latent_{latent_dims}/{epochs}/{mmd_typ}/model.h5")
                    np.save( f"{path}/{distrib_type}/{loss_type}/{datatype}/{scriterion}/latent_{latent_dims}/{epochs}/{mmd_typ}/loggers.npy", model.loggers)
                    
    #dsentanglement metrics...compute and save
    z_mean, z_std, z = model.model.encoder.predict(x_test, batch_size = batch_size) #\mu, stdev, reparam
    results = compute_metric(x_test.reshape(x_test.shape[0], x_test.shape[1]), z, True, 10).run()
    results['z'] = z
    
    if not loss_type == 'gcvae':
        if not os.path.exists(os.path.join(path, f"{distrib_type}/{loss_type}/{datatype}/latent_{latent_dims}/{epochs}")):
            try:
                os.makedirs(os.path.join(path, f"{distrib_type}/{loss_type}/{datatype}/latent_{latent_dims}/{epochs}"), exist_ok = False)
                if not os.path.exists(os.path.join(path, f"{distrib_type}/{loss_type}/{datatype}/latent_{latent_dims}/{epochs}/results.npy")):
                    np.save(os.path.join(path, f"{distrib_type}/{loss_type}/{datatype}/latent_{latent_dims}/{epochs}/results.npy"), results)
                else:
                    np.save(os.path.join(path, f"{distrib_type}/{loss_type}/{datatype}/latent_{latent_dims}/{epochs}/results.npy"), results)
            except OSError:
                if not os.path.exists(os.path.join(path, f"{distrib_type}/{loss_type}/{datatype}/latent_{latent_dims}/{epochs}/results.npy")):
                    np.save(os.path.join(path, f"{distrib_type}/{loss_type}/{datatype}/latent_{latent_dims}/{epochs}/results.npy"), results)
                else:
                    np.save(os.path.join(path, f"{distrib_type}/{loss_type}/{datatype}/latent_{latent_dims}/{epochs}/results.npy"), results)
        else:
            if not os.path.exists(os.path.join(path, f"{distrib_type}/{loss_type}/{datatype}/latent_{latent_dims}/{epochs}/results.npy")):
                np.save(os.path.join(path, f"{distrib_type}/{loss_type}/{datatype}/latent_{latent_dims}/{epochs}/results.npy"), results)
            else:
                np.save(os.path.join(path, f"{distrib_type}/{loss_type}/{datatype}/latent_{latent_dims}/{epochs}/results.npy"), results)
    else:
        if not scriterion == 'useStop':
            if not os.path.exists(os.path.join(path, f"{distrib_type}/{loss_type}/{datatype}/latent_{latent_dims}/{epochs}/{mmd_typ}")):
                try:
                    os.makedirs(os.path.join(path, f"{distrib_type}/{loss_type}/{datatype}/latent_{latent_dims}/{epochs}/{mmd_typ}"), exist_ok = False)
                    if not os.path.exists(os.path.join(path, f"{distrib_type}/{loss_type}/{datatype}/latent_{latent_dims}/{epochs}/{mmd_typ}/results.npy")):
                        np.save(os.path.join(path, f"{distrib_type}/{loss_type}/{datatype}/latent_{latent_dims}/{epochs}/{mmd_typ}/results.npy"), results)
                    else:
                        np.save(os.path.join(path, f"{distrib_type}/{loss_type}/{datatype}/latent_{latent_dims}/{epochs}/{mmd_typ}/results.npy"), results)
                except OSError:
                    if not os.path.exists(os.path.join(path, f"{distrib_type}/{loss_type}/{datatype}/latent_{latent_dims}/{epochs}/{mmd_typ}/results.npy")):
                        np.save(os.path.join(path, f"{distrib_type}/{loss_type}/{datatype}/latent_{latent_dims}/{epochs}/{mmd_typ}/results.npy"), results)
                    else:
                        np.save(os.path.join(path, f"{distrib_type}/{loss_type}/{datatype}/latent_{latent_dims}/{epochs}/{mmd_typ}/results.npy"), results)
            else:
                if not os.path.exists(os.path.join(path, f"{distrib_type}/{loss_type}/{datatype}/latent_{latent_dims}/{epochs}/{mmd_typ}/results.npy")):
                    np.save(os.path.join(path, f"{distrib_type}/{loss_type}/{datatype}/latent_{latent_dims}/{epochs}/{mmd_typ}/results.npy"), results)
                else:
                    np.save(os.path.join(path, f"{distrib_type}/{loss_type}/{datatype}/latent_{latent_dims}/{epochs}/{mmd_typ}/results.npy"), results)
        else:
            if not os.path.exists(os.path.join(path, f"{distrib_type}/{loss_type}/{datatype}/latent_{latent_dims}/{epochs}/{mmd_typ}")):
                try:
                    os.makedirs(os.path.join(path, f"{distrib_type}/{loss_type}/{datatype}/{scriterion}/latent_{latent_dims}/{epochs}/{mmd_typ}"), exist_ok = False)
                    if not os.path.exists(os.path.join(path, f"{distrib_type}/{loss_type}/{datatype}/{scriterion}/latent_{latent_dims}/{epochs}/{mmd_typ}/results.npy")):
                        np.save(os.path.join(path, f"{distrib_type}/{loss_type}/{datatype}/{scriterion}/latent_{latent_dims}/{epochs}/{mmd_typ}/results.npy"), results)
                    else:
                        np.save(os.path.join(path, f"{distrib_type}/{loss_type}/{datatype}/{scriterion}/latent_{latent_dims}/{epochs}/{mmd_typ}/results.npy"), results)
                except OSError:
                    if not os.path.exists(os.path.join(path, f"{distrib_type}/{loss_type}/{datatype}/{scriterion}/latent_{latent_dims}/{epochs}/{mmd_typ}/results.npy")):
                        np.save(os.path.join(path, f"{distrib_type}/{loss_type}/{datatype}/{scriterion}/latent_{latent_dims}/{epochs}/{mmd_typ}/results.npy"), results)
                    else:
                        np.save(os.path.join(path, f"{distrib_type}/{loss_type}/{datatype}/{scriterion}/latent_{latent_dims}/{epochs}/{mmd_typ}/results.npy"), results)
            else:
                if not os.path.exists(os.path.join(path, f"{distrib_type}/{loss_type}/{datatype}/{scriterion}/latent_{latent_dims}/{epochs}/{mmd_typ}/results.npy")):
                    np.save(os.path.join(path, f"{distrib_type}/{loss_type}/{datatype}/{scriterion}/latent_{latent_dims}/{epochs}/{mmd_typ}/results.npy"), results)
                else:
                    np.save(os.path.join(path, f"{distrib_type}/{loss_type}/{datatype}/{scriterion}/latent_{latent_dims}/{epochs}/{mmd_typ}/results.npy"), results)
                
                

def model_saver_dsprites(model, 
                x_test,
                factors,
                hidden_dim, 
                latent_dims, 
                batch_size, 
                beta,
                gamma,
                distrib_type,
                loss_type,
                lr, 
                epochs,
                archi_type,
                mmd_typ,
                datatype):
    """
    Model saver: save model with attributes (loggers)

    Parameters
    ----------
    model : tf.keras.model
        trained model.
    x_test : np.array
        test data.
    factors : np.array
        original factors of the data.
    hidden_dim : int
        hidden number of neurons.
    latent_dims : int
        latent dimension.
    batch_size : int
        batch size for training mini-batch algorithm.
    beta : int
        parameter for training beta VAE.
    gamma : int
        parameter for training GCVAE.
    distrib_type : str
        distribution type: Gaussian or Bernoulli.
    loss_type : loss type
        loss type is the vea loss model in use.
    lr : float
        learning rate.
    epochs : int
        Numbers of epochs to run algorithm.
    archi_type : str
        deep learning architecture used.
    mmd_typ : str
        type of KL(q||p) used. options [mmd, mah, mah_gcvae].
    datatype : str
        Type of data model is training. ex, mnist, celebA etc.


    Returns
    -------
    None.

    """
    path= os.getcwd() #get working directory
    if not loss_type == 'gcvae':
        if not os.path.exists(os.path.join(path, f"{distrib_type}/{loss_type}/{datatype}/latent_{latent_dims}/{epochs}")):
            try:
                os.makedirs(os.path.join(path, f"{distrib_type}/{loss_type}/{datatype}/latent_{latent_dims}/{epochs}"), exist_ok = False)
                if not os.path.exists(os.path.join(path, f"{distrib_type}/{loss_type}/{datatype}/latent_{latent_dims}/{epochs}/model.h5")):
                    tf.saved_model.save(model.model, f"{path}/{distrib_type}/{loss_type}/{datatype}/latent_{latent_dims}/{epochs}/model.h5")
                    np.save( f"{path}/{distrib_type}/{loss_type}/{datatype}/latent_{latent_dims}/{epochs}/loggers.npy", model.loggers)
                else:
                    tf.saved_model.save(model.model, f"{path}/{distrib_type}/{loss_type}/{datatype}/latent_{latent_dims}/{epochs}/model.h5")
                    np.save( f"{path}/{distrib_type}/{loss_type}/{datatype}/latent_{latent_dims}/{epochs}/loggers.npy", model.loggers)
            except OSError:
                if not os.path.exists(os.path.join(path, f"{distrib_type}/{loss_type}/{datatype}/latent_{latent_dims}/{epochs}/model.h5")):
                    tf.saved_model.save(model.model, f"{path}/{distrib_type}/{loss_type}/{datatype}/latent_{latent_dims}/{epochs}/model.h5")
                    np.save( f"{path}/{distrib_type}/{loss_type}/{datatype}/latent_{latent_dims}/{epochs}/loggers.npy", model.loggers)
                else:
                    tf.saved_model.save(model.model, f"{path}/{distrib_type}/{loss_type}/{datatype}/latent_{latent_dims}/{epochs}/model.h5")
                    np.save( f"{path}/{distrib_type}/{loss_type}/{datatype}/latent_{latent_dims}/{epochs}/loggers.npy", model.loggers)
        else:
            if not os.path.exists(os.path.join(path, f"{distrib_type}/{loss_type}/{datatype}/latent_{latent_dims}/{epochs}/model.h5")):
                tf.saved_model.save(model.model, f"{path}/{distrib_type}/{loss_type}/{datatype}/latent_{latent_dims}/{epochs}/model.h5")
                np.save( f"{path}/{distrib_type}/{loss_type}/{datatype}/latent_{latent_dims}/{epochs}/loggers.npy", model.loggers)
            else:
                tf.saved_model.save(model.model, f"{path}/{distrib_type}/{loss_type}/{datatype}/latent_{latent_dims}/{epochs}/model.h5")
                np.save( f"{path}/{distrib_type}/{loss_type}/{datatype}/latent_{latent_dims}/{epochs}/loggers.npy", model.loggers)
    else:
        if not os.path.exists(os.path.join(path, f"{distrib_type}/{loss_type}/{datatype}/latent_{latent_dims}/{epochs}/{mmd_typ}")):
            try:
                os.makedirs(os.path.join(path, f"{distrib_type}/{loss_type}/{datatype}/latent_{latent_dims}/{epochs}/{mmd_typ}"), exist_ok = False)
                if not os.path.exists(os.path.join(path, f"{distrib_type}/{loss_type}/{datatype}/latent_{latent_dims}/{epochs}/{mmd_typ}/model.h5")):
                    tf.saved_model.save(model.model, f"{path}/{distrib_type}/{loss_type}/{datatype}/latent_{latent_dims}/{epochs}/{mmd_typ}/model.h5")
                    np.save( f"{path}/{distrib_type}/{loss_type}/{datatype}/latent_{latent_dims}/{epochs}/{mmd_typ}/loggers.npy", model.loggers)
                else:
                    tf.saved_model.save(model.model, f"{path}/{distrib_type}/{loss_type}/{datatype}/latent_{latent_dims}/{epochs}/{mmd_typ}/model.h5")
                    np.save( f"{path}/{distrib_type}/{loss_type}/{datatype}/latent_{latent_dims}/{epochs}/{mmd_typ}/loggers.npy", model.loggers)
            except OSError:
                if not os.path.exists(os.path.join(path, f"{distrib_type}/{loss_type}/{datatype}/latent_{latent_dims}/{epochs}/{mmd_typ}/model.h5")):
                    tf.saved_model.save(model.model, f"{path}/{distrib_type}/{loss_type}/{datatype}/latent_{latent_dims}/{epochs}/{mmd_typ}/model.h5")
                    np.save( f"{path}/{distrib_type}/{loss_type}/{datatype}/latent_{latent_dims}/{epochs}/{mmd_typ}/loggers.npy", model.loggers)
                else:
                    tf.saved_model.save(model.model, f"{path}/{distrib_type}/{loss_type}/{datatype}/latent_{latent_dims}/{epochs}/{mmd_typ}/model.h5")
                    np.save( f"{path}/{distrib_type}/{loss_type}/{datatype}/latent_{latent_dims}/{epochs}/{mmd_typ}/loggers.npy", model.loggers)
        else:
            if not os.path.exists(os.path.join(path, f"{distrib_type}/{loss_type}/{datatype}/latent_{latent_dims}/{epochs}/{mmd_typ}/model.h5")):
                tf.saved_model.save(model.model, f"{path}/{distrib_type}/{loss_type}/{datatype}/latent_{latent_dims}/{epochs}/{mmd_typ}/model.h5")
                np.save( f"{path}/{distrib_type}/{loss_type}/{datatype}/latent_{latent_dims}/{epochs}/{mmd_typ}/loggers.npy", model.loggers)
            else:
                tf.saved_model.save(model.model, f"{path}/{distrib_type}/{loss_type}/{datatype}/latent_{latent_dims}/{epochs}/{mmd_typ}/model.h5")
                np.save( f"{path}/{distrib_type}/{loss_type}/{datatype}/latent_{latent_dims}/{epochs}/{mmd_typ}/loggers.npy", model.loggers)
                
    #dsentanglement metrics...compute and save
    z_mean, z_std, z = model.model.encoder.predict(x_test, batch_size = batch_size) #\mu, \stdev, reparam
    results = compute_metric(factors, z, True, 10).run()
    results['z'] = z
    
    if not loss_type == 'gcvae':
        if not os.path.exists(os.path.join(path, f"{distrib_type}/{loss_type}/{datatype}/latent_{latent_dims}/{epochs}")):
            try:
                os.makedirs(os.path.join(path, f"{distrib_type}/{loss_type}/{datatype}/latent_{latent_dims}/{epochs}"), exist_ok = False)
                if not os.path.exists(os.path.join(path, f"{distrib_type}/{loss_type}/{datatype}/latent_{latent_dims}/{epochs}/results.npy")):
                    np.save(os.path.join(path, f"{distrib_type}/{loss_type}/{datatype}/latent_{latent_dims}/{epochs}/results.npy"), results)
                else:
                    np.save(os.path.join(path, f"{distrib_type}/{loss_type}/{datatype}/latent_{latent_dims}/{epochs}/results.npy"), results)
            except OSError:
                if not os.path.exists(os.path.join(path, f"{distrib_type}/{loss_type}/{datatype}/latent_{latent_dims}/{epochs}/results.npy")):
                    np.save(os.path.join(path, f"{distrib_type}/{loss_type}/{datatype}/latent_{latent_dims}/{epochs}/results.npy"), results)
                else:
                    np.save(os.path.join(path, f"{distrib_type}/{loss_type}/{datatype}/latent_{latent_dims}/{epochs}/results.npy"), results)
        else:
            if not os.path.exists(os.path.join(path, f"{distrib_type}/{loss_type}/{datatype}/latent_{latent_dims}/{epochs}/results.npy")):
                np.save(os.path.join(path, f"{distrib_type}/{loss_type}/{datatype}/latent_{latent_dims}/{epochs}/results.npy"), results)
            else:
                np.save(os.path.join(path, f"{distrib_type}/{loss_type}/{datatype}/latent_{latent_dims}/{epochs}/results.npy"), results)
    else:
        if not os.path.exists(os.path.join(path, f"{distrib_type}/{loss_type}/{datatype}/latent_{latent_dims}/{epochs}/{mmd_typ}")):
            try:
                os.makedirs(os.path.join(path, f"{distrib_type}/{loss_type}/{datatype}/latent_{latent_dims}/{epochs}/{mmd_typ}"), exist_ok = False)
                if not os.path.exists(os.path.join(path, f"{distrib_type}/{loss_type}/{datatype}/latent_{latent_dims}/{epochs}/{mmd_typ}/results.npy")):
                    np.save(os.path.join(path, f"{distrib_type}/{loss_type}/{datatype}/latent_{latent_dims}/{epochs}/{mmd_typ}/results.npy"), results)
                else:
                    np.save(os.path.join(path, f"{distrib_type}/{loss_type}/{datatype}/latent_{latent_dims}/{epochs}/{mmd_typ}/results.npy"), results)
            except OSError:
                if not os.path.exists(os.path.join(path, f"{distrib_type}/{loss_type}/{datatype}/latent_{latent_dims}/{epochs}/{mmd_typ}/results.npy")):
                    np.save(os.path.join(path, f"{distrib_type}/{loss_type}/{datatype}/latent_{latent_dims}/{epochs}/{mmd_typ}/results.npy"), results)
                else:
                    np.save(os.path.join(path, f"{distrib_type}/{loss_type}/{datatype}/latent_{latent_dims}/{epochs}/{mmd_typ}/results.npy"), results)
        else:
            if not os.path.exists(os.path.join(path, f"{distrib_type}/{loss_type}/{datatype}/latent_{latent_dims}/{epochs}/{mmd_typ}/results.npy")):
                np.save(os.path.join(path, f"{distrib_type}/{loss_type}/{datatype}/latent_{latent_dims}/{epochs}/{mmd_typ}/results.npy"), results)
            else:
                np.save(os.path.join(path, f"{distrib_type}/{loss_type}/{datatype}/latent_{latent_dims}/{epochs}/{mmd_typ}/results.npy"), results)


def model_saver_scriterion_sc(model, 
                x_test,
                factors,
                hidden_dim, 
                latent_dims, 
                batch_size, 
                beta,
                gamma,
                distrib_type,
                loss_type,
                lr, 
                epochs,
                archi_type,
                mmd_typ,
                datatype,
                scriterion):
    """
    Model saver: save model with attributes (loggers)if stopping criterions is in use.

    Parameters
    ----------
    model : tf.keras.model
        trained model.
    x_test : np.array
        test data.
    factors : np.array
        original factors of the data.
    hidden_dim : int
        hidden number of neurons.
    latent_dims : int
        latent dimension.
    batch_size : int
        batch size for training mini-batch algorithm.
    beta : int
        parameter for training beta VAE.
    gamma : int
        parameter for training GCVAE.
    distrib_type : str
        distribution type: Gaussian or Bernoulli.
    loss_type : loss type
        loss type is the vea loss model in use.
    lr : float
        learning rate.
    epochs : int
        Numbers of epochs to run algorithm.
    archi_type : str
        deep learning architecture used.
    mmd_typ : str
        type of KL(q||p) used. options [mmd, mah, mah_gcvae].
    datatype : str
        Type of data model is training. ex, mnist, celebA etc.


    Returns
    -------
    None.

    """
    path= os.getcwd() #get working directory
    if not loss_type == 'gcvae':
        if not os.path.exists(os.path.join(path, f"{distrib_type}/{loss_type}/{datatype}/latent_{latent_dims}/{epochs}")):
            try:
                os.makedirs(os.path.join(path, f"{distrib_type}/{loss_type}/{datatype}/latent_{latent_dims}/{epochs}"), exist_ok = False)
                if not os.path.exists(os.path.join(path, f"{distrib_type}/{loss_type}/{datatype}/latent_{latent_dims}/{epochs}/model.h5")):
                    tf.saved_model.save(model.model, f"{path}/{distrib_type}/{loss_type}/{datatype}/latent_{latent_dims}/{epochs}/model.h5")
                    np.save( f"{path}/{distrib_type}/{loss_type}/{datatype}/latent_{latent_dims}/{epochs}/loggers.npy", model.loggers)
                else:
                    tf.saved_model.save(model.model, f"{path}/{distrib_type}/{loss_type}/{datatype}/latent_{latent_dims}/{epochs}/model.h5")
                    np.save( f"{path}/{distrib_type}/{loss_type}/{datatype}/latent_{latent_dims}/{epochs}/loggers.npy", model.loggers)
            except OSError:
                if not os.path.exists(os.path.join(path, f"{distrib_type}/{loss_type}/{datatype}/latent_{latent_dims}/{epochs}/model.h5")):
                    tf.saved_model.save(model.model, f"{path}/{distrib_type}/{loss_type}/{datatype}/latent_{latent_dims}/{epochs}/model.h5")
                    np.save( f"{path}/{distrib_type}/{loss_type}/{datatype}/latent_{latent_dims}/{epochs}/loggers.npy", model.loggers)
                else:
                    tf.saved_model.save(model.model, f"{path}/{distrib_type}/{loss_type}/{datatype}/latent_{latent_dims}/{epochs}/model.h5")
                    np.save( f"{path}/{distrib_type}/{loss_type}/{datatype}/latent_{latent_dims}/{epochs}/loggers.npy", model.loggers)
        else:
            if not os.path.exists(os.path.join(path, f"{distrib_type}/{loss_type}/{datatype}/latent_{latent_dims}/{epochs}/model.h5")):
                tf.saved_model.save(model.model, f"{path}/{distrib_type}/{loss_type}/{datatype}/latent_{latent_dims}/{epochs}/model.h5")
                np.save( f"{path}/{distrib_type}/{loss_type}/{datatype}/latent_{latent_dims}/{epochs}/loggers.npy", model.loggers)
            else:
                tf.saved_model.save(model.model, f"{path}/{distrib_type}/{loss_type}/{datatype}/latent_{latent_dims}/{epochs}/model.h5")
                np.save( f"{path}/{distrib_type}/{loss_type}/{datatype}/latent_{latent_dims}/{epochs}/loggers.npy", model.loggers)
    else:
        if not scriterion == 'useStop':
            if not os.path.exists(os.path.join(path, f"{distrib_type}/{loss_type}/{datatype}/latent_{latent_dims}/{epochs}/{mmd_typ}")):
                try:
                    os.makedirs(os.path.join(path, f"{distrib_type}/{loss_type}/{datatype}/latent_{latent_dims}/{epochs}/{mmd_typ}"), exist_ok = False)
                    if not os.path.exists(os.path.join(path, f"{distrib_type}/{loss_type}/{datatype}/latent_{latent_dims}/{epochs}/{mmd_typ}/model.h5")):
                        tf.saved_model.save(model.model, f"{path}/{distrib_type}/{loss_type}/{datatype}/latent_{latent_dims}/{epochs}/{mmd_typ}/model.h5")
                        np.save( f"{path}/{distrib_type}/{loss_type}/{datatype}/latent_{latent_dims}/{epochs}/{mmd_typ}/loggers.npy", model.loggers)
                    else:
                        tf.saved_model.save(model.model, f"{path}/{distrib_type}/{loss_type}/{datatype}/latent_{latent_dims}/{epochs}/{mmd_typ}/model.h5")
                        np.save( f"{path}/{distrib_type}/{loss_type}/{datatype}/latent_{latent_dims}/{epochs}/{mmd_typ}/loggers.npy", model.loggers)
                except OSError:
                    if not os.path.exists(os.path.join(path, f"{distrib_type}/{loss_type}/{datatype}/latent_{latent_dims}/{epochs}/{mmd_typ}/model.h5")):
                        tf.saved_model.save(model.model, f"{path}/{distrib_type}/{loss_type}/{datatype}/latent_{latent_dims}/{epochs}/{mmd_typ}/model.h5")
                        np.save( f"{path}/{distrib_type}/{loss_type}/{datatype}/latent_{latent_dims}/{epochs}/{mmd_typ}/loggers.npy", model.loggers)
                    else:
                        tf.saved_model.save(model.model, f"{path}/{distrib_type}/{loss_type}/{datatype}/latent_{latent_dims}/{epochs}/{mmd_typ}/model.h5")
                        np.save( f"{path}/{distrib_type}/{loss_type}/{datatype}/latent_{latent_dims}/{epochs}/{mmd_typ}/loggers.npy", model.loggers)
            else:
                if not os.path.exists(os.path.join(path, f"{distrib_type}/{loss_type}/{datatype}/latent_{latent_dims}/{epochs}/{mmd_typ}/model.h5")):
                    tf.saved_model.save(model.model, f"{path}/{distrib_type}/{loss_type}/{datatype}/latent_{latent_dims}/{epochs}/{mmd_typ}/model.h5")
                    np.save( f"{path}/{distrib_type}/{loss_type}/{datatype}/latent_{latent_dims}/{epochs}/{mmd_typ}/loggers.npy", model.loggers)
                else:
                    tf.saved_model.save(model.model, f"{path}/{distrib_type}/{loss_type}/{datatype}/latent_{latent_dims}/{epochs}/{mmd_typ}/model.h5")
                    np.save( f"{path}/{distrib_type}/{loss_type}/{datatype}/latent_{latent_dims}/{epochs}/{mmd_typ}/loggers.npy", model.loggers)
        else:
            if not os.path.exists(os.path.join(path, f"{distrib_type}/{loss_type}/{datatype}/{scriterion}/latent_{latent_dims}/{epochs}/{mmd_typ}")):
                try:
                    os.makedirs(os.path.join(path, f"{distrib_type}/{loss_type}/{datatype}/{scriterion}/latent_{latent_dims}/{epochs}/{mmd_typ}"), exist_ok = False)
                    if not os.path.exists(os.path.join(path, f"{distrib_type}/{loss_type}/{datatype}/{scriterion}/latent_{latent_dims}/{epochs}/{mmd_typ}/model.h5")):
                        tf.saved_model.save(model.model, f"{path}/{distrib_type}/{loss_type}/{datatype}/{scriterion}/latent_{latent_dims}/{epochs}/{mmd_typ}/model.h5")
                        np.save( f"{path}/{distrib_type}/{loss_type}/{datatype}/{scriterion}/latent_{latent_dims}/{epochs}/{mmd_typ}/loggers.npy", model.loggers)
                    else:
                        tf.saved_model.save(model.model, f"{path}/{distrib_type}/{loss_type}/{datatype}/{scriterion}/latent_{latent_dims}/{epochs}/{mmd_typ}/model.h5")
                        np.save( f"{path}/{distrib_type}/{loss_type}/{datatype}/{scriterion}/latent_{latent_dims}/{epochs}/{mmd_typ}/loggers.npy", model.loggers)
                except OSError:
                    if not os.path.exists(os.path.join(path, f"{distrib_type}/{loss_type}/{datatype}/{scriterion}/latent_{latent_dims}/{epochs}/{mmd_typ}/model.h5")):
                        tf.saved_model.save(model.model, f"{path}/{distrib_type}/{loss_type}/{datatype}/{scriterion}/latent_{latent_dims}/{epochs}/{mmd_typ}/model.h5")
                        np.save( f"{path}/{distrib_type}/{loss_type}/{datatype}/{scriterion}/latent_{latent_dims}/{epochs}/{mmd_typ}/loggers.npy", model.loggers)
                    else:
                        tf.saved_model.save(model.model, f"{path}/{distrib_type}/{loss_type}/{datatype}/{scriterion}/latent_{latent_dims}/{epochs}/{mmd_typ}/model.h5")
                        np.save( f"{path}/{distrib_type}/{loss_type}/{datatype}/{scriterion}/latent_{latent_dims}/{epochs}/{mmd_typ}/loggers.npy", model.loggers)
            else:
                if not os.path.exists(os.path.join(path, f"{distrib_type}/{loss_type}/{datatype}/{scriterion}/latent_{latent_dims}/{epochs}/{mmd_typ}/model.h5")):
                    tf.saved_model.save(model.model, f"{path}/{distrib_type}/{loss_type}/{datatype}/{scriterion}/latent_{latent_dims}/{epochs}/{mmd_typ}/model.h5")
                    np.save( f"{path}/{distrib_type}/{loss_type}/{datatype}/{scriterion}/latent_{latent_dims}/{epochs}/{mmd_typ}/loggers.npy", model.loggers)
                else:
                    tf.saved_model.save(model.model, f"{path}/{distrib_type}/{loss_type}/{datatype}/{scriterion}/latent_{latent_dims}/{epochs}/{mmd_typ}/model.h5")
                    np.save( f"{path}/{distrib_type}/{loss_type}/{datatype}/{scriterion}/latent_{latent_dims}/{epochs}/{mmd_typ}/loggers.npy", model.loggers)
                    
    #dsentanglement metrics...compute and save
    z_mean, z_std, z = model.model.encoder.predict(x_test, batch_size = batch_size) #\mu, \stdev, reparam
    results = compute_metric(factors, z, True, 10).run()
    results['z'] = z
    
    if not loss_type == 'gcvae':
        if not os.path.exists(os.path.join(path, f"{distrib_type}/{loss_type}/{datatype}/latent_{latent_dims}/{epochs}")):
            try:
                os.makedirs(os.path.join(path, f"{distrib_type}/{loss_type}/{datatype}/latent_{latent_dims}/{epochs}"), exist_ok = False)
                if not os.path.exists(os.path.join(path, f"{distrib_type}/{loss_type}/{datatype}/latent_{latent_dims}/{epochs}/results.npy")):
                    np.save(os.path.join(path, f"{distrib_type}/{loss_type}/{datatype}/latent_{latent_dims}/{epochs}/results.npy"), results)
                else:
                    np.save(os.path.join(path, f"{distrib_type}/{loss_type}/{datatype}/latent_{latent_dims}/{epochs}/results.npy"), results)
            except OSError:
                if not os.path.exists(os.path.join(path, f"{distrib_type}/{loss_type}/{datatype}/latent_{latent_dims}/{epochs}/results.npy")):
                    np.save(os.path.join(path, f"{distrib_type}/{loss_type}/{datatype}/latent_{latent_dims}/{epochs}/results.npy"), results)
                else:
                    np.save(os.path.join(path, f"{distrib_type}/{loss_type}/{datatype}/latent_{latent_dims}/{epochs}/results.npy"), results)
        else:
            if not os.path.exists(os.path.join(path, f"{distrib_type}/{loss_type}/{datatype}/latent_{latent_dims}/{epochs}/results.npy")):
                np.save(os.path.join(path, f"{distrib_type}/{loss_type}/{datatype}/latent_{latent_dims}/{epochs}/results.npy"), results)
            else:
                np.save(os.path.join(path, f"{distrib_type}/{loss_type}/{datatype}/latent_{latent_dims}/{epochs}/results.npy"), results)
    else:
        if not scriterion == 'useStop':
            if not os.path.exists(os.path.join(path, f"{distrib_type}/{loss_type}/{datatype}/latent_{latent_dims}/{epochs}/{mmd_typ}")):
                try:
                    os.makedirs(os.path.join(path, f"{distrib_type}/{loss_type}/{datatype}/latent_{latent_dims}/{epochs}/{mmd_typ}"), exist_ok = False)
                    if not os.path.exists(os.path.join(path, f"{distrib_type}/{loss_type}/{datatype}/latent_{latent_dims}/{epochs}/{mmd_typ}/results.npy")):
                        np.save(os.path.join(path, f"{distrib_type}/{loss_type}/{datatype}/latent_{latent_dims}/{epochs}/{mmd_typ}/results.npy"), results)
                    else:
                        np.save(os.path.join(path, f"{distrib_type}/{loss_type}/{datatype}/latent_{latent_dims}/{epochs}/{mmd_typ}/results.npy"), results)
                except OSError:
                    if not os.path.exists(os.path.join(path, f"{distrib_type}/{loss_type}/{datatype}/latent_{latent_dims}/{epochs}/{mmd_typ}/results.npy")):
                        np.save(os.path.join(path, f"{distrib_type}/{loss_type}/{datatype}/latent_{latent_dims}/{epochs}/{mmd_typ}/results.npy"), results)
                    else:
                        np.save(os.path.join(path, f"{distrib_type}/{loss_type}/{datatype}/latent_{latent_dims}/{epochs}/{mmd_typ}/results.npy"), results)
            else:
                if not os.path.exists(os.path.join(path, f"{distrib_type}/{loss_type}/{datatype}/latent_{latent_dims}/{epochs}/{mmd_typ}/results.npy")):
                    np.save(os.path.join(path, f"{distrib_type}/{loss_type}/{datatype}/latent_{latent_dims}/{epochs}/{mmd_typ}/results.npy"), results)
                else:
                    np.save(os.path.join(path, f"{distrib_type}/{loss_type}/{datatype}/latent_{latent_dims}/{epochs}/{mmd_typ}/results.npy"), results)
        else:
            if not os.path.exists(os.path.join(path, f"{distrib_type}/{loss_type}/{datatype}/latent_{latent_dims}/{epochs}/{mmd_typ}")):
                try:
                    os.makedirs(os.path.join(path, f"{distrib_type}/{loss_type}/{datatype}/{scriterion}/latent_{latent_dims}/{epochs}/{mmd_typ}"), exist_ok = False)
                    if not os.path.exists(os.path.join(path, f"{distrib_type}/{loss_type}/{datatype}/{scriterion}/latent_{latent_dims}/{epochs}/{mmd_typ}/results.npy")):
                        np.save(os.path.join(path, f"{distrib_type}/{loss_type}/{datatype}/{scriterion}/latent_{latent_dims}/{epochs}/{mmd_typ}/results.npy"), results)
                    else:
                        np.save(os.path.join(path, f"{distrib_type}/{loss_type}/{datatype}/{scriterion}/latent_{latent_dims}/{epochs}/{mmd_typ}/results.npy"), results)
                except OSError:
                    if not os.path.exists(os.path.join(path, f"{distrib_type}/{loss_type}/{datatype}/{scriterion}/latent_{latent_dims}/{epochs}/{mmd_typ}/results.npy")):
                        np.save(os.path.join(path, f"{distrib_type}/{loss_type}/{datatype}/{scriterion}/latent_{latent_dims}/{epochs}/{mmd_typ}/results.npy"), results)
                    else:
                        np.save(os.path.join(path, f"{distrib_type}/{loss_type}/{datatype}/{scriterion}/latent_{latent_dims}/{epochs}/{mmd_typ}/results.npy"), results)
            else:
                if not os.path.exists(os.path.join(path, f"{distrib_type}/{loss_type}/{datatype}/{scriterion}/latent_{latent_dims}/{epochs}/{mmd_typ}/results.npy")):
                    np.save(os.path.join(path, f"{distrib_type}/{loss_type}/{datatype}/{scriterion}/latent_{latent_dims}/{epochs}/{mmd_typ}/results.npy"), results)
                else:
                    np.save(os.path.join(path, f"{distrib_type}/{loss_type}/{datatype}/{scriterion}/latent_{latent_dims}/{epochs}/{mmd_typ}/results.npy"), results)
                

#%%

def model_saver_2d(model, 
                x_test, 
                hidden_dim, 
                latent_dims, 
                batch_size, 
                beta,
                gamma,
                distrib_type,
                loss_type,
                lr, 
                epochs,
                archi_type,
                mmd_typ,
                datatype):
    """
    Model saver: save model with attributes (loggers)

    Parameters
    ----------
    model : tf.keras.model
        trained model.
    x_test : np.array
        test data.
    hidden_dim : int
        hidden number of neurons.
    latent_dims : int
        latent dimension.
    batch_size : int
        batch size for training mini-batch algorithm.
    beta : int
        parameter for training beta VAE.
    gamma : int
        parameter for training GCVAE.
    distrib_type : str
        distribution type: Gaussian or Bernoulli.
    loss_type : loss type
        loss type is the vea loss model in use.
    lr : float
        learning rate.
    epochs : int
        Numbers of epochs to run algorithm.
    archi_type : str
        deep learning architecture used.
    mmd_typ : str
        type of KL(q||p) used. options [mmd, mah, mah_gcvae].
    datatype : str
        Type of data model is training. ex, mnist, celebA etc.


    Returns
    -------
    None.

    """
    N, L, M, _ = x_test.shape
    path= os.getcwd() #get working directory
    if not loss_type == 'gcvae':
        if not os.path.exists(os.path.join(path, f"{distrib_type}/{loss_type}/{datatype}/latent_{latent_dims}/{epochs}")):
            try:
                os.makedirs(os.path.join(path, f"{distrib_type}/{loss_type}/{datatype}/latent_{latent_dims}/{epochs}"), exist_ok = False)
                if not os.path.exists(os.path.join(path, f"{distrib_type}/{loss_type}/{datatype}/latent_{latent_dims}/{epochs}/model.h5")):
                    tf.saved_model.save(model.model, f"{path}/{distrib_type}/{loss_type}/{datatype}/latent_{latent_dims}/{epochs}/model.h5")
                    np.save( f"{path}/{distrib_type}/{loss_type}/{datatype}/latent_{latent_dims}/{epochs}/loggers.npy", model.loggers)
                else:
                    tf.saved_model.save(model.model, f"{path}/{distrib_type}/{loss_type}/{datatype}/latent_{latent_dims}/{epochs}/model.h5")
                    np.save( f"{path}/{distrib_type}/{loss_type}/{datatype}/latent_{latent_dims}/{epochs}/loggers.npy", model.loggers)
            except OSError:
                if not os.path.exists(os.path.join(path, f"{distrib_type}/{loss_type}/{datatype}/latent_{latent_dims}/{epochs}/model.h5")):
                    tf.saved_model.save(model.model, f"{path}/{distrib_type}/{loss_type}/{datatype}/latent_{latent_dims}/{epochs}/model.h5")
                    np.save( f"{path}/{distrib_type}/{loss_type}/{datatype}/latent_{latent_dims}/{epochs}/loggers.npy", model.loggers)
                else:
                    tf.saved_model.save(model.model, f"{path}/{distrib_type}/{loss_type}/{datatype}/latent_{latent_dims}/{epochs}/model.h5")
                    np.save( f"{path}/{distrib_type}/{loss_type}/{datatype}/latent_{latent_dims}/{epochs}/loggers.npy", model.loggers)
        else:
            if not os.path.exists(os.path.join(path, f"{distrib_type}/{loss_type}/{datatype}/latent_{latent_dims}/{epochs}/model.h5")):
                tf.saved_model.save(model.model, f"{path}/{distrib_type}/{loss_type}/{datatype}/latent_{latent_dims}/{epochs}/model.h5")
                np.save( f"{path}/{distrib_type}/{loss_type}/{datatype}/latent_{latent_dims}/{epochs}/loggers.npy", model.loggers)
            else:
                tf.saved_model.save(model.model, f"{path}/{distrib_type}/{loss_type}/{datatype}/latent_{latent_dims}/{epochs}/model.h5")
                np.save( f"{path}/{distrib_type}/{loss_type}/{datatype}/latent_{latent_dims}/{epochs}/loggers.npy", model.loggers)
    else:
        if not os.path.exists(os.path.join(path, f"{distrib_type}/{loss_type}/{datatype}/latent_{latent_dims}/{epochs}/{mmd_typ}")):
            try:
                os.makedirs(os.path.join(path, f"{distrib_type}/{loss_type}/{datatype}/latent_{latent_dims}/{epochs}/{mmd_typ}"), exist_ok = False)
                if not os.path.exists(os.path.join(path, f"{distrib_type}/{loss_type}/{datatype}/latent_{latent_dims}/{epochs}/{mmd_typ}/model.h5")):
                    tf.saved_model.save(model.model, f"{path}/{distrib_type}/{loss_type}/{datatype}/latent_{latent_dims}/{epochs}/{mmd_typ}/model.h5")
                    np.save( f"{path}/{distrib_type}/{loss_type}/{datatype}/latent_{latent_dims}/{epochs}/{mmd_typ}/loggers.npy", model.loggers)
                else:
                    tf.saved_model.save(model.model, f"{path}/{distrib_type}/{loss_type}/{datatype}/latent_{latent_dims}/{epochs}/{mmd_typ}/model.h5")
                    np.save( f"{path}/{distrib_type}/{loss_type}/{datatype}/latent_{latent_dims}/{epochs}/{mmd_typ}/loggers.npy", model.loggers)
            except OSError:
                if not os.path.exists(os.path.join(path, f"{distrib_type}/{loss_type}/{datatype}/latent_{latent_dims}/{epochs}/{mmd_typ}/model.h5")):
                    tf.saved_model.save(model.model, f"{path}/{distrib_type}/{loss_type}/{datatype}/latent_{latent_dims}/{epochs}/{mmd_typ}/model.h5")
                    np.save( f"{path}/{distrib_type}/{loss_type}/{datatype}/latent_{latent_dims}/{epochs}/{mmd_typ}/loggers.npy", model.loggers)
                else:
                    tf.saved_model.save(model.model, f"{path}/{distrib_type}/{loss_type}/{datatype}/latent_{latent_dims}/{epochs}/{mmd_typ}/model.h5")
                    np.save( f"{path}/{distrib_type}/{loss_type}/{datatype}/latent_{latent_dims}/{epochs}/{mmd_typ}/loggers.npy", model.loggers)
        else:
            if not os.path.exists(os.path.join(path, f"{distrib_type}/{loss_type}/{datatype}/latent_{latent_dims}/{epochs}/{mmd_typ}/model.h5")):
                tf.saved_model.save(model.model, f"{path}/{distrib_type}/{loss_type}/{datatype}/latent_{latent_dims}/{epochs}/{mmd_typ}/model.h5")
                np.save( f"{path}/{distrib_type}/{loss_type}/{datatype}/latent_{latent_dims}/{epochs}/{mmd_typ}/loggers.npy", model.loggers)
            else:
                tf.saved_model.save(model.model, f"{path}/{distrib_type}/{loss_type}/{datatype}/latent_{latent_dims}/{epochs}/{mmd_typ}/model.h5")
                np.save( f"{path}/{distrib_type}/{loss_type}/{datatype}/latent_{latent_dims}/{epochs}/{mmd_typ}/loggers.npy", model.loggers)
                
    #dsentanglement metrics...compute and save
    z_mean, z_std, z = model.model.encoder.predict(x_test, batch_size = batch_size) #\mu, stdev, reparam
    results = compute_metric(x_test.reshape(N, L*M), z, True, 10).run()
    results['z'] = z
    
    if not loss_type == 'gcvae':
        if not os.path.exists(os.path.join(path, f"{distrib_type}/{loss_type}/{datatype}/latent_{latent_dims}/{epochs}")):
            try:
                os.makedirs(os.path.join(path, f"{distrib_type}/{loss_type}/{datatype}/latent_{latent_dims}/{epochs}"), exist_ok = False)
                if not os.path.exists(os.path.join(path, f"{distrib_type}/{loss_type}/{datatype}/latent_{latent_dims}/{epochs}/results.npy")):
                    np.save(os.path.join(path, f"{distrib_type}/{loss_type}/{datatype}/latent_{latent_dims}/{epochs}/results.npy"), results)
                else:
                    np.save(os.path.join(path, f"{distrib_type}/{loss_type}/{datatype}/latent_{latent_dims}/{epochs}/results.npy"), results)
            except OSError:
                if not os.path.exists(os.path.join(path, f"{distrib_type}/{loss_type}/{datatype}/latent_{latent_dims}/{epochs}/results.npy")):
                    np.save(os.path.join(path, f"{distrib_type}/{loss_type}/{datatype}/latent_{latent_dims}/{epochs}/results.npy"), results)
                else:
                    np.save(os.path.join(path, f"{distrib_type}/{loss_type}/{datatype}/latent_{latent_dims}/{epochs}/results.npy"), results)
        else:
            if not os.path.exists(os.path.join(path, f"{distrib_type}/{loss_type}/{datatype}/latent_{latent_dims}/{epochs}/results.npy")):
                np.save(os.path.join(path, f"{distrib_type}/{loss_type}/{datatype}/latent_{latent_dims}/{epochs}/results.npy"), results)
            else:
                np.save(os.path.join(path, f"{distrib_type}/{loss_type}/{datatype}/latent_{latent_dims}/{epochs}/results.npy"), results)
    else:
        if not os.path.exists(os.path.join(path, f"{distrib_type}/{loss_type}/{datatype}/latent_{latent_dims}/{epochs}/{mmd_typ}")):
            try:
                os.makedirs(os.path.join(path, f"{distrib_type}/{loss_type}/{datatype}/latent_{latent_dims}/{epochs}/{mmd_typ}"), exist_ok = False)
                if not os.path.exists(os.path.join(path, f"{distrib_type}/{loss_type}/{datatype}/latent_{latent_dims}/{epochs}/{mmd_typ}/results.npy")):
                    np.save(os.path.join(path, f"{distrib_type}/{loss_type}/{datatype}/latent_{latent_dims}/{epochs}/{mmd_typ}/results.npy"), results)
                else:
                    np.save(os.path.join(path, f"{distrib_type}/{loss_type}/{datatype}/latent_{latent_dims}/{epochs}/{mmd_typ}/results.npy"), results)
            except OSError:
                if not os.path.exists(os.path.join(path, f"{distrib_type}/{loss_type}/{datatype}/latent_{latent_dims}/{epochs}/{mmd_typ}/results.npy")):
                    np.save(os.path.join(path, f"{distrib_type}/{loss_type}/{datatype}/latent_{latent_dims}/{epochs}/{mmd_typ}/results.npy"), results)
                else:
                    np.save(os.path.join(path, f"{distrib_type}/{loss_type}/{datatype}/latent_{latent_dims}/{epochs}/{mmd_typ}/results.npy"), results)
        else:
            if not os.path.exists(os.path.join(path, f"{distrib_type}/{loss_type}/{datatype}/latent_{latent_dims}/{epochs}/{mmd_typ}/results.npy")):
                np.save(os.path.join(path, f"{distrib_type}/{loss_type}/{datatype}/latent_{latent_dims}/{epochs}/{mmd_typ}/results.npy"), results)
            else:
                np.save(os.path.join(path, f"{distrib_type}/{loss_type}/{datatype}/latent_{latent_dims}/{epochs}/{mmd_typ}/results.npy"), results)
                



def model_saver_scriterion_2d(model, 
                x_test, 
                hidden_dim, 
                latent_dims, 
                batch_size, 
                beta,
                gamma,
                distrib_type,
                loss_type,
                lr, 
                epochs,
                archi_type,
                mmd_typ,
                datatype,
                scriterion):
    """
    Model saver: save model with attributes (loggers)if stopping criterions is in use.

    Parameters
    ----------
    model : tf.keras.model
        trained model.
    x_test : np.array
        test data.
    hidden_dim : int
        hidden number of neurons.
    latent_dims : int
        latent dimension.
    batch_size : int
        batch size for training mini-batch algorithm.
    beta : int
        parameter for training beta VAE.
    gamma : int
        parameter for training GCVAE.
    distrib_type : str
        distribution type: Gaussian or Bernoulli.
    loss_type : loss type
        loss type is the vea loss model in use.
    lr : float
        learning rate.
    epochs : int
        Numbers of epochs to run algorithm.
    archi_type : str
        deep learning architecture used.
    mmd_typ : str
        type of KL(q||p) used. options [mmd, mah, mah_gcvae].
    datatype : str
        Type of data model is training. ex, mnist, celebA etc.


    Returns
    -------
    None.

    """
    N, L, M, _ = x_test.shape
    path= os.getcwd() #get working directory
    if not loss_type == 'gcvae':
        if not os.path.exists(os.path.join(path, f"{distrib_type}/{loss_type}/{datatype}/latent_{latent_dims}/{epochs}")):
            try:
                os.makedirs(os.path.join(path, f"{distrib_type}/{loss_type}/{datatype}/latent_{latent_dims}/{epochs}"), exist_ok = False)
                if not os.path.exists(os.path.join(path, f"{distrib_type}/{loss_type}/{datatype}/latent_{latent_dims}/{epochs}/model.h5")):
                    tf.saved_model.save(model.model, f"{path}/{distrib_type}/{loss_type}/{datatype}/latent_{latent_dims}/{epochs}/model.h5")
                    np.save( f"{path}/{distrib_type}/{loss_type}/{datatype}/latent_{latent_dims}/{epochs}/loggers.npy", model.loggers)
                else:
                    tf.saved_model.save(model.model, f"{path}/{distrib_type}/{loss_type}/{datatype}/latent_{latent_dims}/{epochs}/model.h5")
                    np.save( f"{path}/{distrib_type}/{loss_type}/{datatype}/latent_{latent_dims}/{epochs}/loggers.npy", model.loggers)
            except OSError:
                if not os.path.exists(os.path.join(path, f"{distrib_type}/{loss_type}/{datatype}/latent_{latent_dims}/{epochs}/model.h5")):
                    tf.saved_model.save(model.model, f"{path}/{distrib_type}/{loss_type}/{datatype}/latent_{latent_dims}/{epochs}/model.h5")
                    np.save( f"{path}/{distrib_type}/{loss_type}/{datatype}/latent_{latent_dims}/{epochs}/loggers.npy", model.loggers)
                else:
                    tf.saved_model.save(model.model, f"{path}/{distrib_type}/{loss_type}/{datatype}/latent_{latent_dims}/{epochs}/model.h5")
                    np.save( f"{path}/{distrib_type}/{loss_type}/{datatype}/latent_{latent_dims}/{epochs}/loggers.npy", model.loggers)
        else:
            if not os.path.exists(os.path.join(path, f"{distrib_type}/{loss_type}/{datatype}/latent_{latent_dims}/{epochs}/model.h5")):
                tf.saved_model.save(model.model, f"{path}/{distrib_type}/{loss_type}/{datatype}/latent_{latent_dims}/{epochs}/model.h5")
                np.save( f"{path}/{distrib_type}/{loss_type}/{datatype}/latent_{latent_dims}/{epochs}/loggers.npy", model.loggers)
            else:
                tf.saved_model.save(model.model, f"{path}/{distrib_type}/{loss_type}/{datatype}/latent_{latent_dims}/{epochs}/model.h5")
                np.save( f"{path}/{distrib_type}/{loss_type}/{datatype}/latent_{latent_dims}/{epochs}/loggers.npy", model.loggers)
    else:
        if not scriterion == 'useStop':
            if not os.path.exists(os.path.join(path, f"{distrib_type}/{loss_type}/{datatype}/latent_{latent_dims}/{epochs}/{mmd_typ}")):
                try:
                    os.makedirs(os.path.join(path, f"{distrib_type}/{loss_type}/{datatype}/latent_{latent_dims}/{epochs}/{mmd_typ}"), exist_ok = False)
                    if not os.path.exists(os.path.join(path, f"{distrib_type}/{loss_type}/{datatype}/latent_{latent_dims}/{epochs}/{mmd_typ}/model.h5")):
                        tf.saved_model.save(model.model, f"{path}/{distrib_type}/{loss_type}/{datatype}/latent_{latent_dims}/{epochs}/{mmd_typ}/model.h5")
                        np.save( f"{path}/{distrib_type}/{loss_type}/{datatype}/latent_{latent_dims}/{epochs}/{mmd_typ}/loggers.npy", model.loggers)
                    else:
                        tf.saved_model.save(model.model, f"{path}/{distrib_type}/{loss_type}/{datatype}/latent_{latent_dims}/{epochs}/{mmd_typ}/model.h5")
                        np.save( f"{path}/{distrib_type}/{loss_type}/{datatype}/latent_{latent_dims}/{epochs}/{mmd_typ}/loggers.npy", model.loggers)
                except OSError:
                    if not os.path.exists(os.path.join(path, f"{distrib_type}/{loss_type}/{datatype}/latent_{latent_dims}/{epochs}/{mmd_typ}/model.h5")):
                        tf.saved_model.save(model.model, f"{path}/{distrib_type}/{loss_type}/{datatype}/latent_{latent_dims}/{epochs}/{mmd_typ}/model.h5")
                        np.save( f"{path}/{distrib_type}/{loss_type}/{datatype}/latent_{latent_dims}/{epochs}/{mmd_typ}/loggers.npy", model.loggers)
                    else:
                        tf.saved_model.save(model.model, f"{path}/{distrib_type}/{loss_type}/{datatype}/latent_{latent_dims}/{epochs}/{mmd_typ}/model.h5")
                        np.save( f"{path}/{distrib_type}/{loss_type}/{datatype}/latent_{latent_dims}/{epochs}/{mmd_typ}/loggers.npy", model.loggers)
            else:
                if not os.path.exists(os.path.join(path, f"{distrib_type}/{loss_type}/{datatype}/latent_{latent_dims}/{epochs}/{mmd_typ}/model.h5")):
                    tf.saved_model.save(model.model, f"{path}/{distrib_type}/{loss_type}/{datatype}/latent_{latent_dims}/{epochs}/{mmd_typ}/model.h5")
                    np.save( f"{path}/{distrib_type}/{loss_type}/{datatype}/latent_{latent_dims}/{epochs}/{mmd_typ}/loggers.npy", model.loggers)
                else:
                    tf.saved_model.save(model.model, f"{path}/{distrib_type}/{loss_type}/{datatype}/latent_{latent_dims}/{epochs}/{mmd_typ}/model.h5")
                    np.save( f"{path}/{distrib_type}/{loss_type}/{datatype}/latent_{latent_dims}/{epochs}/{mmd_typ}/loggers.npy", model.loggers)
        else:
            if not os.path.exists(os.path.join(path, f"{distrib_type}/{loss_type}/{datatype}/{scriterion}/latent_{latent_dims}/{epochs}/{mmd_typ}")):
                try:
                    os.makedirs(os.path.join(path, f"{distrib_type}/{loss_type}/{datatype}/{scriterion}/latent_{latent_dims}/{epochs}/{mmd_typ}"), exist_ok = False)
                    if not os.path.exists(os.path.join(path, f"{distrib_type}/{loss_type}/{datatype}/{scriterion}/latent_{latent_dims}/{epochs}/{mmd_typ}/model.h5")):
                        tf.saved_model.save(model.model, f"{path}/{distrib_type}/{loss_type}/{datatype}/{scriterion}/latent_{latent_dims}/{epochs}/{mmd_typ}/model.h5")
                        np.save( f"{path}/{distrib_type}/{loss_type}/{datatype}/{scriterion}/latent_{latent_dims}/{epochs}/{mmd_typ}/loggers.npy", model.loggers)
                    else:
                        tf.saved_model.save(model.model, f"{path}/{distrib_type}/{loss_type}/{datatype}/{scriterion}/latent_{latent_dims}/{epochs}/{mmd_typ}/model.h5")
                        np.save( f"{path}/{distrib_type}/{loss_type}/{datatype}/{scriterion}/latent_{latent_dims}/{epochs}/{mmd_typ}/loggers.npy", model.loggers)
                except OSError:
                    if not os.path.exists(os.path.join(path, f"{distrib_type}/{loss_type}/{datatype}/{scriterion}/latent_{latent_dims}/{epochs}/{mmd_typ}/model.h5")):
                        tf.saved_model.save(model.model, f"{path}/{distrib_type}/{loss_type}/{datatype}/{scriterion}/latent_{latent_dims}/{epochs}/{mmd_typ}/model.h5")
                        np.save( f"{path}/{distrib_type}/{loss_type}/{datatype}/{scriterion}/latent_{latent_dims}/{epochs}/{mmd_typ}/loggers.npy", model.loggers)
                    else:
                        tf.saved_model.save(model.model, f"{path}/{distrib_type}/{loss_type}/{datatype}/{scriterion}/latent_{latent_dims}/{epochs}/{mmd_typ}/model.h5")
                        np.save( f"{path}/{distrib_type}/{loss_type}/{datatype}/{scriterion}/latent_{latent_dims}/{epochs}/{mmd_typ}/loggers.npy", model.loggers)
            else:
                if not os.path.exists(os.path.join(path, f"{distrib_type}/{loss_type}/{datatype}/{scriterion}/latent_{latent_dims}/{epochs}/{mmd_typ}/model.h5")):
                    tf.saved_model.save(model.model, f"{path}/{distrib_type}/{loss_type}/{datatype}/{scriterion}/latent_{latent_dims}/{epochs}/{mmd_typ}/model.h5")
                    np.save( f"{path}/{distrib_type}/{loss_type}/{datatype}/{scriterion}/latent_{latent_dims}/{epochs}/{mmd_typ}/loggers.npy", model.loggers)
                else:
                    tf.saved_model.save(model.model, f"{path}/{distrib_type}/{loss_type}/{datatype}/{scriterion}/latent_{latent_dims}/{epochs}/{mmd_typ}/model.h5")
                    np.save( f"{path}/{distrib_type}/{loss_type}/{datatype}/{scriterion}/latent_{latent_dims}/{epochs}/{mmd_typ}/loggers.npy", model.loggers)
                    
    #dsentanglement metrics...compute and save
    z_mean, z_std, z = model.model.encoder.predict(x_test, batch_size = batch_size) #\mu, stdev, reparam
    results = compute_metric(x_test.reshape(N, L*M), z, True, 10).run()
    results['z'] = z
    
    if not loss_type == 'gcvae':
        if not os.path.exists(os.path.join(path, f"{distrib_type}/{loss_type}/{datatype}/latent_{latent_dims}/{epochs}")):
            try:
                os.makedirs(os.path.join(path, f"{distrib_type}/{loss_type}/{datatype}/latent_{latent_dims}/{epochs}"), exist_ok = False)
                if not os.path.exists(os.path.join(path, f"{distrib_type}/{loss_type}/{datatype}/latent_{latent_dims}/{epochs}/results.npy")):
                    np.save(os.path.join(path, f"{distrib_type}/{loss_type}/{datatype}/latent_{latent_dims}/{epochs}/results.npy"), results)
                else:
                    np.save(os.path.join(path, f"{distrib_type}/{loss_type}/{datatype}/latent_{latent_dims}/{epochs}/results.npy"), results)
            except OSError:
                if not os.path.exists(os.path.join(path, f"{distrib_type}/{loss_type}/{datatype}/latent_{latent_dims}/{epochs}/results.npy")):
                    np.save(os.path.join(path, f"{distrib_type}/{loss_type}/{datatype}/latent_{latent_dims}/{epochs}/results.npy"), results)
                else:
                    np.save(os.path.join(path, f"{distrib_type}/{loss_type}/{datatype}/latent_{latent_dims}/{epochs}/results.npy"), results)
        else:
            if not os.path.exists(os.path.join(path, f"{distrib_type}/{loss_type}/{datatype}/latent_{latent_dims}/{epochs}/results.npy")):
                np.save(os.path.join(path, f"{distrib_type}/{loss_type}/{datatype}/latent_{latent_dims}/{epochs}/results.npy"), results)
            else:
                np.save(os.path.join(path, f"{distrib_type}/{loss_type}/{datatype}/latent_{latent_dims}/{epochs}/results.npy"), results)
    else:
        if not scriterion == 'useStop':
            if not os.path.exists(os.path.join(path, f"{distrib_type}/{loss_type}/{datatype}/latent_{latent_dims}/{epochs}/{mmd_typ}")):
                try:
                    os.makedirs(os.path.join(path, f"{distrib_type}/{loss_type}/{datatype}/latent_{latent_dims}/{epochs}/{mmd_typ}"), exist_ok = False)
                    if not os.path.exists(os.path.join(path, f"{distrib_type}/{loss_type}/{datatype}/latent_{latent_dims}/{epochs}/{mmd_typ}/results.npy")):
                        np.save(os.path.join(path, f"{distrib_type}/{loss_type}/{datatype}/latent_{latent_dims}/{epochs}/{mmd_typ}/results.npy"), results)
                    else:
                        np.save(os.path.join(path, f"{distrib_type}/{loss_type}/{datatype}/latent_{latent_dims}/{epochs}/{mmd_typ}/results.npy"), results)
                except OSError:
                    if not os.path.exists(os.path.join(path, f"{distrib_type}/{loss_type}/{datatype}/latent_{latent_dims}/{epochs}/{mmd_typ}/results.npy")):
                        np.save(os.path.join(path, f"{distrib_type}/{loss_type}/{datatype}/latent_{latent_dims}/{epochs}/{mmd_typ}/results.npy"), results)
                    else:
                        np.save(os.path.join(path, f"{distrib_type}/{loss_type}/{datatype}/latent_{latent_dims}/{epochs}/{mmd_typ}/results.npy"), results)
            else:
                if not os.path.exists(os.path.join(path, f"{distrib_type}/{loss_type}/{datatype}/latent_{latent_dims}/{epochs}/{mmd_typ}/results.npy")):
                    np.save(os.path.join(path, f"{distrib_type}/{loss_type}/{datatype}/latent_{latent_dims}/{epochs}/{mmd_typ}/results.npy"), results)
                else:
                    np.save(os.path.join(path, f"{distrib_type}/{loss_type}/{datatype}/latent_{latent_dims}/{epochs}/{mmd_typ}/results.npy"), results)
        else:
            if not os.path.exists(os.path.join(path, f"{distrib_type}/{loss_type}/{datatype}/latent_{latent_dims}/{epochs}/{mmd_typ}")):
                try:
                    os.makedirs(os.path.join(path, f"{distrib_type}/{loss_type}/{datatype}/{scriterion}/latent_{latent_dims}/{epochs}/{mmd_typ}"), exist_ok = False)
                    if not os.path.exists(os.path.join(path, f"{distrib_type}/{loss_type}/{datatype}/{scriterion}/latent_{latent_dims}/{epochs}/{mmd_typ}/results.npy")):
                        np.save(os.path.join(path, f"{distrib_type}/{loss_type}/{datatype}/{scriterion}/latent_{latent_dims}/{epochs}/{mmd_typ}/results.npy"), results)
                    else:
                        np.save(os.path.join(path, f"{distrib_type}/{loss_type}/{datatype}/{scriterion}/latent_{latent_dims}/{epochs}/{mmd_typ}/results.npy"), results)
                except OSError:
                    if not os.path.exists(os.path.join(path, f"{distrib_type}/{loss_type}/{datatype}/{scriterion}/latent_{latent_dims}/{epochs}/{mmd_typ}/results.npy")):
                        np.save(os.path.join(path, f"{distrib_type}/{loss_type}/{datatype}/{scriterion}/latent_{latent_dims}/{epochs}/{mmd_typ}/results.npy"), results)
                    else:
                        np.save(os.path.join(path, f"{distrib_type}/{loss_type}/{datatype}/{scriterion}/latent_{latent_dims}/{epochs}/{mmd_typ}/results.npy"), results)
            else:
                if not os.path.exists(os.path.join(path, f"{distrib_type}/{loss_type}/{datatype}/{scriterion}/latent_{latent_dims}/{epochs}/{mmd_typ}/results.npy")):
                    np.save(os.path.join(path, f"{distrib_type}/{loss_type}/{datatype}/{scriterion}/latent_{latent_dims}/{epochs}/{mmd_typ}/results.npy"), results)
                else:
                    np.save(os.path.join(path, f"{distrib_type}/{loss_type}/{datatype}/{scriterion}/latent_{latent_dims}/{epochs}/{mmd_typ}/results.npy"), results)
                
                
#%% utilities for variational AutoEncoding Large Language Model

# import numpy as np
# import os, sys
import torch
from torch import nn, optim
import subprocess
from tqdm import tqdm, trange
from torch.nn.utils.rnn import pad_sequence

import json
import pdb

import torch.nn.init as init

import glob
import logging
import pickle
import random
from torch.utils.data.distributed import DistributedSampler

def frange_cycle_linear(n_iter, start=0.0, stop=1.0,  n_cycle=4, ratio=0.5):
    L = np.ones(n_iter) * stop
    period = n_iter/n_cycle
    step = (stop-start)/(period*ratio) # linear schedule

    for c in range(n_cycle):
        v, i = start, 0
        while v <= stop and (int(i+c*period) < n_iter):
            L[int(i+c*period)] = v
            v += step
            i += 1
    return L 

def frange_cycle_zero_linear(n_iter, start=0.0, stop=1.0,  n_cycle=4, ratio_increase=0.5, ratio_zero=0.3):
    L = np.ones(n_iter) * stop
    period = n_iter/n_cycle
    step = (stop-start)/(period*ratio_increase) # linear schedule

    for c in range(n_cycle):
        v, i = start, 0
        while v <= stop and (int(i+c*period) < n_iter):
            if i < period*ratio_zero:
                L[int(i+c*period)] = start
            else: 
                L[int(i+c*period)] = v
                v += step
            i += 1
    return L 


class uniform_initializer(object):
        def __init__(self, stdv):
            self.stdv = stdv
        def __call__(self, tensor):
            nn.init.uniform_(tensor, -self.stdv, self.stdv)


class xavier_normal_initializer(object):
    def __call__(self, tensor):
        nn.init.xavier_normal_(tensor)

def reconstruct(model, test_data_batch, vocab, strategy, fname):
    hyps = []
    refs = []
    with open(fname, "w") as fout:
        #for i in range(10):
            # batch_data = test_data_batch[i]

        for batch_data in test_data_batch:
            decoded_batch = model.reconstruct(batch_data, strategy)

            source = [[vocab.id2word(id_.item()) for id_ in sent] for sent in batch_data]
            for j in range(len(batch_data)):
                ref = " ".join(source[j])
                hyp = " ".join(decoded_batch[j])
                fout.write("SOURCE: {}\n".format(ref))
                fout.write("RECON: {}\n\n".format(hyp))

                refs += [ref[len("<s>"): -len("</s>")]]
                if strategy == "beam":
                    hyps += [hyp[len("<s>"): -len("</s>")]]
                else:
                    hyps += [hyp[: -len("</s>")]]

    fname_ref = fname + ".ref"
    fname_hyp = fname + ".hyp"
    with open(fname_ref, "w") as f:
        f.write("\n".join(refs))
    with open(fname_hyp, "w") as f:
        f.write("\n".join(hyps))
    call_multi_bleu_perl("scripts/multi-bleu.perl", fname_hyp, fname_ref, verbose=True)




def calc_iwnll(model_vae, eval_dataloader, args, ns=20):

    eval_loss = 0.0
    ############ Perplexity ############
    report_kl_loss = report_rec_loss = report_loss = 0
    report_num_words = report_num_sents = 0

    for batch in tqdm(eval_dataloader, desc="Evaluating PPL"):
        # pdb.set_trace()
        x0, x1, x_lengths = batch

        max_len_values, _ = x_lengths.max(0)
        x0 = x0[:,:max_len_values[0]]
        x1 = x1[:,:max_len_values[1]]

        x0 = x0.to(args.device)
        x1 = x1.to(args.device)
        x_lengths = x_lengths.to(args.device)

        # pdb.set_trace()
        # not predict start symbol
        report_num_words += x_lengths[:,1].sum().item()
        report_num_sents += args.eval_batch_size

        with torch.no_grad():
            loss, loss_rc, loss_kl = model_vae.loss_iw(x0, x1, nsamples=100, ns=5)

        loss_rc = loss_rc.sum()
        loss_kl = loss_kl.sum()
        loss = loss.sum()

        report_rec_loss += loss_rc.item()
        report_kl_loss += loss_kl.item()
        report_loss += loss.item()

        # pdb.set_trace()
        
    test_loss = report_loss / report_num_sents
    
    elbo = (report_kl_loss - report_rec_loss) / report_num_sents
    nll  = - report_rec_loss / report_num_sents
    kl   = report_kl_loss / report_num_sents
    ppl  = np.exp(-report_loss / report_num_words)

    return ppl, elbo, nll, kl



def calc_rec(model_vae, eval_dataloader, args, ns=1):

    eval_loss = 0.0
    ############ Perplexity ############
    report_kl_loss = report_rec_loss = report_loss = 0
    report_num_words = report_num_sents = 0

    i = 0
    for batch in tqdm(eval_dataloader, desc="Evaluating PPL"):
        # pdb.set_trace()
        x0, x1, x_lengths = batch

        max_len_values, _ = x_lengths.max(0)
        x0 = x0[:,:max_len_values[0]]
        x1 = x1[:,:max_len_values[1]]

        x0 = x0.to(args.device)
        x1 = x1.to(args.device)
        x_lengths = x_lengths.to(args.device)

        # pdb.set_trace()
        # not predict start symbol
        report_num_words += x_lengths[:,1].sum().item()
        report_num_sents += args.eval_batch_size

        with torch.no_grad():
            loss, loss_rc, loss_kl = model_vae.loss_iw(x0, x1, nsamples=1, ns=1)

        loss_rc = loss_rc.sum()
        report_rec_loss += loss_rc.item()

        i += 1
        if i > 500:
            break


        # pdb.set_trace()

    nll_s  = - report_rec_loss / report_num_sents
    nll_w  = - report_rec_loss / report_num_words

    return nll_s, nll_w



# def calc_mi(model, test_data_batch):
#     mi = 0
#     num_examples = 0
#     for batch_data in test_data_batch:
#         batch_size = batch_data.size(0)
#         num_examples += batch_size
#         mutual_info = model.calc_mi_q(batch_data)
#         mi += mutual_info * batch_size

#     return mi / num_examples



def calc_mi(model_vae, test_data_batch, args):
    # calc_mi_v3
    import math 
    from modules.utils import log_sum_exp

    mi = 0
    num_examples = 0

    mu_batch_list, logvar_batch_list = [], []
    neg_entropy = 0.
    for batch in tqdm(test_data_batch, desc="Evaluating MI, Stage 1"):

        x0, _, x_lengths = batch

        max_len_values, _ = x_lengths.max(0)
        x0 = x0[:,:max_len_values[0]]

        x0 = x0.to(args.device)

        with torch.no_grad():
            # encoding into bert features
            bert_fea = model_vae.encoder(x0)[1]

            # (batch_size, nz)
            mu, logvar = model_vae.encoder.linear(bert_fea).chunk(2, -1)

        x_batch, nz = mu.size()

        #print(x_batch, end=' ')

        num_examples += x_batch

        # E_{q(z|x)}log(q(z|x)) = -0.5*nz*log(2*\pi) - 0.5*(1+logvar).sum(-1)

        neg_entropy += (-0.5 * nz * math.log(2 * math.pi)- 0.5 * (1 + logvar).sum(-1)).sum().item()
        mu_batch_list += [mu.cpu()]
        logvar_batch_list += [logvar.cpu()]


    neg_entropy = neg_entropy / num_examples
    ##print()

    num_examples = 0
    log_qz = 0.
    for i in tqdm(range(len(mu_batch_list)), desc="Evaluating MI, Stage 2"):

        ###############
        # get z_samples
        ###############
        mu, logvar = mu_batch_list[i].cuda(), logvar_batch_list[i].cuda()
        
        # [z_batch, 1, nz]
        with torch.no_grad():
            z_samples = model_vae.reparameterize(mu, logvar, 1)

        z_samples = z_samples.view(-1, 1, nz)
        num_examples += z_samples.size(0)

        ###############
        # compute density
        ###############
        # [1, x_batch, nz]
        #mu, logvar = mu_batch_list[i].cuda(), logvar_batch_list[i].cuda()
        #indices = list(np.random.choice(np.arange(len(mu_batch_list)), 10)) + [i]
        indices = np.arange(len(mu_batch_list))
        mu = torch.cat([mu_batch_list[_] for _ in indices], dim=0).cuda()
        logvar = torch.cat([logvar_batch_list[_] for _ in indices], dim=0).cuda()
        x_batch, nz = mu.size()

        mu, logvar = mu.unsqueeze(0), logvar.unsqueeze(0)
        var = logvar.exp()

        # (z_batch, x_batch, nz)
        dev = z_samples - mu

        # (z_batch, x_batch)
        log_density = -0.5 * ((dev ** 2) / var).sum(dim=-1) - \
            0.5 * (nz * math.log(2 * math.pi) + logvar.sum(-1))

        # log q(z): aggregate posterior
        # [z_batch]
        log_qz += (log_sum_exp(log_density, dim=1) - math.log(x_batch)).sum(-1)

    log_qz /= num_examples
    mi = neg_entropy - log_qz

    return mi.item()





def calc_au(model_vae, eval_dataloader, args, delta=0.01):
    """compute the number of active units
    """
    cnt = 0
    for batch in tqdm(eval_dataloader, desc="Evaluating AU, Stage 1"):

        x0, _, x_lengths = batch
        max_len_values, _ = x_lengths.max(0)
        x0 = x0[:,:max_len_values[0]]
        x0 = x0.to(args.device)

        with torch.no_grad():
            # encoding into bert features
            bert_fea = model_vae.encoder(x0)[1]

            # (batch_size, nz)
            mean, logvar = model_vae.encoder.linear(bert_fea).chunk(2, -1)

        if cnt == 0:
            means_sum = mean.sum(dim=0, keepdim=True)
        else:
            means_sum = means_sum + mean.sum(dim=0, keepdim=True)
        cnt += mean.size(0)

    # (1, nz)
    mean_mean = means_sum / cnt

    cnt = 0
    for batch in tqdm(eval_dataloader, desc="Evaluating AU, Stage 2"):

        x0, _, _ = batch
        x0 = x0.to(args.device)

        with torch.no_grad():
            # encoding into bert features
            bert_fea = model_vae.encoder(x0)[1]

            # (batch_size, nz)
            mean, _ = model_vae.encoder.linear(bert_fea).chunk(2, -1)

        if cnt == 0:
            var_sum = ((mean - mean_mean) ** 2).sum(dim=0)
        else:
            var_sum = var_sum + ((mean - mean_mean) ** 2).sum(dim=0)
        cnt += mean.size(0)

    # (nz)
    au_var = var_sum / (cnt - 1)

    # pdb.set_trace()
    return (au_var >= delta).sum().item(), au_var


def sample_sentences(vae, vocab, device, num_sentences):
    global logging

    vae.eval()
    sampled_sents = []
    for i in range(num_sentences):
        z = vae.sample_from_prior(1)
        z = z.view(1,1,-1)
        start = vocab.word2id['<s>']
        # START = torch.tensor([[[start]]])
        START = torch.tensor([[start]])
        end = vocab.word2id['</s>']
        START = START.to(device)
        z = z.to(device)
        vae.eval()
        sentence = vae.decoder.sample_text(START, z, end, device)
        decoded_sentence = vocab.decode_sentence(sentence)
        sampled_sents.append(decoded_sentence)
    for i, sent in enumerate(sampled_sents):
        logging(i,":",' '.join(sent))


def visualize_latent(args, epoch, vae, device, test_data):
    nsamples = 1

    with open(os.path.join(args.exp_dir, f'synthetic_latent_{epoch}.txt'),'w') as f:
        test_data_batch, test_label_batch = test_data.create_data_batch_labels(batch_size=args.batch_size, device=device, batch_first=True)
        for i in range(len(test_data_batch)):
            batch_data = test_data_batch[i]
            batch_label = test_label_batch[i]
            batch_size, sent_len = batch_data.size()
            samples, _ = vae.encoder.encode(batch_data, nsamples)
            for i in range(batch_size):
                for j in range(nsamples):
                    sample = samples[i,j,:].cpu().detach().numpy().tolist()
                    f.write(batch_label[i] + '\t' + ' '.join([str(val) for val in sample]) + '\n')


def call_multi_bleu_perl(fname_bleu_script, fname_hyp, fname_ref, verbose=True):
    cmd = "perl %s %s < %s" % (fname_bleu_script, fname_ref, fname_hyp)
    popen = subprocess.Popen(cmd, stdout=subprocess.PIPE, \
        stderr=subprocess.PIPE, shell=True)
    popen.wait()
    try:
        bleu_result = popen.stdout.readline().strip().decode("utf-8")
        if verbose:
            print(bleu_result)
        bleu = float(bleu_result[7:bleu_result.index(',')])
        stderrs = popen.stderr.readlines()
        if len(stderrs) > 1:
            for line in stderrs:
                print(line.strip())
    except Exception as e:
        print(e)
        bleu = 0.
    return bleu




def weight_init(m):
    '''
    Usage:
        model = Model()
        model.apply(weight_init)
    '''
    if isinstance(m, nn.Conv1d):
        init.normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.Conv2d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.Conv3d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose1d):
        init.normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose2d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose3d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.BatchNorm1d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm2d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm3d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.Linear):
        init.xavier_normal_(m.weight.data)
        init.normal_(m.bias.data)
    elif isinstance(m, nn.LSTM):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.LSTMCell):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.GRU):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.GRUCell):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)




























                
                




