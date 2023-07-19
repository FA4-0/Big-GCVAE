#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 16 16:16:17 2022

@author: ifeanyi.ezukwoke
"""

#importing the libraries
seed_val = seed_trn = 42
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
from features import foi, ic, num, obj, ct, triplets, cl_features, x_n
import numpy as np
import os
import re
import time
import random
import datetime
from rouge import Rouge
#import dependencies
from itertools import chain
#import required libraries
import os #operating system utils
import pandas as pd #data manipulation package
import numpy as np #numerical operation package
import nltk
import pickle
import re
import glob
import langid
import shutil
from sklearn.manifold import TSNE
from tqdm import tqdm, trange
from sklearn.mixture import GaussianMixture
from sklearn.utils.class_weight import compute_sample_weight #for using sample weight...
from os.path import join
from collections import Counter
from multiprocessing.dummy import Pool as ThreadPool
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
# from Similarity import similarity as sm
from collections import Counter, defaultdict
from sklearn.preprocessing import StandardScaler
from nltk.tokenize import RegexpTokenizer
from nltk.stem.snowball import FrenchStemmer, PorterStemmer, ItalianStemmer
#--------transformer utils -------------------------------------------------------
import gc
from torch.utils.tensorboard import SummaryWriter
#from pytorchtools import EarlyStopping
import torch.nn.init as init
from torch.utils.data import Dataset, random_split
# from transformers import (WEIGHTS_NAME, CONFIG_NAME, 
#                             AutoTokenizer, AutoModelForCausalLM, AutoConfig,
#                             GPT2Tokenizer, GPT2LMHeadModel, GPT2Config, GPT2Model, #GPT Model
#                             BertTokenizer, EncoderDecoderModel, EncoderDecoderConfig, BertConfig, #Bert Model #---
#                             RobertaTokenizer, RobertaForCausalLM, RobertaConfig, RobertaConfig, RobertaForMaskedLM, #Roberta Model #---
#                             XLNetTokenizer, XLNetLMHeadModel, XLNetConfig, #XLNET Model
#                             XLMTokenizer, XLMWithLMHeadModel, XLMConfig, #XLM Model
#                             TransfoXLTokenizer, TransfoXLLMHeadModel, TransfoXLConfig, #TransfoXL Model
#                             OpenAIGPTTokenizer, OpenAIGPTLMHeadModel, OpenAIGPTConfig, #OpenAIGPTT Model
#                             BartTokenizer, BartForConditionalGeneration, BartConfig, #---
#                             T5Tokenizer, T5ForConditionalGeneration, T5Config,
#                             )
from pytorch_transformers import (WEIGHTS_NAME, AdamW, WarmupLinearSchedule,
                                  BertConfig, BertForLatentConnector, BertTokenizer,
                                  GPT2Config, GPT2ForLatentConnector, GPT2Tokenizer,
                                  OpenAIGPTConfig, OpenAIGPTLMHeadModel, OpenAIGPTTokenizer,
                                  RobertaConfig, RobertaForMaskedLM, RobertaTokenizer)

from transformers import AdamW, get_linear_schedule_with_warmup
from torch.utils.data import Dataset, DataLoader, random_split, RandomSampler, SequentialSampler, DistributedSampler
from transformers import pipeline, set_seed
torch.manual_seed(seed_trn)
#---VAE model------------------------------------
# from vae_pretrainer import VAE
from modules import VAEGCVAE
#---bleu evaluation------------------------------------
from bert_score import score as bert_score
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import SmoothingFunction
from nltk.translate import ribes_score, meteor_score
#--LESE eveluation 
from LESE import LESE
#--- logging
import logging
import argparse
logging.basicConfig(format="", level=logging.INFO)
logging.getLogger().setLevel(logging.INFO)
logger = logging.getLogger(__name__)
#----utils functions
from utils import (weight_init, calc_iwnll, calc_rec, calc_mi, calc_au, 
                   frange_cycle_linear, frange_cycle_zero_linear)

#-- 
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
#---clear cache
gc.collect()
torch.cuda.empty_cache()
#-- set root path
path = os.getcwd()

#%% Preprocessing...

class Mergefeatures(object):
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
    
def prepretreat(x, stopword = None, threshold = None):
    '''Docstring

    Parameters
    ----------
    x : string type
        word/sentence string.
    threshold : TYPE, optional
        threshold for cutting words. The default is None.

    Returns
    -------
    list
        list of pretreated words.

    '''
    if not threshold:
        threshold = 3
    else:
        threshold = threshold
    if not stopword:
        with open(join(path, 'stopwords.txt'), 'r+', encoding="utf8") as st: #note that you need to define path in the function
            stopwords = set([x for x in st.read().split()])
    else:
        stopwords = stopword
    txt = ','.join(list(set([re.sub(r'[^\w+]', '', x.lower()) for x in set(''.join(str([str(ii).strip() for ii in x])).split())])))
    txt = ' '.join(x for x in txt.split(',') if x not in stopwords if not len(x) < threshold if not any(z.isdigit() for z in x)) #remove stowords etc
    return ' '.join(re.sub('\[^a-zA-Z0-9\n\.]', ' ', x) for x in txt.split(' ') if not len(x) < threshold if not any(z.isdigit() for z in x)) #remove special characters from string


#%% Training function
    
class PIDControl():
    """PID controller for functions with Lagrangian hyper-parameters"""
    def __init__(self):
        """define them out of loop"""
        self.I_k1 = 0.0
        self.W_k1 = 0.0
        self.e_k1 = 0.0
        
    def _Kp_fun(self, Err, scale = 1):
        return 1.0/(1.0 + float(scale)*torch.exp(Err))
        
    def pid(self, exp_KL, kl_loss, Kp = 0.001, Ki = -0.001):
        #Kp = 0.001, Ki = -0.001 <-- Try this if results are unsatisfactory.
        """
        position PID algorithm
        Input: kl_loss
        return: weight for KL loss, beta
        """
        self.exp_KL = exp_KL
        error_k = torch.tensor(self.exp_KL - kl_loss, requires_grad = False)
        ## comput U as the control factor
        Pk = Kp * self._Kp_fun(error_k)
        Ik = self.I_k1 + Ki * error_k

        ## window up for integrator
        if self.W_k1 < 0 and self.W_k1 > 1:
            Ik = self.I_k1
            
        Wk = Pk + Ik
        self.W_k1 = Wk
        self.I_k1 = Ik
        self.e_k1 = error_k
        
        ## min and max value
        if Wk > 1:
            Wk = 1.0
        if Wk < 0:
            Wk = 0.0
        
        return Wk
    
    
def gcvae_loss(latent_code, loss_rec, loss_kl, args):
    #-- utility functions ...
    def compute_kernel(x, y):
        if len(x.size()) > 2:
            x, y = x[-1, :, :], y[-1, :, :]
        x, y = x[:, :args.latent_dim], y[:, :args.latent_dim]
        x_size, y_size = x.size(0), y.size(0)
        dim = x.size(1)
        x = x.unsqueeze(1) # (x_size, 1, dim)
        y = y.unsqueeze(0) # (1, y_size, dim)
        tiled_x = x.expand(x_size, y_size, dim)
        tiled_y = y.expand(x_size, y_size, dim)
        kernel_input = (tiled_x - tiled_y).pow(2).mean(2)/float(dim)
        return torch.exp(-kernel_input) # (x_size, y_size)
    
    def compute_mmd(x, y):
        x_kernel = compute_kernel(x, x)
        y_kernel = compute_kernel(y, y)
        xy_kernel = compute_kernel(x, y)
        mmd = x_kernel.mean() + y_kernel.mean() - 2*xy_kernel.mean()
        return mmd
    
    def z_mahalanobis_fn(z, diag:bool = True, psd = True)->float:
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
        if len(z.size()) > 2:
            z = z[-1, :, :] #--covert [1, N, M] --> [N, M]
        z = z[:, :args.latent_dim]
        m = lambda z: z - z.mean(axis = 0) #mean of vectors
        z_m = m(z) #mean centered data
        # logger.info(f'shape of z_m: {z_m.shape}')
        len_z = len(z_m)-1
        len_z = 1 if len_z == 0 else len_z
        #check if matrix entries are 
        if not psd:
            cov = 1/(len_z)*torch.matmul(z_m.T, z_m)
            cov = torch.eye(cov.shape[0]) if cov[0][0] == 0.0 else cov
            diag_cov = torch.diag(torch.diag(cov))
            diag_cov = torch.eye(diag_cov.shape[0]) if diag_cov[0][0] == 0.0 else diag_cov
        else:
            cov = 1/(len_z)*torch.matmul(z_m.T, z_m)
            cov = torch.eye(cov.shape[0]) if cov[0][0] == 0.0 else cov
            cov = torch.where(cov < 0, 0, cov)
            diag_cov = torch.diag(torch.diag(cov))
            diag_cov = torch.eye(diag_cov.shape[0]) if diag_cov[0][0] == 0.0 else diag_cov
            diag_cov = torch.where(diag_cov < 0, 0, diag_cov)
            # logger.info(f'shape of cov: {cov.shape}')
            # logger.info(f'shape of diag_cov: {diag_cov.shape}')
        if not diag:
            inv_cov = torch.linalg.inv(cov.cpu()).to(args.device) #inverse of a full covariance matrix
        else:
            inv_cov = torch.linalg.inv(diag_cov.cpu()).to(args.device) #inverse of diagonal covariance matrix
        # logger.info(f'shape of inv_cov: {inv_cov.shape}')
        trans_x = torch.matmul(torch.matmul(z_m, inv_cov), z_m.T)
        mah_mat_mean = trans_x.diagonal().mean() #torch.diagonal()
        return mah_mat_mean
    
    def z_mahalanobis_gcvae(z, diag:bool = True, psd = False)->float:
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
        if len(z.size()) > 2:
            z = z[-1, :, :] #--covert [1, N, M] --> [N, M]
        z = z[:, :args.latent_dim]
        m = lambda z: z - z.mean(axis = 0) #mean of vectors
        z_m = m(z) #mean centered data
        #check if matrix entries are 
        if not psd:
            cov = 1/(len(z)-1)*torch.matmul(z_m.T, z_m)
            diag_cov = torch.diag(torch.diag(cov))
        else:
            cov = 1/(len(z)-1)*torch.matmul(z_m.T, z_m)
            cov = torch.where(cov < 0, 0, cov)
            diag_cov = torch.diag(torch.diag(cov))
            diag_cov = torch.where(diag_cov < 0, 0, diag_cov)
        if not diag:
            inv_cov = torch.linalg.inv(cov) #inverse of a full covariance matrix
        else:
            inv_cov = torch.linalg.inv(diag_cov) #inverse of diagonal covariance matrix
        z_sample = torch.randn(z.size(), dtype = torch.float32)
        mah_gcvae = inv_cov * compute_mmd(z_sample, z) #-- compute  MMD
        mah_gcvae_mean = mah_gcvae.diagonal().mean()
        return mah_gcvae_mean
    
    #--MMD
    def mmd(z):
        z_sample = torch.randn(z.size(), dtype = torch.float32).to(args.device)
        return compute_mmd(z_sample, z)
    
    #--Mahalanobis 
    def z_mahalanobis(z):
        return z_mahalanobis_fn(z)
    
    #--Mahalanobis GCVAE
    def z_mah_gcvae(z):
        return z_mahalanobis_gcvae(z)
    
    #--define latent space using logits
    z = latent_code
    #--Maximum Mean Discrepancy (MMD)
    if args.mmd_type == 'mmd':
        mmd_fn = mmd
    #-- Mahalanobolis distance
    elif args.mmd_type == 'mah':
        mmd_fn = z_mahalanobis
    #-- Expectation of 'Mah'
    elif args.mmd_type == 'mah_gcvae':
        mmd_fn = z_mah_gcvae
    
    #-- compute variational losses..
    #select parameters...
    if args.vae_model_name.lower() == 'vae':
        alpha, beta, gamma = -1, 1, 0
        mmd_xy = 0
    elif args.vae_model_name == 'betavae':
        alpha, beta, gamma = -1, args.beta, 0
        mmd_xy = 0
    elif args.vae_model_name.lower() == 'controlvae':
        alpha = 0 
        beta = PIDControl().pid(args.init_kld, loss_kl)
        gamma = 0
        mmd_xy = 0
    elif args.vae_model_name.lower() == 'infovae':
        alpha, beta = 0, 0
        gamma = args.gamma
        mmd_xy = mmd_fn(z)
    elif args.vae_model_name.lower() == 'gcvae':
        mmd_xy = mmd_fn(z)
        alpha = PIDControl().pid(args.init_bce, loss_rec) #reconstruction weight --> cross entropy weight
        beta = PIDControl().pid(args.init_kld, loss_kl) #weight on KL-divergence --> Kullback-Leibler divergence.
        gamma = PIDControl().pid(args.init_mmd, mmd_xy) #weight if correlation measure.
    else:
        return ValueError(f'Unknown loss type: {args.vae_model_name}')
    #--
    loss = (1-alpha-beta)*loss_rec + beta*loss_kl + gamma*mmd_xy
    return loss, loss_rec, loss_kl, alpha, beta, gamma
    
        
def _rotate_checkpoints(args, checkpoint_prefix = 'checkpoint', use_mtime = False):
    if not args.save_total_limit:
        return
    if args.save_total_limit <= 0:
        return
    # Check if we should delete older checkpoint(s)
    output_dir = os.path.abspath(args.output_dir)
    checkpoints = [output_dir]
    if os.path.isdir(output_dir):
        checkpoints = list(os.path.join(output_dir, n) for n in os.listdir(output_dir))
        if args.local_rank not in [-1, 0]:
            checkpoints = [checkpoint for checkpoint in checkpoints if torch.distributed.get_rank() == int(checkpoint.split('-')[-1])]
        checkpoints.sort(key=lambda x: int(x.split('-')[-1]) if len(x.split('-')) > 1 else 0)
        if len(checkpoints) > args.save_total_limit:
            logger.info("Deleting older checkpoint [{}] due to args.save_total_limit".format(checkpoints[0]))
            shutil.rmtree(checkpoints[0])
          

def filter_fdr_frm_generated_text(txt, stp_sstech):
    stp_sstech_join = ' '.join(stp_sstech).split(' ')
    ai_br = txt.split(' ') #original
    ai_br_lwr = txt.lower().split(' ')
    
    index_zer = len(stp_sstech[0].split(' ')) - 1  # the scaler -1 indicates count starting with zero
    ai_pair = []
    st_pair = []
    len_ai_pair, len_stp_sstech_pair = len(ai_br_lwr), len(stp_sstech_join)
    #N-grams in ai_br (i.e AI generated FAs)
    for i in range(len_ai_pair - index_zer):
        pair_ai = ''
        for j in range(index_zer + 1):
            if (i + j) < len_ai_pair:
                pair_ai += ai_br_lwr[i + j] + ' '
        ai_pair.append(pair_ai.strip())
    # N-grams in stp_sstech_join (i.e ground truth FAs- from all possible 
    # combinations of Step-type + Substep technique)
    for i in range(len_stp_sstech_pair - index_zer):
        pair_st = ''
        for j in range(index_zer + 1):
            if (i + j) < len_stp_sstech_pair:
                pair_st += stp_sstech_join[i + j] + ' '
        st_pair.append(pair_st.strip())
    #filter here
    count = 0
    for i in ai_pair:
        if not i in st_pair:
            count += 1
        else:
            break
    ai_pair_filt = ' '.join(ai_br[count:])
    return ai_pair_filt


#%% clustering utils

def GMMClustering(code, nc_trials = 30):
    lowest_bic = np.infty
    bic = []
    n_components_range = range(1, nc_trials)
    cv_types = ['spherical', 'tied', 'diag', 'full']
    for cv_type in cv_types:
        for n_components in n_components_range:
            # Fit a Gaussian mixture with EM
            gmm = GaussianMixture(n_components = n_components, covariance_type = cv_type)
            gmm.fit(code)
            bic.append(gmm.bic(code))
            if bic[-1] < lowest_bic:
                lowest_bic = bic[-1]
                best_gmm = gmm
    c_label = best_gmm.predict(code)
    return c_label, bic


#%% Making the tokens and weight initialization...

def mask_tokens(inputs, tokenizer, args):
    """ Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original. """
    labels = inputs.clone()
    # We sample a few tokens in each sequence for masked-LM training (with probability args.mlm_probability defaults to 0.15 in Bert/RoBERTa)
    
    masked_indices = torch.bernoulli(torch.full(labels.shape, args.mlm_probability)).to(torch.uint8)
    labels[masked_indices == 1] = -1  # We only compute loss on masked tokens

    # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
    indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).to(torch.uint8) & masked_indices
    inputs[indices_replaced] = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)

    # 10% of the time, we replace masked input tokens with random word
    indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).to(torch.uint8) & masked_indices & ~indices_replaced
    indices_random = indices_random.to(args.device)
    random_words = torch.randint(len(tokenizer), labels.shape, dtype = torch.long)
    inputs[indices_random] = random_words[indices_random].to(args.device)

    # The rest of the time (10% of the time) we keep the masked input tokens unchanged
    return inputs, labels

def weights_init_rondom(model):
    model = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
    model_state_dict = model.state_dict()
    for key in model_state_dict:
        if 'encoder' in key:
            init.normal_(model_state_dict[key].data)  
     

def evaluate_generation_from_gpt2(model, decoder_tokenizer, args):
    context_tokens = decoder_tokenizer.encode('<BOS>')
    with torch.no_grad():
        out = sample_sequence(
                            model=model,
                            context=context_tokens,
                            length=args.max_seq_length, # Chunyuan: Fix length; or use <EOS> to complete a sentence
                            temperature=args.temperature,
                            top_k=args.top_k,
                            top_p=args.top_p,
                            device=args.device,
                            decoder_tokenizer = decoder_tokenizer, 
                            max_seq_length = args.max_seq_length
                        )
        text = decoder_tokenizer.decode(out[0,:].tolist(), clean_up_tokenization_spaces=True)
        text = text.split()[1:-1]
        text = ' '.join(text) + '\n'
    return text


def evaluate_generation_fromp_prior(model_vae, decoder_tokenizer, args):
    loc = torch.zeros([args.nz]).to(args.device)
    scale = torch.ones([args.nz]).to(args.device)
    prior = torch.distributions.normal.Normal(loc, scale)
    
    context_tokens = decoder_tokenizer.encode('<BOS>')

    with torch.no_grad():
        latent_z = prior.sample()
        # pdb.set_trace()
        past = model_vae.decoder.linear(latent_z.unsqueeze(0))
        
        # pdb.set_trace()
        out = sample_sequence_conditional(
                                        model=model_vae.decoder,
                                        context=context_tokens,
                                        past=past,
                                        length=args.max_seq_length, # Chunyuan: Fix length; or use <EOS> to complete a sentence
                                        temperature=args.temperature,
                                        top_k=args.top_k,
                                        top_p=args.top_p,
                                        device=args.device,
                                        decoder_tokenizer = decoder_tokenizer, 
                                        max_seq_length = args.max_seq_length
                                    )
        text = decoder_tokenizer.decode(out[0,:].tolist(), clean_up_tokenization_spaces=True)
        text = text.split()[1:-1]
        text = ' '.join(text) + '\n'
    return text


def save_checkpoint(model, optimizer, global_step, tokenizer_enc, tokenizer_dec, args):
    # Create output directory if needed
    # Save model checkpoint
    args.output_encoder_dir = os.path.join(args.output_dir, 'checkpoint-encoder-{}'.format(global_step))
    args.output_decoder_dir = os.path.join(args.output_dir, 'checkpoint-decoder-{}'.format(global_step))
    args.output_full_dir = os.path.join(args.output_dir, 'checkpoint-full-{}'.format(global_step))
    
    if not os.path.exists(args.output_encoder_dir) and args.local_rank in [-1, 0]:
        os.makedirs(args.output_encoder_dir)
    if not os.path.exists(args.output_decoder_dir) and args.local_rank in [-1, 0]:
        os.makedirs(args.output_decoder_dir)
    #--
    logger.info("   Saving encoder model checkpoint to %s", args.output_encoder_dir)
    logger.info("   Saving decoder model checkpoint to %s", args.output_decoder_dir)
    
    # #-- savingoptimizer and scheduler
    # torch.save(optimizer.state_dict(), os.path.join(args.output_dir, "optimizer.pt"))
    # torch.save(scheduler.state_dict(), os.path.join(args.output_dir, "scheduler.pt"))
    
    # Save a trained model, configuration and tokenizer using `save_pretrained()`.
    # They can then be reloaded using `from_pretrained()`
    model_encoder_to_save = model.module.encoder if hasattr(model, 'module') else model.encoder  # Take care of distributed/parallel training
    model_decoder_to_save = model.module.decoder if hasattr(model, 'module') else model.decoder  # Take care of distributed/parallel training
    
    logger.info("   Saving encoder and decoder tokenizers")
    tokenizer_enc.save_pretrained(args.output_encoder_dir)
    tokenizer_dec.save_pretrained(args.output_decoder_dir)
    
    # Good practice: save your training arguments together with the trained model
    model_encoder_to_save.save_pretrained(args.output_encoder_dir)
    torch.save(args, os.path.join(args.output_encoder_dir, 'training_encoder_args.bin'))

    model_decoder_to_save.save_pretrained(args.output_decoder_dir)
    torch.save(args, os.path.join(args.output_decoder_dir, 'training_decoder_args.bin'))

    # save the full model and optmizer into a checkpoint
    model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training

    checkpoint = {
                    'iter': global_step,
                    'model_state_dict': model_to_save.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'beta': model_to_save.args.beta,
                    'args': args
                    }

    if not os.path.exists(args.output_full_dir) and args.local_rank in [-1, 0]:
        os.makedirs(args.output_full_dir)

    logger.info("   Start saving full model checkpoint to %s", args.output_full_dir)
    torch.save(checkpoint, os.path.join(args.output_full_dir, 'training.bin'))
    logger.info("   Saving full checkpoint to %s", args.output_full_dir)
        
  
def load_checkpoint(args, MODEL_CLASSES):
    #--load directory of checkpoints
    files_outdir = os.listdir(args.output_dir)
    global_step = files_outdir[-1].split('-')[-1]

    output_encoder_dir = os.path.join(args.output_dir, 'checkpoint-encoder-{}'.format(global_step))
    output_decoder_dir = os.path.join(args.output_dir, 'checkpoint-decoder-{}'.format(global_step)) 
    output_full_dir    = os.path.join(args.output_dir, 'checkpoint-full-{}'.format(global_step)) 

    checkpoints = [ [output_encoder_dir, output_decoder_dir] ]
    logger.info("   Evaluate the following checkpoints: %s", checkpoints)
    
    # Load a trained Encoder model and vocabulary
    #---------Encoder
    config_class_enc, model_class_enc, tokenizer_class_enc = MODEL_CLASSES[args.model_type_enc]
    #encoder tokenization...load from checkpoint
    tokenizer_enc = tokenizer_class_enc.from_pretrained(output_encoder_dir) #Tokenization
    # config_enc = config_class_enc.from_pretrained(args.config_name_enc if args.config_name_enc else args.model_name_or_path_enc)
    model_enc = model_class_enc.from_pretrained(output_encoder_dir, 
                                                latent_size = args.latent_size) #Encoder LM class
    model_enc.to(args.device)
    if args.block_size <= 0:
        args.block_size = tokenizer_enc.max_len_single_sentence  # Our input block size will be the max possible for the model
    args.block_size = min(args.block_size, tokenizer_enc.max_len_single_sentence)
    
    # Load a trained Decoder model and vocabulary
    #---------Decoder
    config_class_dec, model_class_dec, tokenizer_class_dec = MODEL_CLASSES[args.model_type_dec]
    #decoder tokenization...
    tokenizer_dec = tokenizer_class_dec.from_pretrained(output_decoder_dir) #Tokenization
    # config_dec = config_class_dec.from_pretrained(args.config_name_dec if args.config_name_dec else args.model_name_or_path_dec)
    model_dec = model_class_dec.from_pretrained(output_decoder_dir, 
                                                latent_size = args.latent_size) #Decoder LM class
    model_dec.to(args.device)
    if args.block_size <= 0:
        args.block_size = tokenizer_dec.max_len_single_sentence  # Our input block size will be the max possible for the model
    args.block_size = min(args.block_size, tokenizer_dec.max_len_single_sentence) #check this if 'RuntimeError: The size of tensor a (1025) must match the size of tensor b (1024) at non-singleton dimension 3'
    
    #--Adding special tokens to GPTx
    special_tokens_dict = {'pad_token': '<PAD>', 'bos_token': '<BOS>', 'eos_token': '<EOS>'}
    num_added_toks = tokenizer_dec.add_special_tokens(special_tokens_dict)
    logger.info('   We have added', num_added_toks, 'tokens to GPTx')
    model_dec.resize_token_embeddings(len(tokenizer_dec))  # Notice: resize_token_embeddings expect to receive the full size of the new vocabulary, i.e. the length of the tokenizer.
    assert tokenizer_dec.pad_token == '<PAD>'
    
    # Load full VAE (BERT/RoBERTa/BART <-> GPTx) model
    checkpoint = torch.load(os.path.join(output_full_dir, 'training.bin'))
    model = VAEGCVAE(model_enc, model_dec, tokenizer_enc, tokenizer_dec, args)
    model.load_state_dict(checkpoint['model_state_dict'])
    logger.info("Pre-trained Opti-style model is successfully loaded")
    model.to(args.device)
    return (tokenizer_enc, model_enc, tokenizer_dec, model_dec, model)
    
def top_k_top_p_filtering(logits, top_k = 0, top_p = 0.0, filter_value = -float('Inf')):
    """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
        Args:
            logits: logits distribution shape (vocabulary size)
            top_k > 0: keep only top k tokens with highest probability (top-k filtering).
            top_p > 0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
                Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
        From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    """
    assert logits.dim() == 1  # batch size 1 for now - could be updated for more but the code would be less clear
    top_k = min(top_k, logits.size(-1))  # Safety check
    if top_k > 0:
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value
    return logits


def sample_sequence(model, length, context, num_samples = 1, temperature = 1, top_k = 0, top_p= 0.0, is_xlnet = False, device = 'cpu'):
    context = torch.tensor(context, dtype=torch.long, device=device)
    context = context.unsqueeze(0).repeat(num_samples, 1)
    generated = context
    with torch.no_grad():
        for _ in trange(length):
            inputs = {'input_ids': generated}
            if is_xlnet: 
                # XLNet is a direct (predict same token, not next token) and bi-directional model by default
                # => need one additional dummy token in the input (will be masked), attention mask and target mapping (see model docstring)
                input_ids = torch.cat((generated, torch.zeros((1, 1), dtype=torch.long, device=device)), dim=1)
                perm_mask = torch.zeros((1, input_ids.shape[1], input_ids.shape[1]), dtype=torch.float, device=device)
                perm_mask[:, :, -1] = 1.0  # Previous tokens don't see last token
                target_mapping = torch.zeros((1, 1, input_ids.shape[1]), dtype=torch.float, device=device)
                target_mapping[0, 0, -1] = 1.0  # predict last token
                inputs = {'input_ids': input_ids, 'perm_mask': perm_mask, 'target_mapping': target_mapping}

            outputs = model(**inputs)  # Note: we could also use 'past' with GPT-2/Transfo-XL/XLNet (cached hidden-states)
            next_token_logits = outputs[0][0, -1, :] / temperature
            filtered_logits = top_k_top_p_filtering(next_token_logits, top_k=top_k, top_p=top_p)
            next_token = torch.multinomial(F.softmax(filtered_logits, dim=-1), num_samples=1)
            generated = torch.cat((generated, next_token.unsqueeze(0)), dim=1)
    return generated


def sample_sequence_conditional(model, length, context, past = None, num_samples = 1, temperature = 1, top_k = 0, top_p = 0.0, device = 'cpu', decoder_tokenizer = None, max_seq_length = -1):
    context = torch.tensor(context, dtype = torch.long, device = device)
    context = context.unsqueeze(0).repeat(num_samples, 1)
    generated = context
    gen_seq_length = 0
    with torch.no_grad():
        while True:
            inputs = {'input_ids': generated, 'past': past}
            outputs = model(**inputs)  # Note: we could also use 'past' with GPT-2/Transfo-XL/XLNet (cached hidden-states)
            next_token_logits = outputs[0][0, -1, :] / temperature
            filtered_logits = top_k_top_p_filtering(next_token_logits, top_k = top_k, top_p = top_p)
            next_token = torch.multinomial(F.softmax(filtered_logits, dim=-1), num_samples=1)
            generated = torch.cat((generated, next_token.unsqueeze(0)), dim=1)
            gen_seq_length += 1
            # pdb.set_trace()
            if next_token.unsqueeze(0)[0,0].item() == decoder_tokenizer.encode('<EOS>')[0]:
                break
            if max_seq_length>0 and gen_seq_length>max_seq_length:
                break
    return generated

        
def latent_code_from_text(text, tokenizer_encoder, model_vae, args):
    tokenized1 = tokenizer_encoder.encode(text)
    tokenized1 = [101] + tokenized1 + [102]
    coded1 = torch.Tensor([tokenized1])
    coded1 = torch.Tensor.long(coded1)
    with torch.no_grad():
        x0 = coded1
        x0 = x0.to(args.device)
        pooled_hidden_fea = model_vae.encoder(x0, attention_mask=(x0 > 0).float())[1]
        mean, logvar = model_vae.encoder.linear(pooled_hidden_fea).chunk(2, -1)
        latent_z = mean.squeeze(1)  
        coded_length = len(tokenized1)
        return latent_z, coded_length

def latent_code_to_text(latent_z, tokenizer_decoder, model_vae, args):
    past = latent_z
    context_tokens = tokenizer_decoder.encode('<BOS>')

    length = 128 # maximum length, but not used 
    out = sample_sequence_conditional(
                                    model = model_vae.decoder,
                                    context = context_tokens,
                                    past = past,
                                    length = length, # Chunyuan: Fix length; or use <EOS> to complete a sentence
                                    temperature = args.temperature,
                                    top_k = args.top_k, #must be integer
                                    top_p = args.top_p,
                                    device = args.device,
                                    decoder_tokenizer = tokenizer_decoder
                                )
    text_x1 = tokenizer_decoder.decode(out[0,:].tolist(), 
                                       clean_up_tokenization_spaces = True,
                                       skip_special_tokens = True)
    text_x1 = text_x1.split()[1:-1]
    text_x1 = ' '.join(text_x1)
    return text_x1

def interpolate(model, tokenizer_encoder, tokenizer_decoder, args):
    # and then in the main function         
    latent_z1, coded_length1 = latent_code_from_text(args.sent_source, tokenizer_encoder, model, args)
    latent_z2, coded_length2 = latent_code_from_text(args.sent_target, tokenizer_encoder, model, args)
    #--
    if args.num_interpolation_steps > 1:
        result = defaultdict(str)
        num_steps = args.num_interpolation_steps + 1
        for step in range(num_steps+1):
            latent_z = latent_z1 + (latent_z2 - latent_z1) * step * 1.0/num_steps
            text_interpolate = latent_code_to_text(latent_z, tokenizer_decoder, model, args)
            result[step] = text_interpolate
            logger.info(f'Interpolation step {step}: {text_interpolate}')
    else:
        num_steps = 1
        latent_z = latent_z1 + (latent_z2 - latent_z1) * 1.0/num_steps
        text_interpolate = latent_code_to_text(latent_z, tokenizer_decoder, model, args)
        result = text_interpolate
        logger.info(f'Interpolation step {num_steps}: {result}')

    return result, latent_z


#%% Evaluation function

def evaluate(args, eval_dataset, model, tokenizer = None, tokenizer_enc = None, tokenizer_dec = None, prefix=""):
    eval_output_dir = args.eval_dir
    tokenizer = tokenizer if tokenizer != None else None
    tokenizer_enc = tokenizer_enc if tokenizer_enc != None else None
    tokenizer_dec = tokenizer_dec if tokenizer_dec != None else None

    if not os.path.exists(eval_output_dir) and args.local_rank in [-1, 0]:
        os.makedirs(eval_output_dir)

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    # Note that DistributedSampler samples randomly
    eval_sampler = SequentialSampler(eval_dataset) if args.local_rank == -1 else DistributedSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler = eval_sampler, batch_size = args.eval_batch_size)

    # multi-gpu eval
    if args.n_gpu > 1 and not isinstance(model, torch.nn.DataParallel):
        model = torch.nn.DataParallel(model)

    # Evaluation!
    logger.info(f"\n   ***** Running evaluation {prefix} *****")
    logger.info(f"  Num examples = {len(eval_dataset)}")
    logger.info(f"  Batch size = {args.eval_batch_size}")
    eval_loss, perplexity = 0.0, 0.0
    kl_loss, reconstr = 0.0, 0.0
    nb_eval_steps = 0
    model.eval()
    for batch in tqdm(eval_dataloader, desc = "Evaluating"):
        batch = tuple(t.to(args.device) for t in batch)
        #No optimization during evaluation. i.e mini-batch gradient descent not needed.
        with torch.no_grad():
            if args.encoder_decoder_sep:
                tokenized_text0, tokenized_text1, tokenized_text_lengths = batch #returns encoder_tokens, decoder_tokens, _ 
                inputs, labels = mask_tokens(tokenized_text0, tokenizer_enc, args) if args.mlm else (tokenized_text0, tokenized_text1)
                labels = tokenized_text1
                # tokenized_text1 = tokenized_text1.to(args.device)
                inputs = inputs.to(args.device)
                labels = labels.to(args.device)
            else:
                inputs = {
                          'input_ids':      batch[0],
                          'labels':         batch[0],
                          'attention_mask': batch[1],
                          }
            #varying beta...
            model.args.fb_mode = 1
            
            if args.use_deterministic_connect:
                model.args.fb_mode = 2
            # loss_rec, loss_kl, loss = model(inputs, labels) #VAE  LLM
            loss_rec, loss_kl, latent_z = model(inputs, labels) #VAE LLM throws reconstruction, KL-divergence and Latent code
            loss, loss_rec, loss_kl, alpha, beta, gamma = gcvae_loss(latent_z, loss_rec, loss_kl, args)
                
            #------
            if args.n_gpu > 1:
                loss_rec = loss_rec.mean()
                loss_kl = loss_kl.mean()
                loss = loss.mean()
                
            #--compute averages...
            avg_tmp_eval_loss = loss.mean().item() #average batch evaluation loss
            avg_templ_ppl =torch.exp(torch.tensor(avg_tmp_eval_loss).to(args.device)) #average batch perplexity
            avg_templ_rec = loss_rec.mean().item() #average reconstruction loss
            avg_templ_kl = loss_kl.mean().item() #average KL loss
            eval_loss += avg_tmp_eval_loss #total inreamental loss
            perplexity += avg_templ_ppl #perplexity
            kl_loss += avg_templ_kl #kl divergence
            reconstr += avg_templ_rec #reconstruction loss
                
        nb_eval_steps += 1
    #--
    eval_loss /= nb_eval_steps
    perplexity /= nb_eval_steps
    kl_loss /= nb_eval_steps
    reconstr /= nb_eval_steps
    #---
    output_eval_file = os.path.join(eval_output_dir, "eval_results.txt")
    if not args.encoder_decoder_sep:
        if not os.path.exists(output_eval_file):
            with open(output_eval_file, "w+") as writer:
                logger.info("   ***** Eval loss results *****")
                writer.write("   ***** Eval loss results *****\n")
                writer.write(f"Evaluation loss: {eval_loss} PPL: {perplexity}\n")
        else:
            with open(output_eval_file, "a+") as writer:
                writer.write(f"Evaluation loss: {eval_loss} PPL: {perplexity}\n")
    else:
        if not os.path.exists(output_eval_file):
            with open(output_eval_file, "w+") as writer:
                logger.info("   ***** Eval loss results *****")
                writer.write("   ***** Eval loss results *****\n")
                writer.write(f"Evaluation loss: {eval_loss} PPL: {perplexity} KL: {kl_loss} RECONSTRUCTION: {reconstr}\n")
        else:
            with open(output_eval_file, "a+") as writer:
                writer.write(f"Evaluation loss: {eval_loss} PPL: {perplexity} KL: {kl_loss} RECONSTRUCTION: {reconstr}\n")
    writer.close()
    
    return eval_loss, perplexity


#%% Defining the training loop

def train(args, train_dataset, eval_dataset, model, tokenizer = None, tokenizer_enc = None, tokenizer_dec = None):
    tokenizer = tokenizer if tokenizer != None else None
    tokenizer_enc = tokenizer_enc if tokenizer_enc != None else None
    tokenizer_dec = tokenizer_dec if tokenizer_dec != None else None
    """ Train the model """
    if args.local_rank in [-1, 0]:
        tb_writer = SummaryWriter()

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler = train_sampler, batch_size = args.train_batch_size)

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs
        
    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
    optimizer = AdamW(optimizer_grouped_parameters, lr = args.learning_rate, eps = args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps = args.warmup_steps, num_training_steps = t_total)

    # Check if saved optimizer or scheduler states exist
    if os.path.isfile(os.path.join(args.model_name_or_path, 'optimizer.pt')) and os.path.isfile(os.path.join(args.model_name_or_path, 'scheduler.pt')):
            # Load in optimizer and scheduler states
            optimizer.load_state_dict(torch.load(os.path.join(args.model_name_or_path, 'optimizer.pt')))
            scheduler.load_state_dict(torch.load(os.path.join(args.model_name_or_path, 'scheduler.pt')))    

    #--> numeric precision
    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level = args.fp16_opt_level)

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Distributed training (should be after apex fp16 initialization) set local_rank = -1 for Non-distributed training
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, 
                                                          device_ids = [args.local_rank],
                                                          output_device = args.local_rank,
                                                          find_unused_parameters = True)

    # Train!
    logger.info("  ***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per GPU = {args.per_gpu_train_batch_size}")
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                args.train_batch_size * args.gradient_accumulation_steps * (torch.distributed.get_world_size() if args.local_rank != -1 else 1),)
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {t_total}")

    global_step = 0
    epochs_trained = 0
    steps_trained_in_current_epoch = 0
    # Check if continuing training from a checkpoint
    if os.path.exists(args.model_name_or_path):
        # set global_step to gobal_step of last saved checkpoint from model path
        try:
            global_step = int(args.model_name_or_path.split('-')[-1].split('/')[0])
            epochs_trained = global_step // (len(train_dataloader) // args.gradient_accumulation_steps)
            steps_trained_in_current_epoch = global_step % (len(train_dataloader) // args.gradient_accumulation_steps)
            logger.info("  Continuing training from checkpoint, will skip to saved global_step")
            logger.info("  Continuing training from epoch %d", epochs_trained)
            logger.info("  Continuing training from global step %d", global_step)
            logger.info("  Will skip the first %d steps in the first epoch", steps_trained_in_current_epoch)
        except ValueError:
            global_step = 0
            logger.info("  Start fine-tuning...")

    avg_tr_loss, avg_eval_loss, avg_ppl = [], [], []
    avg_kl, avg_alpha, avg_beta, avg_gamma, avg_rec = [], [], [], [], []
    tr_loss, logging_loss = 0.0, 0.0
    tr_kl_loss, tr_alpha, tr_beta, tr_gamma, tr_loss_rec = 0.0, 0.0, 0.0, 0.0, 0.0
    model.zero_grad()
    train_iterator = trange(epochs_trained, int(args.num_train_epochs), desc = "Epoch", disable = args.local_rank not in [-1, 0])
    set_seed(args.seed)  # Added here for reproducibility (even between python 2 and 3)
    #-------
    n_iter = int(args.num_train_epochs) * len(train_dataloader)
    beta_t_list = frange_cycle_zero_linear(n_iter, 
                                           start = 0.0, 
                                           stop = args.beta,  
                                           n_cycle = 1, 
                                           ratio_increase = args.ratio_increase, 
                                           ratio_zero = args.ratio_zero)
    #-------
    for epoch in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable = args.local_rank not in [-1, 0])
        for step, batch in enumerate(epoch_iterator):
            # Skip past any already trained steps if resuming training
            if steps_trained_in_current_epoch > 0:
                steps_trained_in_current_epoch -= 1
                continue
            
            #begin training
            model.train()
            batch = tuple(t.to(args.device) for t in batch)
            if args.encoder_decoder_sep:
                tokenized_text0, tokenized_text1, tokenized_text_lengths = batch #returns encoder_tokens, decoder_tokens, _ 
                inputs, labels = mask_tokens(tokenized_text0, tokenizer_enc, args) if args.mlm else (tokenized_text0, tokenized_text1) #the label is the encoding of GPTx
                labels = tokenized_text1 #the label is the encoding of GPTx
                inputs = inputs.to(args.device)
                labels = labels.to(args.device)
            else:
                #--from using Failure Analysis dataset without masking...
                inputs = {
                          'input_ids':      batch[0],
                          'labels':         batch[0],
                          'attention_mask': batch[1],
                          }
            
            #varying beta...
            model.args.fb_mode = 1
            
            if args.use_deterministic_connect:
                model.args.fb_mode = 2
            # loss_rec, loss_kl, loss = model(inputs, labels) #VAE  LLM
            loss_rec, loss_kl, latent_z = model(inputs, labels) #VAE LLM throws reconstruction, KL-divergence and Latent code
            loss, loss_rec, loss_kl, alpha, beta, gamma = gcvae_loss(latent_z, loss_rec, loss_kl, args)
                
            #------
            if args.n_gpu > 1:
                loss_rec = loss_rec.mean()
                loss_kl = loss_kl.mean()
                loss = loss.mean()
            
            epoch_iterator.set_description(
                (
                    f'iter: {step +  epoch*len(epoch_iterator) }; loss: {loss.item():.3f}; '
                    f'loss_rec: {loss_rec.item():.3f}; loss_kl: {loss_kl.item():.3f}; '
                    #f'alpha: {alpha:.3f}; beta: {beta:.3f}; gamma: {gamma:.3f};'
                )
            )
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    if not args.freeze_params:
                        scaled_loss.backward()
                torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            
            tr_loss += loss.item()
            tr_kl_loss += loss_kl.item()
            tr_alpha += alpha
            tr_beta += beta
            tr_gamma += gamma
                
            if (step + 1) % args.gradient_accumulation_steps == 0:
                if args.fp16:
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                #---
                optimizer.step()
                scheduler.step()
                model.zero_grad()
                global_step += 1
        tr_loss_tmp = tr_loss / global_step
        tr_kl_tmp = tr_kl_loss / global_step
        tr_alpha_tmp = tr_alpha / global_step
        tr_beta_tmp = tr_beta / global_step
        tr_gamma_tmp = tr_gamma / global_step
        avg_tr_loss.append(tr_loss_tmp)
        avg_kl.append(tr_kl_tmp)
        avg_alpha.append(tr_alpha_tmp)
        avg_beta.append(tr_beta_tmp)
        avg_gamma.append(tr_gamma_tmp)

        #---model evaluation
        if args.local_rank in [-1, 0]:
            # Log metrics
            if args.local_rank == -1 and args.evaluate_during_training:
                eval_loss, ppl = evaluate(args, 
                                          eval_dataset, 
                                          model, 
                                          tokenizer_enc = tokenizer_enc, 
                                          tokenizer_dec = tokenizer_dec) #evaluation here
                avg_eval_loss.append(eval_loss)
                avg_ppl.append(ppl)
                logger.info(f'Train loss: {tr_loss_tmp} Eval loss: {eval_loss} Perplexity: {ppl}')
                tb_writer.add_scalar(f'Eval loss: {eval_loss} Perplexity: {ppl}', global_step)
            tb_writer.add_scalar('lr', scheduler.get_lr()[0], global_step)
            tb_writer.add_scalar('loss', (tr_loss - logging_loss)/args.logging_steps, global_step)
            logging_loss = tr_loss
    #----
    np.save(join(args.eval_dir, 'training_loss.npy'), avg_tr_loss.cpu() if not isinstance(avg_tr_loss, list) else avg_tr_loss) #final training loss
    np.save(join(args.eval_dir, 'evaluation_loss.npy'), avg_eval_loss.cpu() if not isinstance(avg_eval_loss, list) else avg_eval_loss) #final evluation loss
    np.save(join(args.eval_dir, 'training_kl.npy'), avg_kl.cpu() if not isinstance(avg_kl, list) else avg_kl) #final training kl-divergence loss
    np.save(join(args.eval_dir, 'training_alpha.npy'), torch.tensor(avg_alpha).cpu()) #final training alpha
    np.save(join(args.eval_dir, 'training_beta.npy'), torch.tensor(avg_beta).cpu()) #final training beta
    np.save(join(args.eval_dir, 'training_gamma.npy'), torch.tensor(avg_gamma).cpu()) #final training gamma
    np.save(join(args.eval_dir, 'evaluation_loss.npy'), avg_eval_loss) #final evluation loss
    np.save(join(args.eval_dir, 'perplexity.npy'), torch.tensor(avg_ppl).cpu()) #final evaluation perplexity
        
    #--Saving best-practices: if you use defaults names for the model, you can reload it using from_pretrained()
    if args.do_train and args.local_rank in [-1, 0]:
        if not args.encoder_decoder_sep:
            # Save model checkpoint
            output_dir = os.path.join(args.output_dir, f'checkpoint-{global_step}')
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            model_to_save = model.module if hasattr(model, 'module') else model
            model_to_save.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)
    
            torch.save(args, os.path.join(output_dir, 'training_args.bin'))
            logger.info(f"  Saving model checkpoint to {output_dir}")
    
            _rotate_checkpoints(args, checkpoint_prefix = 'checkpoint')
            torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
            torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
            logger.info("  Saving optimizer and scheduler states to {output_dir}")
        else:
            save_checkpoint(model, optimizer, global_step, tokenizer_enc, tokenizer_dec, args)

    if args.local_rank in [-1, 0]:
        tb_writer.close()

    return global_step, tr_loss / global_step


#%% Main

def main():
    #Model config, Model and their respective tokenizers
    # MODEL_CLASSES = {
    #                 'facebook/bart-large-cnn': (BartConfig, BartForConditionalGeneration, BartTokenizer),
    #                 'bert-base-uncased': (BertConfig, EncoderDecoderModel, BertTokenizer), #Causal model I
    #                 'roberta-base': (RobertaConfig, EncoderDecoderModel, RobertaTokenizer), #Causal model II
    #                 #'xlnet-base-cased': (XLNetConfig, XLNetLMHeadModel, XLNetTokenizer),
    #                 'openai-gpt': (OpenAIGPTConfig, OpenAIGPTLMHeadModel, OpenAIGPTTokenizer),
    #                 'gpt2': (GPT2Config, GPT2LMHeadModel, GPT2Tokenizer),
    #                 'gpt2-medium': (GPT2Config, GPT2LMHeadModel, GPT2Tokenizer),
    #                 'gpt2-large': (GPT2Config, GPT2LMHeadModel, GPT2Tokenizer),
    #                 #'gpt2-xl': (GPT2Config, GPT2Model, GPT2Tokenizer),
    #                 'EleutherAI/gpt-neo-1.3B': (AutoConfig, AutoModelForCausalLM, AutoTokenizer),
    #                 'EleutherAI/gpt-neo-2.7B': (AutoConfig, AutoModelForCausalLM, AutoTokenizer),
    #                 'EleutherAI/gpt-j-6B': (AutoConfig, AutoModelForCausalLM, AutoTokenizer),
    #                 # 'EleutherAI/gpt-neox-20b': (AutoConfig, AutoModelForCausalLM, AutoTokenizer),
    #                 # 't5-base': (T5Config, T5ForConditionalGeneration, T5Tokenizer),
    #                 # 't5-small': (T5Config, T5ForConditionalGeneration, T5Tokenizer ),
    #                 # 't5-large': (T5Config, T5ForConditionalGeneration, T5Tokenizer ),
    #                 }
    
    #For Encoder-Decoder Modeling
    MODEL_CLASSES = {
                    # 'facebook/bart-large-cnn': (BartConfig, BartForConditionalGeneration, BartTokenizer),
                    'bert-base-uncased': (BertConfig, BertForLatentConnector, BertTokenizer), #Causal model I
                    # 'roberta-base': (RobertaConfig, EncoderDecoderModel, RobertaTokenizer), #Causal model II
                    #'xlnet-base-cased': (XLNetConfig, XLNetLMHeadModel, XLNetTokenizer),
                    'openai-gpt': (OpenAIGPTConfig, OpenAIGPTLMHeadModel, OpenAIGPTTokenizer),
                    'gpt2': (GPT2Config, GPT2ForLatentConnector, GPT2Tokenizer),
                    'gpt2-medium': (GPT2Config, GPT2ForLatentConnector, GPT2Tokenizer),
                    'gpt2-large': (GPT2Config, GPT2ForLatentConnector, GPT2Tokenizer),
                    #'gpt2-xl': (GPT2Config, GPT2Model, GPT2Tokenizer),
                    # 'EleutherAI/gpt-neo-1.3B': (AutoConfig, AutoModelForCausalLM, AutoTokenizer),
                    # 'EleutherAI/gpt-neo-2.7B': (AutoConfig, AutoModelForCausalLM, AutoTokenizer),
                    # 'EleutherAI/gpt-j-6B': (AutoConfig, AutoModelForCausalLM, AutoTokenizer),
                    # 'EleutherAI/gpt-neox-20b': (AutoConfig, AutoModelForCausalLM, AutoTokenizer),
                    # 't5-base': (T5Config, T5ForConditionalGeneration, T5Tokenizer),
                    # 't5-small': (T5Config, T5ForConditionalGeneration, T5Tokenizer ),
                    # 't5-large': (T5Config, T5ForConditionalGeneration, T5Tokenizer ),
                    }
    
    parser = argparse.ArgumentParser()

    #--Required parameters if not encoder_decoder_seperate
    # parser.add_argument("--model_type",
    #                     default = None,
    #                     type = str,
    #                     required = True,
    #                     help = "Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()))
    # parser.add_argument("--model_name_or_path",
    #                     default = None,
    #                     type = str,
    #                     required = True,
    #                     help = "Path to pre-trained model or shortcut name selected in the list")
    #--If using BERT/RoBERTa Encoder with GPT2/3 Decoder
    parser.add_argument("--model_type_enc",
                        default = None,
                        type = str,
                        required = True,
                        help = "Encoder model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()))
    parser.add_argument("--model_name_or_path_enc",
                        default = None,
                        type = str,
                        required = True,
                        help = "Path to pre-trained encoder model or shortcut name selected in the list")
    parser.add_argument("--model_type_dec",
                        default = None,
                        type = str,
                        required = True,
                        help = "Decoder model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()))
    parser.add_argument("--model_name_or_path_dec",
                        default = None,
                        type = str,
                        required = True,
                        help = "Path to pre-trained decoder model or shortcut name selected in the list")
    parser.add_argument("--output_dir",
                        default = None,
                        type = str,
                        required = True,
                        help = "The output directory where the model results and checkpoints will be written.")
    parser.add_argument("--eval_dir",
                        default = None,
                        type = str,
                        required = True,
                        help = "The output directory where the evaluation metrics and losses are stored.")
    
    #--Variational auto-encoder BERT-GPT2 args
    parser.add_argument("--latent_size", 
                        default = 32, 
                        type = int, 
                        help = "Latent space dimension.")
    parser.add_argument("--use_deterministic_connect", 
                        action = 'store_true',
                        help = "Use deterministic inference to generate latent codes, i.e., standard auto-encoders.")
    parser.add_argument("--latent_as_gpt_memory", 
                        default = 1, 
                        type = int, 
                        help = "Latent vector as memery for GPT2 to attend.")
    parser.add_argument("--latent_as_gpt_emb", 
                        default = 1, 
                        type = int, 
                        help = "Latent vector as embeddings for GPT2.")
    #--Objective functions
    parser.add_argument("--mlm", 
                        action = 'store_true',
                        help = "Train with masked-language modeling loss instead of language modeling.")
    parser.add_argument("--mlm_probability", 
                        type = float, 
                        default = 0.15,
                        help = "Ratio of tokens to mask for masked language modeling loss")
    #--Other parameters
    parser.add_argument("--config_name",
                        default = "",
                        type = str,
                        help = "Pretrained config name or path if not the same as model_name")
    parser.add_argument("--config_name_enc",
                        default = "",
                        type = str,
                        help = "Pretrained encoder config name or path if not the same as model_name")
    parser.add_argument("--config_name_dec",
                        default = "",
                        type = str,
                        help = "Pretrained decoder config name or path if not the same as model_name")
    parser.add_argument("--tokenizer_name",
                        default = "",
                        type = str,
                        help = "Pretrained tokenizer name or path if not the same as model_name")
    parser.add_argument("--tokenizer_name_enc",
                        default = "",
                        type = str,
                        help = "Pretrained encoder tokenizer name or path if not the same as model_name")
    parser.add_argument("--tokenizer_name_dec",
                        default = "",
                        type = str,
                        help = "Pretrained decoder tokenizer name or path if not the same as model_name")
    parser.add_argument("--year",
                        default = 2019,
                        type = int,
                        help = "Year reference for failure analysis dataset")
    parser.add_argument("--max_seq_length",
                        default = 512,
                        type = int,
                        help = "The maximum total input sequence length after tokenization. Sequences longer "
                               "than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--block_size", default = -1, type = int,
                        help = "Optional input sequence length after tokenization."
                                 "The training dataset will be truncated in block of this size for training."
                                 "Default to the model max input length for single sentence inputs (take into account special tokens).")
    #---
    parser.add_argument("--seed",
                        type = int,
                        default = 42,
                        help = "random seed for initialization")
    parser.add_argument("--bos_token",
                        type = str,
                        default = '<|startoftext|>',
                        help = "Beginning of sentence token")
    parser.add_argument("--eos_token",
                        type = str,
                        default = '<|endoftext|>',
                        help = "End of sentence token")
    parser.add_argument("--pad_token",
                        type = str,
                        default = '<|pad|>',
                        help = "padding token")
    parser.add_argument("--do_train",
                        action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval",
                        action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--use_weights",
                        action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--evaluate_during_training",
                        action = 'store_true',
                        help = "Rule evaluation during training at each logging step.")
    parser.add_argument("--do_lower_case",
                        action = 'store_true',
                        help = "Set this flag if you are using an uncased model.")
    parser.add_argument("--encoder_decoder",
                        action = 'store_true',
                        help = "Set this flag if model is Encoder-Decoder type model like BERT and RoBerta.")
    parser.add_argument("--encoder_decoder_sep",
                        action = 'store_true',
                        help = "Set this flag if model is Encoder-Decoder type model using BERT/RoBERTa Encoder and GPT2 decoder")
    parser.add_argument("--use_variational_loss",
                        action = 'store_true',
                        help = "Use variational loss instead of simply CrossEntropy loss")
    parser.add_argument("--vae_model_name",
                        type = str,
                        default = 'vae',
                        help = "Indicate name of variational name e.x VAE, ControlVAE, InfoVAE, GCVAE")
    parser.add_argument("--mmd_type",
                        type = str,
                        default = 'mah',
                        help = "Type of distance metric to use. Applie to InfoVAE and GCVAE")
    parser.add_argument("--beta",
                        type = float,
                        default = 1.0,
                        help = "Parameter for training beta-VAE only")
    parser.add_argument("--gamma",
                        type = float,
                        default = 500.0,
                        help = "Parameter for training InfoVAE (MMD-VAE) only")
    parser.add_argument("--init_kld",
                        type = float,
                        default = 10.0,
                        help = "Initial KL-divergence loss when using PID-controller only")
    parser.add_argument("--init_bce",
                        type = float,
                        default = 10.0,
                        help = "Initial Binary-Cross Entropy loss when using PID-controller only")
    parser.add_argument("--init_mmd",
                        type = float,
                        default = 0.1,
                        help = "Initial Maximum Mean Discrepancy when using PID-controller only")
    parser.add_argument("--latent_dim",
                        type = int,
                        default = 100,
                        help = "Dimension of the latent space used for computating variational loss")
    parser.add_argument("--return_token_type_ids",
                        action = 'store_true',
                        help = "Return return_token_type_ids...useful for some models.")
    parser.add_argument("--ratio_increase", 
                        default = 0.25, 
                        type = float,
                        help = "Learning schedule, the percentage for the annealing stage.") 
    parser.add_argument("--ratio_zero", 
                        default = 0.25, 
                        type = float,
                        help = "Learning schedule, the percentage for the pure auto-encoding stage.")
    parser.add_argument("--temperature", 
                        default = 1.9, 
                        type = float,
                        help = "Temprature parameter reduces hallucination .")
    parser.add_argument("--top_k", 
                        default = 10,
                        type = int,
                        help = "Selectt Top-k (usually integer) generated next tokens")
    parser.add_argument("--top_p", 
                        default = 0.95, 
                        type = float,
                        help = "Similar to Top-k paprameter. Selects keywords above predefined probability threshold p.")
    parser.add_argument("--fb_mode", 
                        default = 0, 
                        type = int,
                        help = "free bit training mode.")   
    parser.add_argument("--dim_target_kl", 
                        default = 3.0, 
                        type = float,
                        help = "dim_target_kl free bit training mode.")       
    parser.add_argument("--per_gpu_train_batch_size",
                        default = 1,
                        type = int,
                        help = "Batch size per GPU/CPU for training.")
    parser.add_argument("--per_gpu_eval_batch_size",
                        default = 1,  
                        type = int,
                        help = "Batch size per GPU/CPU for evaluation.")
    parser.add_argument("--gradient_accumulation_steps",
                        type = int,
                        default = 1,
                        help = "Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--save_total_limit",
                        type = int,
                        default = 0,
                        help = "Limit the total amount of checkpoints, delete the older checkpoints in the output_dir, does not delete by default")
    parser.add_argument("--learning_rate", 
                        default = 5e-5,
                        type = float,
                        help  ="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay",
                        default = 0.0,
                        type = float,
                        help = "Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon",
                        default = 1e-8,
                        type = float,
                        help = "Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm",
                        default = 1.0,
                        type = float,
                        help = "Max gradient norm.")
    parser.add_argument("--num_train_epochs",
                        default = 3.0,
                        type = float,
                        help = "Total number of training epochs to perform.")
    parser.add_argument("--max_steps",
                        default = -1,
                        type = int,
                        help = "If > 0: set total number of training steps to perform. Override num_train_epochs.")  
    parser.add_argument("--warmup_steps",
                        default = 10,
                        type = int,
                        help = "Linear warmup over warmup_steps.")
    parser.add_argument("--logging_steps",
                        type = int,
                        default = 500,
                        help="Log every N-updates steps.")
    parser.add_argument("--save_steps",
                        type = int,
                        default = 500,
                        help = "Save checkpoint every N-updates steps.")
    parser.add_argument("--use_philly", 
                        action = 'store_true',
                        help = "Use Philly for computing.")
    parser.add_argument("--length_weighted_loss", 
                        action = 'store_true',
                        help = "Normalizing reconstruction loss.")
    parser.add_argument("--num_interpolation_steps", 
                        type = int,
                        default = 10,
                        help = "Number of interpolation steps to perform.")
    parser.add_argument("--use_pretrained_vae", 
                        action = 'store_true',
                        help = "Use use_pretrained_vae as initialization, where beta value is specified in the folder")
    parser.add_argument("--use_random_weight", action='store_true',
                        help="Use random weights as initialization")
    parser.add_argument("--eval_all_checkpoints", #checck this to know if it is worth evaluating all checkpoints and the significance
                        action = 'store_true',
                        help = "Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number")
    parser.add_argument("--delete_model", #checck this to know if it is worth evaluating all checkpoints and the significance
                        action = 'store_true',
                        help = "Delete all model from memory.")
    parser.add_argument("--freeze_params", #checck this to know if it is worth evaluating all checkpoints and the significance
                        action = 'store_true',
                        help = "Freeze all weights so the pretrained model. Avoids updating the weights during gradient/backprop computation.")
    parser.add_argument("--overwrite_output_dir",
                        action = 'store_true',
                        help = "Overwrite the content of the output directory")
    parser.add_argument("--overwrite_cache",
                        action = 'store_true',
                        help = "Overwrite the cached training and evaluation sets")
    parser.add_argument("--fp16",
                        action = 'store_true',
                        help = "Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
    parser.add_argument("--fp16_opt_level",
                        type = str,
                        default = "O1",
                        help = "For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                                "See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument("--local_rank",
                        type = int,
                        default = -1,
                        help = "For distributed training: local_rank is 0 and -1 for unit gpu")
    args = parser.parse_args()
    #-----------------set root directories
    if not args.encoder_decoder_sep:
        if not args.use_variational_loss:
            if args.use_weights:
                absolute_dir = f'plm/use_weight/{args.year}'
            else:
                absolute_dir = f'plm/finetuning_freeze/{args.year}'
        else:
            if not args.mmd_type:
                absolute_dir = f'plm/vfinetuning/{args.vae_model_name}/{args.year}'
            else:
                absolute_dir = f'plm/vfinetuning/{args.vae_model_name}/{args.mmd_type}/{args.year}'
        #----
        args.output_dir = join(join(absolute_dir, args.model_name_or_path.split('-')[0]), args.output_dir)
        args.eval_dir = join(join(absolute_dir, args.model_name_or_path.split('-')[0]), args.eval_dir)
    else:
        absolute_dir = f'plm/{args.vae_model_name}/{args.mmd_type}/{args.year}'
        #----
        args.model_name_or_path = f"{args.model_type_enc.split('-')[0]}{args.model_type_dec.split('-')[0]}"
        args.output_dir = join(join(absolute_dir, f"{args.model_type_enc.split('-')[0]}{args.model_type_dec.split('-')[0]}"), args.output_dir)
        args.eval_dir = join(join(absolute_dir, f"{args.model_type_enc.split('-')[0]}{args.model_type_dec.split('-')[0]}"), args.eval_dir)
    #--------------------------------- main
    if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train and not args.overwrite_output_dir:
        raise ValueError("Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(args.output_dir))
        
    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        args.n_gpu = 1
    args.device = device

    # Setup logging
    logging.basicConfig(
                        format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt = '%m/%d/%Y %H:%M:%S',
                        level = logging.INFO if args.local_rank in [-1, 0] else logging.WARN,
                        )
    logger.warning("   Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s", args.local_rank, args.device, args.n_gpu, bool(args.local_rank != -1), args.fp16)

    # Set seed
    set_seed(args.seed)

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab
    # encoder_decoder_mod = ['bert-base-uncased', 'roberta-large']
    #args.model_type = args.model_type.lower()
    #---- check if we are using EncoderDecoder Model or not
    if args.do_train and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        if args.encoder_decoder:
            logger.info('   Loading Tied Encoder-Decoder Model')
            config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
            tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name if args.tokenizer_name else args.model_name_or_path, do_lower_case = args.do_lower_case,
                                                        bos_token = args.bos_token, eos_token = args.eos_token, pad_token = args.pad_token ) #Tokenization
            config_enc = config_class.from_pretrained(args.config_name if args.config_name else args.model_name_or_path)
            config_dec = config_class.from_pretrained(args.config_name if args.config_name else args.model_name_or_path)
            model = model_class.from_encoder_decoder_pretrained(args.model_name_or_path, args.model_name_or_path, 
                                                                encoder_config = config_enc, decoder_config = config_dec, 
                                                                tie_encoder_decoder = True)
            model.config.decoder_start_token_id = tokenizer.cls_token_id
            model.config.pad_token_id = tokenizer.pad_token_id
            model.config.eos_token_id = tokenizer.eos_token_id
            model.config.bos_token_id = tokenizer.bos_token_id
            model.config.pad_token = tokenizer.pad_token
            model.config.decoder_start_token_id = 0
            model.config.no_repeat_ngram_size = 3
            model.config.length_penalty = 2.0
            model.config.vocab_size = model.config.decoder.vocab_size
            embedding_size = model.get_input_embeddings().weight.shape[0]
        elif args.encoder_decoder_sep:
            logger.info('   Loading Seperate Encoder-Decoder Model')
            #---------Encoder
            config_class_enc, model_class_enc, tokenizer_class_enc = MODEL_CLASSES[args.model_type_enc]
            #encoder tokenization...
            tokenizer_enc = tokenizer_class_enc.from_pretrained(args.tokenizer_name_enc if args.tokenizer_name_enc else args.model_name_or_path_enc, 
                                                                do_lower_case = args.do_lower_case, 
                                                                bos_token = args.bos_token, 
                                                                eos_token = args.eos_token, 
                                                                pad_token = args.pad_token) #Tokenization
            config_enc = config_class_enc.from_pretrained(args.config_name_enc if args.config_name_enc else args.model_name_or_path_enc)
            model_enc = model_class_enc.from_pretrained(args.model_name_or_path_enc, 
                                                        from_tf = bool('.ckpt' in args.model_name_or_path_enc), 
                                                        config = config_enc, 
                                                        latent_size = args.latent_size) #Encoder LM class
            model_enc.to(args.device) #transfers parameters of encoder to "cuda"
            if args.block_size <= 0:
                args.block_size = tokenizer_enc.max_len_single_sentence  # Our input block size will be the max possible for the model
            args.block_size = min(args.block_size, tokenizer_enc.max_len_single_sentence)
            #---------Decoder
            config_class_dec, model_class_dec, tokenizer_class_dec = MODEL_CLASSES[args.model_type_dec]
            #decoder tokenization...
            tokenizer_dec = tokenizer_class_dec.from_pretrained(args.tokenizer_name_dec if args.tokenizer_name_dec else args.model_name_or_path_dec, 
                                                                do_lower_case = args.do_lower_case, 
                                                                bos_token = args.bos_token, eos_token = args.eos_token, 
                                                                pad_token = args.pad_token) #Tokenization
            config_dec = config_class_dec.from_pretrained(args.config_name_dec if args.config_name_dec else args.model_name_or_path_dec)
            if args.block_size <= 0:
                args.block_size = tokenizer_dec.max_len_single_sentence  # Our input block size will be the max possible for the model
            args.block_size = min(args.block_size, tokenizer_enc.max_len_single_sentence)
            if args.latent_as_gpt_emb + args.latent_as_gpt_memory == 0:
                return # latent vector should pass into GPT to decode 
            else: 
                latent_as_gpt_emb = True if args.latent_as_gpt_emb == 1 else False
                latent_as_gpt_memory = True if args.latent_as_gpt_memory == 1 else False
            setattr(config_dec, "latent_size", args.latent_size)
            model_dec = model_class_dec.from_pretrained(args.model_name_or_path_dec, 
                                                        from_tf = bool('.ckpt' in args.model_name_or_path_dec), 
                                                        config = config_dec, 
                                                        latent_size = args.latent_size, 
                                                        latent_as_gpt_emb = latent_as_gpt_emb, 
                                                        latent_as_gpt_memory = latent_as_gpt_memory) #Decoder LM class
            model_dec.to(args.device) #transfers parameters of decoder to "cuda" (gpu)
        else:
            logger.info('   Loading Encoder-only or Decoder-only Model')
            config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
            tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name if args.tokenizer_name else args.model_name_or_path, do_lower_case = args.do_lower_case,
                                                        bos_token = args.bos_token, eos_token = args.eos_token, pad_token = args.pad_token ) #Tokenization
            config = config_class.from_pretrained(args.config_name if args.config_name else args.model_name_or_path)
            model = model_class.from_pretrained(args.model_name_or_path, from_tf = bool('.ckpt' in args.model_name_or_path), config = config) #LM class
            embedding_size = model.get_input_embeddings().weight.shape[0]
    #----
    if args.do_train and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        if not args.encoder_decoder_sep:
            if len(tokenizer) > embedding_size:
                if not args.encoder_decoder:
                    model.resize_token_embeddings(len(tokenizer))
                else:
                    model.encoder.resize_token_embeddings(len(tokenizer))
                    model.decoder.resize_token_embeddings(len(tokenizer))
        else:
            special_tokens_dict = {'pad_token': '<PAD>', 'bos_token': '<BOS>', 'eos_token': '<EOS>'}
            num_added_toks = tokenizer_dec.add_special_tokens(special_tokens_dict)
            logger.info(f'   We have added {num_added_toks} tokens to GPT2')
            model_dec.resize_token_embeddings(len(tokenizer_dec))  # Notice: resize_token_embeddings expect to receive the full size of the new vocabulary, i.e. the length of the tokenizer.
            assert tokenizer_dec.pad_token == '<PAD>'
            #---- logging ...
            # embedding_size = model_dec.get_input_embeddings().weight.shape[0]
            # logging.info(f'  Embedding size: {embedding_size}')
            model = VAEGCVAE(model_enc, model_dec, tokenizer_enc, tokenizer_dec, args) #Variational AutoEncoder (VAE) model loading
            logger.info('   Finnished loading Variational model')
            #---- use random initialization weights
            if args.use_random_weight:
                model.apply(weights_init_rondom)
            #---- 
            model.to(args.device)
    
    logger.info(f"  Training/evaluation parameters: \n{args}")
    if args.do_train and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        if args.freeze_params:
            logger.info(f"  Freezing weights of {args.model_type}")
            for param in model.parameters():
                param.requires_grad = False
            for n, p in model.named_parameters():
                if param.requires_grad:
                    logger.info(f'{n}, {p.data}')
                    
    #--Evaluating Model performance
    if args.do_eval and args.local_rank in [-1, 0]:
        tokenizer_enc, model_enc, tokenizer_dec, model_dec, model = load_checkpoint(args, MODEL_CLASSES)
        logger.info(f'   Loaded {args.vae_model_name} w/ {args.mmd_type} kernel')
        
    # Dataset
    class FailureAnalysisDataset(Dataset):
        def __init__(self, args, txt_list, tokenizer, max_length, wts = None, use_weights = False):
            self.input_ids = []
            self.attn_masks = []
            self.wts = wts       #weights GCVAE+GMM...Fixing failure analysis yearly imbalance in dataset
            self.use_weights = use_weights     #probabilistic weights from GCVAE + GMM
            self.token_type_ids = []
            for txt in txt_list:
                encodings_dict = tokenizer(args.bos_token + txt + args.eos_token, truncation = True,
                                            max_length = max_length, padding = "max_length", 
                                            return_token_type_ids = args.return_token_type_ids)
                self.input_ids.append(torch.tensor(encodings_dict['input_ids']))
                self.attn_masks.append(torch.tensor(encodings_dict['attention_mask']))
                self.token_type_ids.append(torch.tensor(encodings_dict['token_type_ids']))
        def __len__(self):
            return len(self.input_ids)
    
        def __getitem__(self, idx):
            if not self.use_weights:
                return self.input_ids[idx], self.attn_masks[idx], self.token_type_ids[idx]
            else:
                return self.input_ids[idx], self.attn_masks[idx], self.wts[idx], self.token_type_ids[idx]
    
    class TextDataset_2Tokenizers(Dataset):
        def __init__(self, args, txt_list, tokenizers, text_split_mode = 'natural', block_size = 512):
            self.txt_list = txt_list
            self.examples = []
            self.tokenizers = tokenizers #contains tuple of (encoder, decoder)
            #--- GPTx tokenizers
            self.pad_token = self.tokenizers[1].convert_tokens_to_ids([self.tokenizers[1].pad_token])[0]
            self.bos_token = self.tokenizers[1].convert_tokens_to_ids([self.tokenizers[1].bos_token])[0]
            self.eos_token = self.tokenizers[1].convert_tokens_to_ids([self.tokenizers[1].eos_token])[0]
    
            if text_split_mode == 'block':
                self._read_corpus_block_split(block_size = block_size)
            elif text_split_mode == 'natural': 
                self._read_corpus_natural_split(max_length = block_size, 
                                                block_size = block_size, 
                                                args = args)
            else:
                return
    
        def __len__(self):
            return len(self.examples)
    
        def __getitem__(self, idx):
            # pdb.set_trace()
            # Convert to Tensors and build dataset
            tokenized_text0 = torch.tensor(self.examples[idx][0], dtype = torch.long)
            tokenized_text1 = torch.tensor(self.examples[idx][2], dtype = torch.long)
            tokenized_text_lengths = torch.tensor([self.examples[idx][1], self.examples[idx][3]], dtype = torch.long)
            # pdb.set_trace()
            return (tokenized_text0, tokenized_text1, tokenized_text_lengths)
    
        def _read_corpus_natural_split(self, max_length, block_size, args):
    
            for line in self.txt_list:
                # pdb.set_trace()
                #---encoding text --> WPE --> BERT/RoBERTa/BART/... encoder
                tokenized_text0 = self.tokenizers[0].convert_tokens_to_ids(self.tokenizers[0].tokenize(line))
                tokenized_text0 = self.tokenizers[0].add_special_tokens_single_sentence(tokenized_text0)
                tokenized_text0_length = len(tokenized_text0) 
                pad_token = self.tokenizers[0].convert_tokens_to_ids([self.tokenizers[0].pad_token])[0]
                if block_size > tokenized_text0_length:
                    tokenized_text0 = tokenized_text0 + ([pad_token] * (block_size - tokenized_text0_length)  ) # Pad up to the sequence length.

                assert len(tokenized_text0) == block_size
                #---decoding text --> BPE --> GPT2x/... decoder
                tokenized_text1 = self.tokenizers[1].convert_tokens_to_ids(self.tokenizers[1].tokenize(line))
                tokenized_text1 = self.tokenizers[1].add_special_tokens_single_sentence(tokenized_text1)
                tokenized_text1 = [self.bos_token] + tokenized_text1 + [self.eos_token]
                tokenized_text1_length = len(tokenized_text1)
                
                if block_size>tokenized_text1_length:
                    tokenized_text1 = tokenized_text1 + ([self.pad_token] *  (block_size - tokenized_text1_length) ) # Pad up to the sequence length.
                
                assert len(tokenized_text1) == block_size
                self.examples.append([tokenized_text0, tokenized_text0_length, tokenized_text1, tokenized_text1_length])
    
        def _read_corpus_block_split(self, block_size):
            # Chunyuan: divide the linguistic text into the same length, then different tokenization schemes are applied
            while len(self.txt_list) >= block_size:  # Truncate in block of block_size
                #---encoding text --> WPE --> BERT/... encoder
                tokenized_text0 = self.tokenizers[0].convert_tokens_to_ids(self.tokenizers[0].tokenize(self.txt_list[:block_size]))
                tokenized_text0 = self.tokenizers[0].add_special_tokens_single_sentence(tokenized_text0)
                tokenized_text0_length = len(tokenized_text0) 
                pad_token = self.tokenizers[0].convert_tokens_to_ids([self.tokenizers[0].pad_token])[0]
                tokenized_text0 = tokenized_text0 + ([pad_token] * (block_size - tokenized_text0_length)  ) # Pad up to the sequence length.
                assert len(tokenized_text0) == block_size
                #---decoding text --> BPE --> GPT2x/... decoder
                tokenized_text1 = self.tokenizers[1].convert_tokens_to_ids(self.tokenizers[1].tokenize(self.txt_list[:block_size]))
                tokenized_text1 = self.tokenizers[1].add_special_tokens_single_sentence(tokenized_text1)
                tokenized_text1_length = len(tokenized_text1)
                
                tokenized_text1 = [self.bos_token] + tokenized_text1 + [self.eos_token]
                tokenized_text1 = tokenized_text1 + ([pad_token] *  (block_size - tokenized_text1_length - 2) ) # Pad up to the sequence length.
                assert len(tokenized_text1) == block_size
    
                self.examples.append([tokenized_text0, tokenized_text0_length, tokenized_text1, tokenized_text1_length])
                self.txt_list = self.txt_list[block_size:]
            
    TRAIN_DATA_FILE = join(path, f'combine_corpus_{args.year}.csv')
    df_df = pd.read_csv(TRAIN_DATA_FILE, sep = ',')['text']
    
    #-- Probabilistic weights
    z_size = 2
    gcvaemodel = args.mmd_type
    pl = np.load(join(path, f'b/gcvae/fagcvaegmm/latent_{z_size}/100/{gcvaemodel}/gmm_proba.npy')) #local
    pl = np.max(pl, 1) #returns maximum along horizontal axis...
    if not args.encoder_decoder_sep:
        max_length = max([len(tokenizer.encode(fa)) for fa in df_df])
        if args.use_weights:
            dataset = FailureAnalysisDataset(args, df_df, tokenizer, max_length = max_length, wts = pl, use_weights = args.use_weights)
        else:
            dataset = FailureAnalysisDataset(args, df_df, tokenizer, max_length = max_length)
    else:
        dataset = TextDataset_2Tokenizers(args, df_df, [tokenizer_enc, tokenizer_dec], text_split_mode = 'natural', block_size = 512)
    train_size = int(0.7 * len(dataset)) #split data into 70/30 train-test proportion
    train_dataset, eval_dataset = random_split(dataset, [train_size, len(dataset) - train_size]) #validation size = len(dataset) - train_size
    
    # Training
    if args.do_train:
        if args.encoder_decoder_sep:
            #--add pretrained Encoder-Decoder to Variational form
            logger.info(f"   Loading VAE model w/ Encoder: {args.model_type_enc}, Decoder: {args.model_type_dec}")
            global_step, tr_loss = train(args, 
                                          train_dataset, 
                                          eval_dataset, 
                                          model, 
                                          tokenizer_enc = tokenizer_enc, 
                                          tokenizer_dec = tokenizer_dec) #train_dataloader ==> train-dataset
            logger.info(f" Global step = {global_step}, Average loss = {tr_loss}")
        else:
            global_step, tr_loss = train(args, 
                                         train_dataset, 
                                         eval_dataset, 
                                         model, 
                                         tokenizer = tokenizer)
            logger.info(f" Global step = {global_step}, Average loss = {tr_loss}")

    
    if args.do_train and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        if not args.encoder_decoder_sep:
            # Load a trained model and vocabulary that you have fine-tuned
            model = model_class.from_pretrained(args.output_dir)
            tokenizer = tokenizer_class.from_pretrained(args.output_dir)
            model.to(args.device)
        else:
            load_checkpoint(args, MODEL_CLASSES) # load model after training...
            logger.info(f'   Loaded {args.vae_model_name} w/ {args.mmd_type} kernel')
            
    # Evaluation - we can ask to evaluate all the checkpoints (sub-directories) in a directory
    if args.eval_all_checkpoints and args.local_rank in [-1, 0]:
        checkpoints = [args.output_dir]
        checkpoints = list(os.path.dirname(c) for c in sorted(glob.glob(args.output_dir + '/**/' + WEIGHTS_NAME, recursive=True)))
        logging.getLogger("pytorch_transformers.modeling_utils").setLevel(logging.WARN)  # Reduce logging
        logger.info("Evaluate the following checkpoints: %s", checkpoints)
        for checkpoint in checkpoints:
            global_step = checkpoint.split('-')[-1] if len(checkpoints) > 1 else ""
            model = model_class.from_pretrained(checkpoint)
            model.to(args.device)
            eval_loss, ppl = evaluate(args, model, tokenizer, prefix=global_step)
            logger.info(f"  Evaluation loss: {eval_loss} PPL: {ppl}")

    #---- Generation evaluation
    step_type = [x.lower() for x in list(np.load(join(path, 'Step_type.npy'), allow_pickle = True).ravel()[0].keys()) if x!= ' ']
    substep_technique = [x.lower() for x in list(np.load(join(path, 'Substep_technique.npy'), allow_pickle = True).ravel()[0].keys()) if x!= ' ']
    stp_sstech = [f'{i} {j}' for i in step_type for j in substep_technique] # possible combinations of Step_type - Substepstep_technique
    x_nns = ['REFERENCE',
              'Subject',
              'Site',
              'Requested activity',
              'Priority level',
              'High confidentiality',
              'Context',
              'Objectives / Work description',
              'Source of failure / request',
              'Source of failure (Detailled)',
              ]
    
    df_xn_lda = pd.read_csv(join(path, f'XN_LAMBDA_BF_PREF_{args.year}.csv'), sep = ',')
    df_xn_lda_xx = pd.read_csv(join(path, 'XN_STDATA_EN.csv'), sep = ',')
    df_xn_lda_xx['LASTTRANSITIONDATE'] = df_xn_lda_xx['LASTTRANSITIONDATE'].apply(pd.to_datetime)
    df_xn_lda_xx = df_xn_lda_xx.iloc[:df_xn_lda.shape[0], :]
    
    lambdas = [x for x in df_xn_lda.columns if 'lambda' in x] #for extracting triplets \lambda = {step type, substep technique; equipment}
    x_df = df_xn_lda_xx[x_nns].astype(str).apply(lambda x: Mergefeatures(x).concat(), axis = 1)
    x_df = x_df.apply(lambda x: prepretreat(x),)
    predictor_corpus = list(x_df) 
    #targets
    target = df_xn_lda[lambdas].astype(str).apply(lambda x: Mergefeatures(x).concat(), axis = 1)
    target = target.apply(lambda x: ' '.join(x))
    target = list(target)
    #---- convert all character to lower case. GPTx generates in lowercase
    target = [x.lower() for x in target]
    
    # Generative metric evaluation
    latent_code = []
    bluescore = []
    bluescore_3 = []
    meteor_score_s = []
    model_generated_fas = []
    lev_d_1, prec_lev_1, rec_lev_1, fs_lev_1 = [], [], [], []
    lev_d, prec_lev, rec_lev, fs_lev = [], [], [], []
    for ii, ij in zip(predictor_corpus, target):
        logger.info(f'Failure description: {ii}\n')
        logger.info(f'Expert FA: {ij}\n')
        start_prompt = ii
        target_path = ij
        if not args.encoder_decoder_sep:
            # start_prompt = 'data castelletto customer manufacturing limit axis analysis failure complaint failed thd abnormal'
            start_tokens = tokenizer(start_prompt, return_tensors = "pt").input_ids.to(device)
            sampled_generated_outputs_token = model.generate(start_tokens, do_sample = True, top_k = 10, max_length = max_length, top_p = 0.95, 
                                                             temperature = 1.9, num_return_sequences = 1, pad_token_id = tokenizer.eos_token_id
                                                             )
            generated_text_is = []
            for _, s_output in enumerate(sampled_generated_outputs_token):
                logger.info(f"AI generated FA: {tokenizer.decode(s_output[len(start_tokens[0]):], skip_special_tokens = True)}")
                generated_text_is.append(tokenizer.decode(s_output[len(start_tokens[0]):], skip_special_tokens = True))
            prediction = " ".join(generated_text_is).lower()
            model_generated_fas.append(prediction) #to be used for scoring ROUGE and BLEU
        else:
            args.sent_source = start_prompt
            args.sent_target = target_path
            latent_z, _ = latent_code_from_text(start_prompt, tokenizer_enc, model, args)
            prediction = latent_code_to_text(latent_z, tokenizer_dec, model, args)
            prediction = filter_fdr_frm_generated_text(prediction, stp_sstech) #filter out generated failure description
            logger.info(f"AI generated FA: {prediction}")
            model_generated_fas.append(prediction) #to be used for scoring ROUGE and BLEU
            #--Use interpolation instead of conditional generation -------------------------------------------------------------------- INTERPOLATION ------------------------
            # latent_z, _ = latent_code_from_text(start_prompt, tokenizer_enc, model, args)
            # prediction, latent_z = interpolate(model, tokenizer_enc, tokenizer_dec, args)
            # prediction = filter_fdr_frm_generated_text(prediction, stp_sstech) #filter out generated failure description
            # logger.info(f"AI generated FA: {prediction}")
            # model_generated_fas.append(prediction) #to be used for scoring ROUGE and BLEU
            latent_z = latent_z.cpu().numpy() if len(latent_z.shape) <= 2 else latent_z.cpu().squeeze(1).numpy()
            latent_code.append(latent_z)
        #--------------------
        #bleu score
        chencherry = SmoothingFunction()
        bluescore.append(sentence_bleu([ij.split(' ')], prediction.split(' '), weights = (1, 0, 0, 0), smoothing_function = chencherry.method2))
        bluescore_3.append(sentence_bleu([ij.split(' ')], prediction.split(' '), weights = (0.33, 0.33, 0.33, 0), smoothing_function = chencherry.method2))
        #meteor scores
        meteor_score_s.append(nltk.translate.meteor_score.meteor_score([ij.split(' ')], prediction.split(' ')))
        #lese score
        lese_1 = LESE(ij.split(' '), prediction.split(' '), 1, False)
        lese = LESE(ij.split(' '), prediction.split(' '), 3, False)
        #---------
        lev_d_1.append(lese_1.levenshstein_distance)
        prec_lev_1.append(lese_1.precision_)
        rec_lev_1.append(lese_1.recall_)
        fs_lev_1.append(lese_1.f_score_)
        #-------
        lev_d.append(lese.levenshstein_distance)
        prec_lev.append(lese.precision_)
        rec_lev.append(lese.recall_)
        fs_lev.append(lese.f_score_)
        logger.info('*'*50)
    less_scores_1 = {'lev_d': lev_d_1, 'prec_lev': prec_lev_1, 'rec_lev': rec_lev_1, 'fs_lev': fs_lev_1} #LESE-1 ..variable name is lese_scores_1 not *less*
    less_scores = {'lev_d': lev_d, 'prec_lev': prec_lev, 'rec_lev': rec_lev, 'fs_lev': fs_lev} #LESE-3
    logger.info(f"  Average blue-1 score: {np.mean(bluescore)}")
    np.save(os.path.join(args.eval_dir, f"pt_{args.model_name_or_path.split('-')[0]}_bleuscore_{len(x_n)}_{args.num_train_epochs}_{args.year}.npy"), bluescore)
    logger.info('  *********************************Done computing self-BELU score*********************************')
    logger.info(f"  Average blue-3 score: {np.mean(bluescore_3)}")
    np.save(os.path.join(args.eval_dir, f"pt_{args.model_name_or_path.split('-')[0]}_bleuscore3_{len(x_n)}_{args.num_train_epochs}_{args.year}.npy"), bluescore_3)
    logger.info('  *********************************Done computing self-BELU score**********************************************')
    np.save(os.path.join(args.eval_dir, f"pt_{args.model_name_or_path.split('-')[0]}_lese1_{len(x_n)}_{args.num_train_epochs}_{args.year}.npy"), less_scores_1)
    logger.info(f'  LESE Precision: {np.mean(prec_lev_1)}\nLESE Recall: {np.mean(rec_lev_1)}\nLESE F1-score: {np.mean(fs_lev_1)}')
    logger.info('   ****************************************Done computing LESE score**********************************************')
    np.save(os.path.join(args.eval_dir, f"pt_{args.model_name_or_path.split('-')[0]}_lese3_{len(x_n)}_{args.num_train_epochs}_{args.year}.npy"), less_scores)
    logger.info(f'  LESE Precision: {np.mean(prec_lev)}\nLESE Recall: {np.mean(rec_lev)}\nLESE F1-score: {np.mean(fs_lev)}')
    logger.info('   *************************************Done computing LESE score***********************************************')
    #------------------ Remove empty/null hypothesis before computing ROUGE and METEOR scores -------------------------
    hyps_and_refs = zip(model_generated_fas, target)
    hyps_and_refs = [x for x in hyps_and_refs if len(x[0]) > 0]
    model_generated_fas, target = zip(*hyps_and_refs)
    #--------------------------------------------------meteor -----------------------------------------------------------
    logger.info(f"  Average metoer score: {np.mean(meteor_score_s)}")
    np.save(os.path.join(args.eval_dir, f"pt_{args.model_name_or_path.split('-')[0]}_meteor_{len(x_n)}_{args.num_train_epochs}_{args.year}.npy"), meteor_score_s)
    logger.info('  **************************************Done computing METEOR score**************************************')
    #------------------------------------------------Rouge -------------------------------------------------------------
    rouge = Rouge()
    rouge_score = rouge.get_scores(model_generated_fas, target, avg = True)
    logger.info(f'  ROUGE SCORES: {rouge_score}')
    np.save(os.path.join(args.eval_dir, f"pt_{args.model_name_or_path.split('-')[0]}_rouge_{len(x_nns)}_{args.num_train_epochs}_{args.year}.npy"), rouge_score)
    logger.info('  ************************************Done computing ROUGE score*****************************************')
    #--- save complete evaluation results in seperate file
    output_eval_file = os.path.join(args.eval_dir, "eval_metric_results.txt")
    #---Saving latent code w/ GMM-clustering
    latent_code = np.array(latent_code)
    latent_code = torch.Tensor(latent_code).squeeze(1).numpy()
    np.save(os.path.join(args.eval_dir, "latent_code.npy"), latent_code)
    logger.info('  Saving latent code')
    c_label, bic = GMMClustering(latent_code, nc_trials = 30)
    np.save(os.path.join(args.eval_dir, "latent_label.npy"), c_label)
    np.save(os.path.join(args.eval_dir, "latent_bic.npy"), bic)
    logger.info('  Saving latent code GMM clustering labels')
    #--TSNE on Latent_code
    # Perform t-SNE w/ GMM-clustering
    tsne = TSNE(n_components = 2, random_state = 42)
    latent_tsne = tsne.fit_transform(latent_code)
    np.save(os.path.join(args.eval_dir, "latent_tsne.npy"), latent_tsne)
    logger.info('  Saving latent code projection of TSNE')
    c_label_tsne, bic_tsne = GMMClustering(latent_tsne, nc_trials = 30)
    np.save(os.path.join(args.eval_dir, "latent_label_tsne.npy"), c_label_tsne)
    np.save(os.path.join(args.eval_dir, "latent_bic_tsne.npy"), bic_tsne)
    logger.info('  Saving TSNE latent code GMM clustering labels')
    #-- 
    with open(output_eval_file, "w+") as writer:
        logger.info("   ***** Storing complete evaluation results *****")
        writer.write("   ***** Complete eval results *****\n")
        writer.write(f"Average blue-1 score: {np.mean(bluescore)}\n")
        writer.write(f"Average blue-3 score: {np.mean(bluescore_3)}\n")
        writer.write(f'LESE-1 Precision: {np.mean(prec_lev_1)}\nLESE-1 Recall: {np.mean(rec_lev_1)}\nLESE-1 F1-score: {np.mean(fs_lev_1)}\nLevenshstein distance-1: {np.mean(lev_d_1)//1}\n')
        writer.write(f'LESE-3 Precision: {np.mean(prec_lev)}\nLESE-3 Recall: {np.mean(rec_lev)}\nLESE-3 F1-score: {np.mean(fs_lev)}\nLevenshstein distance-3: {np.mean(lev_d)//3}\n')
        for i, j in rouge_score.items():
            writer.write(f"{i}: Prec: {j['p']} Rec: {j['r']} F1: {j['f']}\n")
        writer.write(f"Average metoer score: {np.mean(meteor_score_s)}")
    writer.close()
    logger.info("   ***** Evaluation completed! *****")
    #if space is a problem --> wipe model & .jsons to save memory space
    #wipe model and trained parameters from memory
    if args.delete_model:
        os.system(f"rm -r {args.output_dir}")
    
    
if __name__ == "__main__":
    main()
    
    
    