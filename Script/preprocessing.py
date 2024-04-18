# -*- coding: utf-8 -*-
"""
Created on Mon Dec 14 11:29:01 2020

@author: kenneth
"""


# Preprocessing pipeline
import re
import numpy as np
from os.path import join
import pandas as pd
from itertools import chain
from multiprocessing.dummy import Pool as ThreadPool
from features import foi, ic, num, obj, ct, triplets, cl_features, x_n
from nltk.stem.snowball import FrenchStemmer, PorterStemmer, ItalianStemmer



#--cluster directory
paths = {'data': '/export/home/ezukwoke/pourcluster/data', #hdf5 database
        'utils': '/export/home/ezukwoke/pourcluster/utils' #kenneth utils
        }


class Preprocess(object):
    '''Docstring
    Processing main class
    '''
    
    def __init__(self, stopwords:set, path:str = None):
        '''
        

        Parameters
        ----------
        stopwords : str
            stopwords.

        Returns
        -------
        None.

        '''
        self.stopwords = stopwords
        if not path:
            path = paths
            self.path = path
        else:
            self.path = path
        self.pstem = PorterStemmer()
        return
    
    #--first step
    def remove_hyp_uds(self, text:str):
        '''
        

        Parameters
        ----------
        text : str
            text for which we want to remove hyphens and underscore.

        Returns
        -------
        TYPE
            string without hyphen and underscore.

        '''
        text_ls = list(chain(*[x.split('-') for x in text.split('_')])) #remove and underscore
        text_ls = ' '.join(text_ls)
        text_ls =  text_ls.split('/') #check if there are any forward slashes
        return ' '.join(text_ls)
    
    #--second step
    def convert_to_lowercase(self, text:str):
        '''
        

        Parameters
        ----------
        text : str
            text.

        Returns
        -------
        str
            lowercase strings.

        '''
        return text.lower()
    

    #--third step
    def remove_special_characters(self, text:str):
        '''
        

        Parameters
        ----------
        text : str
            text.

        Returns
        -------
        str
            text without punctuations.

        '''
        return ''.join(re.sub('[^A-Za-z0-9]+', ' ', text))
    
    #--fourth step
    def map_abbreviations(self, text:str):
        '''
        

        Parameters
        ----------
        text : str
            text/context.

        Returns
        -------
        xd : str
            text containing mapped abbreviation meaning.

        '''
        abbr = pd.read_csv(join(paths['data'], 'abbr/Abbreviation_complete.csv'), sep = ',')
        kw = {k.lower():v.lower() for k,v in zip(abbr.Abbreviations.astype(str), abbr.Meaning.astype(str))}
        xs = self.convert_to_lowercase(text)
        xs = self.remove_special_characters(xs)
        xc = {k:v for k,v in zip(xs.split(' '), xs.split(' '))} #create a disctionary of words in senetence with same key and value
        xd = (pd.Series(xc)).map(kw) #map abbreviation to keys
        xd = np.where(xd.isna() == True, xd.index, xd.values) #replace value with keys if NaN else abbreviations
        xd = ' '.join(xd) #join words to form back words...
        return xd
    
    #--fifth step
    def tokenize(self, text:str, stopword:bool = None, alphanum:bool = None, threshold:int = None, useThresh:bool = None):
        '''Word tokenization
        
        After tokenization
        --------------------
        [1] Remove words that are less than length of 3
        [2] Remove words containing digits
        [3] Remove words not in stopword list
        [4] Remove alpha numerical words
        

        Parameters
        ----------
        text : str
            DESCRIPTION.
        stopword : bool, optional
            Request to use stopwrds or not. The default is None.
        alphanum : bool, optional
            Request to remove alha numerical data or not. The default is None.
        threshold : int, optional
            THreshold length if threshold is in use. The default is None.
        useThresh : bool, optional
            Request to use threshold or not. The default is None.

        Returns
        -------
        list
            list of tokenized words.

        '''
        self.sw = stopword
        if alphanum == None:
            alphanum = False
            self.alphanum = alphanum
        else:
            self.alphanum = alphanum
    
        if stopword == None:
            stopword = True
            self.sw = stopword
        else:
            self.sw = stopword
            
        if threshold == None:
            threshold = 2
            self.threshold = threshold
        else:
            self.threshold = threshold
            
        if useThresh == None:
            useThresh = True
            self.useThresh = useThresh
        else:
            self.useThresh = useThresh
        
        if self.useThresh:
            if not self.alphanum:
                if self.sw:
                    return [z.strip() for z in text.split(' ') if not z.isdigit() if not z in self.stopwords if not len(z) < self.threshold if not z == '' if not z == ' ']
                else:
                    return [z.strip() for z in text.split(' ') if not z.isdigit() if not len(z) < self.threshold if not z == '' if not z == ' ']
            else:
                if self.sw:
                    return [z.strip() for z in text.split(' ') if not z.isdigit() if not z in self.stopwords if z.isalpha() if not len(z) < self.threshold if not z == '' if not z == ' ']
                else:
                    return [z.strip() for z in text.split(' ') if not z.isdigit() if z.isalpha() if not len(z) < self.threshold if not z == '' if not z == ' ']
        else:
            if not self.alphanum:
                if self.sw:
                    return [z.strip() for z in text.split(' ') if not z.isdigit() if not z in self.stopwords if not z == '' if not z == ' ']
                else:
                    return [z.strip() for z in text.split(' ') if not z.isdigit() if not z == '' if not z == ' ']
            else:
                if self.sw:
                    return [z.strip() for z in text.split(' ') if not z.isdigit() if not z in self.stopwords if z.isalpha() if not z == '' if not z == ' ']
                else:
                    return [z.strip() for z in text.split(' ') if not z.isdigit() if z.isalpha() if not z == '' if not z == ' ']
                
    #--sixth step
    def stemmatize(self, text_ls:list):
        '''
        

        Parameters
        ----------
        text_ls : list
            list containing tokenized words.

        Returns
        -------
        list
            stemmatized tokens.

        '''
        return [self.pstem.stem(x) for x in text_ls]
    
    
    #----organising dates and arranging in pandas date format
    def dates(self, date_sm:str):
        '''
        

        Parameters
        ----------
        date_sm : str
            date strings sample.

        Returns
        -------
        pd.datetime
            Pandas datetime format

        '''
        return pd.to_datetime(date_sm)
    
    @staticmethod
    def prepre_context(text):
        '''
        

        Parameters
        ----------
        text : TYPE
            Context data to pre-preprocess.

        Returns
        -------
        TYPE
            pre-preprocess string.

        '''
        tt = [x.split(':')[-1].strip() for x in text.strip().replace('\t', '').replace('\r\n', ',').split(',') if not x == '' if not x == ' ']
        return ' '.join(tt)


    #preprocess one--> Remove stopwords by default
    def prep_one(self, text:str):
        '''
        

        Parameters
        ----------
        text : str
            DESCRIPTION.

        Returns
        -------
        e : str
            preprocessed text.

        '''
        a = self.remove_hyp_uds(text) #remove hyphen and underscore
        b = self.convert_to_lowercase(a)
        c = self.remove_special_characters(b)
        c = self.map_abbreviations(c)
        d = self.tokenize(c)
        e = self.stemmatize(d)
        return e
    
    #preprocess two--> preprocessing without removing stopwords
    def prep_two(self, text:str):
        '''
        

        Parameters
        ----------
        text : str
            DESCRIPTION.

        Returns
        -------
        e : str
            preprocessed text.

        '''
        a = self.remove_hyp_uds(text)
        b = self.convert_to_lowercase(a)
        c = self.remove_special_characters(b)
        c = self.map_abbreviations(c)
        d = self.tokenize(c, stopword = False)
        e = self.stemmatize(d)
        return e
    
    #preprocess three--> preprocessing without removing stopwords; remove alphanumerics
    def prep_three(self, text:str):
        '''
        

        Parameters
        ----------
        text : str
            DESCRIPTION.

        Returns
        -------
        e : str
            preprocessed text.

        '''
        a = self.remove_hyp_uds(text)
        b = self.convert_to_lowercase(a)
        c = self.remove_special_characters(b)
        c = self.map_abbreviations(c)
        d = self.tokenize(c, stopword = False, alphanum = True)
        e = self.stemmatize(d)
        return e
    
    #preprocess four--> preprocessing removing stopwords; without removing alphanumerics
    def prep_four(self, text:str):
        '''
        

        Parameters
        ----------
        text : str
            DESCRIPTION.

        Returns
        -------
        e : str
            preprocessed text.

        '''
        a = self.remove_hyp_uds(text)
        b = self.convert_to_lowercase(a)
        c = self.remove_special_characters(b)
        c = self.map_abbreviations(c)
        d = self.tokenize(c, stopword = True, alphanum = False)
        e = self.stemmatize(d)
        return e
    
    #preprocess five--> preprocessing without removing stopwords, alphanumerics or applying thresholds
    def prep_five(self, text:str):
        '''
        

        Parameters
        ----------
        text : str
            DESCRIPTION.

        Returns
        -------
        c : TYPE
            preprocessed text.

        '''
        a = self.convert_to_lowercase(text)
        a = self.map_abbreviations(a)
        b = self.tokenize(a, stopword = False, alphanum = False, useThresh = False)
        c = self.stemmatize(b)
        return c
    
    #preprocess five--> preprocessing witjout removing stopwords, alphanumerics or applying thresholds
    def prep_six(self, text:str):
        '''
        

        Parameters
        ----------
        text : str
            DESCRIPTION.

        Returns
        -------
        e : str
            preprocessed text.

        '''
        a = self.remove_hyp_uds(text)
        b = self.convert_to_lowercase(a)
        c = self.remove_special_characters(b)
        d = self.map_abbreviations(c)
        e = self.tokenize(d, stopword = True, alphanum = False, useThresh = True)
        e = self.stemmatize(e)
        return e
    
     #preprocess five--> preprocessing witjout removing stopwords, alphanumerics or applying thresholds
    def prep_seven(self, text:str, stem:bool = None):
        '''
        

        Parameters
        ----------
        text : str
            DESCRIPTION.

        Returns
        -------
        d : str
            preprocessed tokens without stemmatization.
        e : str
            preprocessed tokens with stemmatization.

        '''
        if stem == None:
            stem = True
            self.stem = stem
        else:
            self.stem = stem
        a = self.remove_hyp_uds(text)
        b = self.convert_to_lowercase(a)
        c = self.remove_special_characters(b)
        d = self.tokenize(c, stopword = False, alphanum = False, useThresh = True)
        if not self.stem:
            return d
        else:
            e = self.stemmatize(d)
            return e
        
    #preprocessign for dates...
    def prep_eight(self, text:str):
        '''
        

        Parameters
        ----------
        text : str
            date format in form of text.

        Returns
        -------
        TYPE
            Pandas datetime (pd.datetime) format.

        '''
        return self.dates(text)




#%% example how to use...

if __name__ == '__main__':
    #import stopwords
    with open(join(paths['utils'], 'stopwords.txt'), 'r+',encoding="utf8") as st:
        stopwords = set([x for x in st.read().split()]) #mix of Italia, English and French words...
    #--load data set
    x_n_df = pd.read_csv(join(paths['data'], 'x_n/x_n_en.csv'), sep = ';', low_memory = False)[x_n]
    #x_n_df_cp = x_n_df.copy(deep = True)
    #apply preprocessing...
    x_n_df['REFERENCE'] = x_n_df['REFERENCE'].astype(str).apply(lambda x: Preprocess(stopwords).prep_one(x))
    x_n_df['Subject'] = x_n_df['Subject'].astype(str).apply(lambda x: Preprocess(stopwords).prep_one(x))
    x_n_df['Context'] = x_n_df['Context'].astype(str).apply(lambda x: Preprocess(stopwords).prep_one(Preprocess.prepre_context(x)))
    x_n_df['Objectives / Work description'] = x_n_df['Objectives / Work description'].astype(str).apply(lambda x: Preprocess(stopwords).prep_one(x))
    x_n_df['Source of failure / request'] = x_n_df['Source of failure / request'].astype(str).apply(lambda x: Preprocess(stopwords).prep_two(x))
    x_n_df['Source of failure (Detailled)'] = x_n_df['Source of failure (Detailled)'].astype(str).apply(lambda x: Preprocess(stopwords).prep_two(x))
    x_n_df['Organization'] = x_n_df['Organization'].astype(str).apply(lambda x: Preprocess(stopwords).prep_six(x)) #1
    x_n_df['Organization / Division'] = x_n_df['Organization / Division'].astype(str).apply(lambda x: Preprocess(stopwords).prep_six(x)) #2
    x_n_df['Department'] = x_n_df['Department'].astype(str).apply(lambda x: Preprocess(stopwords).prep_one(x))
    x_n_df['Requestor'] = x_n_df['Requestor'].astype(str).apply(lambda x: Preprocess(stopwords).prep_one(x))
    x_n_df['Cost center'] = x_n_df['Cost center'].astype(str).apply(lambda x: Preprocess(stopwords).prep_four(x))
    #adjust here---> ignore use of abbreviation
    x_n_df['Confidentiality'] = x_n_df['Confidentiality'].astype(str).apply(lambda x: Preprocess(stopwords).prep_seven(x))
    #--
    x_n_df['Site'] = x_n_df['Site'].astype(str).apply(lambda x: Preprocess(stopwords).prep_three(x))
    x_n_df['Lab'] = x_n_df['Lab'].astype(str).apply(lambda x: Preprocess(stopwords).prep_six(x)) #3
    x_n_df['Lab team'] = x_n_df['Lab team'].astype(str).apply(lambda x: Preprocess(stopwords).prep_six(x)) #4
    x_n_df['Requested activity'] = x_n_df['Requested activity'].astype(str).apply(lambda x: Preprocess(stopwords).prep_five(x))
    #---
    x_n_df['Project'] = x_n_df['Project'].astype(str).apply(lambda x: Preprocess(stopwords).prep_seven(x))
    x_n_df['Priority level'] = x_n_df['Priority level'].astype(str).apply(lambda x: Preprocess(stopwords).prep_seven(x))
    #adjust here---> ignore use of abbreviation
    x_n_df['High confidentiality'] = x_n_df['High confidentiality'].astype(str).apply(lambda x: Preprocess(stopwords).prep_seven(x, stem = False))
    #--
    x_n_df['LASTTRANSITIONDATE'] = x_n_df['LASTTRANSITIONDATE'].astype(str).apply(lambda x: Preprocess(stopwords).prep_eight(x))
    x_n_df['Date of creation'] = x_n_df['Date of creation'].apply(lambda x: Preprocess(stopwords).prep_eight(x))
    x_n_df['Date of validation'] = x_n_df['Date of validation'].apply(lambda x: Preprocess(stopwords).prep_eight(x))
    x_n_df['Requestor expected date'] = x_n_df['Requestor expected date'].apply(lambda x: Preprocess(stopwords).prep_eight(x))
    x_n_df['Lab team forecast date'] = x_n_df['Lab team forecast date'].apply(lambda x: Preprocess(stopwords).prep_eight(x))
    x_n_df['Starting date of request'] = x_n_df['Starting date of request'].apply(lambda x: Preprocess(stopwords).prep_eight(x))
    #save preprocessed/tokenized data
    x_n_df.to_csv(join(paths['data'], "x_n/x_n_pp2.csv"), index = False, sep = ';')




















