# -*- coding: utf-8 -*-
"""
Created on Wed Mar 30 12:26:06 2022

@author: samuel
"""

## importing the required general modules and packages
import numpy as np
import pandas as pd
import re
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import matplotlib.pyplot as plt
import seaborn as sns




class textFeaturePurning():
    
    def __init__(self):
        """
            the class will do the all sort of token analysis for text data. 
            with this we can come to know which classes share the same vector space and can have learning issues.
            
            1. total tokens for each class
            2. unique tokens for each class
            3. common tokens of each class, across all classes
            
            to see how this tool works use the test function which loads the 20-news dataset from sklearn.datasets
            
            sample example call:
                textFP = textFeaturePurning()
                textFP.testing()
                
            to run on custom dataset and ngram_ranges:
                textFP = textFeaturePurning()
                textFP.startAnalysis(dataset, text_column_name, label_column_name, ngram_range)
                
                
        Returns
        -------
        None.

        """
        pass
  
    
    def textPreprocess(self, text, word_len=3):
        ## text pre-processing

        try:
            return " ".join([token for token in re.findall(r'[a-zA-Z]+', text) if len(token) > word_len])
        except Exception as ecp:
            print(f"[ ! ] ERR : failed to do text-preprocessing. err_msg : {ecp}")
            return None
            

    def transformFeatures(self, documents, vectorizer_name='tfidf', ngram_range=(2,3)):
        ## text transformation
        print("[ * ] taking the transformation of the text | ", end=" ")
        try:
            if vectorizer_name == 'tfidf':
                self.vectorizer = TfidfVectorizer(ngram_range=ngram_range)
            elif vectorizer_name == 'count':
                self.vectorizer = CountVectorizer(ngram_range=ngram_range)
            else:
                print(f"[ ! ] {vectorizer_name} vectorizer not supported. only supports tfidf, count")
                return -1
            
            self.documents_vector = self.vectorizer.fit_transform(documents)
            print("vector : ", self.documents_vector.shape)
            return 0
        
        except Exception as ecp:
            print(f"[ ! ] ERR : failed to do vectorization. err_msg : {ecp}")
            return -1
            
    def supportTokens(self):
        ## token support says how much tokens have value > 0 for given category / class wrt to total tokens in the vecabulary
        try:
            print('[ * ] taking support tokens')
            total_tokens = self.documents_vector.shape[1]
            
            self.token_support = {}
            self.token_category = {}
            
            for category_id in self.dataset[self.label_column_name].unique():
                cat_index = self.dataset[self.dataset[self.label_column_name] == category_id].index
                token_weights = self.documents_vector[cat_index].sum(axis=0)
                _, cat_token_index = np.where(token_weights>0.0)
                
                self.token_support[category_id] = (cat_token_index.shape[0] / total_tokens)
                self.token_category[category_id] = cat_token_index
                
            self.token_support = dict(sorted(self.token_support.items(), key=lambda item: item[0], reverse=False))
            
            return 0
        except Exception as ecp:
            print(f"[ ! ] ERR : failed to calculate support tokens. err_msg : {ecp}")
            return -1
            
    def commonTokens(self):
        ## common tokens sharing across categories / classes
        
        try:
            print('[ * ] taking common token')
            self.category_keys = sorted(self.token_category.keys())
            self.token_confusion_matrix = np.zeros(shape=(len(self.category_keys), len(self.category_keys)))
            total_tokens = self.documents_vector.shape[1]
            
            for idx, source_cat_id in enumerate(self.category_keys):
                for jdx, compare_cat_id in enumerate(self.category_keys):
                    common_tokens = set(self.token_category[source_cat_id]).intersection(set(self.token_category[compare_cat_id]))
                    common_token_ratio = len(common_tokens) / total_tokens
                    self.token_confusion_matrix[idx][jdx] = common_token_ratio
                    
            ## making sure when we make the max query these 
            ## should not come as its similarity to one and the same data point
            np.fill_diagonal(self.token_confusion_matrix, 0.0) 
            
            return 0            
        except Exception as ecp:
            print(f"[ ! ] ERR : failed to calculate common tokens. err_msg : {ecp}")
            return -1
            
    def uniqueTokens(self):
        
        ## check unique tokens in each class wrt to other classes
        try:
            print('[ * ] taking unique tokens')
            total_tokens = self.documents_vector.shape[1]
            self.category_unique_tokens = {}
            self.unique_token_ratio = {}
            
            for category_id in self.category_keys:
                compare_classes = self.category_keys.copy()
                compare_classes.remove(category_id)
                
                source_cat_tokens = set(self.token_category[category_id])
                compare_tokens = []
                [compare_tokens.extend(self.token_category[cat_id].tolist()) for cat_id in compare_classes]
                compare_tokens = set(compare_tokens)
                
                unique_tokens = source_cat_tokens - compare_tokens
                self.unique_token_ratio[category_id] = len(unique_tokens) / total_tokens
                self.category_unique_tokens[category_id] = unique_tokens
                
                
            self.unique_token_ratio = dict(sorted(self.unique_token_ratio.items(), key=lambda item: item[0], reverse=False))
            
            return 0
        except Exception as ecp:
            print(f"[ ! ] ERR : failed to calculate unique tokens. err_msg : {ecp}")
            return -1
        
    def plotAnalysis(self, savefile=True):
        ## analysis of the plots of text features
        
        print('[ * ] plotting the analysis')
        fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(25,10), gridspec_kw={'width_ratios': [1, 1, 12]})

        sns.heatmap(pd.DataFrame(pd.Series(self.token_support), columns=["support_token"])*100, annot=True, cmap="YlGnBu", ax=axes[0])
        sns.heatmap(pd.DataFrame(pd.Series(self.unique_token_ratio), columns=["unique_token"])*100, annot=True, cmap="YlGnBu", ax=axes[1])
        sns.heatmap(np.round(self.token_confusion_matrix,3)*100, annot=True, cmap="YlGnBu", ax=axes[2])
        
        axes[0].set_ylabel('classes')
        axes[0].set_xlabel('percentage')
        axes[1].set_xlabel('percentage')
        axes[2].set_xlabel('classes')
        axes[2].set_title(f'common token-percentage across differnt classes | total tokens : {self.documents_vector.shape[1]}')
        
        if savefile == True:
            plt.savefig('text_feature_analysis.png')

        plt.show()
        
        
    def startAnalysis(self,  dataset, text_column_name, label_column_name, ngram_range):
        """
            this function will start all the execution one after the other

        Parameters
        ----------
        dataset : pandas dataframe
            dataset dataframe
        text_column_name : str
            text column
        label_column_name : str
            label / class column
        ngram_range : tuple
            number of ngrams range

        Returns
        -------
        None.

        """
        self.dataset = dataset
        self.text_column_name = text_column_name
        self.label_column_name = label_column_name
        
        documents = self.dataset[self.text_column_name].map(self.textPreprocess)
        self.transformFeatures(documents, ngram_range)
        self.supportTokens()
        self.commonTokens()
        self.uniqueTokens()
        self.plotAnalysis()


    
    def testing(self):
        
        """
            this funtion will load the new-20 dataset and plot the analysis
        """
        ## loading the datasets

        from sklearn.datasets import fetch_20newsgroups

        train_dataset = fetch_20newsgroups(subset='train')
        train_dataset = pd.DataFrame({'articles' : train_dataset.data, 'category' : train_dataset.target})
        
        #test_dataset = fetch_20newsgroups(subset='test')
        #test_dataset = pd.DataFrame({'articles' : test_dataset.data, 'category' : test_dataset.target})

        textFP = textFeaturePurning()        
        textFP.startAnalysis(train_dataset, 
                             text_column_name="articles", 
                             label_column_name="category",
                             ngram_range=(2,3))
        

    
    