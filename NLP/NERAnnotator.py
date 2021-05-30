# -*- coding: utf-8 -*-
"""
Created on Sat May 29 20:38:20 2021

@author: samuel
"""

import re
import math
import random
import itertools
from tqdm import tqdm

class prepareTrainingData():
    
    ## list of dict of raw sentences
    def __init__(self, sentences):
        """
        Parameters
        ----------
        sentences : list
            list of dict as per below example
            
            sentences = [{
                            "example_sentence" : "Apple is looking at buying U.K. startup for $1 billion. Meanwhile Microsoft took over India's AI Startup.",
                            "sentence_structure" : "# is looking at buying @ startup for $. Meanwhile # took over @ AI Startup.",
                            "entitymap" : {"#" : {"entities" : ["Apple", "Microsoft"], "label" : "ORG"}, 
                                           "@" : {"entities" : ["U.K.", "India's"], "label" : "LOC"},
                                           "$" : {"entities" : ["$1 billion", "100 corer rupees"], "label" : "CURR"}}
                            }]

        Returns
        -------
        None.
        """
        print("[ * ] intialized NER Annotator")
        self.seteneces = sentences
        
            
    def transformSentences(self):
        """
        DESCRIPTION : this function will loop through all the input sentences given in class initialization.
        
        Returns
        -------
        final transformed list of sentences as per required format will be in 
        self.transformed_setences variable.

        """
        self.transformed_sentences = []
        self.stats = []
        sent_number = 0
        
        for raw_sentence in tqdm(self.seteneces, desc="annotating sentences"):
            ts = self.generateSentences(raw_sentence)
            self.transformed_sentences.extend(ts)
            self.stats.append({"sentence_id" : sent_number, "annotation_count" : len(ts)})
            sent_number += 1
        
        
    def makeEntitiesCombinations(self, entitymap):
        """
        Parameters
        ----------
        entitymap : dict
            takes input of raw_sentence["entitymap"] as per in above example
            
            entitymap = {"#" : {"entities" : ["Apple", "Microsoft"], "label" : "ORG"}, 
                         "@" : {"entities" : ["U.K.", "India's"], "label" : "LOC"},
                         "$" : {"entities" : ["$1 billion", "100 corer rupees"], "label" : "CURR"}

        Returns
        -------
        combinations_list : list
            aftet getting the combinations count, now this function will generate all list of combinations
            this list used as mapping and imputting sentence into various possible combinations.
            
            output looks like as below : 
            ------------------------------------------------------------------------
            combinations_list = [{'$': '$1 billion', '#': 'Apple', '@': "India's"},
                                 {'@': 'U.K.', '#': 'Microsoft', '$': '100 corer rupees'},
                                 {'#': 'Microsoft', '$': '$1 billion', '@': "India's"},
                                 {'@': 'U.K.', '#': 'Apple', '$': '100 corer rupees'},
                                 {'@': 'U.K.', '#': 'Microsoft', '$': '$1 billion'},
                                 {'#': 'Apple', '@': "India's", '$': '100 corer rupees'},
                                 {'@': 'U.K.', '#': 'Apple', '$': '$1 billion'},
                                 {'#': 'Microsoft', '@': "India's", '$': '100 corer rupees'}]

        """        
        
        #print("[ ! ] creating all entities combinations")
        combinations_list = []
        list_of_entities = []
        emkeys = []
        for emkey, emvalue in entitymap.items():
            list_of_entities.append(emvalue["entities"])
            emkeys.append(emkey)
            
        combo_list = list(itertools.product(*list_of_entities))
        for combos in combo_list:
            lc = {emkeys[i]: combos[i] for i in range(len(emkeys))}
            combinations_list.append(lc)
            
        return combinations_list
    
    
    def findspecialChars(self, text):
        """
        Parameters
        ----------
        text : str
            this function will check for special character in entity            
    

        Returns
        -------
        bool
            return True if input string have any special char as this is 
                required for regex pattern search for entities having special chars
                else False.

        """
        regex = re.compile('[@_!#$%^&*()<>?/\|}{~:]') 
        if(regex.search(text) == None):
            return False
        else: 
            return True
        
    def generateSentences(self, raw_sentence):
        """
        Parameters
        ----------
        raw_sentence : dict
            input takes as below example
            
            raw_sentence = {
                "example_sentence" : "Apple is looking at buying U.K. startup for $1 billion. Meanwhile Microsoft took over India's AI Startup.",
                "sentence_structure" : "# is looking at buying @ startup for $. Meanwhile # took over @ AI Startup.",
                "entitymap" : {"#" : {"entities" : ["Apple", "Microsoft"], "label" : "ORG"}, 
                               "@" : {"entities" : ["U.K.", "India's"], "label" : "LOC"},
                               "$" : {"entities" : ["$1 billion", "100 corer rupees"], "label" : "CURR"}}
                }

        Returns 
        -------
        formated_sentences : list
        return list of all possible combinations of given raw sentences with all entities

        """
        #print("[ ! ] generating annotated sentences")
        entites_combinations = self.makeEntitiesCombinations(raw_sentence["entitymap"])
        formated_sentences = []
        for em in entites_combinations: 
            sm = raw_sentence["sentence_structure"]
            used_entities = {}
            for et, el in em.items():
                sm = sm.replace(et, el)
                entity_label = raw_sentence["entitymap"][et]["label"]
                used_entities.update({el:entity_label})
                
            word_indexes = []
            for entity, lab in used_entities.items():
                if self.findspecialChars(entity):
                    sentity = f'\\'+entity
                else:
                    sentity = entity
                word_indexes += [(wi.start(), wi.end(), lab) for wi in re.finditer(sentity,sm)]
            formated_sentences.append((sm, {"entities":word_indexes}))
        return formated_sentences



## ======================== code execution example ============================== ##
"""
sentences = [{
    "example_sentence" : "Apple is looking at buying U.K. startup for $1 billion. Meanwhile Microsoft took over India's AI Startup.",
    "sentence_structure" : "# is looking at buying @ startup for $. Meanwhile # took over @ AI Startup.",
    "entitymap" : {"#" : {"entities" : ["Apple", "Microsoft"], "label" : "ORG"}, 
                   "@" : {"entities" : ["U.K.", "India's"], "label" : "LOC"},
                   "$" : {"entities" : ["$1 billion", "100 corer rupees"], "label" : "CURR"}}
    },{
    "example_sentence" : "i am data scientist and works in Google, California.",
    "sentence_structure" : "i am data scientist and works in #, @.",
    "entitymap" : {"#" : {"entities" : ["Apple", "Microsoft"], "label" : "ORG"}, 
                   "@" : {"entities" : ["U.K.", "India's", "California"], "label" : "LOC"}}
    }
    ]

annot = prepareTrainingData(sentences)
annot.transformSentences()

training_sentences = annot.transformed_sentences
# to get stats of each sentences combination count use as below
annot.stats
"""
        