# -*- coding: utf-8 -*-
"""
Created on Wed Jul 14 07:16:56 2021

@author: samuel
"""

import json
import re
import os
import time
from tqdm import tqdm
import random
from pathlib import Path



class Loaddatasetinfo():
    
    def __init__(self):
        self.raw_data_foldername = "raw_data"
    
    def listAvaliableEntities(self, entity_meta_info=False):
        
        for entity_name in os.listdir(self.raw_data_foldername):
            entity_name = entity_name.split(".")[0]
            print(" * entity : ", entity_name)
            if entity_meta_info == True:
                self.getEntityInfo(entity_name=entity_name)
            
    def getEntityInfo(self, entity_name, return_data=False):
        entity_filename = os.path.join(self.raw_data_foldername, entity_name+".json")
        with open(entity_filename, "r") as fp:
            fruits_data = json.load(fp)

        entities = []
        sentences = []
        for entry in fruits_data:
            entities.append(entry["object"])
            sentences.extend(entry["sentences"])
            
        print("\t >> entities : {} sentences : {}".format(len(entities), len(sentences)))
        
        if return_data == True:
            return entities, sentences
        
class PrepareDataset():
    
    def __init__(self, entity_name, entity_label):
        print("[ * ] ner dataset preparaing from raw data. entity : {} entity label : {}".format(entity_name, entity_label))
        self.entity_name = entity_name
        lsi = Loaddatasetinfo()
        self.entity_label = entity_label
        self.entities, self.sentences = lsi.getEntityInfo(entity_name=entity_name, return_data=True)
    
    
    def findnearset_tag_sentence(self, sentence_index):
        nearest_symantic_sentence = ""
        search_index = -1
        
        ## first half reverse-search
        for sent in self.sentences[0:sentence_index][::-1]:
            if "@@@" in sent:
                nearest_symantic_sentence = sent
                search_index = -1
                break
                
        ## second half forward-search
        for sent in self.sentences[sentence_index:]:
            if "@@@" in sent:
                nearest_symantic_sentence = sent
                search_index = 1
                break
                
        return nearest_symantic_sentence, search_index
    
    def getSentenceStatistics(self, training_dataset):
        SSNE = 0 # single sentence, no entity
        SSSE = 0 # single sentence, single entity
        SSME = 0 # single sentence, multiple entity
        MSNE = 0 # multiple sentence, no entity
        MSSE = 0 # multiple sentence, single entity
        MSME = 0 # multiple sentence, multiple entities
    
        ssne = []
        msne = []
        len_training = len(training_dataset)
        for row_data in training_dataset:
            sentence = row_data[0]
            entity = row_data[1]
    
            if len(sentence.split("."))>1:
                ## multi-setences
                if len(entity["entities"]) == 0:
                    MSNE += 1
                    msne.append(sentence)
                if len(entity["entities"]) == 1:
                    MSSE += 1
                if len(entity["entities"])>1:
                    MSME += 1
            else:
                ## single sentence
                if len(entity["entities"]) == 0:
                    SSNE += 1
                    ssne.append(sentence)
                if len(entity["entities"]) == 1:
                    SSSE += 1
                if len(entity["entities"])>1:
                    SSME += 1
    
        training_entities_distribution_count = {"MSNE":MSNE, "MSSE":MSSE, "MSME":MSME, 
                                                 "SSNE":SSNE, "SSSE":SSSE, "SSME":SSME}
        training_entities_distribution_percentage = {en:round((ec/len_training*100),2) for en, ec in training_entities_distribution_count.items()}
        
        sentence_statistics = {"entities_distribution_count" : training_entities_distribution_count,
                              "entities_distribution_percentage" : training_entities_distribution_percentage}
        return sentence_statistics
    
    
    def format_training_dataset(self, ner_sentences, entities_list, label):
        generated_sentences = []
        entity_sentences_statistics = []
        
        for entity in tqdm(entities_list, desc="processing entities"):
            entity_generated_sentences = []
            for raw_sentence in ner_sentences:
                sentence = raw_sentence.replace("@@@", entity).strip(".")
                
                entities = []
                entities.extend([(wi.start(), wi.end(), label) for wi in re.finditer(r"\b"+entity+"\\b", sentence)])
                
                ## adding as sentences are in original symmatic format
                capital_case_ner_sent = (sentence, {"entities" : entities}) 
                entity_generated_sentences.append(capital_case_ner_sent)
                
                ## adding sentences by lowering the cases
                lower_case_ner_sent = (sentence.lower(), {"entities" : entities}) 
                entity_generated_sentences.append(lower_case_ner_sent)
            
            generated_sentences.extend(entity_generated_sentences)    
            entity_sentences_statistics.append({"entity" : entity, "statistics" : self.getSentenceStatistics(entity_generated_sentences)})
            
        return generated_sentences, entity_sentences_statistics


    def transformdata(self, number_of_sentences=-1, auto_save_output=True):
        ner_sentences = []
        non_entity_count = 0
        entity_count = 0
        
        if number_of_sentences == -1:
            final_sentences = self.sentences
        else:
            final_sentences = random.sample(population=self.sentences, k=number_of_sentences)
    
        
        for i in tqdm(range(len(final_sentences)), desc="transforming sentences"):
            if "@@@" not in final_sentences[i]:
                non_entity_count += 1
                searched_sentence, searched_index = self.findnearset_tag_sentence(sentence_index=self.sentences.index(final_sentences[i]))
                
                if searched_index == -1:
                    ner_sentences.append(searched_sentence+" "+final_sentences[i])
                else:
                    ner_sentences.append(final_sentences[i]+" "+searched_sentence)
            else:
                entity_count += 1
                ner_sentences.append(final_sentences[i])
    
        self.ner_training_dataset, self.dataset_statistics = self.format_training_dataset(ner_sentences, self.entities, label=self.entity_label)
        
        if auto_save_output == True:
            output_filepath = os.path.join(".", self.entity_name+"_"+self.entity_label+".json")
            self.saveDataset(output_filepath)
        
    def saveDataset(self, output_filepath):
        print("saving the generated dataset : ", output_filepath)
        with open(output_filepath, "w") as fp:
            json.dump(self.ner_training_dataset, fp)


    
## ================= code testing =================== ##
## loading already present raw data
#  nerdatainfo = Loaddatasetinfo()

## getting basic count of senteneces and unique entities for each label (i.e fruit)
#  nerdatainfo.listAvaliableEntities(entity_meta_info=True)

## preparing dataset from above avaliable raw data entities (i.e fruit)
#  nerdata = PrepareDataset(entity_name="fruits", entity_label="FRUIT")
    
## generate and write data of 5000 sentences (default -1 flag will generate all posible sentences) and save to default filename.
#  nerdata.transformdata(number_of_sentences=5000)
