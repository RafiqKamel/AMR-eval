from transition_amr_parser.parse import AMRParser
from general_parser.models._base_parser import AbstractParser
from general_parser.helpers.helper import vectorize_function
from typing import List
import numpy as np

class transition_AMR_parser(AbstractParser):
    def __init__(self, model_name: str):
        self.parser = AMRParser.from_pretrained(model_name)
    
    def parse_amr(self, text: str, plot: bool = True):
        tokens, _ = self.parser.tokenize(text)
        annotations, machines = self.parser.parse_sentence(tokens)
        amr = machines.get_amr()
        if plot:
            amr.plot()  
        return amr.to_penman(jamr=False, isi=False)     
    
    def parse_amr_batches(self, sentences):
        tokens_list = []
        amrs = []
        for sent in sentences:
             tokens, _ = self.parser.tokenize(sent)
             tokens_list.append(tokens)
        _ , machines = self.parser.parse_sentences(tokens_list, batch_size=256)
        for machine in machines:
            amrs.append(machine.get_amr().to_penman(jamr=False, isi=False)  )  
        return amrs
        
    
    def tokenize(self, text):
        tokens, _ = self.parser.tokenize(text)
        return tokens
    
    def return_amr_from_machine(machine):
        return machine.get_amr().to_penman(jamr=False, isi=False)     
        
    def parse_doc_amr(self, doc, plot: bool = True):
        
        tok_sentences = []
        for sen in doc:
            tokens, _ = self.parser.tokenize(sen)
            tok_sentences.append(tokens)
        annotations, machines = self.parser.parse_docs([tok_sentences])
        amr = machines[0].get_amr()
        if plot:
            amr.plot() 
        return(amr.to_penman(jamr=False, isi=True))
    