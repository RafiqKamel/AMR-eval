from amr_utils.amr_readers import AMR_Reader
from general_parser.helpers.helper import vectorize_function


class dataset_loader():
    def __init__(self,
                 dataset_path: str, 
                 sentence_reference: str = "snt"
                 ):
        self.reader = AMR_Reader()
        self.dataset_path = dataset_path
        self.amrs = self.return_amrs()
        self.sentences = self.return_sentences(self.amrs)
        self.text_amrs = self.return_amr_text(self.amrs)
        
    def return_amrs(self):
        return self.reader.load(self.dataset_path, remove_wiki=True)
    
    def return_sentences(self, amrs):
        return vectorize_function(function = extract_sentence, vector = amrs)
    
    def return_amr_text(self, amrs):
        return vectorize_function(function = extract_amr_text, vector = amrs)

def extract_sentence(amr, sentence_reference: str = "snt"):
    return amr.metadata[sentence_reference]            
            
def extract_amr_text(amr):
    return amr.graph_string()         

    

        
        