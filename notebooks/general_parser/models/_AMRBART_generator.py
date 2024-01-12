from transformers import BartForConditionalGeneration
from transformers import AutoTokenizer
from transformers import pipeline
from transformers.pipelines.pt_utils import KeyDataset

class AMRBART_generator():
    
    def __init__(self):
        model = BartForConditionalGeneration.from_pretrained("xfbai/AMRBART-large-finetuned-AMR3.0-AMR2Text")
        tokenizer = AutoTokenizer.from_pretrained("xfbai/AMRBART-large-finetuned-AMR3.0-AMR2Text")
        self.pipeline = pipeline("text2text-generation", model=model, tokenizer=tokenizer)
    def generate_text(self, amrs):
        return self.pipeline(KeyDataset(amr, "text"))
            
        
