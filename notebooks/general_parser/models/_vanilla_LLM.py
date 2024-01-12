from transition_amr_parser.parse import AMRParser
from general_parser.models._base_parser import AbstractParser
from transformers import BartTokenizer, BartForConditionalGeneration

class vanilla_parser(AbstractParser):
    def __init(self, model_name: str):
        self.tokenizer = BartTokenizer.from_pretrained(model_name)
        self.model = BartForConditionalGeneration.from_pretrained(model_name)
    
    def tokenize(self, input: str):
        input_ids = tokenizer.encode(amr_input, return_tensors="pt")
        return input_ids    
    def encode(self, tokens):
        output_ids = model.generate(input_ids)
        return output_ids
    def return_amr(self, output_ids):
        output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        return output_text   

