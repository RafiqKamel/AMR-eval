from general_parser.models._base_parser import AbstractParser
from spring_amr.utils import instantiate_model_and_tokenizer

class spring_parser(AbstractParser):
    def __init__(self, 
            model_name, 
            dropout: float = 0.,
            attention_dropout: float = 0.,
            penman_linearization: bool = True,
            use_pointer_tokens: bool = False,
            raw_graph: bool = False,  
             ):
        self.super()
        self.model, self.tokenizer = instantiate_model_and_tokenizer(
        model_name,
        dropout=0.,
        attention_dropout=0.,
        penman_linearization=penman_linearization,
        use_pointer_tokens=use_pointer_tokens,
        raw_graph=raw_graph,
        )   
    
    def tokenizer(self, input):
        pass
        