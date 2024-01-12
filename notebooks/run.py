from general_parser.models import transition_AMR_parser
from general_parser.loaders import dataset_loader
from general_parser.evaluation.evaluation import eval_bottom_up, eval_top_down, eval_at_each_level, plot_smatch, plot_buckets

parser = transition_AMR_parser(model_name='AMR3-structbart-L')
loader = dataset_loader("/Users/rafiqmazen/GR/AMR/bio/amr/full/t.txt")
amr_pred = []
for sent in loader.sentences:
    amr_pred.append(parser.parse_amr(sent,plot=False))

r = eval_bottom_up(amr_pred,loader.amr_texts, loader.sentences, evaluation_function="smatch")
plot_smatch(r, "bottom up structBART")
plot_buckets(r,"bottom up structBART")
r = eval_top_down(amr_pred,loader.amr_texts, loader.sentences, evaluation_function="smatch")
plot_smatch(r, "top down structBART")
plot_buckets(r,"top down structBART")
r = eval_at_each_level(amr_pred,loader.amr_texts, loader.sentences, evaluation_function="smatch")
plot_smatch(r, "exact level structBART")
plot_buckets(r,"exact level structBART")
