
from amrlib.evaluate.smatch_enhanced import compute_scores, get_entries, compute_smatch
from amrlib.evaluate.bleu_scorer import BLEUScorer
from nltk.tokenize import word_tokenize
import pandas as pd
import seaborn.objects as so
import matplotlib.pyplot as plt
import seaborn as sns


def evaluate_text_bleu(ref, hyp):
    bleu_scorer = BLEUScorer()
    bleu_score, ref_len, hyp_len = bleu_scorer.compute_bleu(ref, hyp)
    print('BLEU score: %5.2f' % (bleu_score*100.))
    return bleu_score


def smatch_score(test, gold):
    test_entries = get_entries(test)
    gold_entries = get_entries(gold)
    print(test_entries)
    print(gold_entries)
    return compute_smatch(test_entries, gold_entries)
    

def evaluate_text(preds, gold, evaluation_function= "bleu"):
    if evaluation_function == "bleu":
        evaluation_function = evaluate_text_bleu
    return evaluation_function(gold, preds)    
    
def evaluate_AMRs(preds, gold, gold_save_path=None, pred_save_path=None, evaluation_function= None):
    if evaluation_function is None:
        evaluation_function = compute_scores
    elif evaluation_function == "smatch":
        evaluation_function = smatch_score
    preds_file_path = pred_save_path+".txt" if pred_save_path else "amr_pred_file.txt"
    gold_file_path = gold_save_path+".txt" if gold_save_path else "amr_gold_file.txt"
    # Open the file in write mode, creating it if it doesn't exist
    with open(preds_file_path, 'w') as file:
        # Write a string to the file
        for amr in preds:
            file.write(amr)
            file.write('\n')
            
    with open(gold_file_path, 'w') as file:
        # Write a string to the file
        for amr in gold:
            file.write(amr)
            file.write('\n') 
            file.write('\n')        
            
    return evaluation_function(preds_file_path, gold_file_path)

def create_eval_dataframe(preds, gold, sentences):
    df = pd.DataFrame()
    df['pred'] = preds
    df['gold'] = gold
    df['sentences'] = sentences
    df['n_tokens'] = df['sentences'].apply(num_of_tokens)
    return df

def eval_bottom_up(preds, gold, sentences, evaluation_mode = "AMR", evaluation_function=None):
    if evaluation_mode == "AMR":
        evaluation_mode = evaluate_AMR
    elif evaluation_mode == "text":
        evaluation_mode = evaluate_text
    if evaluation_function is None:
        evaluation_function = compute_scores
    eval_df = create_eval_dataframe(preds=preds,gold=gold, sentences=sentences)
    max_n_tokens = eval_df['n_tokens'].max()
    min_n_tokens = eval_df['n_tokens'].min()
    previous_len=-1
    results = pd.DataFrame()
    for current_max in range(min_n_tokens, max_n_tokens+1):
        row = {}
        df_tmp = eval_df[eval_df['n_tokens'] <= current_max]
        if (len(df_tmp)==previous_len):
            continue
        previous_len = len(df_tmp)
        print("max number of tokens: " + str(current_max), "number of entries:",len(df_tmp))
        row["n_tokens"] = current_max
        row["n_entries"] = len(df_tmp)
        row["bleu"]=evaluation_mode(
            preds = df_tmp['pred'], 
            gold = df_tmp['gold'],
            evaluation_function = evaluation_function
        )
        results = pd.concat([results, pd.DataFrame([row])], ignore_index=True)
        print(" ")
    return results
def eval_at_each_level(preds, gold, sentences, evaluation_mode = "AMR", evaluation_function=None):
    if evaluation_mode == "AMR":
        evaluation_mode = evaluate_AMR
    elif evaluation_mode == "text":
        evaluation_mode = evaluate_text
    if evaluation_function is None:
        evaluation_function = compute_scores
    eval_df = create_eval_dataframe(preds=preds,gold=gold, sentences=sentences)
    max_n_tokens = eval_df['n_tokens'].max()
    min_n_tokens = eval_df['n_tokens'].min()
    results = pd.DataFrame()
    for current in range(max_n_tokens, min_n_tokens-1,-1):
        row = {}
        df_tmp = eval_df[eval_df['n_tokens'] == current]
        if (len(df_tmp)==0):
            continue
        print("number of tokens: " + str(current), "number of entries:",len(df_tmp))
        row["n_tokens"] = current
        row["n_entries"] = len(df_tmp)
        row["metric"]=evaluation_mode(
            preds = df_tmp['pred'], 
            gold = df_tmp['gold'],
            evaluation_function = evaluation_function
        )
        results = pd.concat([results, pd.DataFrame([row])], ignore_index=True)
        print(" ")   
    return results
        
def eval_top_down(preds, gold, sentences, evaluation_mode = "AMR", evaluation_function=None):
    if evaluation_mode == "AMR":
        evaluation_mode = evaluate_AMR
    elif evaluation_mode == "text":
        evaluation_mode = evaluate_text   
    if evaluation_function is None:
        evaluation_function = compute_scores
    eval_df = create_eval_dataframe(preds=preds,gold=gold, sentences=sentences)
    max_n_tokens = eval_df['n_tokens'].max()
    min_n_tokens = eval_df['n_tokens'].min()
    previous_len = -1
    results = pd.DataFrame()
    for current_min in range(max_n_tokens, min_n_tokens-1,-1):
        row = {}
        df_tmp = eval_df[eval_df['n_tokens'] >= current_min]
        if (len(df_tmp)==previous_len):
            continue
        previous_len = len(df_tmp)
        print("min number of tokens: " + str(current_min), "number of entries:",len(df_tmp))
        row["n_tokens"] = current_min
        row["n_entries"] = len(df_tmp)
        row["BLEU"]=evaluation_mode(
            preds = df_tmp['pred'], 
            gold = df_tmp['gold'],
            evaluation_function = evaluation_function
        )
        results = pd.concat([results, pd.DataFrame([row])], ignore_index=True)
        print(" ")   
    return results      
        
def num_of_tokens(sent):
    return len(word_tokenize(sent))


def plot_smatch(results, title= ""):
    fig,ax = plt.subplots(2,1)
    plt.figure(figsize=(100,120))
    sns.scatterplot(results,x='n_tokens',y='metric', ax = ax[0]).set(title=title) 
    sns.scatterplot(results,x='n_tokens',y='n_entries', ax = ax[1])
 
def plot_buckets(results, title= ""):
    bins = [0,20,40,60,80, results['n_tokens'].max()]
    results['n_tokens_binned'] = pd.cut(results["n_tokens"], bins)  
    fig,ax = plt.subplots(2,1)
    plt.figure(figsize=(100,120))
    sns.boxenplot(
    results, x="n_tokens_binned", y="BLEU",
    color="b",  width_method="linear", ax = ax[0]
    ).set(title=title) 
    sns.barplot(results, x="n_tokens_binned", y="n_entries", ax = ax[1])