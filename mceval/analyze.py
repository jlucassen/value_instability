import pandas as pd
import argparse

def wwo_reasoning(df_decoded_answers):
    df_decoded_answers['reasoning_decoded_top_0'] = df_decoded_answers['reasoning_decoded_top_0'].apply(lambda x: x.strip())
    df_decoded_answers['no_reasoning_decoded_top_0'] = df_decoded_answers['no_reasoning_decoded_top_0'].apply(lambda x: x.strip())

    df_decoded_answers['reasoning_final_A'] = df_decoded_answers['reasoning_decoded_top_0'] == "A"
    df_decoded_answers['no_reasoning_final_A'] = df_decoded_answers['no_reasoning_decoded_top_0']== "A"
    df_decoded_answers['reasoning_final_B'] = df_decoded_answers['reasoning_decoded_top_0'] == 'B'
    df_decoded_answers['no_reasoning_final_B'] = df_decoded_answers['no_reasoning_decoded_top_0'] == 'B'
    # valid_answers = ['A', 'B']
    df_decoded_answers['reasoning_final_other'] = df_decoded_answers['reasoning_decoded_top_0'].apply(lambda x: False if x in ['A', 'B'] else True)
    df_decoded_answers['no_reasoning_final_other'] = df_decoded_answers['no_reasoning_decoded_top_0'].apply(lambda x: False if x in ['A', 'B'] else True)
    
    # print number of other answers
    print("Number of other answers in reasoning_decoded_top_0: ", df_decoded_answers['reasoning_final_other'].value_counts())

    # number of rows where reasoning_final_A changes from reasoning to no reasoning
    print("Number of rows where reasoning_final_A changes from reasoning to no reasoning: ", df_decoded_answers['reasoning_final_A'].ne(df_decoded_answers['no_reasoning_final_A']).value_counts())

    

# load dataset from json
dataset_name = 'harmless'
# df_leading_newline = pd.read_json('dilemmas_decoded_answers_newline.jsonl', orient='records', lines=True)
# df_standard_prompt = pd.read_json('dilemmas_decoded_answers.jsonl', orient='records', lines=True)
df_leading_newline = pd.read_json(f'{dataset_name}_decoded_answers_newline.jsonl', orient='records', lines=True)
df_standard_prompt = pd.read_json(f'{dataset_name}_decoded_answers.jsonl', orient='records', lines=True)

wwo_reasoning(df_leading_newline)
wwo_reasoning(df_standard_prompt)