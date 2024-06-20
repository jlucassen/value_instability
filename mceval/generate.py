import datasets as HFDatasets

import torch

import pandas as pd

import argparse

import os

from transformers import AutoTokenizer, AutoModelForCausalLM

from mceval.utils import LabelledDataset, GenerationPipeline, PromptFormatter, LlamaGenerationPipeline

def generate_prompts(prompt_formatter, base_prompts, out_path, decoded_colname, generation_pipeline, debug=False):


    # Add reasoning prompt
    base_and_reasoning_prompt = pd.DataFrame()
    # base_and_one_sentence_reasoning_prompt = pd.DataFrame()
    # base_and_ten_tokens_reasoning_prompt = pd.DataFrame()
    base_and_reasoning_prompt['base_and_reasoning_prompt'] = base_prompts['base_prompt'] + prompt_formatter.reasoning_prompt
    # base_and_one_sentence_reasoning_prompt['base_and_one_sentence_reasoning_prompt'] = base_prompts['base_prompt'] + prompt_formatter.one_sentence_reasoning_prompt
    # base_and_ten_tokens_reasoning_prompt['base_and_ten_tokens_reasoning_prompt'] = base_prompts['base_prompt'] + prompt_formatter.ten_tokens_reasoning_prompt

    # if debug:
    #     print(base_prompts['base_prompt'][0])
    # Convert back to Hugging Face Dataset
    df_base_and_reasoning_prompt = HFDatasets.Dataset.from_pandas(base_and_reasoning_prompt)
    # df_base_and_one_sentence_reasoning_prompt = HFDatasets.Dataset.from_pandas(base_and_one_sentence_reasoning_prompt)
    # df_base_and_ten_tokens_reasoning_prompt = HFDatasets.Dataset.from_pandas(base_and_ten_tokens_reasoning_prompt)

    # Tokenize
    if debug:
        print("Tokenizing prompts")
    tokenized_base_and_reasoning_prompt = generation_pipeline.tokenize_dataset(df_base_and_reasoning_prompt, 'base_and_reasoning_prompt')
    # tokenized_base_and_one_sentence_reasoning_prompt = generation_pipeline.tokenize_dataset(df_base_and_one_sentence_reasoning_prompt, 'base_and_one_sentence_reasoning_prompt')
    # tokenized_base_and_ten_tokens_reasoning_prompt = generation_pipeline.tokenize_dataset(df_base_and_ten_tokens_reasoning_prompt, 'base_and_ten_tokens_reasoning_prompt')
    # dataset.map(
    #     lambda row: generation_pipeline.tokenize_function(row, 'base_prompt'), 
    #     batched=False, 
    #     remove_columns=['base_prompt'],
    #     # num_proc=num_cpus,
    #     )
    # tokenized_base_and_reasoning_prompt.set_format(type='torch', columns=['input_ids', 'attention_mask'])

    if debug:
        print(tokenized_base_and_reasoning_prompt)
        print(tokenized_base_and_reasoning_prompt['input_ids'][0].shape)

    # if debug:
    #     input_ids = tokenized_prompts_with_reasoning['input_ids']
    #     # print type of input_ids~
    #     print(input_ids)
    #     print(len(input_ids))
    #     print(input_ids[0].shape)
        
    # Generate reasoning
    if debug:
        print("Generating reasoning")
    reasoning_tensors = tokenized_base_and_reasoning_prompt.map(
        lambda x: generation_pipeline.generate_reasoning(x, max_new_tokens=200), 
        batched=True, 
        batch_size=15,
        # num_proc=num_gpus,
        )
    # only keep column ['reasoning_output_tensors']

    # one_sentence_reasoning_tensors = tokenized_base_and_one_sentence_reasoning_prompt.map(
    #     generation_pipeline.generate_reasoning, 
    #     # batched=True, 
    #     # num_proc=num_gpus,
    #     )

    # ten_tokens_reasoning_tensors = tokenized_base_and_ten_tokens_reasoning_prompt.map(
    #     lambda example: generation_pipeline.generate_reasoning(example, max_new_tokens=10), 
    #     # batched=True, 
    #     # num_proc=num_gpus,
    #     )


    # Decode reasoning tensors
    if debug:
        # print(reasoning_tensors['reasoning_output_tensors'])
        print("Decoding reasoning tensors")

    
    reasoning_decoded = reasoning_tensors.map(
        lambda row: generation_pipeline.decode_generations(row, 'reasoning_output_tensors', decoded_colname),
        batched=True, 
        # num_proc=num_cpus
        )

    # one_sentence_reasoning_decoded = one_sentence_reasoning_tensors.map(
    #     lambda row: generation_pipeline.decode_generations(row, 'reasoning_output_tensors', decoded_colname),
    #     batched=True, 
    #     # num_proc=num_cpus
    #     )

    # ten_tokens_reasoning_decoded = ten_tokens_reasoning_tensors.map(
    #     lambda row: generation_pipeline.decode_generations(row, 'reasoning_output_tensors', decoded_colname),
    #     batched=True, 
    #     # num_proc=num_cpus
    #     )

    # Convert reasoning to dataframe
    df_reasoning = reasoning_decoded.to_pandas()
    # df_one_sentence_reasoning = one_sentence_reasoning_decoded.to_pandas()
    # df_ten_tokens_reasoning = ten_tokens_reasoning_decoded.to_pandas()
    
    # df_reasoning.to_csv(out_path, index=False)
    # save to jsonl
    df_reasoning.to_json(out_path, orient='records', lines=True)
    return df_reasoning

def get_generation_pipeline(model_identifier):
    if model_identifier == "llama2":
        generation_pipeline = LlamaGenerationPipeline(
            chat=False,
            # device='cpu',
        )
    elif model_identifier == "llama2-chat":
        generation_pipeline = LlamaGenerationPipeline(
            chat=True,
            # device='cpu',
        )
    elif model_identifier == "llama3":
        generation_pipeline = LlamaGenerationPipeline(
            model_series=3,
            chat=False,
            # device='cpu',
        )
    elif model_identifier == "llama3-instruct":
        generation_pipeline = LlamaGenerationPipeline(
            model_series=3,
            chat=True,
            # device='cpu',
        )
    else:
        raise ValueError(f"Model {model_identifier} not supported. Supported models: llama2, llama2-chat")
    return generation_pipeline

def run_eval(labelled_dataset, generation_pipeline, model_identifier, leading_newline=False, debug=False, reasoning_only=False):
    prompt_variation = 'standard'
    if leading_newline:
        prompt_variation = 'newline'
    # out_path = f'./{model_identifier}/{dataset_identifier}_decoded_answers_{prompt_variation}.jsonl'
    scores_without_reasoning_path = f'./{model_identifier}/scores_without_reasoning_{labelled_dataset.dataset_nickname}_{prompt_variation}.jsonl'  
    scores_with_reasoning_path = f'./{model_identifier}/scores_with_reasoning_{labelled_dataset.dataset_nickname}_{prompt_variation}.jsonl'
    # # skip if  path already exists
    if os.path.exists(scores_without_reasoning_path) and os.path.exists(scores_with_reasoning_path):
        print(f"Skipping because {scores_without_reasoning_path} and {scores_with_reasoning_path} already exists")
        return
    # if os.path.exists(out_path):
    #     print(f"Skipping {dataset_identifier} with {model_identifier} because {out_path} already exists")
    #     return

    #     # Load model
    
    
    
    dataset_nickname = labelled_dataset.dataset_nickname
    dataset = labelled_dataset.dataset
    if debug:
        print("Loaded dataset")
        # print(dataset)
    # prompt_formatter = PromptFormatter(dataset_nickname, new_prompt_configs)
    prompt_formatter = PromptFormatter(
        dataset_nickname, 
        {
            'leading_newline': leading_newline,
            }
        )

    decoded_colname = 'decoded_reasoning'
    # Convert dataset to pandas
    df = dataset.to_pandas()
    # Add base prompts to the dataset
    base_prompts_path = f'./{model_identifier}/base_prompts_{dataset_nickname}_{prompt_variation}.jsonl'
    if os.path.exists(base_prompts_path):
        print(f"Loading base prompts from {base_prompts_path}")
        base_prompts = pd.read_json(base_prompts_path, lines=True, orient='records')
    else:
        print(f"Generating base prompts and saving to {base_prompts_path}")
        base_prompts = prompt_formatter.df_to_base_prompts(df)
        # base prompts to jsonl
        base_prompts.to_json(base_prompts_path, orient='records', lines=True)
    
    reasoning_path =  f'./{model_identifier}/reasoning_transcripts_{dataset_nickname}_{prompt_variation}.jsonl'
    # check if reasoning path exists
    if os.path.exists(reasoning_path):
        if reasoning_only:
            return
        print(f"Loading reasoning from {reasoning_path}")
        # load
        df_reasoning = pd.read_json(reasoning_path, lines=True, orient='records')
    else:
        print(f"Generating reasoning and saving to {reasoning_path}")
        df_reasoning = generate_prompts(prompt_formatter, base_prompts, reasoning_path, decoded_colname, generation_pipeline, debug=debug)
        if reasoning_only:
            return
        
        
    
    

    # Create df final_prompts
    prompts_with_reasoning = pd.DataFrame()
    prompts_with_reasoning['prompts_with_reasoning'] = df_reasoning[decoded_colname] + "\n" + prompt_formatter.answer_prompt

    prompts_without_reasoning = pd.DataFrame()
    prompts_without_reasoning['prompts_without_reasoning'] = base_prompts['base_prompt'] + prompt_formatter.answer_prompt

    # prompts_with_one_sentence_reasoning = pd.DataFrame()
    # prompts_with_one_sentence_reasoning['prompts_with_one_sentence_reasoning'] = df_one_sentence_reasoning[decoded_colname] + "\n" + prompt_formatter.answer_prompt

    # prompts_with_ten_tokens_reasoning = pd.DataFrame()
    # prompts_with_ten_tokens_reasoning['prompts_with_ten_tokens_reasoning'] = df_ten_tokens_reasoning[decoded_colname] + "\n" + prompt_formatter.answer_prompt

    # Tokenize final prompts
    if debug:
        # print(f"prompts_with_reasoning: {prompts_with_reasoning}")
        # print(f"prompts_without_reasoning: {prompts_without_reasoning}")
        # print(f"prompts_with_one_sentence_reasoning: {prompts_with_one_sentence_reasoning}")
        # print(f"prompts_with_ten_tokens_reasoning: {prompts_with_ten_tokens_reasoning}")

        print("Tokenizing final prompts")
        
    prompts_with_reasoning = HFDatasets.Dataset.from_pandas(prompts_with_reasoning)
    tokenized_prompts_with_reasoning = generation_pipeline.tokenize_dataset(prompts_with_reasoning, 'prompts_with_reasoning')
    # .map(
    #     lambda row: generation_pipeline.tokenize_function(row, 'prompts_with_reasoning'), 
    #     batched=False, 
    #     remove_columns=['prompts_with_reasoning'],
    #     # num_proc=num_cpus,
    #     )
    # tokenized_prompts_with_reasoning.set_format(type='torch', columns=['input_ids', 'attention_mask'])
    prompts_without_reasoning = HFDatasets.Dataset.from_pandas(prompts_without_reasoning)
    tokenized_prompts_without_reasoning =  generation_pipeline.tokenize_dataset(prompts_without_reasoning, 'prompts_without_reasoning')
    # HFDatasets.Dataset.from_pandas(prompts_without_reasoning).map(
    #     lambda row: generation_pipeline.tokenize_function(row, 'prompts_without_reasoning'), 
    #     batched=False, 
    #     remove_columns=['prompts_without_reasoning'],
    #     # num_proc=num_cpus,
    #     )
    # tokenized_prompts_without_reasoning.set_format(type='torch', columns=['input_ids', 'attention_mask'])

    # prompts_with_one_sentence_reasoning = HFDatasets.Dataset.from_pandas(prompts_with_one_sentence_reasoning)
    # tokenized_prompts_with_one_sentence_reasoning = generation_pipeline.tokenize_dataset(prompts_with_one_sentence_reasoning, 'prompts_with_one_sentence_reasoning')

    # prompts_with_ten_tokens_reasoning = HFDatasets.Dataset.from_pandas(prompts_with_ten_tokens_reasoning)
    # tokenized_prompts_with_ten_tokens_reasoning = generation_pipeline.tokenize_dataset(prompts_with_ten_tokens_reasoning, 'prompts_with_ten_tokens_reasoning')


    # Pass tokenized prompts into get_logits
    if debug:
        print("Getting logits")

    # dict_tokenized_prompts_without_reasoning = {feature: tokenized_prompts_without_reasoning[feature] for feature in tokenized_prompts_without_reasoning.features}
    # dict_tokenized_prompts_with_reasoning = {feature: tokenized_prompts_with_reasoning[feature] for feature in tokenized_prompts_with_reasoning.features}

    # if model_identifier in ["llama3", "llama3-instruct"]:
    #     top_k_func = generation_pipeline.get_top_k_scores
    # else:
    #     top_k_func = lambda row: generation_pipeline.get_top_k_scores(row, k=5, scores_is_tuple=False)

    # out_dict = {'top_k_scores': [], 'top_k_ids': []}
    scores_with_reasoning = tokenized_prompts_with_reasoning.map(
        lambda batch: generation_pipeline.get_top_k_scores(batch), #ERROR: bsz, q_len, _ = hidden_states.size() ValueError: not enough values to unpack (expected 3, got 2)
        # batched=True,
        # batch_size=58,
        remove_columns=['input_ids', 'attention_mask'],
    )

    # out_dict_2 = {'top_k_scores': [], 'top_k_ids': []}
    scores_without_reasoning = tokenized_prompts_without_reasoning.map(
        lambda batch: generation_pipeline.get_top_k_scores(batch),
        # batched=True,
        # batch_size=58,
        remove_columns=['input_ids', 'attention_mask'],
    )
    
    # check if scores_without_reasoning has same number of rows as tokenized_prompts_without_reasoning
    if len(scores_with_reasoning) == len(tokenized_prompts_with_reasoning):
        print("scores_without_reasoning has the same number of rows as tokenized_prompts_without_reasoning")
    else:
        # print("scores_without_reasoning does not have the same number of rows as tokenized_prompts_without_reasoning")
        print(len(scores_with_reasoning))
        print(len(tokenized_prompts_with_reasoning))
        raise ValueError("scores_without_reasoning does not have the same number of rows as tokenized_prompts_without_reasoning")
    
    
    if len(scores_without_reasoning) == len(tokenized_prompts_without_reasoning):
        print("scores_without_reasoning has the same number of rows as tokenized_prompts_without_reasoning")
    else:
        # print("scores_without_reasoning does not have the same number of rows as tokenized_prompts_without_reasoning")
        print(len(scores_without_reasoning))
        print(len(tokenized_prompts_without_reasoning))
        raise ValueError("scores_without_reasoning does not have the same number of rows as tokenized_prompts_without_reasoning")


    df_scores_without_reasoning = scores_without_reasoning.to_pandas()
    df_scores_with_reasoning = scores_with_reasoning.to_pandas()

    df_scores_without_reasoning.to_json(scores_without_reasoning_path, orient='records', lines=True)
    df_scores_with_reasoning.to_json(scores_with_reasoning_path, orient='records', lines=True)

    # final_df = pd.DataFrame()

    # for i in range(5):
    #     df[f'no_reasoning_top_id_{i}'] = df_scores_without_reasoning[f'top_k_ids_{i}']
    #     df[f'no_reasoning_top_score_{i}'] = df_scores_without_reasoning[f'top_k_scores_{i}']
    #     df[f'reasoning_top_id_{i}'] = df_scores_with_reasoning[f'top_k_ids_{i}']
    #     df[f'reasoning_top_score_{i}'] = df_scores_with_reasoning[f'top_k_scores_{i}']

    # dataset_decoded_answers.to_json(out_path, orient='records', lines=True)

    # # df back to dataset
    # dataset_decoded_answers = HFDatasets.Dataset.from_pandas(df)
    # # decode ids with map
    # dataset_decoded_answers = dataset_decoded_answers.map(
    #     lambda row: generation_pipeline.decode_top_k_ids(row, 'reasoning_top_id', 'reasoning_decoded_top', k=5),
    #     # batched=True,
    #     remove_columns=['reasoning_top_id_0', 'reasoning_top_id_1', 'reasoning_top_id_2', 'reasoning_top_id_3', 'reasoning_top_id_4'],
    # )
    # dataset_decoded_answers = dataset_decoded_answers.map(
    #     lambda row: generation_pipeline.decode_top_k_ids(row, 'no_reasoning_top_id', 'no_reasoning_decoded_top', k=5),
    #     # batched=True,
    #     remove_columns=['no_reasoning_top_id_0', 'no_reasoning_top_id_1', 'no_reasoning_top_id_2', 'no_reasoning_top_id_3', 'no_reasoning_top_id_4'],
    # )

    # dataset_decoded_answers.to_json(out_path, orient='records', lines=True)

def __main__():
    parser =  argparse.ArgumentParser(description='Run evaluation on a dataset with a model')    
    parser.add_argument('--dataset', type=str, nargs='+', help='The identifier for the dataset to evaluate.', default=['harmless', 'dilemmas'])
    parser.add_argument('--model', type=str, nargs='+', help='The identifier for the model to evaluate.', default=['llama2-chat', 'llama3-instruct'], ) 
    parser.add_argument('--prompt_var', type=str, nargs='+', default=['standard', 'leading_newline'],)
    parser.add_argument('--debug', action='store_true', help='Whether to print debug statements.', default=False)
    parser.add_argument('--reasoning_only', action='store_true', help='Whether to only generate reasoning.', default=False)
    args = parser.parse_args()

    for dataset_identifier in args.dataset:
        
        for model_identifier in args.model:
            print("Running eval on", dataset_identifier, "with", model_identifier)
            
            if 'standard' in args.prompt_var:
                generation_pipeline = get_generation_pipeline(model_identifier)
                labelled_dataset = LabelledDataset(dataset_identifier)
                print("Running standard prompt variation")
                run_eval(labelled_dataset, generation_pipeline, model_identifier, debug=args.debug, reasoning_only=args.reasoning_only)
                torch.cuda.empty_cache()
            if 'newline' or 'leading_newline' in args.prompt_var:
                generation_pipeline = get_generation_pipeline(model_identifier)
                labelled_dataset = LabelledDataset(dataset_identifier)
                print("Running leading_newline prompt variation")
                run_eval(labelled_dataset, generation_pipeline, model_identifier, leading_newline=True, debug=args.debug, reasoning_only=args.reasoning_only)
                torch.cuda.empty_cache()
                         

if __name__ == "__main__":
    __main__()