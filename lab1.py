"""
Lab 1 from course: https://www.coursera.org/learn/generative-ai-with-llms/

dependencies

From Hugging face
transformers==4.45.1
datasets==3.0.1
"""
from datasets import load_dataset, get_dataset_config_names, DatasetDict
from transformers import AutoModelForSeq2SeqLM, T5ForConditionalGeneration
from transformers import AutoTokenizer
from transformers import GenerationConfig

import textwrap

# too big to run on my local machine
# MODEL_NAME = 'google/flan-t5-base'
MODEL_NAME = 'google/flan-t5-small'
DATASET_NAME = 'knkarthick/dialogsum'
DASH_LINE = '-' * 100

def explore_dataset(dataset: DatasetDict):
    print(f'{DASH_LINE}\nEXPLORE DATASET\n{DASH_LINE}')
    print("let's check out this dataset...")
    print(f'all the splits with the number of columns in each {dataset.num_columns}')
    print(f'now the number of rows in each {dataset.num_rows}')
    print(f'or we could just check the shape of each {dataset.shape}')
    print(f'now the column names {dataset.column_names}')
    print(f'and the underlying data is stored here: {dataset.cache_files}')
    print(DASH_LINE + '\n')


def display_data(dataset: DatasetDict, indices: list):
    split = 'test'

    for i, data_index in enumerate(indices):
        dialogue = dataset[split][data_index]['dialogue']
        summary = dataset[split][data_index]['summary']

        print(f'{DASH_LINE}\nExample {i + 1}\n{DASH_LINE}')
        print(f'INPUT DIALOGUE: \n{dialogue}\n{DASH_LINE}')

        print(f'BASELINE HUMAN SUMMARY: \n{summary}\n{DASH_LINE}\n')


def summarize_without_prompting(model, tokenizer, dataset, indices: list):
    print(f'{DASH_LINE}\nSUMMARIZE WITHOUT PROMPTING\n{DASH_LINE}')
    split = 'test'

    for i, data_index in enumerate(indices):
        dialogue = dataset[split][data_index]['dialogue']
        summary = dataset[split][data_index]['summary']

        dialogue_encoded = tokenizer(dialogue, return_tensors='pt')
        model_summary = tokenizer.decode(
            model.generate(
                dialogue_encoded['input_ids'],
                max_new_tokens=50
            )[0],
            skip_special_tokens=True
        )

        print(f'{DASH_LINE}\nExample {i + 1}\n{DASH_LINE}')
        print(f'INPUT DIALOGUE: \n{dialogue}\n{DASH_LINE}')

        print(f'BASELINE HUMAN SUMMARY: \n{summary}\n{DASH_LINE}')
        print(f'MODEL SUMMARY: \n{model_summary}\n{DASH_LINE}\n')


def summarize_zero_shot(model, tokenizer, dataset, indices: list):
    print(f'{DASH_LINE}\nSUMMARIZE ZERO SHOT INFERENCE\n{DASH_LINE}')
    split = 'test'

    for i, data_index in enumerate(indices):
        dialogue = dataset[split][data_index]['dialogue']
        summary = dataset[split][data_index]['summary']

        # TODO - does formatting matter to an LLM? wondering if this will effect accuracy
        prompt = textwrap.dedent(f'''
            Dialogue:
                
            {dialogue}
            
            What was going on? 
        ''').strip()

        dialogue_encoded = tokenizer(prompt, return_tensors='pt')
        model_summary = tokenizer.decode(
            model.generate(
                dialogue_encoded['input_ids'],
                max_new_tokens=50
            )[0],
            skip_special_tokens=True
        )

        print(f'{DASH_LINE}\nExample {i + 1}\n{DASH_LINE}')
        print(f'INPUT DIALOGUE: \n{dialogue}\n{DASH_LINE}')

        print(f'BASELINE HUMAN SUMMARY: \n{summary}\n{DASH_LINE}')
        print(f'MODEL SUMMARY - ZERO SHOT: \n{model_summary}\n{DASH_LINE}\n')


def summarize_with_prompt(model, tokenizer, prompt):
    print(f'{DASH_LINE}\nSUMMARIZE WITH PROMPT\n{DASH_LINE}')

    dialogue_encoded = tokenizer(prompt, return_tensors='pt')
    model_summary = tokenizer.decode(
        model.generate(
            dialogue_encoded['input_ids'],
            max_new_tokens=50
        )[0],
        skip_special_tokens=True
    )

    '''
    print(f'{DASH_LINE}\nExample {i + 1}\n{DASH_LINE}')
    print(f'INPUT DIALOGUE: \n{dialogue}\n{DASH_LINE}')
    print(f'BASELINE HUMAN SUMMARY: \n{summary}\n{DASH_LINE}')
    '''

    print(f'PROMPT: \n{prompt}\n{DASH_LINE}\n')
    print(f'MODEL SUMMARY: \n{model_summary}\n{DASH_LINE}\n')


def _build_few_shot_prompt(dataset, example_indices: list, example_index_to_summarize: int):
    split = 'test'
    prompt = ''
    for i in example_indices:
        dialogue = dataset[split][i]['dialogue']
        summary = dataset[split][i]['summary']

        # Stop sequence '{summary}\n\n\n' is important for FLAN-T5 according to the deep learning ai course
        prompt += f'''
            Dialogue:
            
            {dialogue}
            
            What was going on?
            {summary}\n\n
        '''

    dialogue = dataset[split][example_index_to_summarize]['dialogue']
    prompt += f'''
        Dialogue:

        {dialogue}
    
        What was going on?
    '''
    return prompt


def main():
    test_indices = [40, 200]

    # Load data set into memory
    dataset = load_dataset(DATASET_NAME)

    # Look into the data set a bit
    explore_dataset(dataset)
    display_data(dataset, test_indices)

    # Load in a pre trained model
    print('creating model')
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
    print('created model')

    # Load in tokenizer for that model
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True, clean_up_tokenization_spaces=True)

    # Example of tokenizer
    '''
    sentence = "What time is it, Tom?"
    sentence_encoded = tokenizer(sentence, return_tensors='pt')
    sentence_decoded = tokenizer.decode(sentence_encoded['input_ids'][0], skip_special_tokens=True)
    print(f"Encoded sentence: {sentence_encoded['input_ids'][0]}")
    print(f"Decoded sentence: {sentence_decoded}")
    '''

    summarize_without_prompting(model, tokenizer, dataset, test_indices)
    summarize_zero_shot(model, tokenizer, dataset, test_indices)

    one_shot_prompt = _build_few_shot_prompt(dataset, [test_indices[0]], test_indices[1])
    print(f'{DASH_LINE}\nONE SHOT INFERENCE\n{DASH_LINE}')
    summarize_with_prompt(model, tokenizer, one_shot_prompt)

    three_shot_prompt = _build_few_shot_prompt(dataset, [40, 80, 120], 200)
    print(f'{DASH_LINE}\nTHREE SHOT INFERENCE\n{DASH_LINE}')
    summarize_with_prompt(model, tokenizer, three_shot_prompt)


if __name__ == '__main__':
    main()
