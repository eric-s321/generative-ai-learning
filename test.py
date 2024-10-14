import textwrap

for i, data_index in enumerate([0]):

    # Create the prompt without leading spaces
    prompt = textwrap.dedent(f'''
        Summarize the following conversation.

        test dialog

        Summary:
    ''').strip()

    print('printing prompt')
    print(prompt)
    print('prompt printed')



def collator(data):
    return dict((key, [d[key] for d in data]) for key in data[0])

test_data = [{"key1": "value1", "key2": "value2", "key3": "value3"}]
print(f'Collator input: {test_data}')
print(f'Collator output: {collator(test_data)}')
