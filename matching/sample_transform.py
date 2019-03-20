import os

from collections import defaultdict
import csv

import pandas as pd

source_path = 'sample3.txt'


def load_data(base_dir):
    """
    Load dataset from File

    source_text = load_data(source_path)
    print("source_text:", source_text)
    """
    input_file = os.path.join(base_dir, source_path)
    with open(input_file, 'r', encoding='utf-8') as f:
        return f.read()


def load_table(base_dir):
    """
    source_text = load_table(source_path)
    print("source_text:\n", source_text)
    """
    path = os.path.join(base_dir, source_path)
    table = pd.read_csv(path, sep='\t', header=0, quoting=csv.QUOTE_NONE)
    # print(table.columns)
    df = pd.DataFrame({
        'text_id': table.index,
        'text': table['sentence'],
        'label': table['category']
    })
    return df


def produce_standard_data(df):
    standard = defaultdict(list)
    for i in range(df.shape[0]):
        for j in range(i+1, df.shape[0]):
            standard['text_left'].append(df.ix[i, 'text'])
            standard['text_right'].append(df.ix[j, 'text'])
            standard['id_left'].append(df.ix[i, 'text_id'])
            standard['id_right'].append(df.ix[j, 'text_id'])
            if df.ix[i, 'label'] == df.ix[j, 'label']:
                standard['label'].append(1)
            else:
                standard['label'].append(0)

    return pd.DataFrame(standard)


if __name__ == '__main__':

    source_text = load_table('../data/')
    print("standard data:\n", produce_standard_data(source_text))
