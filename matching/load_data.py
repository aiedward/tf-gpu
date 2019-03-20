"""data loader."""

from pathlib import Path

import matchzoo

from matching.sample_transform import load_table, produce_standard_data

base_dir = '../data/'


def load_data(stage='train', task='ranking'):
    """
    Load data.

    :param stage: One of `train`, `dev`, and `test`.
    :param task: Could be one of `ranking`, `classification` or a
        :class:`matchzoo.engine.BaseTask` instance.
    :return: A DataPack if `ranking`, a tuple of (DataPack, classes) if
        `classification`.
    """
    if stage not in ('train', 'dev', 'test'):
        raise ValueError(f"{stage} is not a valid stage."
                         f"Must be one of `train`, `dev`, and `test`.")

    data_root = Path(base_dir)
    # file_path = data_root.joinpath(f'WikiQA-{stage}.tsv')
    data_pack = _read_data(data_root)

    if task == 'ranking':
        task = matchzoo.tasks.Ranking()
    if task == 'classification':
        task = matchzoo.tasks.Classification()

    if isinstance(task, matchzoo.tasks.Ranking):
        return data_pack
    elif isinstance(task, matchzoo.tasks.Classification):
        data_pack.one_hot_encode_label(task.num_classes, inplace=True)
        return data_pack, [False, True]
    else:
        raise ValueError(f"{task} is not a valid task.")


def _read_data(path):
    source_text = load_table(path)
    df = produce_standard_data(source_text)
    return matchzoo.pack(df)


if __name__ == '__main__':
    load_data()
