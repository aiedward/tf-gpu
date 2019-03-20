import matchzoo as mz

train_data_pack = mz.datasets.wiki_qa.load_data(stage='train', task='ranking')
test_data_pack = mz.datasets.wiki_qa.load_data(stage='test', task='ranking')

