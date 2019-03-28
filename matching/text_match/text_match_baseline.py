import matchzoo as mz

from matching.load_data import load_data
from matching.train_test_split import train_test_split


if __name__ == '__main__':
    sample_pack = load_data()
    train_data_pack, test_data_pack = train_test_split(sample_pack)
    preprocessor = mz.preprocessors.NaivePreprocessor()
    preprocessor.fit(train_data_pack)
    print(preprocessor.context)
    vocab_unit = preprocessor.context['vocab_unit']
    print(vocab_unit.state['term_index']['match'])
    print(vocab_unit.state['term_index']['zoo'])
    print(vocab_unit.state['index_term'][1])
    print(vocab_unit.state['index_term'][2])
    train_data_pack_processed = preprocessor.transform(train_data_pack)
    test_data_pack_processed = preprocessor.transform(test_data_pack)
    print(train_data_pack_processed.left.head())

    # print('Before:', train_data_pack.left.loc[1]['text_left'])
    # sequence = train_data_pack_processed.left.loc[1]['text_left']
    # print('After:', sequence)
    # print('Translated:',
    #       '_'.join([vocab_unit.state['index_term'][i] for i in sequence]))

    model = mz.models.DenseBaseline()
    print(model.params, end='\n\n')
    model.params['name'] = 'My First Model'
    model.params['mlp_num_units'] = 3
    print(model.params, end='\n\n')
    model.guess_and_fill_missing_params()
    print(model.params)
    print('model completed: ', model.params.completed())
    model.build()
    model.compile()
    print(model.backend.summary())

    x, y = train_data_pack_processed.unpack()
    test_x, test_y = test_data_pack_processed.unpack()
    model.fit(x, y, batch_size=32, epochs=5)
    print(model.evaluate(test_x, test_y))
    print(model.predict(test_x))
