from matching.load_data import load_data


def train_test_split(data_pack):
    num_train = int(len(data_pack) * 0.8)
    data_pack.shuffle(inplace=True)
    train_slice = data_pack[:num_train]
    test_slice = data_pack[num_train:]
    return train_slice, test_slice


if __name__ == '__main__':
    sample_pack = load_data()
    train_data, test_data = train_test_split(sample_pack)
    print(train_data.frame())
    print(test_data.frame())
