from matching.sample_transform import load_table


def train_test_split(data_frame):
    num_train = int(data_frame.shape[0] * 0.8)
    data_frame = data_frame.sample(frac=1).reset_index(drop=True)
    # data_frame = data_frame.sample(frac=1)
    train_slice = data_frame[:num_train]
    test_slice = data_frame[num_train:]
    return train_slice, test_slice


if __name__ == '__main__':
    data_table = load_table('../data/')
    print("Before split: ", data_table)
    train_data, test_data = train_test_split(data_table)
    print("After split: \n\n", train_data)
    print(test_data)
