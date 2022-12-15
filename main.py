import gzip, pickle
def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo)
    return dict

if __name__ == '__main__':
    dataset_filename = 'bikes.tar.gz'
    print(unpickle(dataset_filename))