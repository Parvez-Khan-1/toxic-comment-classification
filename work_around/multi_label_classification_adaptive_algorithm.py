from skmultilearn.dataset import Dataset

## some information about the data set
# number of labels
labelcount = 16

# where the labels are located,
# big = at the beginning of the file
endianness = 'little'

# dtype used in the feature space
feature_type = 'float'

# whether the nominal attributes should be encoded as integers
encode_nominal = True

# if True - use the sparse loading mechanism from liac-arff
# if False - load dense representation and convert to sparse
load_sparse = True

# load data
X_train, y_train = Dataset.load_arff_to_numpy("path_to_data/dataset-train.dump.bz2",
    labelcount = labelcount,
    endian = "big",
    input_feature_type = feature_type,
    encode_nominal = encode_nominal,
    load_sparse = load_sparse)

X_test, y_test = Dataset.load_arff_to_numpy("path_to_data/dataset-train.dump.bz2",
    labelcount = labelcount,
    endian = "big",
    input_feature_type = feature_type,
    encode_nominal = encode_nominal,
    load_sparse = load_sparse)