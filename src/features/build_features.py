def feature_describe(data, lst_features):
    for feature in lst_features:
        print(data[feature].describe())
        print("\n")
    return None

