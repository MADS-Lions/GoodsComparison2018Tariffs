"""This script is to give ideas for selecting relevant features for the visualization and analysis. It contains functions that describe the features."""

def feature_describe(data, feature, time_feature, time_period_pre, time_period_post, which_dataset):
    """This function describes the features in the data.
    Args:
        data::pandas_df: The data.
        lst_features::list: The list of features to describe.
        time_feature::str: The time feature.
        time_period_pre::str: The pre-treatment period.
        time_period_post::str: The post-treatment period.
        

    Returns:
        pre_treatment::pandas_df: The data for the pre-treatment period.
        treatment::pandas_df: The data for the treatment period.
        post_treatment::pandas_df: The data for the post-treatment period.
    """
    if which_dataset == "USA":
        pre_treatment = data[data[time_feature] < time_period_pre][feature]
        treatment = data[(data[time_feature] >= time_period_pre) & (data[time_feature] < time_period_post)][feature]
        post_treatment = data[data[time_feature] >= time_period_post][feature]

        print("Feature: ", feature)
        print("Pre-treatment period: ")
        print(pre_treatment.describe())
        print("\n")
        print("Treatment period: ")
        print(treatment.describe())
        print("\n")
        print("Post-treatment period: ")
        print(post_treatment.describe())
        print("\n")
            
        return pre_treatment, treatment, post_treatment
    else:
        pre_treatment = data[(data[time_feature] < time_period_pre) & (data['Products and product groups'] == feature)]['VALUE']
        treatment = data[(data[time_feature] >= time_period_pre) & (data[time_feature] < time_period_post) & (data['Products and product groups'] == feature)]['VALUE']
        post_treatment = data[(data[time_feature] >= time_period_post) & (data['Products and product groups'] == feature)]['VALUE']

        print("Feature: ", feature)
        print("Pre-treatment period: ")
        print(pre_treatment.describe())
        print("\n")
        print("Treatment period: ")
        print(treatment.describe())
        print("\n")
        print("Post-treatment period: ")
        print(post_treatment.describe())
        print("\n")
            
        return pre_treatment, treatment, post_treatment

