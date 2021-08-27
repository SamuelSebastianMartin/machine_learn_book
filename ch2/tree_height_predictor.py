#! /usr/bin/env python3

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
from pandas.plotting import scatter_matrix


"""
This graphs the location of various tree types in London.
The user is asked to enter the name of a tree from a list
of all tree types.
The location of the trees is then printed on the map.
"""
DATAFILE = "Borough_tree_list_2021July.csv"
TREETYPE = 'London plane'

def prepare_dataframe():
    df = pd.read_csv(DATAFILE, low_memory=False)
    # Drop uninteresting columns
    df = df.drop(['objectid','borough','maintainer','gla_tree_name',
                  'dbh_group','load_date','updated',
                  'canopy_spread_group', 'condition'],
                 axis=1)
    ## Drop rows with any null values in critical columns ('N/A')
    df = df.dropna(subset=['age','diameter_at_breast_height_cm',
                           'height_m', 'spread_m'])
    ## Convert the height column to 'float', from 'object'
    df['height_m'] = df['height_m'].astype('float')
    return df

def select_treetype_option(df):
    """
    This function offers the chance to change the tree type
    that is analysed.
    It returns a dataframe with data on that tree only.
    """
    print("""The data will be prepared for London plane trees\n
          If you wish to look at a different type of tree,""")
    change_tree_choice = input("type 'y'. Otherwise, <ENTER>")
    if change_tree_choice == 'y':
        list_of_treetypes = list(df['common_name'].unique())
        print('Here are all the trees', list_of_treetypes)
        TREETYPE = input("\n\nChoose one tree: ")
        df = df[df['common_name'] == TREETYPE]
        print(df.head(10))
        print(df.info())
        print(df.describe())
    return df

def map_trees(df):
    """
    Plots the location of each tree, forming a kind of map.
    This cannot be used with show_histograms()
    """
    plt.scatter(df['longitude'], df['latitude'], alpha=0.1)
    plt.show()

def show_histograms(df):
    """
    Plots relevant histograms, to give a rough idea of
    the data.
    This cannot be used with map_trees()
    """
    df.hist(bins=50, figsize=(20,15))
    plt.show()

def options(df):
    """
    Allows the option to show a graphic. This will be
    either a map or a histogram
    """
    print("""\n\nWould you like to see a map or histogram?\n
          To see a map, type 'm':\n
          For a histogram, its 'h':""")
    option = input("Or press <ENTER> to skip: ")
    if option == 'm':
        map_trees(df)
    elif option == 'h':
        show_histograms(df)
    else:
        return

def basic_split_test_set(df):
    """
    This splits the test set off from the training set. It is a good-enough
    function, but the categorised_split_test_set() is less prone to bias.
    """
    train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)
    return train_set, test_set

def categorised_split_test_set(df):
    ## NOT WORKING with this dataset ##
    """
    This splits the data into categories before separating the training
    and the testing sets. This means that there will be less chance of
    random bias entering into the selection of the sets.
    In particular, as the diameter of the trunk is a good predictor of
    age, it ensures that the proportion of thin, medium, thick etc 
    trunk diameters will be similar in both sets.
    """
    # Create the stratified categories
    df['diameter_cat'] = pd.cut(df['diameter_at_breast_height_cm'],
                                bins=3,
                                labels=[1, 2, 3])
    # Split the sets
    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train_index, test_index in split.split(df, df["diameter_cat"]):
        strat_train_set = df.loc[train_index]
        strat_test_set = df.loc[test_index]
    # Remove the categories column
    for set_ in (strat_train_set, strat_test_set):
        set_.drop('diameter_cat', axis=1, inplace=True)
    return strat_train_set, strat_test_set

def visualise_data(df):
    """
    Plots the location of the trees with diameter as size of dot 's'
    and height as the color of the dot 'c'.
    """
    df.plot(kind="scatter", x="longitude", y="latitude", alpha=0.4,
            s=df["diameter_at_breast_height_cm"], label="Tree diameter",
                 figsize=(10, 7), c="height_m", cmap=plt.get_cmap("jet"),
                 colorbar=True,
    )
    plt.legend()
    plt.show()


def visualise_correlations(df):
    """
    Uses pandas scatter_matrix to plot a correlation matrix for every attribute.
    """
    attributes = ['height_m', 'diameter_at_breast_height_cm',
                  'spread_m']
    scatter_matrix(df[attributes], figsize=(12, 8))
    plt.show()


def main():
    df = prepare_dataframe()
#    df = select_treetype_option(df)
#    options(df)
    train_set, test_set = basic_split_test_set(df)
#    train_set, test_set = categorised_split_test_set(df)
#    visualise_data(train_set)
    visualise_correlations(df)


if __name__ == '__main__':
    main()
