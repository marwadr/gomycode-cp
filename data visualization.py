#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def plot_correlation_map(df):
    corr = df.corr()
    fig, ax = plt.subplots(figsize=(12, 10))
    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    sns.heatmap(
        corr,
        cmap=cmap,
        square=True,
        cbar_kws={'shrink': .9},
        ax=ax,
        annot=True,
        annot_kws={'fontsize': 12},
    )
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    df = pd.read_csv("titanic-passengers.csv", delimiter=";")

    df.info()
    print(df.columns.values)

    # Missing data summary
    total = df.isnull().sum().sort_values(ascending=False)
    percent = (df.isnull().sum() / len(df) * 100).round(1).sort_values(ascending=False)
    missing_data = pd.concat([total, percent], axis=1, keys=['Total', '%'])
    print(missing_data.head(5))

    print(df.isnull().sum())

    # Fill missing Age values with the mean
    df['Age'] = df['Age'].fillna(df['Age'].mean())

    # Feature engineering
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1

    # Extract Title from Name before dropping the column
    df['Title'] = df['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
    print(pd.crosstab(df['Title'], df['Sex']))

    # Drop columns no longer needed
    df.drop('Cabin', axis=1, inplace=True)
    df.drop('Name', axis=1, inplace=True)

    print(df)
    print(df.groupby('Pclass').Survived.value_counts())

    # Visualisations
    g = sns.FacetGrid(df, col='Survived')
    g.map(plt.hist, 'Age', bins=20)
    plt.show()

    plot_correlation_map(df.select_dtypes(include=[np.number]))




