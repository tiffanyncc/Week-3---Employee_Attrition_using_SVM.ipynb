import matplotlib.pyplot as plt
import seaborn as sns
import os
import pandas as pd

# Ensure the images directory exists
os.makedirs('src/visualization/images', exist_ok=True)

def plot_distributions(df, num_cols):
    df[num_cols].hist(figsize=(14, 14))
    plt.savefig('src/visualization/images/distributions.png')
    plt.show()

def plot_categorical_distribution(df, cat_cols):
    for i in cat_cols:
        print(df[i].value_counts(normalize=True))
        print('*'*40)

def plot_categorical_attrition(df, cat_cols):
    for i in cat_cols:
        if i != 'Attrition':
            (pd.crosstab(df[i], df['Attrition'], normalize='index')*100).plot(kind='bar', figsize=(8, 4), stacked=True)
            plt.ylabel('Percentage Attrition %')
            plt.savefig(f'src/visualization/images/{i}_attrition.png')
            plt.show()

def plot_correlation_heatmap(df, num_cols):
    plt.figure(figsize=(15, 8))
    sns.heatmap(df[num_cols].corr(), annot=True, fmt='.2f', cmap='YlGnBu')
    plt.savefig('src/visualization/images/correlation_heatmap.png')
    plt.show()

def plot_metrics_score(cm):
    plt.figure(figsize=(8, 5))
    sns.heatmap(cm, annot=True, fmt='.2f', xticklabels=['Not Attrite', 'Attrite'], yticklabels=['Not Attrite', 'Attrite'])
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.savefig('src/visualization/images/confusion_matrix.png')
    plt.show()

def plot_bivariate_attrition(df, cat_cols):
    for i in cat_cols:
        if i != 'Attrition':
            try:
                (pd.crosstab(df[i], df['Attrition'], normalize='index')*100).plot(kind='bar', figsize=(8, 4), stacked=True)
                plt.ylabel('Percentage Attrition %')
                plt.savefig(f'src/visualization/images/bivariate_{i}_attrition.png')
                plt.show()
            except Exception as e:
                print(f'Error plotting {i}: {e}')
