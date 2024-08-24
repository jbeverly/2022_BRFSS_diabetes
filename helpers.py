import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

def plot_missing_values(df):
    plt.figure(figsize=(12, 6))
    sns.heatmap(df.isnull(), yticklabels=False, cbar=False, cmap='viridis')
    plt.title('Missing Value Heatmap')
    plt.tight_layout()
    plt.show()

    missing = df.isnull().sum() / len(df) * 100
    missing = missing[missing > 0].sort_values(ascending=False)
    plt.figure(figsize=(12, 6))
    missing.plot(kind='bar')
    plt.title('Percentage of Missing Values by Feature')
    plt.xlabel('Features')
    plt.ylabel('Percentage Missing')
    plt.tight_layout()
    plt.show()

def plot_distribution_comparison(df_before, df_after, column):
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    sns.histplot(df_before[column], kde=True)
    plt.title(f'{column} Distribution (Before Cleaning)')
    
    plt.subplot(1, 2, 2)
    sns.histplot(df_after[column], kde=True)
    plt.title(f'{column} Distribution (After Cleaning)')
    
    plt.tight_layout()
    plt.show()

def plot_boxplot_comparison(df_before, df_after, column):
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    sns.boxplot(x=df_before[column])
    plt.title(f'{column} Boxplot (Before Cleaning)')
    
    plt.subplot(1, 2, 2)
    sns.boxplot(x=df_after[column])
    plt.title(f'{column} Boxplot (After Cleaning)')
    
    plt.tight_layout()
    plt.show()

def plot_data_retention(df_before, df_after):
    labels = ['Before Cleaning', 'After Cleaning']
    rows = [len(df_before), len(df_after)]
    cols = [len(df_before.columns), len(df_after.columns)]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    rects1 = ax.bar(x - width/2, rows, width, label='Rows')
    rects2 = ax.bar(x + width/2, cols, width, label='Columns')

    ax.set_ylabel('Count')
    ax.set_title('Dataset Size Before and After Cleaning')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    ax.bar_label(rects1, padding=3)
    ax.bar_label(rects2, padding=3)

    fig.tight_layout()
    plt.show()

def plot_columnar_correlation(df, columns, figsize=(30,30)):
    correlation_matrix = df[list(columns)].corr()
    plt.figure(figsize=figsize)
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
    plt.title('Correlation Matrix')
    plt.show()

def analyze_missing_data(df):
    total_rows = len(df)    
    missing_counts = df.isnull().sum().sort_values(ascending=False)
    missing_percentages = (missing_counts / total_rows * 100).round(2)
    
    missing_table = pd.concat([missing_counts, missing_percentages], axis=1, 
                              keys=['Missing Count', 'Missing Percentage'])
    
    missing_table = missing_table[missing_table['Missing Count'] > 0]
    
    rows_retained = []
    for i in range(len(missing_table)):
        columns_to_drop = missing_table.index[:i+1]
        rows_kept = df.dropna(subset=columns_to_drop).shape[0]
        rows_retained.append(rows_kept)
    
    missing_table['Rows Retained'] = rows_retained
    missing_table['Retention Percentage'] = (missing_table['Rows Retained'] / total_rows * 100).round(2)
    
    return missing_table

def grid_display(cols, n):
    num_plots = n
    num_cols = cols
    num_rows = (num_plots + num_cols - 1) // num_cols  # Calculate rows needed
    
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(25, 5 * num_rows))
    axes = axes.flatten()
    return fig, axes

def combined_pairplot(data, categorical_columns, continuous_columns):
    n = len(categorical_columns) + len(continuous_columns)
    fig, axes = plt.subplots(n, n, figsize=(n*3, n*3))
    
    for i, col1 in enumerate(categorical_columns + continuous_columns):
        for j, col2 in enumerate(categorical_columns + continuous_columns):
            if i != j:
                if col1 in categorical_columns and col2 in categorical_columns:
                    sns.countplot(data=data, x=col1, hue=col2, ax=axes[i,j])
                elif col1 in continuous_columns and col2 in categorical_columns:
                    sns.boxplot(data=data, x=col2, y=col1, ax=axes[i,j])
                elif col1 in categorical_columns and col2 in continuous_columns:
                    sns.boxplot(data=data, x=col1, y=col2, ax=axes[i,j])
                else:
                    sns.scatterplot(data=data, x=col1, y=col2, ax=axes[i,j])
            else:
                if col1 in categorical_columns:
                    sns.countplot(data=data, x=col1, ax=axes[i,j])
                else:
                    sns.histplot(data=data, x=col1, ax=axes[i,j])
            
            axes[i,j].set_xlabel('')
            axes[i,j].set_ylabel('')
    
    plt.tight_layout()
    plt.show()
