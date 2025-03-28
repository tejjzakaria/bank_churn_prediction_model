import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

dataset = pd.read_csv('../data/churn_data.csv')


# Function to save and show distribution plots
def plot_distribution(column, data, title):
    plt.figure(figsize=(10, 6))
    sns.histplot(data[column], kde=True, color='blue', bins=10)
    plt.title(title)
    plt.xlabel(column)
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.savefig(f'{column}_distribution.png')  # Save the plot as a PNG file
    plt.show()  # Display the plot

# Analyzing the dataset
for column in dataset.columns:
    plot_distribution(column, dataset, f'Distribution of {column}')

# Additional analysis: Pair plot for numerical data
sns.pairplot(dataset)
plt.savefig('pairplot.png')  # Save pairplot as a PNG file
plt.show()

# Boxplot to check for outliers (useful for numerical columns)
def plot_boxplot(column, data):
    plt.figure(figsize=(10, 6))
    sns.boxplot(x=data[column], color='green')
    plt.title(f'Boxplot of {column}')
    plt.xlabel(column)
    plt.grid(True)
    plt.savefig(f'{column}_boxplot.png')  # Save boxplot as PNG file
    plt.show()

# Generating boxplots for each numerical column
for column in dataset.select_dtypes(include=['number']).columns:
    plot_boxplot(column, dataset)

# Correlation heatmap to show relationships between numerical variables
plt.figure(figsize=(10, 6))
correlation_matrix = dataset.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=1)
plt.title('Correlation Heatmap')
plt.savefig('correlation_heatmap.png')  # Save correlation heatmap as PNG file
plt.show()
