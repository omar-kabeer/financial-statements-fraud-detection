
from src.dependencies import *

"""{{Functions}}"""
"""This is a Python script with functions for used for the Financial Statement Fraud Detection Project. 
The script contains methods and techniques used for, EDA, Outlier detection, Modeling."""
import sys
# Insert the parent path relative to this notebook so we can import from the src folder.
sys.path.insert(0, "..")
import src.dependencies
"""{{Functions}}"""
# Functions
import pandas as pd
import os

def load_dataset(file_path):
    """
    Load a dataset from a file while handling errors gracefully.

    Parameters:
        file_path (str): The path to the dataset file.

    Returns:
        pd.DataFrame: The loaded dataframe if successful, else None.
    """
    # Check if file exists
    if not os.path.exists(file_path):
        print(f"Error: File '{file_path}' not found.")
        return None

    # Determine file format and read accordingly
    try:
        if file_path.endswith('.csv'):
            # Try reading with different encodings if UTF-8 fails
            try:
                df = pd.read_csv(file_path, encoding='utf-8')
            except UnicodeDecodeError:
                print("Warning: UTF-8 encoding failed. Trying ISO-8859-1.")
                df = pd.read_csv(file_path, encoding='ISO-8859-1')

        elif file_path.endswith(('.xlsx', '.xls')):
            try:
                df = pd.read_excel(file_path, engine='openpyxl')
            except ValueError:
                print("Error: Excel file might be corrupt or unsupported.")
                return None

        elif file_path.endswith('.json'):
            try:
                df = pd.read_json(file_path)
            except ValueError:
                print("Error: JSON file format is incorrect.")
                return None

        else:
            print("Error: Unsupported file format. Please provide a CSV, Excel, or JSON file.")
            return None

        # Check for empty dataframe
        if df.empty:
            print("Warning: The loaded dataset is empty.")
        
        return df

    except pd.errors.ParserError:
        print("Error: File is corrupted or incorrectly formatted.")
    except Exception as e:
        print(f"Error: Failed to load dataset due to {str(e)}")
    
    return None


def count_percentage(df, column):
    """
    Calculate the percentage of each unique value in the specified column of the given DataFrame.

    Parameters:
    df (DataFrame): The input DataFrame
    column (str): The column for which to calculate the percentage

    Returns:
    DataFrame: A new DataFrame containing the count and percentage of each unique value in the specified column, sorted by percentage in descending order.
    """
    counts = df[column].value_counts()
    percentages = counts / counts.sum() * 100
    _df = pd.concat([counts, percentages], axis=1)
    _df.columns = ['Count', 'Percentage']
    _df = _df.sort_values(by='Percentage', ascending=False)
    _df = _df[_df['Count'] >= 0]
    return _df

def group_plot_timeline(df, cat_col, num_col, time):
    """
    A function to plot a timeline of total numeric values by category over time.
    
    Parameters:
    df (DataFrame): The input dataframe containing the data.
    cat_col (str): The name of the categorical column.
    num_col (str): The name of the numeric column to be plotted.
    time (str): The name of the time column.
    
    Returns:
    None
    """
    total_by_variable = df.groupby([cat_col, time])[num_col].sum().reset_index()
    total_by_variable[num_col] = total_by_variable[num_col] / 1000000000

    sns.set_style('whitegrid')
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.lineplot(x=time, y=num_col, hue=cat_col, data=total_by_variable, ax=ax)

    ax.set_ylabel(f'{num_col} (billions)')
    plt.xticks(rotation=90)

    plt.show()

def plot_percentage(df, x):
    sns.set(style="darkgrid")

    # create the bar chart
    ax = sns.barplot(y=df.index, x="Percentage", orient = 'h', data=df)

    # set the chart title and axis labels
    plt.title("Percentage of Financial Statements by " + x)
    plt.ylabel(x)
    plt.xlabel("Percentage")

    # add percentage values on top of each bar
    for i in ax.containers:
        ax.bar_label(i, label_type='edge', fontsize=8, rotation='horizontal')

    # rotate the x-axis labels
    plt.xticks(rotation='horizontal')

    # show the chart
    plt.show()


def plot_timeline(df, time, feature):
    ft_by_time = df.groupby(df[time])[feature].sum()


    fig, ax = plt.subplots(figsize=(20, 10))
    ax.plot(ft_by_time.index, ft_by_time.values, label='Revenue')
    ax.set_xlabel(time)
    ax.set_ylabel('Amount (in billions)')
    ax.set_title(f'Time-series of {time} by {feature}')
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: '${:.0f}B'.format(y/1e9)))
    ax.legend()
    plt.show()

def skew_kurt(df, col):
    # Calculate skewness and kurtosis of column
    _skewness = skew(df[col] / 1000000000)
    _kurtosis = kurtosis(df[col] / 1000000000)

    # Create histogram of column with mean, median, and mode
    fig, ax = plt.subplots(figsize=(20, 16))
    sns.histplot(data=df, x=col, kde=True, ax=ax)
    ax.axvline(df[col].mean(), color='r', linestyle='--', label='Mean')
    ax.axvline(df[col].median(), color='g', linestyle='--', label='Median')
    ax.axvline(df[col].mode()[0], color='b', linestyle='--', label='Mode')
    ax.legend()

    # Add text annotation for skewness and kurtosis values
    ax.annotate('Skewness: {:.2f}'.format(_skewness), xy=(0.5, 0.9), xycoords='axes fraction', fontsize=12)
    ax.annotate('Kurtosis: {:.2f}'.format(_kurtosis), xy=(0.5, 0.85), xycoords='axes fraction', fontsize=12)

    # Set x-axis label font size
    plt.xticks(fontsize=12)

    # Set x-axis range from 0 to 95th percentile of data
    ax.set_xlim(0, df[col].quantile(0.95))

    plt.show()
        
def plot_month_count(month, name):
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(18,12))

    # Plot the value counts of the 'Country' column in the first subplot
    sns.countplot(x='Country', data=month, ax=axs[0])
    sns.set_style('whitegrid')
    axs[0].set_title(f'Number of Financial Statements for Countries in {name}')
    axs[0].set_xlabel('Country')
    axs[0].set_ylabel('Count')
    axs[0].tick_params(axis='x', rotation=90)

    # Add count values on top of the bars in the first subplot
    for p in axs[0].patches:
        axs[0].annotate(format(p.get_height()), (p.get_x() + p.get_width() / 2., p.get_height()), ha = 'center', va = 'center', xytext = (0, 5), textcoords = 'offset points', fontsize=12)

    # Plot the value counts of the 'Industry' column in the second subplot
    sns.countplot(x='Industry', data=month, ax=axs[1])
    sns.set_style('whitegrid')
    axs[1].set_title(f'Number of Financial Statements for Industries in {name}')
    axs[1].set_xlabel('Industry')
    axs[1].set_ylabel('Count')
    axs[1].tick_params(axis='x', rotation=90)

    # Add count values on top of the bars in the second subplot
    for p in axs[1].patches:
        axs[1].annotate(format(p.get_height()), (p.get_x() + p.get_width() / 2., p.get_height()), ha = 'center', va = 'center', xytext = (0, 5), textcoords = 'offset points', fontsize=12)

    # Adjust the layout of the subplots
    plt.tight_layout()

    # Show the plot
    plt.show()

def plot_corr(df):
    # Set figure size and font sizes
    fig, ax = plt.subplots(figsize=(50, 50))
    sns.set(font_scale=1.9)

    # Plot heatmap with adjusted color map
    sns.heatmap(df, cmap='coolwarm', annot=True, center=0, square=True)

    # Adjust font size of features
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=35)
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=35)

    # Add title and axis labels
    plt.title('Correlation Matrix', fontsize=30)
    plt.xlabel('Features', fontsize=20)
    plt.ylabel('Features', fontsize=20)

    # Show plot
    plt.show()

def mahalanobis_distance(x, data):
    covariance_matrix = np.cov(data.T)
    inv_covariance_matrix = np.linalg.inv(covariance_matrix)
    diff = x - np.mean(data, axis=0)
    md = (diff @ inv_covariance_matrix @ diff.T)**0.5
    return md


def detect_multivariate_outliers(data, threshold=2):
    diffs = data - data.mean()
    md = []
    for i in range(len(diffs)):
        md.append(mahalanobis_distance(diffs.iloc[i], data))
    chi2_threshold = chi2.ppf((1 - 0.01), df=data.shape[1])
    outliers = np.where(np.array(md) > np.sqrt(chi2_threshold))[0]
    return outliers


def pca_outliers(data, nc, deviation):
    # Create a PCA object with 2 principal components
    pca = PCA(n_components=nc)

    # Fit the data to the PCA model
    pca.fit(data)

    # Transform the data using the PCA model
    transformed_data = pca.transform(data)

    # Calculate the distance of each point from the center of the data
    distances = np.linalg.norm(transformed_data - np.mean(transformed_data, axis=0), axis=1)

    # Calculate the mean and standard deviation of the distances
    mean_distance = np.mean(distances)
    std_distance = np.std(distances)

    # Set the threshold for outliers (e.g. 3 standard deviations from the mean)
    threshold = mean_distance + deviation * std_distance

    # Identify the outliers
    outliers = np.where(distances > threshold)[0]
    return outliers
    
    
def lof_outliers(data, nn, ct):
    # create a LOF object with contamination parameter set to 0.1
    lof = LocalOutlierFactor(n_neighbors=nn, contamination=0.1)

    # fit the LOF model and make predictions
    y_pred = lof.fit_predict(data)

    # get the scores and outliers from the LOF model
    scores = lof.negative_outlier_factor_
    outliers = np.where(y_pred==-1)[0]
    return outliers
    
    
def isolation_outliers(data, ne, ct):

    # instantiate the IsolationForest model
    model = IsolationForest(n_estimators=100, contamination=0.01, random_state=0)

    # fit the model to the data
    model.fit(data)

    # predict the outliers
    outlier_labels = model.predict(data)

    # get the indices of the outlier data points
    outlier_indices = np.where(outlier_labels == -1)[0]
    return outlier_labels
    
# Benford's Law
def firstDigit(n) : 

    while n >= 10:  
        n = n / 10

    return int(n)
BENFORD = [30.1, 17.6, 12.5, 9.7, 7.9, 6.7, 5.8, 5.1, 4.6]
def chi_square_test(data_count,expected_counts):
    """Return boolean on chi-square test (8 degrees of freedom & P-val=0.05)."""
    chi_square_stat = 0  # chi square test statistic
    for data, expected in zip(data_count,expected_counts):

        chi_square = math.pow(data - expected, 2)

        chi_square_stat += chi_square / expected

    print("\nChi-squared Test Statistic = {:.3f}".format(chi_square_stat))
    print("Critical value at a P-value of 0.05 is 15.51.")    
    return chi_square_stat < 15.51

def benford_chi2(df, industry_name, col):
    industry = df[df['Industry'] == industry_name]
    industry = industry[industry[col]!=0]
    if(len(industry)>100):
        industry_df = []
        industry_type = industry[col].values

        for i in industry_type:
            industry_df.append(firstDigit(i))
            
        industry_type_counts = pd.Series(industry_df).value_counts().sort_index().values


        if(0 in industry_df):
            industry_type_counts = industry_type_counts[1:]
        industry_type_percent = (industry_type_counts/np.sum(industry_type_counts))*100

        if(chi_square_test(industry_type_percent, BENFORD)):
            title = '{0}  {1} conforms with Benfords Law'.format(industry_name, col)
        else:
            title = ' {0} {1} industry seem to have some manipulation'.format(industry_name, col)

            

        plt.figure(figsize=(8, 5))
        plt.plot(industry_type_percent)
        plt.plot(BENFORD)
        plt.legend(['Industry distribution','Benfords Distribution'])
        plt.title(title)
        plt.show()
    else:
        print('{0} for #{1} cases doesnt have enough records to run Benfords law'.format(industry_name,col))
        
        
def benford_kstest(df, industry_name, col):
    industry = df[df['Industry'] == industry_name]
    industry = industry[industry[col]!=0]
    
    if len(industry) > 100:
        # extract first digit of values
        first_digits = [int(str(abs(i))[0]) for i in industry[col].values]
        
        # compute empirical distribution of first digits
        observed_counts = np.bincount(first_digits)[1:]
        observed_dist = observed_counts / np.sum(observed_counts)
        
        # compute expected distribution of first digits under Benford's Law
        expected_dist = np.array(BENFORD) / 100
        
        # perform KS test to compare empirical and expected distributions
        ks_stat, p_val = kstest(observed_dist, lambda x: expected_dist.cumsum())
        
        # plot distributions and label result based on p-value
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(range(1, 10), observed_dist, marker='o', label='Observed distribution')
        ax.plot(range(1, 10), expected_dist, marker='s', label='Expected distribution')
        ax.legend()
        ax.set_xticks(range(1, 10))
        ax.set_xlabel('First digit')
        ax.set_ylabel('Frequency')
        if p_val > 0.05:
            ax.set_title(f'{industry_name} {col} conforms with Benford\'s Law (KS p-value={p_val:.3f})')
        else:
            ax.set_title(f'{industry_name} {col} does not conform with Benford\'s Law (KS p-value={p_val:.3f})')
            
        plt.show()
    else:
        print(f'{industry_name} for #{col} cases does not have enough records to run Benford\'s Law')
        
        

def benford_linregress(df, industry_name, col):
    industry = df[df['Industry'] == industry_name]
    industry = industry[industry[col]!=0]
    
    if len(industry) > 100:
        # extract first digit of values
        first_digits = [int(str(abs(i))[0]) for i in industry[col].values]
        
        # compute empirical distribution of first digits
        observed_counts = np.bincount(first_digits)[1:]
        observed_dist = observed_counts / np.sum(observed_counts)
        
        # compute expected distribution of first digits under Benford's Law
        expected_dist = np.array(BENFORD) / 100
        
        # perform Kuiper's test to compare empirical and expected distributions
        linregress_stat, p_val = linregress(observed_dist, expected_dist.cumsum())

        
        # plot distributions and label result based on p-value
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(range(1, 10), observed_dist, marker='o', label='Observed distribution')
        ax.plot(range(1, 10), expected_dist, marker='s', label='Expected distribution')
        ax.legend()
        ax.set_xticks(range(1, 10))
        ax.set_xlabel('First digit')
        ax.set_ylabel('Frequency')
        if p_val > 0.05:
            ax.set_title(f'{industry_name} {col} conforms with Benford\'s Law (Kuiper p-value={p_val:.3f})')
        else:
            ax.set_title(f'{industry_name} {col} does not conform with Benford\'s Law (Kuiper p-value={p_val:.3f})')
            
        plt.show()
    else:
        print(f'{industry_name} for #{col} cases does not have enough records to run Benford\'s Law')



def benford_norm(df, industry_name, col):
    industry = df[df['Industry'] == industry_name]
    industry = industry[industry[col]!=0]
    
    if len(industry) > 100:
        # extract first digit of values
        first_digits = [int(str(abs(i))[0]) for i in industry[col].values]
        
        # compute empirical distribution of first digits
        observed_counts = np.bincount(first_digits)[1:]
        observed_dist = observed_counts / np.sum(observed_counts)
        
        # compute expected distribution of first digits under Benford's Law
        expected_dist = np.array(BENFORD) / 100
        expected_counts = expected_dist * np.sum(observed_counts)
        
        # perform Z-test to compare observed and expected counts
        z_stat = (observed_counts - expected_counts) / np.sqrt(expected_counts * (1 - expected_dist))
        p_val = 2 * norm.cdf(-np.abs(z_stat))
        
        # plot distributions and label result based on p-value
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.bar(range(1, 10), observed_counts, label='Observed distribution')
        ax.plot(range(1, 10), expected_counts, marker='s', label='Expected distribution')
        ax.legend()
        ax.set_xticks(range(1, 10))
        ax.set_xlabel('First digit')
        ax.set_ylabel('Frequency')
        if (p_val > 0.05).all():
            ax.set_title(f'{industry_name} {col} conforms with Benford\'s Law (Z-test p-value={p_val:.3f})')
        else:
            ax.set_title(f'{industry_name} {col} does not conform with Benford\'s Law (Z-test p-value={p_val:.3f})')
            
        plt.show()
    else:
        print(f'{industry_name} for #{col} cases does not have enough records to run Benford\'s Law')
        
        
        
def zipf_first_digit(n, N):
    """Return the rank of the first digit of n according to Zipf's Law."""
    rank = 0
    while n >= 1:
        n /= 10
        rank += 1
    return rank

def zipf_distribution(N):
    """Return a list of expected frequencies based on Zipf's Law."""
    freqs = []
    for i in range(1, 10):
        freq = 1 / (i * math.log10(N + 1))
        freqs.append(freq)
    return freqs

def zipf_goodness_of_fit(data):
    """Return boolean on goodness-of-fit test for Zipf's Law."""
    data_size = len(data)
    expected_freqs = zipf_distribution(data_size)
    observed_freqs = [0] * 9
    for n in data:
        rank = zipf_first_digit(n, data_size)
        if rank <= 9:
            observed_freqs[rank-1] += 1
    observed_freqs = np.array(observed_freqs) / data_size
    expected_freqs = np.array(expected_freqs)
    ks_stat, p_value = ks_2samp(observed_freqs, expected_freqs)
    print("\nKS Test Statistic = {:.3f}".format(ks_stat))
    print("Critical value at a P-value of 0.05 is 0.456.")
    return ks_stat < 0.456

def zipf_ksamp(df, industry_name, col):
    industry = df[df['Industry'] == industry_name]
    industry = industry[industry[col]!=0]
    if len(industry) > 100:
        industry_first_digits = []
        industry_values = industry[col].values
        for value in industry_values:
            first_digit = int(str(abs(value))[0])
            industry_first_digits.append(first_digit)
        industry_first_digit_counts = pd.Series(industry_first_digits).value_counts().sort_index().values
        if 0 in industry_first_digits:
            industry_first_digit_counts = industry_first_digit_counts[1:]
        industry_first_digit_freqs = (industry_first_digit_counts / np.sum(industry_first_digit_counts)) * 100
        if zipf_goodness_of_fit(industry_first_digits):
            title = '{0} {1} conforms with Zipf\'s Law'.format(industry_name, col)
        else:
            title = '{0} {1} does not conform with Zipf\'s Law'.format(industry_name, col)
        
        plt.figure(figsize=(8, 5))
        plt.hist(industry_first_digits, bins=range(1, 11), align='left', density=True)
        plt.plot(range(1, 10), zipf_goodness_of_fit(range(1, 10)), marker='o', markersize=4, linestyle='-', color='red')
        plt.title(title)
        plt.show()
    else:
        print('{0} for #{1} cases doesnt have enough records to run Zipf\'s Law'.format(industry_name, col))


def create_tsne_model(X, n_components=None):
    """
    Create a t-SNE model and fit it to the data.

    Args:
    X (array-like): The input data

    Returns:
    X_tsne (array-like): The transformed data
    """
    # Create a t-SNE model
    tsne = TSNE(n_components=n_components)

    # Fit the model to the data
    X_tsne = tsne.fit_transform(X)
    return X_tsne

def create_loop_model(X_tsne, extent=None, n_neighbors=None):
    """
    Create a LOF model and fit it to the data.

    Args:
    X_tsne (array-like): The transformed data
    extent (int): The extent parameter for LOF model
    n_neighbors (int): The number of neighbors for LOF model

    Returns:
    m (loop.LocalOutlierProbability): The LOF model
    """
    # Create a LOOP model
    m = loop.LocalOutlierProbability(X_tsne, extent=extent, n_neighbors=n_neighbors).fit()
    return m

def get_outlier_scores(m):
    """
    Get the outlier scores from the LOF model.

    Args:
    m (loop.LocalOutlierProbability): The LOF model

    Returns:
    scores (array-like): The outlier scores
    """
    # Get the outlier scores
    scores = m.local_outlier_probabilities
    return scores

def camel_to_snake(name):
    """
    Convert a camelCase string to snake_case.
    
    Args:
        name (str): The input camelCase string to be converted.
        
    Returns:
        str: The snake_case version of the input string.
    """
    import re
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()

def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion Matrix', cmap=plt.cm.Blues):
    """
Plot a confusion matrix.

Parameters:
- cm (array-like): The confusion matrix to be plotted.
- classes (list): The class labels.
- normalize (bool, optional): Whether to normalize the confusion matrix. Defaults to False.
- title (str, optional): The title of the plot. Defaults to 'Confusion Matrix'.
- cmap (colormap, optional): The colormap to be used for the plot. Defaults to plt.cm.Blues.

Returns:
None

"""
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), horizontalalignment='center', color='white' if cm[i, j] > thresh else 'black')

    plt.ylabel('True')
    plt.xlabel('Predicted')
    plt.tight_layout()