
# --------------------------------------- IMPORTS ------------------------------------------------------ #


from pyexpat import model
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import openpyxl
from dateutil.relativedelta import relativedelta
from math import ceil

# For Scaling
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

# For Hierarchical Clustering Dendrogram
from scipy.cluster.hierarchy import linkage, dendrogram

# For Clustering Algorithms
from sklearn.cluster import MeanShift, DBSCAN, HDBSCAN, estimate_bandwidth, AgglomerativeClustering, KMeans
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors
from minisom import MiniSom

# Metrics to evaluate kmeans clustering
from sklearn.metrics import silhouette_score, calinski_harabasz_score, silhouette_samples

# For Profiling Clusters
from pandas.plotting import parallel_coordinates

# For Color Maps and Visualizations
from matplotlib import colors as mpl_colors
from matplotlib import colorbar
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize, to_hex
from matplotlib.patches import RegularPolygon
from mpl_toolkits.axes_grid1 import make_axes_locatable

# For Autoencoder
from typing import Tuple
from torch import nn
from torch import Tensor
import torch
from torch.utils.data import DataLoader, TensorDataset

# Cluster Vizualizations
from umap import UMAP
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import plotly.express as px
import plotly.io as pio

# Assess feature importance and outlier cluster label prediction
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

# To ignore warnings
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(1)
torch.manual_seed(1)







# --------------------------------------- FEATURE ENGINEERING ------------------------------------------------------ #

def get_season(month):

    ''' Returns the season for a given month. 
    
    Parameters:
    ----------------------------------------
    month: int
        The month as an integer (1-12).
    
    Returns:
    ----------------------------------------
    season: str
        The corresponding season ('Winter', 'Spring', 'Summer', 'Autumn').
    
    '''

    if month in [12, 1, 2]:
        return 'Winter'
    elif month in [3, 4, 5]:
        return 'Spring'
    elif month in [6, 7, 8]:
        return 'Summer'
    else:
        return 'Autumn'



# -------------------------------------------------- SCALING ------------------------------------------------------ #

def scaling_features(data, method):
    """ 
    Scales the numeric features according to the specified method.

    Parameters:
    ----------------------------------------
    data: dataframe
        data to be scaled

    method : str 
        The scaling method to use. Options are 'minmax', 'minmax2', 'standard', and 'robust'.

    Returns:
    ----------------------------------------
    scaled_data : dataFrame
        The scaled data.
    """

    data_copy = data.copy()

    metric_features = data_copy.select_dtypes(include=[np.number]).columns.tolist()
    non_metric_features = data_copy.drop(columns=metric_features).columns.tolist()

    data_to_scale = data_copy[metric_features]
    
    if method == 'minmax':
        # Create a MinMaxScaler instance
        scaler = MinMaxScaler()
        
    elif method == 'minmax2':
        # Create a MinMaxScaler instance that will range between -1 and 1
        scaler = MinMaxScaler(feature_range=(-1, 1))
        
    elif method == 'standard':
        # Create a StandardScaler instance
        scaler = StandardScaler()
        
    elif method == 'robust':
        # Create a RobustScaler instance
        scaler = RobustScaler()

    scaled = scaler.fit_transform(data_to_scale)

    scaled_data = pd.DataFrame(scaled, columns=metric_features, index=data_copy.index)
    result = pd.concat([scaled_data, data_copy[non_metric_features]], axis=1)
    
    return result



# ------------------------------------------------------ GET R² ------------------------------------------------------ #

def get_ss(df, features):
    """ 
    Calculate total sum of squares (SST)

    Parameters:
    ----------------------------------------
    df: dataframe
        data to calculate sum of squares

    features : list
        List of feature names to consider for the calculation.
    
    Returns:
    ----------------------------------------
    sst : float
        The total sum of squares.
    """
    return np.sum(df[features].var() * (df[features].count() - 1))



def get_ssw(df, features, labels):
    """ 
    Calculate within-cluster sum of squares (SSW)

    Parameters:
    ----------------------------------------
    df: dataframe
        data to calculate sum of squares

    features : list
        List of feature names to consider for the calculation.

    labels : array-like
        Cluster labels for each data point.
    
    Returns:
    ----------------------------------------
    ssw : float
        The within-cluster sum of squares.
    """
    df_temp = df[features].copy()
    df_temp['labels'] = labels
    return df_temp.groupby('labels')[features].apply(lambda x: get_ss(x, features)).sum()



def get_rsq(df, features, labels):
    """ 
    Calculate R² for clustering quality

    Parameters:
    ----------------------------------------
    df: dataframe
        data to calculate R²

    features : list
        List of feature names to consider for the calculation.

    labels : array-like
        Cluster labels for each data point.
    
    Returns:
    ----------------------------------------
    rsq : float
        The R² value.
    """
    sst = get_ss(df, features)
    ssw = get_ssw(df, features, labels)
    return 1 - (ssw / sst)



# -------------------------------------------------- FEATURE SELECTION ------------------------------------------------------ #

def corr_pairs(corr, threshold):
    '''
    
    
    Identifies and prints pairs of features with correlation above a specified threshold, along with the count of other features
    each is significantly correlated with.

    Parameters:
    ----------------------------------------
    corr: DataFrame
        A pandas DataFrame representing the correlation matrix of features.

    threshold: float
        The correlation threshold above which feature pairs are considered significant.

    Returns:
    ----------------------------------------
    None
        Prints the feature pairs and their correlation values, along with counts of significantly correlated features.
    
    '''

    # Create a mask for the lower triangle of the correlation matrix (excluding the diagonal)
    mask = np.tril(np.ones(corr.shape), k=-1).astype(bool)
    
    # Apply the mask to the correlation matrix, keeping only the lower triangle values
    corr_lower = corr.where(mask)
    
    # Unstack the lower triangle into a Series with MultiIndex (feature1, feature2)
    cor_pairs = corr_lower.unstack().dropna()
    
    # Filter pairs where the absolute correlation exceeds the specified threshold
    cor_pairs = cor_pairs[(cor_pairs.abs() >= threshold)]

    # Iterate over each pair of features and their correlation value
    for pair, value in cor_pairs.items():
        # Count the number of other features significantly correlated with the first feature in the pair
        # Significance is defined as abs(corr) > 0.35 and less than the threshold
        feature_count_pair1 = len(corr[(corr[pair[0]].abs() > 0.35) & (corr[pair[0]].abs() < threshold)])
        
        # Same count for the second feature in the pair
        feature_count_pair2 = len(corr[(corr[pair[1]].abs() > 0.35) & (corr[pair[1]].abs() < threshold)])
        
        print(f'{value:.2f} : {pair[0]} | significantly correlated with {feature_count_pair1} more features | or | {pair[1]} | significantly correlated with {feature_count_pair2} more features')




# --------------------------------------- HIERARCHICAL CLUSTERING ------------------------------------------------------ #

def get_r2_hc(df, link_method, max_nclus, min_nclus=1, dist="euclidean"):

    '''
    This function computes the R² for a set of cluster solutions given by the application of a hierarchical method.
    The R² is a measure of the homogenity of a cluster solution. It is based on SSt = SSw + SSb and R² = SSb/SSt. 
    
    Parameters:
    -------------------------------------------------------------
    df (DataFrame):
        Dataset to apply clustering

    link_method (str): 
        either "ward", "complete", "average", "single"

    max_nclus (int): 
        maximum number of clusters to compare the methods

    min_nclus (int): 
        minimum number of clusters to compare the methods. Defaults to 1.

    dist (str): 
        distance to use to compute the clustering solution. Must be a valid distance. Defaults to "euclidean".
    
    Returns:
    -------------------------------------------------------------
    ndarray: 
        R² values for the range of cluster solutions
    '''
    
    r2 = []  # where we will store the R² metrics for each cluster solution
    feats = df.columns.tolist()
    
    for i in range(min_nclus, max_nclus+1):  # iterate over desired ncluster range
        

        cluster = AgglomerativeClustering(linkage=link_method, metric=dist, n_clusters=i)
 
        hclabels = cluster.fit_predict(df[feats])
    
        r2.append(get_rsq(df, feats, hclabels))
        
    return np.array(r2)




def get_r2_hc_link(df, hc_methods, max_clusters, distance="euclidean"):
    '''
    This function computes the R2 for a set of hierarchical clustering methods, for different linkage methods and number of clusters.

    Parameters:
    -------------------------------------------------------------
    df (DataFrame):
        Dataset to apply clustering

    hc_methods (list): 
        list of hierarchical clustering methods to compare, e.g., ["ward", "complete", "average", "single"]

    max_clusters (int): 
        maximum number of clusters to compare the methods

    distance (str): 
        distance to use to compute the clustering solution. Must be a valid distance. Defaults to "euclidean".

    Returns:
    -------------------------------------------------------------
    DataFrame: 
        R² values for each hierarchical clustering method and number of clusters
    
    
    
    '''
    results = []
    for link in hc_methods: # for each method, the R² values for different nclusters are computed and stored
        r2 = get_r2_hc(
            df=df, 
            link_method=link, 
            max_nclus=max_clusters, 
            min_nclus=1, 
            dist=distance)
        results.append(r2)
        r2_hc = np.vstack(results)
    return pd.DataFrame(r2_hc.T, index=range(1, max_clusters + 1), columns=hc_methods)



def plot_r2_hc_methods(datasets):

    '''
    This function plots the R2 values for different hierarchical clustering methods and number of clusters.

    Parameters:
    -------------------------------------------------------------
    datasets (list): 
        list of DataFrames containing R2 values for different hierarchical clustering methods and number of clusters.

    Returns:
    -------------------------------------------------------------
    None: 
        Displays the plots of R2 values for each hierarchical clustering method.


    '''
    sns.set()

    fig, axes = plt.subplots(2, 2, figsize=(18, 10))  
    axes = axes.flatten()

    for i, data in enumerate(datasets):
        sns.lineplot(ax=axes[i], data=data, linewidth=2.5, markers=["o"]*4)
        axes[i].set_title(f"Plot {i+1}", fontsize=15)
        axes[i].set_xticks(range(1, 11))
        axes[i].set_xlabel("Number of clusters", fontsize=12)
        axes[i].set_ylabel("R2 metric", fontsize=12)
        axes[i].legend(title="HC methods", title_fontsize=10)


    fig.suptitle("$R^2$ plots for various hierarchical methods", fontsize=22)
    plt.tight_layout()
    plt.show()





def plot_dendogram(linkage_matrix, y_threshold, distance):
    '''
    Plots a dendrogram for hierarchical clustering.
    
    Parameters:
    -------------------------------------------------------------
    linkage_matrix (ndarray): 
        The linkage matrix obtained from hierarchical clustering.

    y_threshold (float):
        The threshold value for cutting the dendrogram.

    distance (str): 
        The distance metric to use.


    Returns:
    -------------------------------------------------------------
    None: 
        Displays the dendrogram plot.
        
    '''

    sns.set()
  
    plt.figure(figsize=(16, 8))
    
    dendrogram(linkage_matrix, truncate_mode='level', p=5, color_threshold=y_threshold, above_threshold_color='k')
     
    # Add horizontal line for threshold
    plt.hlines(y_threshold, 0, 1000, colors="r", linestyles="dashed")
    plt.title(f'Hierarchical Clustering Dendrogram: Ward Linkage', fontsize=21)
    plt.xlabel('Number of points in node (or index of point if no parenthesis)')
    plt.ylabel(f'{distance.title()} Distance', fontsize=13)
    plt.show()



# --------------------------------------------- KMEANS CLUSTERING ------------------------------------------------------ #


def silhouette_analysis(data, k_range, title_suffix):
    """
    Performs silhouette analysis for different k values
    and plots all silhouettes in a single figure (3 per row)

    Parameters:
    -------------------------------------------------------
    data : numpy.ndarray
        The scaled data for clustering.

    k_range : range
        The range of k values to evaluate.

    title_suffix : str
        Suffix to add to the title of each subplot.

    Returns:
    -------------------------------------------------------
    avg_silhouette : list
        A list of average silhouette scores for each k.
    """
    
    # Keep only k values greater than 1, since silhouette score is undefined for k=1
    k_values = [k for k in k_range if k > 1]
    
    # Compute number of subplots (3 per row)
    n_plots = len(k_values)
    n_cols = 3
    n_rows = ceil(n_plots / n_cols)
    
    # Create subplots
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 5 * n_rows))
    axes = axes.flatten()  # Flatten in case of multiple rows
    
    # List to store average silhouette scores
    avg_silhouette = []
    
    # Loop over each k value
    for idx, nclus in enumerate(k_values):
        ax = axes[idx]  # current subplot
        
        # Fit KMeans with the current number of clusters
        kmclust = KMeans(n_clusters=nclus, init='k-means++', n_init=15, random_state=1)
        cluster_labels = kmclust.fit_predict(data)
        
        # Compute the average silhouette score for this k
        silhouette_avg = silhouette_score(data, cluster_labels)
        avg_silhouette.append(silhouette_avg)
        print(f"For n_clusters = {nclus}, the average silhouette_score is : {silhouette_avg:.4f}")
        
        # Compute silhouette score for each sample
        sample_silhouette_values = silhouette_samples(data, cluster_labels)
        y_lower = 10  # Initial position for plotting silhouette bars
        
        # Plot silhouette bars for each cluster
        for i in range(nclus):
            # Get silhouette scores for cluster i and sort them
            ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]
            ith_cluster_silhouette_values.sort()
            
            # Define vertical boundaries of the cluster in the plot
            size_cluster_i = ith_cluster_silhouette_values.shape[0]
            y_upper = y_lower + size_cluster_i
            
            # Assign a color to the cluster
            color = cm.nipy_spectral(float(i) / nclus)
            
            # Draw the filled silhouette bars for the cluster
            ax.fill_betweenx(
                np.arange(y_lower, y_upper),
                0,
                ith_cluster_silhouette_values,
                facecolor=color,
                edgecolor=color,
                alpha=0.7
            )
            
            # Add cluster label on the plot
            ax.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
            
            # Update y_lower for the next cluster, adding spacing of 10
            y_lower = y_upper + 10
        
        # Draw a vertical line for the average silhouette score
        ax.axvline(
            x=silhouette_avg,
            color="red",
            linestyle="--",
            label=f"Avg: {silhouette_avg:.3f}"
        )
        
        # Set x-axis and y-axis limits
        xmin = np.round(sample_silhouette_values.min() - 0.1, 2)
        xmax = np.round(sample_silhouette_values.max() + 0.1, 2)
        
        ax.set_xlim([xmin, xmax])
        ax.set_ylim([0, len(data) + (nclus + 1) * 10])
        
        # Remove y-ticks, keep x-ticks for reference
        ax.set_yticks([])
        ax.set_xticks(np.arange(xmin, xmax, 0.1))
        
        # Add title and labels
        ax.set_title(f"{title_suffix} (k={nclus})")
        ax.set_xlabel("Silhouette Coefficient")
        ax.legend()
    
    # Remove extra subplots that are empty
    for j in range(n_plots, len(axes)):
        fig.delaxes(axes[j])
    
    # Adjust layout and show the figure
    plt.tight_layout()
    plt.show()
    
    # Return the list of average silhouette scores
    return avg_silhouette






def elbow_and_ch_analysis(X_scaled, k_range, title_suffix):
    """
    Calculates Inertia (Elbow) and Calinski-Harabasz scores.

    Parameters:
    -------------------------------------------------------
    X_scaled : numpy.ndarray
        The scaled data for clustering.

    k_range : range
        The range of k values to evaluate.

    title_suffix : str
        Suffix to add to the title of each subplot.


    Returns:
    -------------------------------------------------------
    inertias : list
        A list of inertia values for each k.

    ch_scores : list
        A list of Calinski-Harabasz scores for each k.

    """
    inertias = []
    ch_scores = []
    
    for k in k_range:
        kmclust = KMeans(n_clusters=k, init='k-means++', n_init=10, random_state=1)
        labels = kmclust.fit_predict(X_scaled)
        
        inertias.append(kmclust.inertia_)
        ch_scores.append(calinski_harabasz_score(X_scaled, labels))
    
    # Plot Elbow Method
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    axes[0].plot(k_range, inertias, 'bo-', linewidth=2, markersize=8)
    axes[0].set_title(f'Elbow Method - {title_suffix}', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Number of Clusters (k)')
    axes[0].set_ylabel('Inertia')
    axes[0].grid(True, alpha=0.3)
    
    # Plot Calinski-Harabasz Score
    axes[1].plot(k_range, ch_scores, 'go-', linewidth=2, markersize=8)
    best_k_ch = k_range[np.argmax(ch_scores)]
    axes[1].axvline(x=best_k_ch, color='red', linestyle='--', linewidth=2, label=f'Best k={int(best_k_ch)}')
    axes[1].set_title(f'Calinski-Harabasz Score - {title_suffix}', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Number of Clusters (k)')
    axes[1].set_ylabel('Calinski-Harabasz Score')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    
    plt.tight_layout()
    plt.show()
    
    return inertias, ch_scores




def summary_metrics_table(X_scaled, k_range, title_suffix):
    """
    Creates a summary table of all metrics for each k.


    Parameters:
    -------------------------------------------------------
    X_scaled : numpy.ndarray
        The scaled data for clustering.

    k_range : range
        The range of k values to evaluate.

    title_suffix : str
        Suffix to add to the title of each subplot.


    Returns:
    -------------------------------------------------------
    results_df : DataFrame
        A DataFrame containing k, Inertia, Silhouette, and Calinski-Harabasz scores.

    """
    results = {
        'k': [],
        'Inertia': [],
        'Silhouette': [],
        'Calinski-Harabasz': []
    }
    
    for k in k_range:
        kmclust = KMeans(n_clusters=k, init='k-means++', n_init=10, random_state=1)
        labels = kmclust.fit_predict(X_scaled)
        
        results['k'].append(k)
        results['Inertia'].append(kmclust.inertia_)
        results['Silhouette'].append(silhouette_score(X_scaled, labels))
        results['Calinski-Harabasz'].append(calinski_harabasz_score(X_scaled, labels))
    
    results_df = pd.DataFrame(results)
    
    print("\n" + "=" * 80)
    print(f"SUMMARY METRICS - {title_suffix}")
    print("=" * 80)
    print(results_df.to_string(index=False, float_format='%.4f'))
    print("=" * 80)
    
    return results_df





# --------------------------------------- DENSITY BASED CLUSTERING ------------------------------------------------------ #

def mean_shift(df, perspective, quantile, random_state=1):

    '''

    Applies Mean Shift clustering to the given DataFrame and computes the R² score.

    Parameters:
    -------------------------------------------------------------
    df : dataframe
        Dataset to apply Mean Shift clustering

    perspective : list
        List of feature names to consider for clustering.

    quantile : float
        The quantile parameter for bandwidth estimation.

    random_state : int, optional
        Random state for reproducibility. Default is 1.

    Returns:
    -------------------------------------------------------------
    None
        Prints the R² score, bandwidth, number of clusters, and cluster label counts.

    '''

    bandwidth = estimate_bandwidth(df[perspective], quantile=quantile, random_state=random_state)

    mst = MeanShift(bandwidth=bandwidth, bin_seeding = True, n_jobs=4)

    ms_labels = mst.fit_predict(df[perspective])

    r2_1 = get_rsq(df, perspective, ms_labels)
    
    print(f"R² score: {r2_1:.4f}")
    print(f"bandwith: {bandwidth:.4f}")
    print(len(np.unique(ms_labels)))
    return print(pd.Series(ms_labels).value_counts().sort_index())





# -------------------------------------------- SELF-ORGANIZING MAP ------------------------------------------------------ #

def train_som (data, topology='hexagonal', N = 20, M = 30):
    """
    Trains a Self-Organizing Map (SOM) on the provided data.

    Parameters:
    --------------------------------------------------------
    data: dataframe
        pandas DataFrame of shape (n_samples, n_features)

    topology: str
        The topology of the SOM grid. Options are 'rectangular' or 'hexagonal'. Default is 'hexagonal'.

    N: int
        Number of rows in the SOM grid. Default is 20.

    M: int
        Number of columns in the SOM grid. Default is 30.

    Returns:
    --------------------------------------------------------
    som: MiniSom
        trained MiniSom object
    
    """
    
    data = data.values  # Convert DataFrame to NumPy array
    n_feats = data.shape[1]

    sm = MiniSom(N, M,              
             n_feats,        
             learning_rate=0.5, 
             topology=topology, 
             neighborhood_function='gaussian', 
             sigma=1.8,
             activation_distance='euclidean',
             random_seed=42
             )
    
    # Initializes the weights of the SOM picking random samples from data
    sm.random_weights_init(data) 

    print("Before training:")
    print("QE", np.round(sm.quantization_error(data),4))
    if topology == 'rectangular':
        print("TE", np.round(sm.topographic_error(data),4))



    # Trains the SOM using all the vectors in data sequentially
    # minisom does not distinguish between unfolding and fine tuning phase;

    sm.train_batch(data, 500_000)

    print("After training:")
    print("QE", np.round(sm.quantization_error(data),4))
    if topology == 'rectangular':
        print("TE", np.round(sm.topographic_error(data),4))
    
    return sm






def plot_hexagons(som, sf, colornorm, matrix_vals, label="", cmap=cm.Grays, annot=False):

    """
    
    Plots a hexagonal grid representing the Self-Organizing Map (SOM).
    
    Each hexagon corresponds to a node in the SOM, and its color represents
    a value from the provided matrix (e.g., U-Matrix, component planes, or custom metrics).
    Optionally, the hexagons can be annotated with their numeric values.

    Parameters:
    --------------------------------------------------------
    som : MiniSom
        A trained MiniSom object. Used to map grid coordinates to Euclidean positions.
    
    sf : matplotlib.figure.Figure
        Figure object on which to plot the hexagonal SOM grid.
    
    colornorm : matplotlib.colors.Normalize
        Normalization object used to scale matrix values to the colormap.
    
    matrix_vals : 2D numpy.ndarray
        A 2D array of values to plot on the SOM grid. Typically a U-Matrix or component plane.
        The shape should match the dimensions of the SOM.
    
    label : str, optional (default="")
        Title for the figure.
    
    cmap : matplotlib.colors.Colormap, optional (default=cm.Grays)
        Colormap used to color the hexagons.
    
    annot : bool, optional (default=False)
        If True, the numeric values from matrix_vals will be annotated inside the hexagons.

    Returns:
    --------------------------------------------------------
    sf : matplotlib.figure.Figure
        The input matplotlib figure object, now containing the hexagonal SOM plot
        with color mapping, optional annotations, and an accompanying colorbar.


    """
    
    axs = sf.subplots(1,1)
    
    for i in range(matrix_vals.shape[0]):
        for j in range(matrix_vals.shape[1]):

            wx, wy = som.convert_map_to_euclidean((i,j)) 

            hex = RegularPolygon((wx, wy), 
                                numVertices=6, 
                                radius= np.sqrt(1/3),
                                facecolor=cmap(colornorm(matrix_vals[i, j])), 
                                alpha=1, 
                                edgecolor='white',
                                linewidth=.5)
            axs.add_patch(hex)
            if annot==True:
                annot_val = np.round(matrix_vals[i,j],2)
                if int(annot_val) == annot_val:
                    annot_val = int(annot_val)
                axs.text(wx,wy, annot_val, 
                        ha='center', va='center', 
                        fontsize='x-small')


    ## Remove axes for hex plot
    axs.margins(.05)
    axs.set_aspect('equal')
    axs.axis("off")
    axs.set_title(label)

    

    # ## Add colorbar
    divider = make_axes_locatable(axs)
    ax_cb = divider.append_axes("right", size="5%", pad="0%")

    ## Create a Mappable object
    cmap_sm = plt.cm.ScalarMappable(cmap=cmap, norm=colornorm)
    cmap_sm.set_array([])

    ## Create custom colorbar 
    cb1 = colorbar.Colorbar(ax_cb,
                            orientation='vertical', 
                            alpha=1,
                            mappable=cmap_sm
                            )
    cb1.ax.get_yaxis().labelpad = 6

    # Add colorbar to plot
    sf.add_axes(ax_cb)

    return sf 




def plot_component_planes(som, data, max_cols=3, figsize=(12, 8)):
    """
    Plots the component planes of the trained SOM.

    Parameters:
    --------------------------------------------------------
    som: MiniSom
        trained MiniSom object

    data: dataframe
        pandas DataFrame used to train the SOM
    
    max_cols: int, optional (default=3)
        maximum number of columns in the grid

    figsize: tuple, optional (default=(12, 8))
        figure size

    Returns:
    --------------------------------------------------------
    None
        Displays the component planes of the SOM for each feature in the data.
    """
    
    weights = som.get_weights()
    n_features = len(data.columns)

    n_cols = min(max_cols, n_features)
    n_rows = ceil(n_features / n_cols)

    fig = plt.figure(figsize=figsize, constrained_layout=True, dpi=128)
    subfigs = fig.subfigures(n_rows, n_cols).ravel()[:n_features]

    for cpi, sf in zip(range(n_features), subfigs):
        matrix_vals = weights[:, :, cpi]
        colornorm = mpl_colors.Normalize(vmin=np.min(matrix_vals), vmax=np.max(matrix_vals))

        sf = plot_hexagons(som, sf, 
                            colornorm,
                            matrix_vals,
                            label=data.columns[cpi],
                            cmap=cm.coolwarm)
    
    plt.show()



def plot_hitmap(som, data ):

    """
    Plots the hits map of the trained SOM.

    Parameters:
    --------------------------------------------------------
    som: MiniSom
        trained MiniSom object

    data: dataframe
        pandas DataFrame used to train the SOM

    Returns:
    --------------------------------------------------------
    None
        Displays the hits map of the SOM.

    """

    hitsmatrix = som.activation_response(data.values)

    fig = plt.figure(figsize=(20,15))

    colornorm = mpl_colors.Normalize(vmin=0, vmax=np.max(hitsmatrix))

    fig = plot_hexagons(som, fig, 
                        colornorm,
                        hitsmatrix,
                        label="SOM Hits Map",
                        cmap=cm.Greens,
                        annot=True
                        )



def plot_u_matrix(som):
    
    """
    Plots the U-Matrix of the trained SOM.

    Parameters:
    --------------------------------------------------------
    som: MiniSom
        trained MiniSom object

    Returns:
    --------------------------------------------------------
    None
        Displays the U-Matrix of the SOM.

    """

    umatrix = som.distance_map(scaling='mean')
    fig = plt.figure(figsize=(20,15))

    colornorm = mpl_colors.Normalize(vmin=np.min(umatrix), vmax=np.max(umatrix))

    fig = plot_hexagons(som, fig, 
                        colornorm,
                        umatrix,
                        label="U-matrix",
                        cmap=cm.RdYlBu_r,
                        annot=True
                        )




# ------------------------------------------------- AUTOENCODER ------------------------------------------------------ #
class AutoEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=32, latent_dim=5):
        super().__init__()
        # keeping the autoencoder simple but testing different model architectures
        # ENCODER: collects the original data and compresses it into a lower dimensional representation
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim)
        )

        # DECODER: tries to reconstruct the original data from the compressed representation
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )
    def forward(self, x):
        z = self.encoder(x) # create the latent space
        x_rebuilt = self.decoder(z) # reconstruct the input data
        return x_rebuilt
    





def train_autoencoder(X_scaled, device, hidden_dim=32, latent_dim=5, lr=1e-3, batch_size=256,
                      epochs=50, patience=8, verbose=False):


    '''
    Trains an autoencoder on the given scaled data.

    Parameters:
    --------------------------------------------------------
    X_scaled: numpy.ndarray or pandas.DataFrame
        Scaled input data for training the autoencoder.

    device: torch.device
        Device to run the training on (CPU or GPU).

    hidden_dim: int, optional (default=32)
        Number of neurons in the hidden layer of the autoencoder.

    latent_dim: int, optional (default=5)
        Dimensionality of the latent space representation.

    lr: float, optional (default=1e-3)
        Learning rate for the optimizer.

    batch_size: int, optional (default=256)
        Number of samples per batch during training.

    epochs: int, optional (default=50)
        Maximum number of training epochs.

    patience: int, optional (default=8)
        Number of epochs to wait for improvement before early stopping.

    verbose: bool, optional (default=False)
        If True, prints training progress.

    Returns:
    --------------------------------------------------------
    model: AutoEncoder
        Trained autoencoder model.

    best_loss: float
        Best training loss achieved during training.
    
    '''
    g = torch.Generator()
    g.manual_seed(1)

    X_t = torch.tensor(X_scaled, dtype=torch.float32)
    ds = TensorDataset(X_t, X_t)  # input == target (reconstrução)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, generator=g)

    model = AutoEncoder(input_dim=X_scaled.shape[1], hidden_dim=hidden_dim, latent_dim=latent_dim).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    best_loss = np.inf
    best_state = None
    no_improve = 0

    for epoch in range(epochs):
        model.train()
        total = 0.0

        for xb, yb in dl:
            xb = xb.to(device)
            yb = yb.to(device)

            optim.zero_grad()
            y_hat = model(xb)
            loss = loss_fn(y_hat, yb)
            loss.backward()
            optim.step()

            total += loss.item() * xb.size(0)

        epoch_loss = total / len(ds)

        if verbose:
            print(f"Epoch {epoch+1:03d} | loss={epoch_loss:.6f}")

        # early stopping
        if epoch_loss < best_loss - 1e-6:
            best_loss = epoch_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break

    # restore best
    model.load_state_dict(best_state)
    return model, best_loss




def encode(model, X_scaled, device):
    '''
    Encodes the scaled data using the trained autoencoder model.

    Parameters:
    --------------------------------------------------------
    model: AutoEncoder
        Trained autoencoder model.

    X_scaled: numpy.ndarray or pandas.DataFrame
        Scaled input data to be encoded.

    Returns:
    --------------------------------------------------------
    Z: numpy.ndarray
        Encoded latent representations of the input data.
    
    '''
    # Set the model to evaluation mode (disables dropout, batchnorm updates, etc.)
    model.eval()
    
    # Disable gradient computation for efficiency and memory savings
    with torch.no_grad():
        # Convert input data to a PyTorch tensor of type float32 and move to the target device (CPU or GPU)
        X_t = torch.tensor(X_scaled, dtype=torch.float32).to(device)
        
        # Pass the data through the encoder part of the autoencoder to get latent representations
        Z = model.encoder(X_t).cpu().numpy()  # Move back to CPU and convert to NumPy array
    
    # Return the encoded latent representations
    return Z






def deep_kmeans_search(df, features, device, k_list=(3, 4, 5, 6), latent_list=(2, 3, 5, 8), hidden_dim=32, lr=1e-3, batch_size=256,
    epochs=60, random_state=1, verbose=False):
    """
    Perform a grid search over KMeans and AutoEncoder + KMeans clustering models.

    The function compares:
    1) KMeans applied directly on the original feature space X
    2) KMeans applied on a latent space Z learned by an AutoEncoder

    For each configuration, clustering quality is evaluated using:
    - Silhouette score (computed in the space where clustering is performed)
    - R² (computed in the original feature space X)

    Parameters
    -----------------------------------------------------------------------
    df : pandas.DataFrame
        Input dataset.
    features : list of str
        Columns used for clustering.
    k_list : iterable of int
        Number of clusters to test.
    latent_list : iterable of int
        Latent dimensions to test for the AutoEncoder.
    hidden_dim : int
        Hidden layer size of the AutoEncoder.
    lr : float
        Learning rate for AutoEncoder training.
    batch_size : int
        Batch size for AutoEncoder training.
    epochs : int
        Maximum number of training epochs.
    random_state : int
        Random seed for KMeans.
    verbose : bool
        Whether to print progress messages.

    Returns
    ----------------------------------------------------------------------
    results : pandas.DataFrame
        One row per model configuration with evaluation metrics.
    best : dict
        Information about the best model according to silhouette score.
    labels_all : list of numpy.ndarray
        Cluster labels corresponding to each row of `results`.
        labels[i] matches results.iloc[i].
    """

    # --- Prepare data ---
    X = df[features].to_numpy().astype(np.float32)

    rows = []
    labels_all = []

    best = {"score": -np.inf, "index": None}

    
    # Baseline: KMeans on original feature space X
    
    for k in k_list:
        km = KMeans(n_clusters=k, n_init=10, random_state=random_state)
        lbls = km.fit_predict(X)

        sil = silhouette_score(X, lbls) if len(np.unique(lbls)) > 1 else np.nan
        r2 = get_rsq(df, features, lbls)

        rows.append({
            "model": "KMeans_X",
            "k": k,
            "latent_dim": None,
            "train_loss": None,
            "silhouette": sil,
            "r2": r2,
        })
        labels_all.append(lbls)


    # AutoEncoder + KMeans
   
    for latent_dim in latent_list:

        ae, ae_loss = train_autoencoder(
            X,
            device = device,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            lr=lr,
            batch_size=batch_size,
            epochs=epochs,
            patience=10,
            verbose=False,
        )

        Z = encode(ae, X, device = device)

        for k in k_list:
            km = KMeans(n_clusters=k, n_init=10, random_state=random_state)
            lbls = km.fit_predict(Z)

            sil = silhouette_score(Z, lbls) if len(np.unique(lbls)) > 1 else np.nan
            r2 = get_rsq(df, features, lbls)

            rows.append({
                "model": "AE+KMeans_Z",
                "k": k,
                "latent_dim": latent_dim,
                "train_loss": ae_loss,
                "silhouette": sil,
                "r2": r2,
            })
            labels_all.append(lbls)

            # Track best model (by silhouette)
            if not np.isnan(sil) and sil > best["score"]:
                best = {
                    "index": len(rows) - 1,
                    "score": sil,
                    "k": k,
                    "latent_dim": latent_dim,
                    "ae_loss": ae_loss,
                    "r2": r2,
                    "ae": ae,
                    "Z": Z,
                }

        if verbose:
            print(f"Finished latent_dim = {latent_dim}")

    results = pd.DataFrame(rows)

    return results, best, labels_all




# --------------------------------------- MERGING SEGMENTS ------------------------------------------------------ #

def compare_models(models_dict, df) :

    '''
    Compares multiple clustering models by computing R-squared scores and the number of clusters,
    and visualizes the results using side-by-side bar plots.

    Parameters:
    --------------------------------------------------------
    models_dict : dict
        Dictionary of models or clustering results, where keys are model names and values are
        arrays of cluster labels for each observation.
    
    df : pandas.DataFrame
        Original dataset used for computing the R-squared score. All columns are used as features.

    Returns:
    --------------------------------------------------------
    None
        Displays two bar plots:
        1. R-squared scores by model
        2. Number of clusters found by each model
    
    
    
    
    '''



    scores = []        # List to store R² scores for each model
    n_clusters = []    # List to store number of clusters for each model

    segment = df.columns.tolist()       # List of feature names
    model_names = list(models_dict.keys())

    # Compute R² score and number of clusters for each model
    for model_labels in models_dict.values():
        scores.append(get_rsq(df, segment, model_labels))
        n_clusters.append(len(np.unique(model_labels)))

    # Number of models
    n_models = len(model_names)
    x = np.arange(n_models)   # X-axis positions
    width = 0.25              # Width of bars

    # Generate colors from matplotlib plasma colormap
    colors = [cm.plasma(i / n_models) for i in range(n_models)]

    sns.set()

    # Create figure with 2 subplots
    f, (ax1, ax2) = plt.subplots(1, 2, facecolor='white', figsize=(22, 15))

    # Add a black border/frame to the figure
    f.patch.set_linewidth(5)
    f.patch.set_edgecolor('black')

    # ---------------- Plot R² scores ----------------
    ax1.set_facecolor("white")
    for i, (score, model_name, color) in enumerate(zip(scores, model_names, colors)):
        ax1.bar(x[i], score, width, label=model_name, color=color)

    ax1.set_ylim(0.0, 1.0)
    ax1.set_ylabel('Scores', fontsize=18)
    ax1.set_title('R-Squared score by Model', fontsize=20)
    ax1.set_xticks(x)
    ax1.set_xticklabels([])
    ax1.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=14)

    # ---------------- Plot Number of Clusters ----------------
    ax2.set_facecolor("white")
    for i, (cluster, model_name, color) in enumerate(zip(n_clusters, model_names, colors)):
        ax2.bar(x[i], cluster, width, label=model_name, color=color)

    ax2.set_ylabel('Clusters', fontsize=18)
    ax2.set_title('Number of Clusters by Model', fontsize=20)
    ax2.set_xticks(x)
    ax2.set_xticklabels([])
    ax2.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=14)

    # Adjust layout and display plots
    f.tight_layout()
    plt.show()






#-------------------------------------------------- PROFILLING ------------------------------------------------------ #

def radar_chart(df, ax=None, title=None):
    """
    Creates a radar chart showing normalized feature averages
    for each cluster.

    Parameters:
    ----------------------------
    df : pandas DataFrame
        DataFrame containing feature columns and a 'cluster_labels' column.


    Returns:
    ----------------------------
    fig, ax : matplotlib Figure and Axes objects
        The radar chart figure and axes.
    """

    # Compute mean values per cluster
    cluster_means = (
        df
        .groupby("cluster_labels")
        .mean()
        .reset_index()
    )

    # Select only feature columns (exclude cluster label)
    feature_names = [col for col in cluster_means.columns if col != "cluster_labels"]

    # Apply Min-Max normalization to features only
    normalized_features = (
        cluster_means[feature_names] - cluster_means[feature_names].min()
    ) / (
        cluster_means[feature_names].max() - cluster_means[feature_names].min()
    )

    # Number of variables in the radar chart
    n_features = len(feature_names)

    # Compute angles for each axis
    angles = np.linspace(0, 2 * np.pi, n_features, endpoint=False).tolist()
    angles += angles[:1]  # Close the circle

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={"polar": True})
    else:
        fig = ax.figure

    # Plot one radar shape per cluster
    for idx in range(len(cluster_means)):
        values = normalized_features.iloc[idx].tolist()
        values += values[:1]  # Close the polygon

        ax.plot(
            angles,
            values,
            linewidth=2,
            label=f"Cluster {cluster_means.loc[idx, 'cluster_labels']}"
        )
        ax.fill(angles, values, alpha=0.25)

    # Set feature labels on the angular axis
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(feature_names)

    # Configure radial axis
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], color="grey", fontsize=9)

    # Add title
    ax.set_title(title if title is not None else "Radar Chart – Normalized Feature Means by Cluster", y=1.08)

    # Add legend
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1))

    return fig, ax





def plot_parallel_coordinates_from_labels(df, features, title="Cluster Profiling - Parallel Coordinates"):
    """
    Plots parallel coordinates for cluster mean profiles.

    Parameters:
    -------------------------------------------------------------
    df: datafrane
        dataframe with features and 'cluster_labels' column


    features: list
        list of features to include

    title: str 
        plot title

    Returns:
    -------------------------------------------------------------
    None
        Displays a parallel coordinates plot of cluster mean profiles.
    """
    
    
    # Compute cluster means
    cluster_profile = df[features + ['cluster_labels']].groupby('cluster_labels').mean().T
    
    # Prepare dataframe for plotting
    cluster_profile2 = cluster_profile.T.reset_index()
    cluster_profile2['cluster_labels'] = cluster_profile2['cluster_labels'].astype(str)
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 5))
    parallel_coordinates(cluster_profile2, 'cluster_labels', colormap='tab10', linewidth=2, ax=ax)
    ax.set_title(title, fontsize=14)
    ax.axhline(0, linestyle='-.', alpha=0.5)
    plt.xticks(rotation=70, ha='right')
    plt.tight_layout()
    plt.show()




def plot_stacked_bars(df, features, top_n=5, n_cols=2, figsize_per_plot=(7, 5)):
    """
    Plots normalized stacked bar charts by cluster for multiple categorical variables.
    The cluster column is assumed to be named 'cluster_labels'.

    Parameters
    ------------------------------------------------------
    df : pd.DataFrame
        DataFrame containing the data.

    features : list
        List of categorical variables to be plotted (e.g. locations).

    top_n : int, default=5
        Number of most frequent categories (globally) to display per variable.

    n_cols : int, default=2
        Number of columns in the subplot grid.

    figsize_per_plot : tuple, default=(7, 5)
        Base figure size for each subplot (width, height).

    Returns
    -----------------------------------------------------------
    None
        Displays stacked bar charts for each categorical variable.

    """

    # Total number of plots
    n_plots = len(features)

    # Compute number of rows needed given the number of columns
    n_rows = ceil(n_plots / n_cols)

    # Create subplot grid
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(figsize_per_plot[0] * n_cols,
                 figsize_per_plot[1] * n_rows)
    )


    # Flatten axes array for easier indexing
    axes = np.atleast_1d(axes).flatten()

    for i, var in enumerate(features):
        # Create a contingency table: clusters x categories
        ct = pd.crosstab(df['cluster_labels'], df[var])

        # Select the top N most frequent categories globally
        top_categories = ct.sum(axis=0).nlargest(top_n).index
        ct_top = ct[top_categories]

        # Normalize counts by cluster to obtain percentages
        ct_pct = ct_top.div(ct_top.sum(axis=1), axis=0) * 100

        # Plot stacked bar chart
        ct_pct.plot(
            kind='bar',
            stacked=True,
            ax=axes[i],
            colormap='plasma'
        )

        # Axis formatting
        axes[i].set_title(f'Top {top_n} {var} by Cluster')
        axes[i].set_xlabel('Cluster')
        axes[i].set_ylabel('Percentage (%)')
        axes[i].set_xticklabels(axes[i].get_xticklabels(), rotation=0)

        # Place legend outside the plot area
        axes[i].legend(title=var, bbox_to_anchor=(1.05, 1), loc='upper left')

    # Remove unused axes if the grid is larger than the number of variables
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    # Add a global title 
    fig.suptitle( f"Top {top_n} {', '.join(features)} per Cluster", fontsize=16, y=1.02)

    # Adjust layout to avoid overlapping elements
    plt.tight_layout()
    plt.show()






def plot_multiple_stacked_bars(df, features, top_n=5, n_cols=3, figsize_per_bar=(4, 6)):
    """
    Plots multiple vertical 100% stacked bars showing the global distribution 
    of categorical variables. Each bar represents one variable, with segments 
    colored by the top N most frequent categories.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the data to be plotted.
    
    features : list of str
        List of categorical variable names (column names) to plot. Each variable will be represented by one vertical stacked bar.
    
    top_n : int, default=5
        Number of most frequent categories to display per variable. Only the top N categories (by count) will be shown in each bar.
    
    n_cols : int, default=3
        Maximum number of bars (subplots) to display per row.

    
    figsize_per_bar : tuple of (float, float), default=(4, 6)
        Base size for each individual bar subplot as (width, height).


    Returns
    -------
    None
        Displays the matplotlib figure with stacked bars.


    """
    
    # Calculate number of rows needed based on features and columns
    n_plots = len(features)
    n_rows = ceil(n_plots / n_cols)
    
    # Calculate actual number of columns (in case n_plots < n_cols)
    actual_cols = min(n_plots, n_cols)
    
    # Create figure with subplots
    fig, axes = plt.subplots(
        n_rows, 
        actual_cols,
        figsize=(figsize_per_bar[0] * actual_cols, figsize_per_bar[1] * n_rows),
        sharey=True  # Share y-axis across all subplots
    )
    
    # Ensure axes is always a flat array for consistent indexing
    axes = np.atleast_1d(axes).flatten()
    
    # Iterate through each feature to create stacked bars
    for idx, var in enumerate(features):
        # Get top N most frequent categories for this variable
        top_categories = df[var].value_counts().head(top_n)
        
        # Calculate percentages (sum = 100%)
        percentages = (top_categories / top_categories.sum()) * 100
        
        # Generate colors using plasma colormap
        colors = plt.cm.plasma(np.linspace(0, 1, len(percentages)))
        
        # Initialize bottom position for stacking
        bottom = 0
        
        # Plot each category as a segment in the stacked bar
        for i, (category, pct) in enumerate(percentages.items()):
            # Add bar segment
            axes[idx].bar(
                0,                    # x-position (single bar at center)
                pct,                  # height of this segment
                bottom=bottom,        # starting y-position
                color=colors[i],      # color for this category
                label=category,       # legend label
                width=0.6,           # bar width
                edgecolor='white',   # white border between segments
                linewidth=1
            )
            
            # Add percentage text in the center of each segment
            axes[idx].text(
                0,                    # x-position (center of bar)
                bottom + pct/2,       # y-position (middle of segment)
                f'{pct:.1f}%',       # formatted percentage text
                ha='center',          # horizontal alignment
                va='center',          # vertical alignment
                fontsize=10, 
                color='white', 
                weight='bold'
            )
            
            # Update bottom position for next segment
            bottom += pct
        
        # Configure subplot appearance
        axes[idx].set_ylim(0, 100)                    # y-axis from 0 to 100%
        axes[idx].set_title(f'{var}', fontsize=12)    # subplot title
        axes[idx].set_xticks([])                      # remove x-axis ticks
        
        # Add legend to the right of the bar
        axes[idx].legend(
            bbox_to_anchor=(1.05, 1),   # position outside plot area
            loc='upper left',             # anchor point
            fontsize=9,
            frameon=True
        )
        
        # Add y-axis label only to leftmost column
        if idx % actual_cols == 0:
            axes[idx].set_ylabel('Percentage (%)', fontsize=11)
    
    # Remove unused subplots if grid is larger than number of features
    for j in range(idx + 1, len(axes)):
        fig.delaxes(axes[j])
    
    # Add overall title to the figure
    fig.suptitle(
        f"Global Distribution - Top {top_n} of {', '.join(features)}", 
        fontsize=16, 
        y=0.98
    )
    
    # Adjust layout to prevent overlapping elements
    plt.tight_layout()
    plt.show()



# -------------------------------------------------- FEATURE IMPORTANCE AND RECLASSIFY OUTLIERS ------------------------------------------------------ #

def get_ss_variables(df):
    """
    Get the SS for each variable

    Parameters:
    --------------------------------------------------------
    df : pandas.DataFrame
        DataFrame containing the data.

    Returns:
    --------------------------------------------------------
    ss_vars : pandas.Series
        Series containing the sum of squares for each variable.
    """
    ss_vars = df.var() * (df.count() - 1)
    return ss_vars


def r2_variables(df, labels):
    """
    Get the R² for each variable

    Parameters:
    --------------------------------------------------------
    df : pandas.DataFrame
        DataFrame containing the data.

    labels : array-like
        Cluster labels for each observation.

    Returns:
    --------------------------------------------------------
    r2_vars : pandas.Series
        Series containing the R² for each variable.
    """
    sst_vars = get_ss_variables(df)
    ssw_vars = np.sum(df.groupby(labels).apply(get_ss_variables))
    return 1 - ssw_vars/sst_vars