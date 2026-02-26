import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import warnings

# customer segmentation model using K-Means

warnings.filterwarnings('ignore')

def load_and_preprocess():
    print("Loading data...")
    df = pd.read_csv('final_outlierTreated.csv')
    
    # Selecting numeric features as requested
    features = ['Recency', 'Monetary', 'payment_value', 'price', 'freight_value']
    X = df[features]
    
    print(f"Features selected: {features}")
    print(f"Data shape: {X.shape}")
    
    # Scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X, X_scaled

def find_optimal_k(X_scaled):
    print("\nCalculating silhouette scores for k=2 to 10...")
    silhouette_scores = []
    k_range = range(2, 11)
    
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
        cluster_labels = kmeans.fit_predict(X_scaled)
        score = silhouette_score(X_scaled, cluster_labels)
        silhouette_scores.append(score)
        print(f"k={k}, Silhouette Score: {score:.4f}")
        
    optimal_k = k_range[np.argmax(silhouette_scores)]
    print(f"\nOptimal number of clusters based on highest silhouette score: {optimal_k}")
    
    return optimal_k, silhouette_scores

def fit_and_visualize(X, X_scaled, optimal_k):
    print(f"Fitting KMeans with k={optimal_k}...")
    kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init='auto')
    clusters = kmeans.fit_predict(X_scaled)
    X['Cluster'] = clusters
    
    # PCA for 2D visualization
    print("Performing PCA for 2D visualization...")
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(X_scaled)
    X['PCA1'] = pca_result[:, 0]
    X['PCA2'] = pca_result[:, 1]
    
    # Plotting
    plt.figure(figsize=(12, 8))
    sns.scatterplot(x='PCA1', y='PCA2', hue='Cluster', data=X, palette='viridis', alpha=0.6)
    plt.title(f'Customer Segments (k={optimal_k}) - PCA Projection')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.grid(True)
    plt.savefig('customer_clusters.png')
    print("Saved cluster visualization to customer_clusters.png")
    
    # Summary of clusters
    print("\nCluster Summary (Mean values):")
    summary = X.groupby('Cluster')[['Recency', 'Monetary', 'payment_value', 'price', 'freight_value']].mean()
    print(summary)
    
    return X, summary

if __name__ == "__main__":
    X, X_scaled = load_and_preprocess()
    
    # Find optimal k
    optimal_k, scores = find_optimal_k(X_scaled)
    
    # Fit and visualize
    X_final, cluster_summary = fit_and_visualize(X, X_scaled, optimal_k)
