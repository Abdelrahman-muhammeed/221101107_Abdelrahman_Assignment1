import pandas as pd
import os
import numpy as np
from scipy.spatial.distance import cosine
from scipy.stats import pearsonr

def preprocess_ratings(file_path, keep='first'):
    try:
        # Read the CSV file
        print(f"Attempting to read file from: {file_path}")
        df = pd.read_csv(file_path)
        
        # Remove duplicates
        df_clean = df.drop_duplicates(keep=keep)
        
        # Fill missing values with 0
        df_clean = df_clean.fillna(0)
        
        # Print info about the preprocessing
        print(f"Original number of rows: {len(df)}")
        print(f"Number of rows after removing duplicates: {len(df_clean)}")
        print(f"Number of missing values filled: {df.isna().sum().sum()}")
        
        return df_clean
        
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return None
    except Exception as e:
        print(f"Error: {str(e)}")
        return None

def create_user_item_matrix(df, n_users=5, n_items=5):
    """
    Create a matrix of n users who have rated the same n items
    """
    try:
        # Count how many ratings each item has
        item_counts = df['movieId'].value_counts()
        
        # Get items that have been rated by at least n_users
        common_items = item_counts[item_counts >= n_users].index[:n_items]
        
        # Filter dataframe to only include these items
        df_common = df[df['movieId'].isin(common_items)]
        
        # Count how many items each user has rated
        user_item_counts = df_common.groupby('userId').size()
        
        # Get users who have rated all n_items
        eligible_users = user_item_counts[user_item_counts >= n_items].index[:n_users]
        
        # Create the final filtered dataframe
        df_final = df_common[df_common['userId'].isin(eligible_users)]
        
        # Create the matrix
        rating_matrix = df_final.pivot(index='userId', columns='movieId', values='rating')
        rating_matrix = rating_matrix.iloc[:n_users, :n_items]
        
        print("\nUser-Item Rating Matrix:")
        print(rating_matrix)
        
        return rating_matrix, df_final
        
    except Exception as e:
        print(f"Error creating matrix: {str(e)}")
        return None, None

def compute_similarity_measures(matrix, method='user'):
    """
    Compute similarity measures for either users or items
    """
    if method not in ['user', 'item']:
        raise ValueError("Method must be either 'user' or 'item'")
    
    # If item-based, transpose the matrix
    if method == 'item':
        matrix = matrix.T
    
    n = len(matrix)
    cosine_sim = np.zeros((n, n))
    pearson_sim = np.zeros((n, n))
    
    for i in range(n):
        for j in range(n):
            # Compute cosine similarity
            if np.any(matrix.iloc[i]) and np.any(matrix.iloc[j]):  # Check for non-zero vectors
                cosine_sim[i][j] = 1 - cosine(matrix.iloc[i], matrix.iloc[j])
            
            # Compute Pearson correlation
            try:
                pearson_sim[i][j], _ = pearsonr(matrix.iloc[i], matrix.iloc[j])
            except:
                pearson_sim[i][j] = 0
    
    # Convert to DataFrames
    index = matrix.index
    cosine_df = pd.DataFrame(cosine_sim, index=index, columns=index)
    pearson_df = pd.DataFrame(pearson_sim, index=index, columns=index)
    
    return cosine_df, pearson_df

def find_peer_groups(similarity_matrix, threshold=0.5):
    """
    Find peer groups based on similarity threshold
    """
    peer_groups = {}
    for idx in similarity_matrix.index:
        # Find similar users/items above threshold
        peers = similarity_matrix[idx][similarity_matrix[idx] > threshold].index.tolist()
        # Remove self from peer group
        peers = [peer for peer in peers if peer != idx]
        if peers:  # Only add if there are peers
            peer_groups[idx] = peers
    
    return peer_groups

# Example usage
if __name__ == "__main__":
    # Use raw string (r prefix) and backslashes for Windows file paths
    file_path = r"C:\Users\manam\OneDrive\Desktop\IRS\Assignment 1\ml-25m\ratings.csv"
    
    # Preprocess the data
    processed_df = preprocess_ratings(file_path, keep='first')
    
    if processed_df is not None:
        # Create user-item matrix
        rating_matrix, filtered_df = create_user_item_matrix(processed_df)
        
        if rating_matrix is not None:
            # Compute user-based similarities
            print("\nComputing User-Based Similarities...")
            user_cosine_sim, user_pearson_sim = compute_similarity_measures(rating_matrix, method='user')
            
            print("\nUser Cosine Similarity Matrix:")
            print(user_cosine_sim)
            print("\nUser Pearson Correlation Matrix:")
            print(user_pearson_sim)
            
            # Compute item-based similarities
            print("\nComputing Item-Based Similarities...")
            item_cosine_sim, item_pearson_sim = compute_similarity_measures(rating_matrix, method='item')
            
            print("\nItem Cosine Similarity Matrix:")
            print(item_cosine_sim)
            print("\nItem Pearson Correlation Matrix:")
            print(item_pearson_sim)
            
            # Find peer groups (threshold can be adjusted)
            user_peers_cosine = find_peer_groups(user_cosine_sim, threshold=0.5)
            user_peers_pearson = find_peer_groups(user_pearson_sim, threshold=0.5)
            item_peers_cosine = find_peer_groups(item_cosine_sim, threshold=0.5)
            item_peers_pearson = find_peer_groups(item_pearson_sim, threshold=0.5)
            
            print("\nUser Peer Groups (Cosine Similarity):")
            print(user_peers_cosine)
            print("\nUser Peer Groups (Pearson Correlation):")
            print(user_peers_pearson)
            print("\nItem Peer Groups (Cosine Similarity):")
            print(item_peers_cosine)
            print("\nItem Peer Groups (Pearson Correlation):")
            print(item_peers_pearson)
            
            # Save results
            output_dir = os.path.dirname(file_path)
            user_cosine_sim.to_csv(os.path.join(output_dir, 'user_cosine_similarity.csv'))
            user_pearson_sim.to_csv(os.path.join(output_dir, 'user_pearson_similarity.csv'))
            item_cosine_sim.to_csv(os.path.join(output_dir, 'item_cosine_similarity.csv'))
            item_pearson_sim.to_csv(os.path.join(output_dir, 'item_pearson_similarity.csv'))
            
            print("\nSimilarity matrices have been saved to CSV files.")