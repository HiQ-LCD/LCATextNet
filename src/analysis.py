# -*- coding: utf-8 -*-
# @Time    : 2024/12/18 18:11
# @Author  : Biao
# @File    : analysis.py

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd
from .config import WORK_DIR
import numpy as np
from decimal import Decimal, getcontext
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.metrics import davies_bouldin_score, calinski_harabasz_score
from scipy.spatial.distance import pdist, squareform
from sklearn.neighbors import NearestNeighbors
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

common_columns = ['sector']

target_columns= [
    "Activity Name",
    "Reference Product Name",
    "CPC Classification",
    "Product Information",
    "SystemBoundary",
    "generalComment",
    "technologyComment"
]

data_path = os.path.join(WORK_DIR, 'data', 'ecoinvent', 'gwp_train_data.xlsx')


impact_list = ['acidification', 'climate change', 'climate change - biogenic', 'climate change - fossil', 'climate change - land use and land use change', 'ecotoxicity - freshwater', 'ecotoxicity - freshwater, inorganics', 'ecotoxicity - freshwater, organics', 'energy resources - non-renewable', 'eutrophication - freshwater', 'eutrophication - marine', 'eutrophication - terrestrial', 'human toxicity - carcinogenic', 'human toxicity - carcinogenic, inorganics', 'human toxicity - carcinogenic, organics', 'human toxicity - non-carcinogenic', 'human toxicity - non-carcinogenic, inorganics', 'human toxicity - non-carcinogenic, organics', 'ionising radiation - human health', 'land use', 'material resources - metals&minerals', 'ozone depletion', 'particulate matter formation', 'photochemical oxidant formation - human health', 'water use']


# Set Decimal precision
getcontext().prec = 28  # Can be adjusted based on needs

def analysis_dataset(data_path):
    stopwords_path = os.path.join(WORK_DIR, 'data', 'stopwords','english')

    STOP_WORDS = set(open(stopwords_path, 'r').read().splitlines())
    # Analyze text statistical features of target fields by sector
    # Analysis dimensions include:
    # 1. Field length distribution: mean, standard deviation
    # 2. Vocabulary distribution: word count, word length distribution
    data = pd.read_excel(data_path)
    data["SystemBoundary"] = data.apply(lambda x: f"{x['includedActivitiesStart']} {x['includedActivitiesEnd']}", axis=1)
    required_columns = common_columns + target_columns
    data = data[required_columns]
    stats = {}
    
    sector_counts = data['sector'].value_counts()
    print("\nSector data counts:")
    for sector, count in sector_counts.items():
        print(f"{sector}: {count} records")
    
    # Analysis by sector
    for sector in data['sector'].unique():
        sector_data = data[data['sector'] == sector]
        sector_stats = {}
        # record count
        sector_stats['record_count'] = len(sector_data)
        # analyze each target field
        for column in target_columns:
            column_stats = {}
            # 1. field length distribution
            lengths = sector_data[column].str.len()
            column_stats['avg_length'] = lengths.mean()
            column_stats['std_length'] = lengths.std()
            # 2. vocabulary distribution
            # split text and count different words
            unique_words = set()
            word_counts = []
            word_lengths = []
            
            for text in sector_data[column].dropna():
                words = text.split()
                unique_words.update(words)  # add unique words
                word_counts.append(len(words))
                word_lengths.extend([len(word) for word in words])
            
            column_stats['unique_word_count'] = len(unique_words)  # 不同词汇的数量
            column_stats['avg_word_count'] = sum(word_counts) / len(word_counts) if word_counts else 0
            column_stats['std_word_count'] = pd.Series(word_counts).std()
            column_stats['avg_word_length'] = sum(word_lengths) / len(word_lengths) if word_lengths else 0
            column_stats['std_word_length'] = pd.Series(word_lengths).std()
            
            # save the most common words (top 10) - modified to filter stopwords
            word_freq = {}
            for text in sector_data[column].dropna():
                for word in text.split():
                    # Skip stopwords and words shorter than 3 characters
                    if word.lower() not in STOP_WORDS and len(word) > 2:
                        word_freq[word] = word_freq.get(word, 0) + 1
            
            top_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:10]
            column_stats['top_words'] = ', '.join([f"{word}({freq})" for word, freq in top_words])
            
            sector_stats[column] = column_stats
            
        stats[sector] = sector_stats
    
    # convert statistics to DataFrame for display
    results = []
    for sector, sector_stats in stats.items():
        # add record count of sector
        record_count = sector_stats.pop('record_count')
        for column, column_stats in sector_stats.items():
            row = {
                'sector': sector,
                'record_count': record_count,
                'column': column,
                **column_stats
            }
            results.append(row)
    
    results_df = pd.DataFrame(results)
    
    # save analysis results
    output_path = os.path.join(os.path.dirname(data_path), 'text_analysis_results.xlsx')
    results_df.to_excel(output_path, index=False)
    
    return results_df


def impact_analysis(folder_path):
    train_data = pd.read_excel(data_path)
    
    
    for impact in impact_list:
        print(f"\nProcessing {impact}...")
        file_name = f"{impact}.csv"
        file_path = os.path.join(folder_path, file_name)
        
        try:
            data = pd.read_csv(file_path)
            
            # Data preprocessing: remove null values and zeros
            train_data_clean = train_data.dropna(subset=[impact, 'sector'])
            train_data_clean = train_data_clean[train_data_clean[impact] != 0]
            
            data_clean = data.dropna(subset=['true_value'])
            data_clean = data_clean[data_clean['true_value'] != 0]
            
            # Convert to float64 for higher precision
            train_values = train_data_clean[impact].astype(np.float64)
            true_values = data_clean['true_value'].astype(np.float64)
            
            def find_best_match(value):
                # calculate relative error: |a-b|/|b|
                relative_errors = np.abs(train_values - value) / np.abs(value)
                min_error_idx = np.argmin(relative_errors)
                min_error = relative_errors[min_error_idx]
                
                # get sector with minimum error
                best_match_sector = train_data_clean['sector'].iloc[min_error_idx]
                
                # if minimum relative error exceeds 1%, return None
                if min_error > 0.01:
                    print(f"Warning: Best match for value {value} has relative error {min_error*100:.4f}%")
                    return None
                
                return {
                    'sector': best_match_sector,
                    'relative_error': min_error,
                    'matched_value': train_values[min_error_idx]
                }
            
            # apply matching function and collect detailed information
            matches = []
            for idx, value in true_values.items():
                match_result = find_best_match(value)
                if match_result is not None:
                    matches.append({
                        'index': idx,
                        'test_value': value,
                        **match_result
                    })
            
            # convert matching results to DataFrame for analysis
            match_df = pd.DataFrame(matches)
            
            # initialize sector column
            data['sector'] = None
            data['relative_error'] = None
            
            # fill matching results
            if not match_df.empty:
                for _, row in match_df.iterrows():
                    data.loc[row['index'], 'sector'] = row['sector']
                    data.loc[row['index'], 'relative_error'] = row['relative_error']
            
            # print matching statistics
            total_rows = len(data)
            matched_rows = data['sector'].notna().sum()
            
            print(f"\nMatching statistics for {impact}:")
            print(f"Total rows (excluding zeros): {total_rows}")
            print(f"Matched rows: {matched_rows}")
            print(f"Match rate: {(matched_rows/total_rows)*100:.2f}%")
            
            if matched_rows > 0:
                # print relative error distribution
                errors = data['relative_error'].dropna()
                print("\nRelative error distribution:")
                print(f"Min error: {errors.min()*100:.4f}%")
                print(f"Max error: {errors.max()*100:.4f}%")
                print(f"Mean error: {errors.mean()*100:.4f}%")
                print(f"Median error: {errors.median()*100:.4f}%")
                
                # print matching count by error range
                error_ranges = [0.0001, 0.0005, 0.001, 0.005, 0.01]
                print("\nMatches by error range:")
                prev_threshold = 0
                for threshold in error_ranges:
                    count = ((errors > prev_threshold) & (errors <= threshold)).sum()
                    print(f"{prev_threshold*100:.3f}% - {threshold*100:.3f}%: {count} matches")
                    prev_threshold = threshold
            
            # display matched values
            unmatched = data[data['sector'].isna()]
            if len(unmatched) > 0:
                print(f"\nSample of unmatched values:")
                print(unmatched['true_value'].head())
            
            # save results
            data.to_csv(file_path, index=False)
            print(f"Results saved to {file_path}")
            
        except Exception as e:
            print(f"Error processing {impact}: {str(e)}")
            continue

def impact_result_analysis_by_sector(folder_path):
    # Initialize results list to store all metrics
    all_results = []
    
    # Process each impact file
    for impact in impact_list:
        print(f"\nAnalyzing {impact}...")
        file_path = os.path.join(folder_path, f"{impact}.csv")
        
        try:
            # Read the data
            data = pd.read_csv(file_path)
            
            # Skip if no sector information or predictions
            if 'sector' not in data.columns or 'prediction' not in data.columns:
                print(f"Skipping {impact}: Missing required columns")
                continue
            
            # Group by sector and calculate metrics
            for sector in data['sector'].dropna().unique():
                sector_data = data[data['sector'] == sector]
                
                # Get true values and predictions
                y_true = sector_data['true_value']
                y_pred = sector_data['prediction']
                
                # Calculate metrics
                metrics = {
                    'impact': impact,
                    'sector': sector,
                    'sample_count': len(sector_data),
                    'r2': r2_score(y_true, y_pred) if len(y_true) > 1 else None,
                    'mse': mean_squared_error(y_true, y_pred),
                    'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
                    'mape': np.mean(np.abs((y_true - y_pred) / y_true)) * 100
                }
                
                all_results.append(metrics)
                
                # Print progress
                print(f"\nSector: {sector}")
                print(f"Samples: {metrics['sample_count']}")
                print(f"R²: {metrics['r2']:.4f}")
                print(f"RMSE: {metrics['rmse']:.4f}")
                print(f"MAPE: {metrics['mape']:.2f}%")
        
        except Exception as e:
            print(f"Error processing {impact}: {str(e)}")
            continue
    
    # Convert results to DataFrame
    results_df = pd.DataFrame(all_results)
    
    # Save results
    output_path = os.path.join(folder_path, 'sector_analysis_results.xlsx')
    results_df.to_excel(output_path, index=False)
    print(f"\nResults saved to {output_path}")
    
    return results_df


def read_vectors():
    from configs.model_config import MODEL_ROOT_PATH,DATA_PATH
    import joblib
    text_columns = [
            "Activity Name", "Reference Product Name", "CPC Classification",
            "Product Information","SystemBoundary",
            "generalComment",
            "technologyComment",
        ]
    for text_column in text_columns:
        vectors = joblib.load(os.path.join(MODEL_ROOT_PATH, f"{text_column}_embedding.joblib"))
        print(f"{text_column}: {len(vectors)}")


def analysis_vector_by_sector():
    from configs.model_config import MODEL_ROOT_PATH, DATA_PATH
    import joblib
    import numpy as np
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    image_output_path = os.path.join(DATA_PATH,'image')
    
    def get_color_list(n):
        # Generate color list for visualization
        # Returns base colors if n is small enough, otherwise generates a color gradient
        base_colors = [
            '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
            '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
            '#393b79', '#637939', '#8c6d31', '#843c39', '#7b4173'
        ]
        if n <= len(base_colors):
            return base_colors[:n]
        
        from matplotlib.colors import LinearSegmentedColormap
        cmap = LinearSegmentedColormap.from_list('custom', base_colors)
        return [mcolors.rgb2hex(cmap(i)) for i in np.linspace(0, 1, n)]

    def calculate_metrics(vectors, sector_index_dict, sector_list):
        """
        Calculate all metrics for given vectors
        
        Parameters:
        - vectors: embedding vectors
        - sector_index_dict: dictionary mapping sectors to their indices
        - sector_list: list of all sectors
        
        Returns:
        - Dictionary containing intra-sector similarity, inter-sector distances,
          and clustering metrics
        - Labels array for visualization
        """
        results = {
            'intra_sector_similarity': {},
            'inter_sector_distances': [],
            'clustering_metrics': {}
        }
        
        # Calculate intra-sector metrics
        for sector in sector_list:
            sector_vectors = vectors[np.array(sector_index_dict[sector])]
            if len(sector_vectors) > 1:
                similarity_matrix = cosine_similarity(sector_vectors)
                upper_tri = similarity_matrix[np.triu_indices_from(similarity_matrix, k=1)]
                distances = pdist(sector_vectors)
                
                results['intra_sector_similarity'][sector] = {
                    'mean': np.mean(upper_tri),
                    'std': np.std(upper_tri),
                    'samples': len(sector_vectors),
                    'compactness': np.mean(distances),
                    'max_distance': np.max(distances)
                }
        
        # Calculate inter-sector metrics
        for i, sector1 in enumerate(sector_list):
            for sector2 in sector_list[i+1:]:
                vectors1 = vectors[np.array(sector_index_dict[sector1])]
                vectors2 = vectors[np.array(sector_index_dict[sector2])]
                mean_similarity = np.mean(cosine_similarity(vectors1, vectors2))
                results['inter_sector_distances'].append({
                    'pair': f"{sector1} vs {sector2}",
                    'similarity': mean_similarity
                })
        
        # Calculate clustering metrics
        labels = np.zeros(len(vectors))
        for i, sector in enumerate(sector_list):
            for idx in sector_index_dict[sector]:
                labels[idx] = i
                
        try:
            results['clustering_metrics'] = {
                'silhouette_score': silhouette_score(vectors, labels),
                'davies_bouldin_score': davies_bouldin_score(vectors, labels),
                'calinski_harabasz_score': calinski_harabasz_score(vectors, labels)
            }
        except Exception as e:
            print(f"Error calculating clustering metrics: {e}")
            
        return results, labels

    def create_visualizations(vectors, labels, sector_list, title_prefix, output_path):
        """
        Create visualizations for vector analysis
        
        Parameters:
        - vectors: embedding vectors to visualize
        - labels: sector labels for each vector
        - sector_list: list of all sectors
        - title_prefix: prefix for plot titles
        - output_path: path to save visualization files
        """
        # t-SNE visualization
        plt.figure(figsize=(12, 8))
        tsne = TSNE(n_components=2, random_state=42)
        vectors_2d = tsne.fit_transform(vectors)
        
        colors = get_color_list(len(sector_list))
        for i, sector in enumerate(sector_list):
            mask = labels == i
            plt.scatter(vectors_2d[mask, 0], vectors_2d[mask, 1],
                       label=sector, color=colors[i], alpha=0.6)
        
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.title(f't-SNE Visualization - {title_prefix}')
        plt.tight_layout()
        plt.savefig(os.path.join(output_path, f'{title_prefix}_tsne.png'), bbox_inches='tight')
        plt.close()
        
        # Distance distributions
        plt.figure(figsize=(15, 10))
        for i, sector in enumerate(sector_list):
            sector_vectors = vectors[labels == i]
            if len(sector_vectors) > 1:
                distances = pdist(sector_vectors)
                plt.subplot(3, (len(sector_list) + 2) // 3, i + 1)
                plt.hist(distances, bins=30, density=True, alpha=0.7)
                plt.title(f'{sector}\n(n={len(sector_vectors)})')
                plt.xlabel('Distance')
                if i % 3 == 0:
                    plt.ylabel('Density')
        
        plt.suptitle(f'Distance Distributions - {title_prefix}')
        plt.tight_layout()
        plt.savefig(os.path.join(output_path, f'{title_prefix}_distances.png'), bbox_inches='tight')
        plt.close()

    # Main execution flow
    train_data = pd.read_excel(data_path)
    sector_list = train_data['sector'].dropna().unique().tolist()
    sector_index_dict = {sector: train_data[train_data['sector'] == sector].index.tolist() 
                        for sector in sector_list}
    
    text_columns = [
        "Activity Name", "Reference Product Name", "CPC Classification",
        "Product Information", "SystemBoundary", "generalComment",
        "technologyComment"
    ]
    
    # Process individual columns
    results = {}
    for text_column in text_columns:
        vectors = np.array(joblib.load(os.path.join(MODEL_ROOT_PATH, f"{text_column}_embedding.joblib")))
        print(f"\nAnalyzing {text_column} embeddings...")
        
        results[text_column], labels = calculate_metrics(vectors, sector_index_dict, sector_list)
        create_visualizations(vectors, labels, sector_list, text_column, image_output_path)
    
    # Process concatenated vectors
    print("\nAnalyzing concatenated embeddings...")
    concatenated_vectors = np.hstack([
        joblib.load(os.path.join(MODEL_ROOT_PATH, f"{col}_embedding.joblib"))
        for col in text_columns
    ])
    
    results['concatenated'], concat_labels = calculate_metrics(
        concatenated_vectors, sector_index_dict, sector_list)
    create_visualizations(concatenated_vectors, concat_labels, sector_list, 
                         'concatenated', image_output_path)
    
    # Save results to Excel
    save_results_to_excel(results, text_columns, image_output_path)
    
    return results

def save_results_to_excel(results, text_columns, output_path):
    """
    Save analysis results to Excel file
    
    Parameters:
    - results: dictionary containing analysis results
    - text_columns: list of text columns analyzed
    - output_path: path to save the Excel file
    """
    with pd.ExcelWriter(os.path.join(output_path, 'sector_embedding_analysis.xlsx')) as writer:
        # Summary sheet
        summary_data = []
        for col, data in results.items():
            metrics = data['clustering_metrics']
            intra_similarities = data['intra_sector_similarity']
            inter_distances = data['inter_sector_distances']
            
            summary_data.append({
                'text_column': col,
                'silhouette_score': metrics.get('silhouette_score'),
                'davies_bouldin_score': metrics.get('davies_bouldin_score'),
                'calinski_harabasz_score': metrics.get('calinski_harabasz_score'),
                'avg_intra_sector_similarity': np.mean([v['mean'] for v in intra_similarities.values()]),
                'avg_inter_sector_similarity': np.mean([p['similarity'] for p in inter_distances])
            })
        
        pd.DataFrame(summary_data).to_excel(writer, sheet_name='Summary', index=False)
        
        # Individual sheets for each text column
        for col in text_columns + ['concatenated']:
            sheet_name = col[:31]  # Excel sheet name length limit
            data = results[col]
            
            # Convert nested dict to flat format for Excel
            intra_data = []
            for sector, metrics in data['intra_sector_similarity'].items():
                intra_data.append({
                    'sector': sector,
                    'samples': metrics['samples'],
                    'mean_similarity': metrics['mean'],
                    'std_similarity': metrics['std'],
                    'compactness': metrics['compactness'],
                    'max_distance': metrics['max_distance']
                })
            
            # Save intra-sector metrics
            pd.DataFrame(intra_data).to_excel(writer, sheet_name=sheet_name, index=False)
            
            # Save inter-sector distances in a separate sheet
            inter_data = pd.DataFrame(data['inter_sector_distances'])
            inter_sheet_name = f"{sheet_name[:27]}_inter" if len(sheet_name) > 27 else f"{sheet_name}_inter"
            inter_data.to_excel(writer, sheet_name=inter_sheet_name, index=False)


if __name__ == "__main__":
    # analysis_dataset(data_path)
    # folder_path = os.path.join(WORK_DIR, 'data', 'gwp','evaluate_results_1218')
    # impact_analysis(folder_path)
    # impact_result_analysis_by_sector(folder_path)
    analysis_vector_by_sector()