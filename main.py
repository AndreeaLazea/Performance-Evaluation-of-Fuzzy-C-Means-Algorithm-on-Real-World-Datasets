import pandas as pd
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns


def load_and_explore_data(filepath):
    data = pd.read_csv(filepath)
    print("\nFirst few rows of the dataset:")
    print(data.head())
    print("\nSummary Statistics:")
    print(data.describe())
    return data

def preprocess_data(data, relevant_columns):
    print("\nChecking for missing values:")
    print(data[relevant_columns].isnull().sum())
    data = data.dropna(subset=relevant_columns)
    return data

def normalize_data(data, columns):
    scaler = MinMaxScaler()
    data[columns] = scaler.fit_transform(data[columns])
    return data

# Histograms and Boxplots
def plot_distributions(data, columns, title):
    for col in columns:
        plt.figure()
        sns.histplot(data[col], kde=True, bins=20)
        plt.title(f'{col} Distribution ({title})')
        plt.xlabel(col)
        plt.ylabel('Frequency')
        plt.show()

        plt.figure()
        sns.boxplot(data[col])
        plt.title(f'{col} Boxplot ({title})')
        plt.xlabel(col)
        plt.show()

# Elbow and Silhouette Method for Clustering
def plot_elbow_and_silhouette(data, features, max_clusters=10):
    inertia = []
    silhouette_scores = []

    for i in range(2, max_clusters + 1):
        kmeans = KMeans(n_clusters=i, random_state=42)
        kmeans.fit(data[features])
        inertia.append(kmeans.inertia_)
        silhouette_scores.append(silhouette_score(data[features], kmeans.labels_))

    # Elbow Plot
    plt.figure()
    plt.plot(range(2, max_clusters + 1), inertia, marker='o')
    plt.title('Elbow Method')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Inertia')
    plt.show()

    # Silhouette Plot
    plt.figure()
    plt.plot(range(2, max_clusters + 1), silhouette_scores, marker='o')
    plt.title('Silhouette Scores')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Silhouette Score')
    plt.show()

# Fuzzy C-Means Clustering
def perform_fuzzy_clustering(data, features, n_clusters=6):
    feature_values = data[features].values.T
    cntr, u, _, _, _, _, _ = fuzz.cluster.cmeans(
        feature_values, c=n_clusters, m=2, error=0.005, maxiter=1000, init=None
    )
    cluster_labels = np.argmax(u, axis=0)
    data['Cluster'] = cluster_labels
    return data, cntr, u

# Visualize Clusters in 2D
def plot_fuzzy_clusters(data, features, cntr, title):
    pca = PCA(n_components=2)
    reduced_data = pca.fit_transform(data[features])
    reduced_cntr = pca.transform(cntr)

    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        x=reduced_data[:, 0],
        y=reduced_data[:, 1],
        hue=data['Cluster'],
        palette='viridis',
        s=100,
        alpha=0.7
    )
    plt.scatter(
        reduced_cntr[:, 0], reduced_cntr[:, 1], c='red', marker='X', s=200, label='Centers'
    )
    plt.title(f'Fuzzy Clustering ({title})')
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.legend()
    plt.show()

# Star Classification Fuzzy System
def create_star_fuzzy_system():
    temperature = ctrl.Antecedent(np.arange(0, 1.1, 0.01), 'temperature')
    luminosity = ctrl.Antecedent(np.arange(0, 1.1, 0.01), 'luminosity')
    radius = ctrl.Antecedent(np.arange(0, 1.1, 0.01), 'radius')
    star_type = ctrl.Consequent(np.arange(0, 1.1, 0.01), 'star_type')

    temperature['cool'] = fuzz.trimf(temperature.universe, [0, 0, 0.4])
    temperature['warm'] = fuzz.trimf(temperature.universe, [0.3, 0.5, 0.7])
    temperature['hot'] = fuzz.trimf(temperature.universe, [0.6, 1, 1])

    luminosity['low'] = fuzz.trimf(luminosity.universe, [0, 0, 0.4])
    luminosity['medium'] = fuzz.trimf(luminosity.universe, [0.3, 0.5, 0.7])
    luminosity['high'] = fuzz.trimf(luminosity.universe, [0.6, 1, 1])

    radius['small'] = fuzz.trimf(radius.universe, [0, 0, 0.4])
    radius['medium'] = fuzz.trimf(radius.universe, [0.3, 0.5, 0.7])
    radius['large'] = fuzz.trimf(radius.universe, [0.6, 1, 1])

    star_type['dwarf'] = fuzz.trimf(star_type.universe, [0, 0, 0.4])
    star_type['main_sequence'] = fuzz.trimf(star_type.universe, [0.3, 0.5, 0.7])
    star_type['giant'] = fuzz.trimf(star_type.universe, [0.6, 1, 1])

    rules = [
        ctrl.Rule(temperature['cool'] & luminosity['low'] & radius['small'], star_type['dwarf']),
        ctrl.Rule(temperature['warm'] & luminosity['medium'] & radius['medium'], star_type['main_sequence']),
        ctrl.Rule(temperature['hot'] & luminosity['high'] & radius['large'], star_type['giant']),
    ]

    star_type_ctrl = ctrl.ControlSystem(rules)
    return ctrl.ControlSystemSimulation(star_type_ctrl)

# Air Quality Fuzzy System
def create_aqi_fuzzy_system():
    pm25 = ctrl.Antecedent(np.arange(0, 1.1, 0.01), 'pm25')
    no2 = ctrl.Antecedent(np.arange(0, 1.1, 0.01), 'no2')
    co = ctrl.Antecedent(np.arange(0, 1.1, 0.01), 'co')
    aqi = ctrl.Consequent(np.arange(0, 1.1, 0.01), 'aqi')

    pm25['low'] = fuzz.trimf(pm25.universe, [0, 0, 0.5])
    pm25['medium'] = fuzz.trimf(pm25.universe, [0.3, 0.5, 0.7])
    pm25['high'] = fuzz.trimf(pm25.universe, [0.6, 1, 1])

    no2['low'] = fuzz.trimf(no2.universe, [0, 0, 0.5])
    no2['medium'] = fuzz.trimf(no2.universe, [0.3, 0.5, 0.7])
    no2['high'] = fuzz.trimf(no2.universe, [0.6, 1, 1])

    co['low'] = fuzz.trimf(co.universe, [0, 0, 0.5])
    co['medium'] = fuzz.trimf(co.universe, [0.3, 0.5, 0.7])
    co['high'] = fuzz.trimf(co.universe, [0.6, 1, 1])

    aqi['good'] = fuzz.trimf(aqi.universe, [0, 0, 0.4])
    aqi['moderate'] = fuzz.trimf(aqi.universe, [0.3, 0.5, 0.7])
    aqi['unhealthy'] = fuzz.trimf(aqi.universe, [0.6, 1, 1])

    rules = [
        ctrl.Rule(pm25['low'] & no2['low'] & co['low'], aqi['good']),
        ctrl.Rule(pm25['medium'] | no2['medium'] | co['medium'], aqi['moderate']),
        ctrl.Rule(pm25['high'] | no2['high'] | co['high'], aqi['unhealthy']),
    ]

    aqi_ctrl = ctrl.ControlSystem(rules)
    return ctrl.ControlSystemSimulation(aqi_ctrl)
# Movie Dataset Functions
def analyze_movie_data(filepath):
    print("\n=== Movie Data Analysis ===")
    try:
        movie_data = load_and_explore_data(filepath)
        relevant_columns = ['Rating', 'Votes', 'Revenue (Millions)']
        movie_data = preprocess_data(movie_data, relevant_columns)
        movie_data = normalize_data(movie_data, relevant_columns)

        # Plot distributions for movies
        plot_distributions(movie_data, relevant_columns, "Movies")

        # Perform Elbow and Silhouette Analysis for Movies
        optimal_clusters = determine_optimal_clusters(movie_data, relevant_columns)

        # Perform Fuzzy Clustering for Movies
        movie_data, cntr, u = perform_fuzzy_clustering(movie_data, relevant_columns, n_clusters=2)
        plot_fuzzy_clusters(movie_data, relevant_columns, cntr, "Movie Data Clustering")

        print("\nMovie Data Clustering Results:")
        print(movie_data[['Rating', 'Votes', 'Revenue (Millions)', 'Cluster']])
    except Exception as e:
        print(f"Error processing Movie Data Analysis: {e}")

def determine_optimal_clusters(data, features, max_clusters=10):
    silhouette_scores = []

    for i in range(2, max_clusters + 1):
        kmeans = KMeans(n_clusters=i, random_state=42)
        kmeans.fit(data[features])
        silhouette_scores.append(silhouette_score(data[features], kmeans.labels_))

    # Determine the cluster count with the highest silhouette score
    optimal_clusters = np.argmax(silhouette_scores) + 2  # +2 because index starts at 0 for n_clusters=2
    print(f"Optimal number of clusters based on silhouette scores: {optimal_clusters}")

    # Silhouette Plot
    plt.figure()
    plt.plot(range(2, max_clusters + 1), silhouette_scores, marker='o')
    plt.title('Silhouette Scores')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Silhouette Score')
    plt.show()

    return optimal_clusters

# Main Function

def main():
    # Star Data
    print("\n=== Star Classification ===")
    try:
        star_data = load_and_explore_data('Stars.csv')
        star_data = preprocess_data(star_data, ['Temperature (K)', 'Luminosity (L/Lo)', 'Radius (R/Ro)'])
        star_data = normalize_data(star_data, ['Temperature (K)', 'Luminosity (L/Lo)', 'Radius (R/Ro)'])

        # Plot distributions for stars
        plot_distributions(star_data, ['Temperature (K)', 'Luminosity (L/Lo)', 'Radius (R/Ro)'], "Stars")

        # Perform Elbow and Silhouette Analysis for Stars
        optimal_clusters = determine_optimal_clusters(star_data,
                                                      ['Temperature (K)', 'Luminosity (L/Lo)', 'Radius (R/Ro)'])

        # Perform Fuzzy Clustering for Stars
        optimal_clusters = 10
        star_data, cntr, u = perform_fuzzy_clustering(star_data, ['Temperature (K)', 'Luminosity (L/Lo)', 'Radius (R/Ro)'], optimal_clusters)
        plot_fuzzy_clusters(star_data, ['Temperature (K)', 'Luminosity (L/Lo)', 'Radius (R/Ro)'], cntr, "Star Data Clustering")

        # Fuzzy Classification for Stars
        star_simulation = create_star_fuzzy_system()
        classifications = []
        for _, row in star_data.iterrows():
            star_simulation.input['temperature'] = row['Temperature (K)']
            star_simulation.input['luminosity'] = row['Luminosity (L/Lo)']
            star_simulation.input['radius'] = row['Radius (R/Ro)']
            star_simulation.compute()
            classifications.append(star_simulation.output['star_type'])
        star_data['Fuzzy Star Type'] = classifications

        print("\nStar Classification Results:")
        print(star_data[['Temperature (K)', 'Luminosity (L/Lo)', 'Radius (R/Ro)', 'Cluster', 'Fuzzy Star Type']])
    except Exception as e:
        print(f"Error processing Star Classification: {e}")

    # Air Quality Data
    print("\n=== Air Quality Analysis ===")
    try:
        air_data = load_and_explore_data('Cleaned_AirQuality.csv')  # Preprocessed air quality dataset
        relevant_columns = ['CO(GT)', 'C6H6(GT)', 'NO2(GT)']
        air_data = preprocess_data(air_data, relevant_columns)
        air_data = normalize_data(air_data, relevant_columns)

        # Plot distributions for air quality
        plot_distributions(air_data, relevant_columns, "Air Quality")

        # Perform Elbow and Silhouette Analysis for Air Quality
        optimal_clusters = determine_optimal_clusters(air_data, relevant_columns)

        # Perform Fuzzy Clustering for Air Quality
        air_data, cntr, u = perform_fuzzy_clustering(air_data, relevant_columns, n_clusters=5)
        plot_fuzzy_clusters(air_data, relevant_columns, cntr, "Air Quality Clustering")

        # Fuzzy Classification for Air Quality
        aqi_simulation = create_aqi_fuzzy_system()
        classifications = []
        for _, row in air_data.iterrows():
            aqi_simulation.input['pm25'] = row['CO(GT)']  # Mapping CO as a proxy for PM2.5
            aqi_simulation.input['no2'] = row['NO2(GT)']
            aqi_simulation.input['co'] = row['C6H6(GT)']
            aqi_simulation.compute()
            classifications.append(aqi_simulation.output['aqi'])
        air_data['Fuzzy AQI'] = classifications

        print("\nAir Quality Classification Results:")
        print(air_data[['CO(GT)', 'C6H6(GT)', 'NO2(GT)', 'Cluster', 'Fuzzy AQI']])
    except Exception as e:
        print(f"Error processing Air Quality Analysis: {e}")

    # IMDb Movie Dataset
    analyze_movie_data('imdb_movie_dataset.csv')


if __name__ == "__main__":
    main()

