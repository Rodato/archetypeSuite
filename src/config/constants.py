SUPPORTED_FILE_TYPES = {".csv", ".xlsx", ".xls"}

MAX_PREVIEW_ROWS = 100
MAX_CATEGORIES_FOR_ENCODING = 20
PCA_VARIANCE_THRESHOLD = 0.95

AVAILABLE_ALGORITHMS = ["KMeans", "DBSCAN", "AgglomerativeClustering", "GaussianMixture"]

DEFAULT_K_RANGE = (2, 10)

QUALITY_THRESHOLDS = {
    "silhouette": 0.25,
    "calinski_harabasz": 50.0,
    "davies_bouldin": 2.0,
}
