from prometheus_client import Counter, Histogram, start_http_server

# Mendefinisikan metrik untuk memantau performa model
REQUEST_COUNT = Counter(
    'model_requests_total', 
    'Total request yang masuk ke endpoint inference'
)

PREDICTION_COUNT = Counter(
    'model_predictions_total', 
    'Distribusi hasil prediksi model', 
    ['predicted_class']
)

LATENCY = Histogram(
    'model_prediction_latency_seconds', 
    'Waktu yang dibutuhkan untuk melakukan satu prediksi'
)

def start_metrics_server(port=8001):
    """Memulai server HTTP terpisah untuk diekspos ke Prometheus"""
    start_http_server(port)
    print(f"Prometheus exporter metrics berjalan di http://localhost:{port}")
