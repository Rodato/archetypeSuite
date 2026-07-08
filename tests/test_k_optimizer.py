import numpy as np
import pytest

from src.data.k_optimizer import KOptimizer


class TestKOptimizer:
    def test_analyze_returns_expected_keys(self, numeric_df):
        optimizer = KOptimizer(k_min=2, k_max=5)
        result = optimizer.analyze(numeric_df.values)

        assert "k_range" in result
        assert "inertias" in result
        assert "silhouette_scores" in result
        assert "best_silhouette_k" in result
        assert "best_silhouette_score" in result
        assert "elbow_k" in result
        assert "optimal_k" in result

    def test_k_range_is_correct(self, numeric_df):
        optimizer = KOptimizer(k_min=2, k_max=5)
        result = optimizer.analyze(numeric_df.values)
        assert result["k_range"] == [2, 3, 4, 5]

    def test_optimal_k_within_range(self, numeric_df):
        optimizer = KOptimizer(k_min=2, k_max=6)
        result = optimizer.analyze(numeric_df.values)
        assert result["optimal_k"] in result["k_range"]

    def test_elbow_k_within_range(self, numeric_df):
        optimizer = KOptimizer(k_min=2, k_max=6)
        result = optimizer.analyze(numeric_df.values)
        assert result["elbow_k"] in result["k_range"]

    def test_silhouette_scores_bounded(self, numeric_df):
        optimizer = KOptimizer(k_min=2, k_max=5)
        result = optimizer.analyze(numeric_df.values)
        for score in result["silhouette_scores"]:
            assert -1.0 <= score <= 1.0

    def test_inertias_decrease_monotonically(self, numeric_df):
        optimizer = KOptimizer(k_min=2, k_max=6)
        result = optimizer.analyze(numeric_df.values)
        inertias = result["inertias"]
        for i in range(len(inertias) - 1):
            assert inertias[i] >= inertias[i + 1]

    def test_k_max_capped_by_sample_size(self):
        small_data = np.random.randn(30, 2)
        optimizer = KOptimizer(k_min=2, k_max=10)
        result = optimizer.analyze(small_data)
        # k_max efectivo = min(10, max(2, 30//5), 29) = 6  (≈5 muestras por cluster)
        assert max(result["k_range"]) <= 6

    def test_optimal_k_equals_best_silhouette_k(self, numeric_df):
        optimizer = KOptimizer(k_min=2, k_max=5)
        result = optimizer.analyze(numeric_df.values)
        assert result["optimal_k"] == result["best_silhouette_k"]


class TestSelectOptimalK:
    """Regla de dos regímenes, fijada con las dos curvas reales que la motivaron."""

    def test_flat_curve_prefers_few_workable_clusters(self):
        # Curva real de estudiantes_portugal.csv (rango 0.022): antes elegía k=10.
        from src.data.k_optimizer import select_optimal_k
        k_range = [2, 3, 4, 5, 6, 7, 8, 9, 10]
        scores = [0.131, 0.131, 0.135, 0.136, 0.142, 0.137, 0.146, 0.146, 0.153]
        k, flat = select_optimal_k(k_range, scores)
        assert flat is True
        assert k == 4  # el mejor entre los "pocos y trabajables" (2-4)

    def test_peaked_curve_keeps_argmax(self):
        # Curva real del demo bienestar_digital.csv (pico claro en k=4).
        from src.data.k_optimizer import select_optimal_k
        k_range = [2, 3, 4, 5, 6, 7, 8, 9, 10]
        scores = [0.361, 0.359, 0.370, 0.298, 0.232, 0.200, 0.177, 0.172, 0.187]
        k, flat = select_optimal_k(k_range, scores)
        assert flat is False
        assert k == 4

    def test_single_k_is_not_flat(self):
        from src.data.k_optimizer import select_optimal_k
        k, flat = select_optimal_k([2], [0.5])
        assert (k, flat) == (2, False)

    def test_flat_fallback_when_no_small_k_available(self):
        from src.data.k_optimizer import select_optimal_k
        # Rango que empieza arriba del flat_max_k: usar todos los candidatos.
        k, flat = select_optimal_k([6, 7, 8], [0.10, 0.11, 0.105])
        assert flat is True
        assert k == 7

    def test_analyze_exposes_flat_flag(self, numeric_df):
        from src.data.k_optimizer import KOptimizer
        analysis = KOptimizer().analyze(numeric_df.values)
        assert "flat_k_curve" in analysis
        assert isinstance(analysis["flat_k_curve"], bool)


class TestDegenerateData:
    """Un k que colapsa a <2 clusters no debe abortar toda la búsqueda (round adversarial)."""

    def test_all_identical_rows_raise_clear_error(self):
        # np.zeros → KMeans colapsa a 1 label para todo k → antes: ValueError opaco de
        # sklearn ('Number of labels is 1'); ahora: mensaje de dominio.
        with pytest.raises(ValueError, match="partición válida"):
            KOptimizer(k_min=2, k_max=3).analyze(np.zeros((10, 2)))

    def test_separable_with_heavy_duplicates_still_works(self):
        # Dos ubicaciones distintas muy repetidas: k=2 es válido → análisis normal, sin abortar.
        data = np.vstack([np.zeros((10, 3)), np.full((10, 3), 5.0)])
        result = KOptimizer(k_min=2, k_max=2).analyze(data)
        assert result["optimal_k"] == 2
        assert result["k_range"] == [2]
        assert len(result["silhouette_scores"]) == len(result["k_range"])

    def test_parallel_lists_stay_aligned(self, numeric_df):
        # Invariante tras el filtrado de k inválidos: k_range/inertias/silhouette_scores
        # se llenan en la MISMA iteración → siempre paralelas (best_sil_idx/select_optimal_k
        # /_find_elbow indexan sobre ellas). Nota: el caso "algunos k colapsan y otros no"
        # no es construible con KMeans — si un k colapsa a 1 label hay <2 puntos distintos,
        # así que TODO k colapsa; por eso se cubre 'todos' (raise) y 'ninguno' (este).
        result = KOptimizer(k_min=2, k_max=6).analyze(numeric_df.values)
        assert len(result["k_range"]) == len(result["inertias"]) == len(result["silhouette_scores"])
        assert result["k_range"] == sorted(result["k_range"])  # orden determinista preservado
