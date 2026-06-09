"""
Pruebas de ml/metrics_export.py: que produzca las claves esperadas a partir de
los artefactos del ML (comparacion_modelos.csv, alertas.csv, informe_modulo_b.md).
"""

from ml.metrics_export import construir_metricas_a, construir_metricas_b


def test_metricas_a_estructura():
    a = construir_metricas_a()
    assert a["primary_metric"] == "WAPE"
    assert isinstance(a["model_comparison"], list) and len(a["model_comparison"]) > 0
    fila = a["model_comparison"][0]
    for col in ("Modelo", "WAPE", "MASE"):
        assert col in fila
    assert a["best_model"]
    assert isinstance(a["forecast"], list)


def test_metricas_b_estructura():
    b = construir_metricas_b()
    # Distribuciones derivadas de alertas.csv (datos vivos).
    assert "level_distribution" in b
    assert "type_distribution" in b
    assert "risk_score_histogram" in b
    # Métricas de evaluación parseadas del informe.
    assert "precision_at_k" in b
    assert b["concordance"]["jaccard"]
    assert len(b["concordance"]["jaccard"]) == 3
