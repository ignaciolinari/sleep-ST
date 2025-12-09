# Análisis de Resultados XGBoost LOSO - Sleep Staging

Análisis completo del entrenamiento XGBoost con validación Leave-One-Subject-Out (LOSO).

## Resumen Ejecutivo

| Métrica | Agregado (78 folds) | Desviación Estándar |
|---------|---------------------|---------------------|
| **Accuracy** | 73.08% | - |
| **Cohen's Kappa** | **0.641** | - |
| **F1 Macro** | 70.02% | - |
| **F1 Weighted** | 73.74% | - |
| **CV Mean Score** | 67.96% | ±9.85% |

> [!NOTE]
> **Kappa 0.641** = Acuerdo "sustancial" según Landis & Koch. La alta variabilidad inter-sujeto (±9.85%) es esperada en LOSO.

---

## Rendimiento por Clase

| Clase | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| **W** (Wake) | 88.34% | 87.95% | **88.14%** | 9,162 |
| **N1** | 35.59% | 47.70% | **40.76%** | 4,067 |
| **N2** | 75.85% | 73.50% | **74.65%** | 12,560 |
| **N3** | 78.53% | 74.10% | **76.25%** | 3,208 |
| **REM** | 76.60% | 64.93% | **70.28%** | 5,076 |

> [!IMPORTANT]
> **N1 es la clase más difícil** con F1=40.76%, consistente con literatura. REM muestra bajo recall (64.93%) indicando confusión con otras etapas.

---

## Matriz de Confusión

```
              W      N1      N2      N3     REM
    W      8058     812     198      26      68
   N1       633    1940    1080      45     369
   N2       179    2016    9231     564     570
   N3        28       1     802    2377       0
  REM       224     682     859      15    3296
```

### Observaciones Clave

1. **N1 → N2**: 1,080 confusiones (26.6% de N1) - Transición más problemática
2. **N2 → N1**: 2,016 confusiones (16.1% de N2) - Bidireccional
3. **N2 → REM**: 570 confusiones - Características espectrales similares
4. **REM → N2**: 859 confusiones (16.9% de REM) - Alta confusión cruzada
5. **N3 → N2**: 802 confusiones (25.0% de N3) - Límite difuso profundo/ligero

---

## Análisis de Validación Cruzada LOSO

### Distribución de Scores por Fold (78 sujetos)

| Estadística | Valor |
|-------------|-------|
| **Media** | 0.680 (Kappa) |
| **Desviación Estándar** | ±0.098 |
| **Mínimo** | 0.466 (Fold 70) |
| **Máximo** | 0.838 (Fold 6) |
| **Rango** | 0.372 |

### Distribución por Cuartiles

| Cuartil | Rango Kappa | N° Folds |
|---------|-------------|----------|
| Q1 (Bajo) | 0.46 - 0.56 | ~20 |
| Q2 | 0.56 - 0.68 | ~19 |
| Q3 | 0.68 - 0.75 | ~20 |
| Q4 (Alto) | 0.75 - 0.84 | ~19 |

> [!WARNING]
> **6 folds tienen clases faltantes en test** (N3 ausente): Folds 33, 34, 63, 69, 70, 71. Esto afecta las métricas y explica parcialmente la variabilidad extrema.

### Top 5 Mejores Folds

| Fold | Score (Kappa) | Test Size |
|------|---------------|-----------|
| 6 | **0.838** | 1,739 |
| 47 | 0.834 | 3,847 |
| 15 | 0.817 | 2,594 |
| 7 | 0.811 | 2,129 |
| 0 | 0.806 | 1,848 |

### Top 5 Peores Folds

| Fold | Score (Kappa) | Test Size | Problema |
|------|---------------|-----------|----------|
| 70 | **0.466** | 4,925 | N3 faltante |
| 58 | 0.504 | 2,981 | - |
| 37 | 0.505 | 2,307 | - |
| 59 | 0.514 | 3,272 | - |
| 63 | 0.520 | 3,200 | N3 faltante |

---

## Configuración del Modelo

```python
{
    "model_type": "xgboost",
    "n_estimators": 413,
    "max_depth": 10,
    "learning_rate": 0.033,
    "subsample": 0.744,
    "colsample_bytree": 0.622,
    "min_child_weight": 4.0,
    "gamma": 0.712,
    "reg_alpha": 0.000486,
    "reg_lambda": 0.027,
    "n_iter_optimize": 50,
    "cv_folds_optimize": 3
}
```

**Dataset Total**: 186,499 epochs (~34,073 por fold en promedio)

---

## Análisis de Variabilidad Inter-Sujeto

La alta variabilidad (σ=0.098) indica diferencias significativas entre sujetos:

### Factores Contribuyentes

1. **Distribución de clases variable**: Sujetos sin N3 (6 folds identificados)
2. **Calidad de señal**: Diferentes niveles de artefactos por sujeto
3. **Patrones de sueño individuales**: Arquitectura de sueño única por persona
4. **Duración de registros**: Tamaños de test desde 764 hasta 4,925 epochs

### Correlación Score vs Test Size

```
Folds con test_size < 1000:  Mean Kappa = 0.67
Folds con test_size > 4000:  Mean Kappa = 0.64
```

No se observa correlación clara entre tamaño de test y rendimiento, lo que sugiere que la variabilidad es principalmente por características del sujeto.

---

## Puntos Fuertes

1. **Validación rigurosa**: LOSO garantiza independencia sujeto-nivel, sin data leakage
2. **Optimización bayesiana**: Más de 50 iteraciones para hiperparámetros óptimos
3. **F1 Macro competitivo**: 70.02% es robusto para clasificación desbalanceada
4. **Interpretabilidad**: Feature importance disponible para análisis clínico
5. **Consistencia W y N2**: F1 >74% en clases mayoritarias

## Áreas de Mejora

1. **Rendimiento N1**: F1=40.76% necesita mejora (data augmentation, SMOTE)
2. **Confusión N2-REM**: 1,429 errores bidireccionales sugieren overlap de features
3. **Sujetos sin N3**: 6/78 (7.7%) afectan métricas agregadas
4. **Recall REM**: 64.93% indica sub-detección de REM

---

## Recomendaciones

### Mejoras Inmediatas

1. **Excluir folds con clases faltantes**: Recalcular métricas sin folds 33, 34, 63, 69, 70, 71
2. **Class balancing**: Aplicar SMOTE o class_weight para N1
3. **Feature engineering adicional**: Agregar features de contexto temporal (adyacentes)

### Mejoras Avanzadas

4. **Análisis de sujetos outliers**: Investigar folds con Kappa <0.55
5. **Threshold optimization**: Ajustar umbrales de decisión por clase

---

## Conclusión

**Entrenamiento exitoso** con validación LOSO rigurosa
**Kappa 0.641** competitivo para modelo basado en features
**Alta variabilidad inter-sujeto** (±9.85%) refleja realidad clínica
