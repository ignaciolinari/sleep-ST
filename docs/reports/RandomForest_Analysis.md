# Análisis de Resultados Random Forest - Sleep Staging

Análisis completo del modelo Random Forest optimizado con búsqueda bayesiana para clasificación de etapas de sueño.

## Resumen Ejecutivo

| Métrica | Valor |
|---------|-------|
| **Accuracy** | 72.82% |
| **Cohen's Kappa** | **0.635** |
| **F1 Macro** | 69.50% |
| **F1 Weighted** | 73.35% |

> [!NOTE]
> **Kappa 0.635** = Acuerdo "sustancial" según Landis & Koch, clasificación para modelo basado en features con balanceo automático.

---

## Rendimiento por Clase

| Clase | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| **W** (Wake) | 88.64% | 87.63% | **88.13%** | 9,162 |
| **N1** | 35.56% | 45.12% | **39.77%** | 4,067 |
| **N2** | 73.24% | 75.68% | **74.44%** | 12,560 |
| **N3** | 85.24% | 68.24% | **75.80%** | 3,208 |
| **REM** | 75.52% | 64.11% | **69.34%** | 5,076 |

### Observaciones por Clase

- **Wake (W)**: Mejor rendimiento global con F1=88.13%. Alta precisión y recall balanceados.
- **N1**: Clase más difícil con F1=39.77%. Recall moderado (45.12%) pero precisión baja (35.56%).
- **N2**: Buen balance precision/recall. Beneficiada por ser la clase mayoritaria.
- **N3**: Alta precisión (85.24%) pero recall bajo (68.24%) - el modelo es conservador para N3.
- **REM**: Recall bajo (64.11%) indica sub-detección de episodios REM.

> [!IMPORTANT]
> **N1 sigue siendo el cuello de botella** con F1<40%, consistente con la dificultad inherente de esta etapa transicional en la literatura de sleep staging.

---

## Matriz de Confusión

```
              W      N1      N2      N3     REM
    W      8029     732     293      17      91
   N1       671    1835    1176      15     370
   N2       169    1946    9505     347     593
   N3        34       0     984    2189       1
  REM       155     647    1020       0    3254
```

### Análisis de Errores Principales

| Confusión | Cantidad | % de Origen | Interpretación |
|-----------|----------|-------------|----------------|
| **N2 → N1** | 1,946 | 15.5% de N2 | Sobre-predicción de transiciones |
| **N1 → N2** | 1,176 | 28.9% de N1 | Confusión bidireccional severa |
| **REM → N2** | 1,020 | 20.1% de REM | Features espectrales similares |
| **N3 → N2** | 984 | 30.7% de N3 | Sueño profundo confundido con ligero |
| **N1 → W** | 671 | 16.5% de N1 | N1 confundido con vigilia |
| **REM → N1** | 647 | 12.7% de REM | Patrón EEG de baja amplitud similar |

> [!WARNING]
> **30.7% de N3 clasificado como N2** indica que el modelo tiene dificultad para distinguir sueño profundo de sueño ligero. Esto puede deberse a overlap en bandas delta.

---

## Configuración del Modelo

```python
{
    "model_type": "random_forest",
    "n_estimators": 495,
    "max_depth": 20,
    "min_samples_split": 14,
    "min_samples_leaf": 1,
    "max_features": "sqrt",
    "class_weight": "balanced_subsample"
}
```

### Parámetros Destacados

- **n_estimators=495**: Alto número de árboles para reducir varianza
- **max_depth=20**: Profundidad moderada, balance entre complejidad y generalización
- **class_weight="balanced_subsample"**: Balanceo dinámico por bootstrap, mitiga desbalance de clases
- **max_features="sqrt"**: Reducción de correlación entre árboles

**Dataset Total**: 34,073 epochs de prueba

---

## Análisis de Trade-offs Precision/Recall

| Clase | Precision | Recall | Trade-off |
|-------|-----------|--------|-----------|
| **N3** | 85.24% | 68.24% | Conservador - alta confianza cuando predice N3 |
| **REM** | 75.52% | 64.11% | Conservador - sub-detección significativa |
| **N1** | 35.56% | 45.12% | Agresivo - sobreclasifica N1 con baja precisión |
| **N2** | 73.24% | 75.68% | Balanceado |
| **W** | 88.64% | 87.63% | Balanceado |

El modelo exhibe comportamiento **conservador para sueño profundo (N3) y REM**, prefiriendo no clasificar estas clases a menos que tenga alta confianza. En contraste, es **más agresivo para N1**, clasificando más muestras como N1 a costa de precisión.

---

## Puntos Fuertes

1. **Balanceo automático**: `balanced_subsample` maneja desbalance sin preprocesamiento
2. **Alta precisión N3**: 85.24% - cuando predice N3, es confiable
3. **Rendimiento Wake robusto**: F1=88.13% es excelente para esta clase
4. **Interpretabilidad**: Feature importance disponible para análisis
5. **Sin overfitting severo**: max_depth=20 y min_samples_split=14 regularizan

## Áreas de Mejora

1. **Recall N3 y REM**: Sub-detección afecta sensibilidad clínica
2. **Confusión N1↔N2**: 3,122 errores bidireccionales
3. **Precisión N1**: 35.56% genera muchos falsos positivos
4. **Sensibilidad REM**: 64.11% puede perder episodios REM importantes

---

## Recomendaciones

### Optimización del Modelo

1. **Threshold adjustment por clase**: Reducir umbral para N3 y REM para mejorar recall
2. **Feature engineering temporal**: Agregar features de epochs adyacentes
3. **Análisis de feature importance**: Identificar qué features discriminan mejor N1

### Mejoras de Data

4. **Oversampling N1**: SMOTE o ADASYN para aumentar representación
5. **Undersampling N2**: Reducir sesgo hacia clase mayoritaria

---

## Conclusión

**Modelo funcional** con rendimiento competitivo para clasificación basada en features
**Kappa 0.635** indica acuerdo sustancial con ground truth
**Puntos críticos**: Recall bajo en N3 (68%) y REM (64%), precisión baja en N1 (36%)
**Fortaleza principal**: Alta confiabilidad cuando predice Wake y N3
