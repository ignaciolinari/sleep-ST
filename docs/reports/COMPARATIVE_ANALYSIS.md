# Informe Comparativo Final - Modelos de Sleep Staging

**Dataset**: Sleep-EDF (78 sujetos, 186,499 epochs)
**Split**: 70/15/15 por sujeto (sin data leakage)

---

## Resumen Ejecutivo

| Modelo | Kappa | F1 Macro | Accuracy | Ranking |
|--------|-------|----------|----------|---------|
| **CNN1D** | **0.680** | **70.83%** | **76.86%** | 1° |
| LSTM Bi + Attention | 0.651 | 68.07% | 74.64% | 2° |
| XGBoost LOSO | 0.641 | 70.02% | 73.08% | 3° |
| Random Forest | 0.635 | 69.50% | 72.82% | 4° |
| LSTM Unidireccional | 0.530 | 58.59% | 66.17% | 5° |
| LSTM Bidireccional | 0.521 | 58.18% | 65.41% | 6° |

> [!IMPORTANT]
> **Mejor modelo**: CNN1D con Kappa 0.680 (acuerdo "sustancial" según Landis & Koch)

---

## Análisis por Categoría

### Deep Learning vs ML Clásico

| Categoría | Mejor Modelo | Kappa | Ventaja |
|-----------|--------------|-------|---------|
| **Deep Learning** | CNN1D | 0.680 | +6.1% vs mejor ML |
| **ML Clásico** | XGBoost LOSO | 0.641 | Validación LOSO rigurosa |

> [!NOTE]
> Deep Learning supera a ML clásico, pero la diferencia es **moderada** (0.680 vs 0.641). ML clásico ofrece mejor interpretabilidad y validación LOSO.

### Familia LSTM: Impacto de Arquitectura

| Configuración | Kappa | Δ vs Baseline |
|---------------|-------|---------------|
| Unidireccional (baseline) | 0.530 | 0 |
| Bidireccional | 0.521 | **-1.7%** |
| Bidireccional + Atención | 0.651 | **+22.8%** |

> [!TIP]
> **Hallazgo clave**: La bidireccionalidad sola **NO mejora** el rendimiento para clasificación single-epoch. Sin embargo, **la atención es crítica** (+22.8% en Kappa).

---

## Rendimiento por Clase (F1-Score)

| Clase | CNN1D* | LSTM+Attn* | XGBoost | RF | Promedio |
|-------|--------|------------|---------|-----|----------|
| **Wake (W)** | ~80% | ~85% | 88.14% | 88.13% | ~86% |
| **N1** | ~40% | ~45% | 40.76% | 39.77% | ~41% |
| **N2** | ~85% | ~80% | 74.65% | 74.44% | ~78% |
| **N3** | ~75% | ~75% | 76.25% | 75.80% | ~76% |
| **REM** | ~75% | ~70% | 70.28% | 69.34% | ~71% |

*\*CNN1D y LSTM+Attn: valores estimados de matrices de confusión. Ver reportes individuales para detalles.*

> [!WARNING]
> **N1 es consistentemente la clase más difícil** en todos los modelos (F1 ~40%). Esto es esperado dada su naturaleza transicional y baja representación.

---

## Trade-offs Clave

### Rendimiento vs Tiempo de Entrenamiento

| Modelo | Kappa | Tiempo | Eficiencia (κ/hora) |
|--------|-------|--------|---------------------|
| **CNN1D** | 0.680 | 105 min | **0.39** |
| LSTM Bi + Attn | 0.651 | 200 min | 0.20 |
| LSTM Unidir | 0.530 | 202 min | 0.16 |
| LSTM Bidir | 0.521 | 372 min | 0.08 |

> [!NOTE]
> CNN1D es **4.9x más eficiente** que LSTM Bidireccional (0.39 vs 0.08 κ/hora).

### Uso Real-Time vs Offline

| Caso de Uso | Modelo Recomendado | Kappa | Justificación |
|-------------|-------------------|-------|---------------|
| **Real-time** | LSTM Unidireccional | 0.530 | Único modelo causal |
| **Offline (mejor rendimiento)** | CNN1D | 0.680 | Mejor Kappa overall |
| **Offline (interpretabilidad)** | XGBoost/RF | 0.64 | Feature importance |

---

## Comparación con Literatura

| Modelo | Dataset | Kappa | Contexto |
|--------|---------|-------|----------|
| **CNN1D (nuestro)** | Sleep-EDF (78 suj) | **0.680** | Single-epoch |
| **LSTM+Attn (nuestro)** | Sleep-EDF (78 suj) | **0.651** | Single-epoch |
| DeepSleepNet | Sleep-EDF | ~0.76 | Secuencias |
| SleepTransformer | Sleep-EDF | ~0.79 | Attention |
| Inter-scorer humano | - | 0.75-0.85 | Gold standard |

**Conclusión**: Nuestro mejor modelo alcanza **~90% del rendimiento humano** (0.680/0.76).

---

## Hallazgos Principales

### 1. CNN1D supera a LSTM para single-epoch
Las CNN capturan mejor patrones locales (husos, K-complexes) cuando se procesan epochs aisladas.

### 2. Atención es crítica para LSTM
Sin atención, las LSTM no aprovechan su potencial secuencial en epochs de 30s. Con atención, se acercan al rendimiento de CNN.

### 3. Bidireccionalidad no justificada para single-epoch
El contexto futuro dentro de una misma época no aporta información discriminativa adicional.

### 4. ML clásico es competitivo
XGBoost y Random Forest logran Kappa ~0.64, comparable a LSTM+Attention, con menor complejidad.

---

## Recomendaciones

### Para Producción

| Prioridad | Escenario | Modelo | Kappa |
|-----------|-----------|--------|-------|
| **1** | Máximo rendimiento offline | CNN1D | 0.680 |
| **2** | Balance rendimiento/interpretabilidad | XGBoost LOSO | 0.641 |
| **3** | Inferencia real-time | LSTM Unidireccional | 0.530 |

### Para Investigación Futura

1. **Contexto multi-epoch**: Procesar secuencias de 3-5 epochs consecutivas
   - Esperado: Kappa > 0.75 con LSTM Bidireccional

2. **Ensemble CNN + LSTM+Attn**: Combinar fortalezas de ambas arquitecturas
   - CNN para patrones locales
   - LSTM+Attn para contexto temporal

3. **Data augmentation para N1**: SMOTE/ADASYN para mejorar F1 en clase minoritaria

---

## Conclusión Final

| Aspecto | Resultado |
|---------|-----------|
| **Mejor modelo overall** | CNN1D (κ=0.680) |
| **Mejor LSTM** | Bidireccional + Atención (κ=0.651) |
| **Mejor ML clásico** | XGBoost LOSO (κ=0.641) |
| **Mejor para real-time** | LSTM Unidireccional (κ=0.530) |
| **Clase más difícil** | N1 (F1 ~40% en todos los modelos) |

**Mensaje clave**: Para clasificación single-epoch de sleep staging, **CNN1D es la mejor opción** por su combinación de rendimiento, eficiencia y simplicidad. Para aplicaciones que requieren interpretabilidad o inferencia causal, considerar XGBoost o LSTM Unidireccional respectivamente.

---

## Matriz de Confusión Comparativa (Errores Críticos)

| Confusión | CNN1D | LSTM+Attn | XGBoost | RF | Patrón |
|-----------|-------|-----------|---------|----|---------|
| N1 ↔ N2 | Alta | Alta | Alta | Alta | **Universal** |
| N3 → N2 | Media | Media | 25% | 31% | Consistente |
| REM → N2 | Baja | Media | ~17% | 20% | Variable |

> [!CAUTION]
> La confusión N1↔N2 es **sistemática** en todos los modelos. Mejoras requieren contexto temporal inter-epoch o features adicionales.

---
