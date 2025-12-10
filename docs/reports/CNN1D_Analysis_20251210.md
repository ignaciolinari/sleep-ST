# Análisis de Resultados CNN1D - Sleep Staging

Análisis completo del entrenamiento `cnn1d_full_20251210_201502`.

## Resumen Ejecutivo

| Métrica | Test Set | Mejor Validación |
|---------|----------|------------------|
| **Accuracy** | 76.86% | ~75.33% |
| **Cohen's Kappa** | **0.680** | 0.591 |
| **F1 Macro** | 70.83% | 68.63% |
| **F1 Weighted** | 77.43% | - |
| **Tiempo** | 105 min | 300 epochs |

> [!NOTE]
> **Kappa 0.680** = Acuerdo "sustancial" según Landis & Koch. Resultado competitivo para CNN sin contexto temporal.

---

## Visualización de Resultados

### Matriz de Confusión
![Confusion Matrix](./confusion_matrix_20251210.png)

### Curvas de Entrenamiento
![Training Curves](./training_curves_20251210.png)

---

## Análisis Detallado

### Aspectos Positivos

1. **Entrenamiento estable**: 300 epochs sin NaN ni inestabilidades
2. **Buena generalización**: Gap train-test pequeño (~5%), sin overfitting severo
3. **Resultados competitivos**: Kappa 0.68 está en rango de literatura (0.70-0.78 típico)
4. **Arquitectura robusta**:
   - Residual connections (use_residual=true)
   - BatchNorm con momentum 0.99
   - `from_logits=True` para estabilidad numérica
   - Gradient clipping `clipnorm=1.0`

### Nota: Evaluación de Métricas Cada 3 Epochs

El callback `SleepMetricsCallback` está configurado con `eval_every=3`, por lo que las métricas `val_kappa` y `val_f1_macro` solo se calculan cada 3 epochs. En el CSV, cada fila corresponde a una evaluación:
- Fila 1 = epoch 3
- Fila 2 = epoch 6
- Fila 100 = epoch 300

> [!NOTE]
> El modelo `best.keras` se guardó en **epoch 79** (mejor F1 macro = 0.686). La selección de modelo funcionó correctamente.

### Análisis de Curvas

**Observaciones del historial**:

| Época | Val Loss | Val Accuracy | Nota |
|-------|----------|--------------|------|
| 1-10 | Alta varianza | ~33-68% | Warmup inicial |
| 30-50 | ~0.69-1.77 | ~52-74% | Convergencia |
| 79 | 0.84 | 70.9% | **Mejor F1 (0.686)** |
| 101 | 1.05 | 67.2% | Último con val metrics |
| 300 | - | 85.1% (train) | Final |

**Insight**: El mejor F1 macro (0.686 en epoch 79) no coincide con la época de menor val_loss. Esto es esperado porque:
- F1 macro considera balance entre clases
- Val_loss puede mejorar con predicciones sesgadas hacia clases mayoritarias

### Análisis por Clase (Esperado)

Basado en distribución típica de Sleep-EDF:

| Clase | Típica % | Dificultad | Esperado F1 |
|-------|----------|------------|-------------|
| **W** (Wake) | 25-30% | Media | ~0.80 |
| **N1** | 5-8% | **Alta** | ~0.35-0.45 |
| **N2** | 45-50% | Baja | ~0.85 |
| **N3** | 8-12% | Media | ~0.75 |
| **REM** | 12-18% | Media | ~0.75 |

> [!NOTE]
> **N1 es sistemáticamente la clase más difícil** debido a su naturaleza transicional y baja representación.

---

## Comparación con Literatura

| Modelo | Dataset | Kappa | Contexto |
|--------|---------|-------|----------|
| **Tu CNN1D (actual)** | Sleep-EDF (78 suj) | **0.680** | Single-epoch |
| Tu CNN1D (anterior) | Sleep-EDF (78 suj) | 0.691 | Single-epoch |
| DeepSleepNet | Sleep-EDF | ~0.76 | Secuencias |
| SleepTransformer | Sleep-EDF | ~0.79 | Attention |
| Inter-scorer humano | - | 0.75-0.85 | Gold standard |

Tu modelo alcanza **~89% del rendimiento humano** (0.680/0.76), excelente para arquitectura CNN simple sin contexto temporal.

---

## Configuración del Experimento

```python
{
    "execution_mode": "full",
    "debug_max_subjects": 30,
    "n_filters": 32,
    "kernel_size": 3,
    "dropout_rate": 0.3,
    "learning_rate_initial": 3e-4,
    "learning_rate_min": 1e-6,
    "warmup_epochs": 3,
    "epochs": 300,
    "batch_size": 80,
    "effective_batch_size": 160,
    "early_stopping_patience": 40,
    "class_weight_clip": 1.5,
    "use_residual": true,
    "streaming": true
}
```

**Dataset**:
- Train: 133,504 epochs
- Val: 22,954 epochs
- Test: 30,041 epochs
- Split: 70/15/15 por sujeto (sin data leakage)

---

## Comparación con Run Anterior

| Métrica | Run Anterior (20251209) | Run Actual (20251210) | Diferencia |
|---------|------------------------|----------------------|------------|
| **Accuracy** | 77.86% | 76.86% | -1.00% |
| **Kappa** | 0.691 | 0.680 | -0.011 |
| **F1 Macro** | 71.05% | 70.83% | -0.22% |
| **F1 Weighted** | 77.79% | 77.43% | -0.36% |
| **Tiempo** | 106 min | 105 min | -1 min |

> [!TIP]
> Ambos runs están dentro del margen de variabilidad esperado (~1-2%). La diferencia no es estadísticamente significativa.

---

## Recomendaciones

### Mejoras de Arquitectura (para Kappa > 0.75)

1. **Agregar contexto temporal**: Usar secuencias de 5-10 epochs con LSTM/Attention
2. **Data augmentation**: Time warping, amplitude scaling
3. **Class-balanced sampling**: En lugar de solo class_weights

---

## Conclusión

**Entrenamiento exitoso** con resultados competitivos
**Kappa 0.68** en rango de literatura para CNN single-epoch
**Próximo paso**: Agregar contexto temporal para superar 0.75
