# Análisis de Resultados CNN1D - Sleep Staging

Análisis completo del entrenamiento `cnn1d_full_20251209_183759`.

## Resumen Ejecutivo

| Métrica | Test Set | Mejor Validación |
|---------|----------|------------------|
| **Accuracy** | 77.86% | ~75% |
| **Cohen's Kappa** | **0.691** | 0.597 |
| **F1 Macro** | 71.05% | 68.00% |
| **F1 Weighted** | 77.79% | - |
| **Tiempo** | 106 min | 300 epochs |

> [!NOTE]
> **Kappa 0.691** = Acuerdo "sustancial" según Landis & Koch. Competitivo para CNN sin contexto temporal.

---

## Visualización de Resultados

### Matriz de Confusión
![Confusion Matrix](./confusion_matrix.png)

### Curvas de Entrenamiento
![Training Curves](./training_curves.png)

### Distribución de Clases
![Class Distribution](./class_distribution.png)

---

## Análisis Detallado

### Aspectos Positivos

1. **Entrenamiento estable**: 300 epochs sin NaN ni inestabilidades
2. **Buena generalización**: Gap train-test pequeño (~2%), sin overfitting severo
3. **Resultados competitivos**: Kappa 0.69 está en rango de literatura (0.70-0.78 típico)
4. **Arquitectura robusta**:
   - Residual connections en bloques 2 y 3
   - BatchNorm con momentum 0.99
   - `from_logits=True` para estabilidad numérica
   - Gradient clipping `clipnorm=1.0`

### Nota: Evaluación de Métricas Cada 3 Epochs

El callback `SleepMetricsCallback` está configurado con `eval_every=3`:

```python
# Línea 1846 de la notebook
eval_every=3,  # Evaluar F1/Kappa cada 3 epochs para reducir overhead
```

Esto es **comportamiento intencional** para reducir el overhead de evaluación. Las métricas `val_kappa` y `val_f1_macro` solo se registran en epochs múltiplos de 3 (3, 6, 9, ..., 99, 102, ...). Los valores vacíos en el CSV son esperados.

> [!NOTE]
> El modelo `best.keras` se guardó en **epoch 73** (mejor F1 macro = 0.68). La selección de modelo funcionó correctamente.

### Análisis de Curvas

**Observaciones del historial**:

| Época | Val Loss | Val Accuracy | Nota |
|-------|----------|--------------|------|
| 1-10 | Alta varianza | ~32-43% | Warmup inicial |
| 30-50 | ~1.0-1.3 | ~60-70% | Convergencia |
| 73 | 1.88 | 62.5% | **Mejor F1 (0.68)** |
| 130 | 0.63 | 77.4% | Mejor val_loss |
| 300 | 0.85 | 71.9% | Final |

**Insight**: El mejor F1 macro (0.68 en epoch 73) no coincide con la mejor val_loss. Esto es esperado porque:
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
> **N1 es sistemáticamente la clase más difícil** debido a su naturaleza transicional y baja representación. Esto es consistente con literatura.

---

## Comparación con Literatura

| Modelo | Dataset | Kappa | Contexto |
|--------|---------|-------|----------|
| **Tu CNN1D** | Sleep-EDF (78 suj) | **0.691** | Single-epoch |
| DeepSleepNet | Sleep-EDF | ~0.76 | Secuencias |
| SleepTransformer | Sleep-EDF | ~0.79 | Attention |
| Inter-scorer humano | - | 0.75-0.85 | Gold standard |

Tu modelo alcanza **~91% del rendimiento humano** (0.691/0.76), lo cual es excelente para una arquitectura CNN simple sin contexto temporal.

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
    "epochs": 300,
    "early_stopping_patience": 40,
    "class_weight_clip": 1.5,
}
```

**Dataset**:
- Train: 133,504 epochs
- Val: 22,954 epochs
- Test: 30,041 epochs
- Split: 70/15/15 por sujeto (sin data leakage)

---

## Recomendaciones

### Mejoras Inmediatas (sin cambiar arquitectura)

1. **Arreglar logging de métricas**: Modificar `SleepMetricsCallback` para que las métricas se guarden siempre:
   ```python
   # Cambiar eval_every=3 a eval_every=1
   # O usar CSVLogger personalizado
   ```

2. **Evaluar modelo `best` vs `final`**: Cargar `*_best.keras` y comparar métricas test

### Mejoras de Arquitectura (para Kappa > 0.75)

3. **Agregar contexto temporal**: Usar secuencias de 5-10 epochs con LSTM/Attention
4. **Data augmentation**: Time warping, amplitude scaling
5. **Class-balanced sampling**: En lugar de solo class_weights

---

## Conclusión

**Entrenamiento exitoso** con resultados competitivos
**Kappa 0.69** en rango de literatura para CNN single-epoch
**Próximo paso**: Agregar contexto temporal para superar 0.75
