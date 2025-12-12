# Trabajo Futuro

Este proyecto representa un **punto de partida** para la clasificación automática de estadios de sueño. A continuación se detallan las líneas de investigación y mejoras que quedan abiertas para futuras iteraciones.

---

## Arquitecturas Deep Learning a Explorar

### LSTM con Secuencias Largas

El modelo LSTM actual procesa cada epoch de 30 segundos de forma independiente. Una dirección natural es explorar **secuencias multi-epoch**, donde el modelo reciba contexto temporal extendido:

- Secuencias de 3-5 epochs consecutivos (~1.5-2.5 minutos de contexto)
- Secuencias más largas que capturen transiciones de fase completas (~10-20 minutos)
- Evaluación del trade-off entre contexto temporal y complejidad computacional

### Arquitecturas Híbridas CNN1D + LSTM

Combinar las fortalezas de ambos enfoques:

- **CNN1D** para extracción automática de features locales de la señal
- **LSTM** para capturar dependencias temporales entre epochs
- Arquitecturas encoder-decoder donde CNN procesa la señal cruda y LSTM modela la secuencia de representaciones

### Transformers

Los Transformers han demostrado rendimiento estado del arte en múltiples dominios:

- **Self-attention** para modelar relaciones entre diferentes partes de la señal
- Arquitecturas como **Sleep Transformer** de la literatura reciente
- Positional encodings adaptados a la estructura temporal del EEG

### Arquitecturas SOTA de la Bibliografía

Modelos propuestos específicamente para clasificación de sueño:

| Arquitectura | Referencia | Características |
|--------------|------------|-----------------|
| DeepSleepNet | Supratak et al., 2017 | CNN + BiLSTM, primera arquitectura end-to-end |
| SeqSleepNet | Phan et al., 2019 | Attention jerárquica para secuencias |
| AttnSleep | Eldele et al., 2021 | Multi-head attention con TCN |
| SleepTransformer | Phan et al., 2022 | Transformer puro para sleep staging |
| GraphSleepNet | Jia et al., 2021 | GNN para modelar relaciones entre canales |
| U-Sleep | Perslev et al., 2021 | U-Net adaptado, generalización cross-dataset |

---

## Mejoras de Datos y Features

- **Data Augmentation**: ruido gaussiano, time warping, mixup de señales
- **Transfer Learning**: pre-entrenar en datasets más grandes (MASS, SHHS) y fine-tune en Sleep-EDF
- **Multi-modal fusion**: mejores estrategias para combinar EEG, EOG y EMG


---

> [!NOTE]
> El objetivo de este proyecto fue establecer una **baseline sólida** con modelos clásicos (Random Forest, XGBoost) y arquitecturas DL fundamentales (CNN1D, LSTM). Las direcciones aquí planteadas representan el siguiente paso natural para mejorar el rendimiento y la aplicabilidad clínica del sistema.
