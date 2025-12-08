"""Utilidades para visualizar el historial de entrenamiento de modelos de deep learning."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TypedDict

import matplotlib.pyplot as plt
import numpy as np

from src.models import load_model


class OverfittingResults(TypedDict):
    """Tipo para los resultados del análisis de sobreajuste."""

    has_validation: bool
    overfitting_detected: bool
    warnings: list[str]
    metrics: dict[str, float]


def plot_training_history(
    history: dict | Path | str,
    model_name: str | None = None,
    save_path: Path | str | None = None,
    figsize: tuple[int, int] = (12, 4),
) -> None:
    """Visualiza el historial de entrenamiento de un modelo de deep learning.

    Parameters
    ----------
    history : dict | Path | str
        Historial de entrenamiento (dict) o ruta al archivo JSON con historial
    model_name : str, optional
        Nombre del modelo para el título
    save_path : Path | str, optional
        Ruta donde guardar la figura
    figsize : tuple[int, int]
        Tamaño de la figura
    """
    # Cargar historial si es una ruta
    if isinstance(history, (Path, str)):
        history_path = Path(history)
        if not history_path.exists():
            raise FileNotFoundError(
                f"Archivo de historial no encontrado: {history_path}"
            )
        with open(history_path) as f:
            history = json.load(f)

    if not isinstance(history, dict):
        raise ValueError("history debe ser un diccionario o ruta a archivo JSON")

    # Determinar métricas disponibles
    has_val = any(key.startswith("val_") for key in history.keys())

    # Crear figura con subplots
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Plot 1: Loss
    ax1 = axes[0]
    ax1.plot(history["loss"], label="Train Loss", linewidth=2)
    if has_val and "val_loss" in history:
        ax1.plot(history["val_loss"], label="Val Loss", linewidth=2)
    ax1.set_xlabel("Epoch", fontsize=12)
    ax1.set_ylabel("Loss", fontsize=12)
    ax1.set_title("Training and Validation Loss", fontsize=14, fontweight="bold")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Accuracy
    ax2 = axes[1]
    if "accuracy" in history:
        ax2.plot(history["accuracy"], label="Train Accuracy", linewidth=2)
        if has_val and "val_accuracy" in history:
            ax2.plot(history["val_accuracy"], label="Val Accuracy", linewidth=2)
        ax2.set_ylabel("Accuracy", fontsize=12)
    else:
        # Si no hay accuracy, mostrar otra métrica disponible
        metric_key = [
            k for k in history.keys() if not k.startswith("val_") and k != "loss"
        ][0]
        ax2.plot(history[metric_key], label=f"Train {metric_key}", linewidth=2)
        if has_val and f"val_{metric_key}" in history:
            ax2.plot(
                history[f"val_{metric_key}"], label=f"Val {metric_key}", linewidth=2
            )
        ax2.set_ylabel(metric_key, fontsize=12)

    ax2.set_xlabel("Epoch", fontsize=12)
    ax2.set_title("Training and Validation Accuracy", fontsize=14, fontweight="bold")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Título general
    if model_name:
        fig.suptitle(
            f"Training History: {model_name}", fontsize=16, fontweight="bold", y=1.02
        )

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Figura guardada en {save_path}")

    plt.show()


def plot_training_history_from_model(
    model_path: Path | str,
    save_path: Path | str | None = None,
    figsize: tuple[int, int] = (12, 4),
) -> None:
    """Carga un modelo y visualiza su historial de entrenamiento.

    Parameters
    ----------
    model_path : Path | str
        Ruta al modelo guardado (directorio para Keras)
    save_path : Path | str, optional
        Ruta donde guardar la figura
    figsize : tuple[int, int]
        Tamaño de la figura
    """
    model_path = Path(model_path)

    # Intentar cargar historial directamente desde archivo JSON primero (más eficiente)
    history_path = model_path.parent / f"{model_path.name}_history.json"
    if history_path.exists():
        plot_training_history(
            history_path,
            model_name=model_path.name,
            save_path=save_path,
            figsize=figsize,
        )
        return

    # Si no existe el archivo JSON, cargar el modelo completo
    model = load_model(model_path)

    if not hasattr(model, "history_"):
        raise ValueError(
            f"El modelo en {model_path} no tiene historial de entrenamiento guardado"
        )

    model_name = model_path.name
    plot_training_history(
        model.history_, model_name=model_name, save_path=save_path, figsize=figsize
    )


def check_overfitting(
    history: dict | Path | str, threshold: float = 0.1
) -> OverfittingResults:
    """Verifica si hay signos de sobreajuste en el historial de entrenamiento.

    Parameters
    ----------
    history : dict | Path | str
        Historial de entrenamiento
    threshold : float
        Umbral para considerar diferencia significativa entre train y val

    Returns
    -------
    OverfittingResults
        Diccionario con resultados del análisis
    """
    # Cargar historial si es una ruta
    if isinstance(history, (Path, str)):
        history_path = Path(history)
        if not history_path.exists():
            raise FileNotFoundError(
                f"Archivo de historial no encontrado: {history_path}"
            )
        with open(history_path) as f:
            history = json.load(f)

    if not isinstance(history, dict):
        raise ValueError("history debe ser un diccionario o ruta a archivo JSON")

    results: OverfittingResults = {
        "has_validation": False,
        "overfitting_detected": False,
        "warnings": [],
        "metrics": {},
    }

    # Verificar si hay datos de validación
    has_val = any(key.startswith("val_") for key in history.keys())
    results["has_validation"] = has_val

    if not has_val:
        results["warnings"].append(
            "No hay datos de validación disponibles para detectar sobreajuste"
        )
        return results

    # Analizar loss
    if "loss" in history and "val_loss" in history:
        train_loss = history["loss"]
        val_loss = history["val_loss"]

        # Diferencia final entre train y val
        final_diff = val_loss[-1] - train_loss[-1]

        # Tendencias: train bajando pero val subiendo = sobreajuste
        if len(train_loss) > 5 and len(val_loss) > 5:
            train_trend = np.mean(train_loss[-5:]) - np.mean(train_loss[:5])
            val_trend = np.mean(val_loss[-5:]) - np.mean(val_loss[:5])

            if train_trend < -threshold and val_trend > threshold:
                results["overfitting_detected"] = True
                results["warnings"].append(
                    f"Posible sobreajuste detectado: train loss bajando ({train_trend:.4f}) "
                    f"pero val loss subiendo ({val_trend:.4f})"
                )

        results["metrics"]["loss_diff"] = final_diff
        if final_diff > threshold:
            results["warnings"].append(
                f"Diferencia significativa entre train y val loss: {final_diff:.4f}"
            )

    # Analizar accuracy
    if "accuracy" in history and "val_accuracy" in history:
        train_acc = history["accuracy"]
        val_acc = history["val_accuracy"]

        # Diferencia final
        final_diff = train_acc[-1] - val_acc[-1]

        results["metrics"]["accuracy_diff"] = final_diff
        if final_diff > threshold:
            results["warnings"].append(
                f"Diferencia significativa entre train y val accuracy: {final_diff:.4f}"
            )

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Visualizar historial de entrenamiento"
    )
    parser.add_argument(
        "model_path",
        type=str,
        help="Ruta al modelo guardado o archivo de historial JSON",
    )
    parser.add_argument(
        "--save",
        type=str,
        default=None,
        help="Ruta donde guardar la figura",
    )

    args = parser.parse_args()

    model_path = Path(args.model_path)

    # Intentar cargar como modelo primero
    if model_path.is_dir() or (
        model_path.suffix == ".json" and "history" in model_path.name
    ):
        try:
            plot_training_history_from_model(model_path, save_path=args.save)
        except Exception as e:
            print(f"Error cargando modelo: {e}")
            # Intentar cargar como JSON directo
            if model_path.suffix == ".json":
                plot_training_history(model_path, save_path=args.save)
    else:
        # Asumir que es un archivo JSON de historial
        plot_training_history(model_path, save_path=args.save)

    # Verificar sobreajuste
    print("\n" + "=" * 60)
    print("ANÁLISIS DE SOBREAJUSTE")
    print("=" * 60)
    results = check_overfitting(model_path)
    if results["overfitting_detected"]:
        print("⚠️  SOBREAJUSTE DETECTADO")
    else:
        print("✓ No se detectó sobreajuste significativo")

    if results["warnings"]:
        print("\nAdvertencias:")
        for warning in results["warnings"]:
            print(f"  - {warning}")

    if results["metrics"]:
        print("\nMétricas:")
        for key, value in results["metrics"].items():
            print(f"  {key}: {value:.4f}")
