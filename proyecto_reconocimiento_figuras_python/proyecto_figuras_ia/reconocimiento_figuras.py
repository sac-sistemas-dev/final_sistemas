"""
Proyecto Semana 8 - Sistemas Expertos
Reconocimiento de figuras geométricas mediante una red neuronal artificial.

Importante:
- Este prototipo usa características numéricas sintéticas, no fotografías.
- Entradas: [relación_ancho_alto, circularidad, número_lados, solidez]
- Salidas: círculo, triángulo, cuadrado, rectángulo.
"""

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

SEMILLA = 42
np.random.seed(SEMILLA)
tf.random.set_seed(SEMILLA)

NOMBRES_FIGURAS = ["Círculo", "Triángulo", "Cuadrado", "Rectángulo"]
NUM_CARACTERISTICAS = 4


def generar_datos(muestras_por_clase=700):
    """Genera datos sintéticos con variaciones pequeñas para cada figura."""
    X, y = [], []

    for _ in range(muestras_por_clase):
        # Círculo: ancho/alto cercano a 1, circularidad cercana a 1,
        # 0 lados poligonales y solidez alta.
        X.append([
            np.random.normal(1.00, 0.06),
            np.random.normal(0.92, 0.06),
            np.random.normal(0.00, 0.15),
            np.random.normal(0.98, 0.025),
        ])
        y.append(0)

        # Triángulo: tres lados, circularidad menor y solidez alta.
        X.append([
            np.random.normal(1.00, 0.18),
            np.random.normal(0.55, 0.10),
            np.random.normal(3.00, 0.25),
            np.random.normal(0.94, 0.04),
        ])
        y.append(1)

        # Cuadrado: proporción 1:1, cuatro lados y alta solidez.
        X.append([
            np.random.normal(1.00, 0.06),
            np.random.normal(0.78, 0.08),
            np.random.normal(4.00, 0.20),
            np.random.normal(0.98, 0.025),
        ])
        y.append(2)

        # Rectángulo: proporción distinta de 1 en la mayoría de muestras,
        # cuatro lados y circularidad algo menor.
        relacion = np.random.uniform(1.25, 2.2)
        X.append([
            np.random.normal(relacion, 0.10),
            np.random.normal(0.68, 0.08),
            np.random.normal(4.00, 0.20),
            np.random.normal(0.98, 0.025),
        ])
        y.append(3)

    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.int32)

    # Limitar pequeñas variaciones generadas por la distribución normal.
    X[:, 0] = np.clip(X[:, 0], -2.5, 2.5)
    X[:, 1] = np.clip(X[:, 1], 0.05, 1.0)
    X[:, 2] = np.clip(X[:, 2], 0.0, 5.0)
    X[:, 3] = np.clip(X[:, 3], 0.5, 1.0)
    return X, y


def construir_modelo():
    """Red neuronal: 4 entradas, 8 neuronas ocultas ReLU y 4 salidas Softmax."""
    modelo = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(NUM_CARACTERISTICAS,)),
        tf.keras.layers.Dense(
            8, activation="relu",
            kernel_regularizer=tf.keras.regularizers.l2(0.001)
        ),
        tf.keras.layers.Dropout(0.10),
        tf.keras.layers.Dense(4, activation="softmax"),
    ])

    modelo.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return modelo


def mostrar_matriz_confusion(matriz):
    """Imprime la matriz de confusión sin requerir bibliotecas de gráficos."""
    ancho = 14
    print("\nMatriz de confusión (filas = valor real, columnas = predicción)")
    print("".ljust(ancho) + "".join(n[:11].ljust(ancho) for n in NOMBRES_FIGURAS))
    for i, fila in enumerate(matriz):
        print(NOMBRES_FIGURAS[i].ljust(ancho) +
              "".join(str(int(v)).ljust(ancho) for v in fila))


def solicitar_figura(modelo, escalador):
    """Permite introducir las características de una figura para clasificarla."""
    print("\nIngrese las características de la figura:")
    print("1. Relación ancho/alto (ej. 1.0 para una figura proporcionada)")
    print("2. Circularidad entre 0 y 1 (aproximada)")
    print("3. Número de lados: 0 para círculo, 3 para triángulo, 4 para cuadriláteros")
    print("4. Solidez entre 0 y 1 (aproximada)")

    try:
        ancho_alto = float(input("Relación ancho/alto: "))
        circularidad = float(input("Circularidad (0-1): "))
        lados = float(input("Número de lados: "))
        solidez = float(input("Solidez (0-1): "))
    except ValueError:
        print("Entrada inválida: debe ingresar valores numéricos.")
        return

    if not (0 <= circularidad <= 1 and 0 <= solidez <= 1):
        print("Aviso: circularidad y solidez normalmente deben estar entre 0 y 1.")
    entrada = np.array([[ancho_alto, circularidad, lados, solidez]], dtype=np.float32)
    entrada = escalador.transform(entrada)
    probabilidades = modelo.predict(entrada, verbose=0)[0]
    indice = int(np.argmax(probabilidades))

    print(f"\nFigura reconocida: {NOMBRES_FIGURAS[indice]}")
    print("Probabilidades estimadas:")
    for nombre, prob in zip(NOMBRES_FIGURAS, probabilidades):
        print(f"  {nombre}: {prob * 100:.2f}%")


def main():
    print("=" * 64)
    print("PROTOTIPO DE RECONOCIMIENTO DE FIGURAS GEOMÉTRICAS")
    print("Red neuronal artificial con TensorFlow/Keras")
    print("=" * 64)

    X, y = generar_datos()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=SEMILLA, stratify=y
    )

    escalador = StandardScaler()
    X_train = escalador.fit_transform(X_train)
    X_test = escalador.transform(X_test)

    modelo = construir_modelo()
    parada_temprana = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=8, restore_best_weights=True
    )

    print(f"\nMuestras totales: {len(X)}")
    print(f"Muestras de entrenamiento: {len(X_train)}")
    print(f"Muestras de prueba: {len(X_test)}")
    print("\nEntrenando la red neuronal...")

    historial = modelo.fit(
        X_train, y_train,
        validation_split=0.20,
        epochs=100,
        batch_size=32,
        callbacks=[parada_temprana],
        verbose=0,
    )

    perdida, precision = modelo.evaluate(X_test, y_test, verbose=0)
    predicciones = np.argmax(modelo.predict(X_test, verbose=0), axis=1)
    exactitud = accuracy_score(y_test, predicciones)

    print(f"Épocas ejecutadas: {len(historial.history['loss'])}")
    print(f"Pérdida en prueba: {perdida:.4f}")
    print(f"Exactitud en prueba: {precision:.4f} ({precision * 100:.2f}%)")
    print(f"Exactitud calculada: {exactitud * 100:.2f}%")

    print("\nReporte de clasificación:")
    print(classification_report(
        y_test, predicciones, target_names=NOMBRES_FIGURAS, zero_division=0
    ))
    mostrar_matriz_confusion(confusion_matrix(y_test, predicciones))

    modelo.save("modelo_figuras.keras")
    np.savez(
        "escalador_figuras.npz",
        media=escalador.mean_,
        escala=escalador.scale_,
    )
    print("\nModelo guardado como 'modelo_figuras.keras'.")
    print("Parámetros del escalador guardados en 'escalador_figuras.npz'.")

    while True:
        opcion = input("\n¿Desea clasificar una figura? (s/n): ").strip().lower()
        if opcion != "s":
            print("Programa finalizado.")
            break
        solicitar_figura(modelo, escalador)


if __name__ == "__main__":
    main()
