# Reconocimiento de figuras geométricas con redes neuronales

## Requisitos

- Python 3.10 a 3.12 recomendado para compatibilidad con TensorFlow.
- Conexión a Internet para instalar las bibliotecas la primera vez.

## Instalación en Windows (VS Code)

1. Abra la carpeta del proyecto en Visual Studio Code.
2. Cree un entorno virtual:
   `py -m venv .venv`
3. Active el entorno:
   PowerShell: `.\.venv\Scripts\Activate.ps1`
   CMD: `.venv\Scripts\activate.bat`
4. Instale las dependencias:
   `python -m pip install --upgrade pip`
   `python -m pip install -r requirements.txt`
5. Ejecute:
   `python reconocimiento_figuras.py`

## Qué hace

- Genera un conjunto de datos sintéticos para cuatro clases.
- Divide los datos en entrenamiento y prueba.
- Estandariza las características.
- Entrena una red 4-8-4 con ReLU y Softmax.
- Aplica regularización L2, Dropout y EarlyStopping.
- Informa exactitud, pérdida, reporte y matriz de confusión.
- Permite ingresar características para una clasificación de prueba.
- Guarda el modelo entrenado en `modelo_figuras.keras`.

## Limitación importante

Este prototipo clasifica características numéricas, no píxeles de imágenes. Para reconocer dibujos o fotografías reales, sería necesario implementar extracción de características de imágenes o una red convolucional (CNN) y entrenarla con imágenes etiquetadas.
