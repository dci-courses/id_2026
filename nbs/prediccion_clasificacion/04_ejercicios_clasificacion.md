---
title: "Ejercicios de clasificación supervisada"
---

# Ejercicios de clasificación supervisada

Esta guía adapta los cuatro ejercicios de la sección *Exercises* del Chapter 3: Classification del libro base. La intención es practicar los conceptos centrales de la unidad sin entregar una solución completa: ustedes deberán diseñar, entrenar, evaluar e interpretar sus propios clasificadores.

Los ejercicios están redactados en español y ajustados al contexto del curso. En algunos casos se proponen alternativas reproducibles con datasets disponibles en `scikit-learn`, para evitar que el avance dependa obligatoriamente de descargas externas.

## Objetivo de la actividad

El objetivo general es que apliques un flujo completo de clasificación supervisada:

- definir un problema de clasificación;
- preparar datos de entrenamiento y prueba;
- entrenar modelos;
- comparar resultados con métricas apropiadas;
- interpretar matrices de confusión;
- analizar errores reales;
- comunicar conclusiones con evidencia.

La actividad combina problemas de imágenes, datos tabulares y texto. No se espera que todos los ejercicios alcancen el mismo nivel de desempeño, sino que el trabajo sea reproducible, razonado y bien interpretado.

## Antes de comenzar

Trabaja en notebooks propios, uno por ejercicio o uno consolidado con secciones claras. Usa semillas aleatorias cuando corresponda y evita depender de resultados obtenidos manualmente.

Para los ejercicios con imágenes de dígitos puedes usar `load_digits`, que viene incluido en scikit-learn. El libro trabaja con MNIST completo, pero para el curso es aceptable usar `load_digits` si quieres mantener el trabajo liviano y sin conexión a internet.

```python
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

digits = load_digits()
X = digits.data
y = digits.target
```

Recuerda separar siempre entrenamiento y prueba antes de evaluar. Si haces búsqueda de hiperparámetros, evita usar el conjunto de prueba durante la selección del modelo.

## Descarga y organización de datasets externos

Los ejercicios 1 y 2 no requieren descargar datos externos: usan `load_digits`, disponible directamente en `scikit-learn`.

Los ejercicios 3 y 4 pueden trabajarse con datasets externos de Kaggle. No debes subir archivos `.csv` ni `.zip` descargados al repositorio del curso; guárdalos localmente en una carpeta de datos y documenta la ruta usada en tu notebook.

### Opción 1: descarga manual desde Kaggle

1. Entra al enlace del dataset.
2. Descarga el archivo `.zip` desde Kaggle.
3. Descomprime el archivo.
4. Copia los archivos resultantes a la carpeta sugerida dentro del repositorio.

Estructura esperada:

```text
data/
└── prediccion_clasificacion/
    ├── titanic/
    │   ├── train.csv
    │   └── test.csv
    └── spam/
        └── spam.csv
```

### Opción 2: descarga usando Kaggle CLI

Esta opción requiere una cuenta de Kaggle y un token de API configurado en tu equipo. La herramienta oficial es `kaggle`, que se instala y configura fuera de este notebook. Si no tienes configurado el token, usa la descarga manual.

Para Titanic, Kaggle lo publica como competencia introductoria. Un comando orientativo es:

```bash
kaggle competitions download -c titanic -p data/prediccion_clasificacion/titanic
```

Luego descomprime el archivo descargado en esa misma carpeta.

Para SMS Spam Collection, el identificador confirmado del dataset es `uciml/sms-spam-collection-dataset`:

```bash
kaggle datasets download -d uciml/sms-spam-collection-dataset -p data/prediccion_clasificacion/spam --unzip
```

Si eliges otro dataset desde Kaggle, ajusta el identificador al que aparece en la página del dataset.

### Resumen de datasets sugeridos

| Ejercicio | Tipo de problema | Dataset sugerido | Fuente | Ruta local sugerida |
|---|---|---|---|---|
| 1 | Imágenes, clasificación multiclase | `load_digits` | `scikit-learn` | No requiere archivos externos |
| 2 | Imágenes, aumento de datos | `load_digits` | `scikit-learn` | No requiere archivos externos |
| 3 | Tabular, clasificación binaria | Titanic: Machine Learning from Disaster | Kaggle | `data/prediccion_clasificacion/titanic/` |
| 3 alternativa | Tabular, clasificación binaria | Breast Cancer Wisconsin | `scikit-learn` | No requiere archivos externos |
| 4 | Texto, clasificación binaria | SMS Spam Collection Dataset | Kaggle | `data/prediccion_clasificacion/spam/` |


## Ejercicio 1: Clasificador con alto desempeño sobre dígitos

### Objetivo

Construir un clasificador de dígitos con buen desempeño y justificar la elección del modelo usando métricas e interpretación de errores.

Este ejercicio está inspirado en la primera actividad del capítulo, que propone mejorar el desempeño de un clasificador sobre imágenes de dígitos. En esta versión se espera que compares modelos y ajustes algunos hiperparámetros, sin limitarte a ejecutar una configuración por defecto.

### Datos sugeridos

Este ejercicio no requiere descargar datos externos ni usar Kaggle. Usa `load_digits` de `scikit-learn`, que está disponible directamente desde la biblioteca. Si cuentas con MNIST completo de forma local y reproducible, puedes usarlo, pero no es obligatorio.

La siguiente imagen recuerda por qué el problema no es trivial: los mismos dígitos pueden escribirse de muchas formas distintas.

![Ejemplos de dígitos escritos a mano usados para clasificación](img/clasificacion_digitos_mnist_libro.png)

Un buen clasificador no aprende una forma única de cada número; debe generalizar a variaciones de escritura.

### Instrucciones

1. Carga el dataset de dígitos.
2. Separa los datos en entrenamiento y prueba.
3. Entrena al menos dos clasificadores. Puedes considerar `KNeighborsClassifier`, `SGDClassifier`, `LogisticRegression` o `RandomForestClassifier`.
4. Si usas `KNeighborsClassifier`, prueba distintos valores de `n_neighbors` y `weights`.
5. Evalúa los modelos con accuracy.
6. Muestra la matriz de confusión del mejor modelo.
7. Identifica al menos dos pares de dígitos que el modelo confunda.
8. Visualiza ejemplos de errores y explica por qué podrían ocurrir.

Fragmento inicial sugerido:

```python
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

digits = load_digits()
X = digits.data
y = digits.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=42
)

# TODO: entrena y compara al menos dos modelos
```

### Preguntas de interpretación

- ¿Qué modelo obtuvo mejor desempeño general?
- ¿La diferencia entre modelos es suficientemente clara o es pequeña?
- ¿Qué clases se confunden con más frecuencia?
- ¿La matriz de confusión revela algo que la accuracy no muestra?
- ¿Qué error te parece más razonable al mirar las imágenes?

### Criterios mínimos de logro

- Entrenar al menos dos modelos.
- Reportar accuracy en test.
- Incluir una matriz de confusión.
- Interpretar errores entre clases específicas.
- Justificar qué modelo parece más adecuado y por qué.

### Reflexión

No basta con afirmar que un modelo es mejor porque tiene una accuracy más alta. Debes conectar la métrica con los errores observados y explicar si esos errores parecen aceptables para el problema.

## Ejercicio 2: Aumento de datos para clasificación de imágenes

### Objetivo

Explorar si aumentar artificialmente el conjunto de entrenamiento mejora el desempeño de un clasificador de dígitos.

El aumento de datos consiste en crear nuevas muestras a partir de transformaciones simples de los ejemplos originales. En imágenes, una transformación frecuente es desplazar ligeramente la imagen hacia arriba, abajo, izquierda o derecha.

### Datos sugeridos

Este ejercicio tampoco requiere Kaggle. Usa nuevamente `load_digits`, disponible directamente en `scikit-learn`. Cada imagen tiene tamaño 8 x 8, por lo que puedes transformar una fila de 64 valores en una matriz, desplazarla y volver a aplanarla.

### Instrucciones

1. Carga `load_digits` y separa entrenamiento/prueba.
2. Entrena un modelo base sin aumento de datos.
3. Crea una función que desplace una imagen un pixel en una dirección.
4. Aplica la función a cada imagen de entrenamiento para generar copias desplazadas.
5. Construye un nuevo conjunto de entrenamiento expandido.
6. Entrena el mismo modelo con el conjunto expandido.
7. Compara el desempeño antes y después del aumento.
8. Interpreta si el aumento ayudó, perjudicó o no cambió mucho el resultado.

Pseudocódigo orientativo:

```python
import numpy as np

def desplazar_imagen(imagen_8x8, direccion):
    # TODO: desplazar la matriz un pixel
    # direccion puede ser "arriba", "abajo", "izquierda" o "derecha"
    # cuida qué valor queda en los bordes nuevos
    return imagen_desplazada

# TODO: crear copias desplazadas del conjunto de entrenamiento
# TODO: entrenar el mismo modelo antes y después del aumento
```

Puedes implementar el desplazamiento con NumPy. Si decides usar `scipy.ndimage.shift`, verifica que esté disponible en el entorno antes de depender de esa función.

### Preguntas de interpretación

- ¿El aumento de datos mejora la accuracy?
- ¿Mejora por igual en todas las clases?
- ¿Qué pares de dígitos siguen siendo difíciles?
- ¿Hay desplazamientos que podrían empeorar algunos ejemplos?
- ¿Por qué el conjunto de prueba no debe aumentarse de la misma forma?

### Criterios mínimos de logro

- Implementar una estrategia de desplazamiento.
- Entrenar el mismo tipo de modelo con y sin aumento.
- Comparar métricas en test.
- Revisar matriz de confusión antes o después del aumento.
- Explicar si el aumento fue útil.

### Reflexión

El aumento de datos no es magia. Sirve cuando las transformaciones generadas son coherentes con variaciones reales del problema. Si las transformaciones producen imágenes poco naturales, pueden confundir al modelo.

## Ejercicio 3: Clasificación tabular

### Objetivo

Resolver un problema tabular de clasificación aplicando exploración, preprocesamiento, entrenamiento y evaluación.

El tercer ejercicio del capítulo propone trabajar con Titanic, un dataset tabular clásico. Ese dataset suele obtenerse desde Kaggle u otras fuentes externas. Para mantener la reproducibilidad del curso, puedes elegir una de estas dos rutas:

- **Ruta A**: usar Titanic si el dataset está disponible localmente o si el docente entrega los archivos.
- **Ruta B**: usar un dataset tabular incluido en scikit-learn, como `load_breast_cancer`, si quieres evitar descargas.

Explica en tu notebook cuál ruta elegiste y por qué.

### Datos sugeridos

**Opción principal en Kaggle:** [Titanic: Machine Learning from Disaster](https://www.kaggle.com/competitions/titanic).

Este dataset plantea una tarea de clasificación binaria: predecir si una persona pasajera sobrevivió al naufragio del Titanic. La variable objetivo es `Survived`, donde usualmente `1` indica que sobrevivió y `0` que no sobrevivió. Entre las columnas principales se encuentran `Pclass`, `Sex`, `Age`, `SibSp`, `Parch`, `Fare`, `Cabin` y `Embarked`, además de identificadores y campos textuales como `Name` y `Ticket`.

Ruta local sugerida:

```text
data/prediccion_clasificacion/titanic/
```

Después de descargar y descomprimir, deberías tener al menos:

```text
data/prediccion_clasificacion/titanic/train.csv
data/prediccion_clasificacion/titanic/test.csv
```

Código mínimo para comenzar:

```python
import pandas as pd

train_path = "data/prediccion_clasificacion/titanic/train.csv"
df = pd.read_csv(train_path)

df.head()
```

**Alternativa sin Kaggle:** usa `load_breast_cancer`, incluido en `scikit-learn`. Esta alternativa permite practicar clasificación tabular sin descargar archivos externos.

```python
from sklearn.datasets import load_breast_cancer

data = load_breast_cancer(as_frame=True)
X = data.data
y = data.target
```

Si usas Titanic, documenta la fuente del archivo y no incluyas credenciales ni pasos manuales imposibles de reproducir. Si usas la alternativa de scikit-learn, explica por qué elegiste una ruta sin descarga externa.

### Instrucciones

1. Carga el dataset elegido.
2. Describe la variable objetivo.
3. Revisa dimensiones, tipos de variables y valores faltantes.
4. Separa entrenamiento y prueba.
5. Construye un pipeline de preprocesamiento si hay variables numéricas y categóricas.
6. Entrena al menos un clasificador.
7. Evalúa con métricas apropiadas: accuracy, matriz de confusión, precisión, recall y F1-score.
8. Interpreta los errores más relevantes.

Estructura orientativa:

```python
from sklearn.model_selection import train_test_split

# TODO: cargar datos
# TODO: separar X e y
# TODO: definir preprocesamiento si corresponde
# TODO: entrenar un clasificador
# TODO: evaluar e interpretar
```

### Preguntas de interpretación

- ¿Qué representa la clase positiva?
- ¿El dataset está balanceado?
- ¿Qué métrica usarías para decidir si el modelo es útil?
- ¿Qué tipo de error sería más costoso en este problema?
- ¿Qué variables parecen aportar más información?

### Criterios mínimos de logro

- Usar un dataset tabular real.
- Separar correctamente `X` e `y`.
- Aplicar preprocesamiento cuando corresponda.
- Entrenar y evaluar un clasificador.
- Interpretar la matriz de confusión en el contexto del problema.

### Reflexión

En datos tabulares, muchas decisiones importantes ocurren antes de entrenar el modelo: selección de variables, tratamiento de valores faltantes, codificación de categorías y prevención de data leakage.

## Ejercicio 4: Clasificación de texto

### Objetivo

Construir una guía de solución para un clasificador de texto tipo spam/ham, transformando correos o mensajes en variables numéricas y evaluando errores.

El cuarto ejercicio del capítulo propone construir un clasificador de spam usando datos públicos de Apache SpamAssassin. Ese dataset puede requerir descarga externa, por lo que en el curso tienes dos alternativas:

- Usar el dataset original si puedes descargarlo y documentar el proceso de forma reproducible.
- Usar un conjunto local preparado por el docente.
- Crear una versión pequeña de prueba para validar el pipeline, dejando claro que no reemplaza una evaluación real.

### Datos sugeridos

**Opción principal en Kaggle:** [SMS Spam Collection Dataset](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset).

Este dataset contiene mensajes SMS etiquetados como `spam` o `ham`. La tarea es clasificación binaria de texto: predecir si un mensaje es spam. Según la versión descargada, el archivo puede venir como `spam.csv` o como un archivo de texto tabulado; revisa los nombres de columnas al cargarlo.

Ruta local sugerida:

```text
data/prediccion_clasificacion/spam/
```

Código mínimo para comenzar si el archivo descargado es `spam.csv`:

```python
import pandas as pd

spam_path = "data/prediccion_clasificacion/spam/spam.csv"
df = pd.read_csv(spam_path)

df.head()
```

Si tu descarga contiene un archivo llamado `SMSSpamCollection`, revisa si viene separado por tabulaciones y adapta la carga con `sep="\t"`.

**Alternativa original del libro:** Apache SpamAssassin Public Corpus. Es una fuente válida para spam/ham, pero requiere más trabajo de descarga, descompresión y parsing de correos. Si eliges esa ruta, documenta cuidadosamente cómo organizaste los archivos.

Para practicar la estructura del pipeline sin resolver el ejercicio completo, puedes comenzar con una lista pequeña:

```python
textos = [
    "gana dinero rápido ahora",
    "reunión de proyecto mañana",
    "oferta exclusiva por tiempo limitado",
    "adjunto el informe revisado",
]
etiquetas = [1, 0, 1, 0]  # 1 = spam, 0 = ham
```

Ese ejemplo mínimo solo sirve para revisar el flujo. El ejercicio completo debe usar más datos.

### Instrucciones

1. Reúne o carga ejemplos de mensajes `spam` y `ham`.
2. Separa entrenamiento y prueba.
3. Convierte texto en variables usando `CountVectorizer` o `TfidfVectorizer`.
4. Entrena un clasificador simple, por ejemplo `LogisticRegression`, `SGDClassifier` o `MultinomialNB`.
5. Evalúa con matriz de confusión, precisión, recall y F1-score.
6. Analiza falsos positivos y falsos negativos.
7. Propón mejoras al preprocesamiento.

Fragmento orientativo:

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

pipeline = Pipeline(steps=[
    ("vectorizador", TfidfVectorizer(lowercase=True)),
    ("modelo", LogisticRegression(max_iter=1000)),
])

# TODO: ajustar pipeline con textos de entrenamiento
# TODO: evaluar en textos de prueba
```

### Preguntas de interpretación

- ¿Qué significa un falso positivo en spam?
- ¿Qué significa un falso negativo?
- ¿Cuál de los dos errores sería más molesto para una persona usuaria?
- ¿Qué métrica priorizarías: precisión o recall?
- ¿Qué palabras o patrones podrían estar influyendo en las predicciones?

### Criterios mínimos de logro

- Transformar texto a variables numéricas.
- Entrenar un clasificador reproducible.
- Evaluar con precisión, recall, F1-score y matriz de confusión.
- Interpretar falsos positivos y falsos negativos.
- Explicar limitaciones del dataset usado.

### Reflexión

La clasificación de texto muestra que el aprendizaje supervisado no depende solo del modelo. La representación de los datos es central: cambiar cómo convertimos palabras en variables puede cambiar completamente el desempeño.

## Entrega esperada

Cada estudiante o grupo debe entregar:

- Un notebook desarrollado con los cuatro ejercicios, o cuatro notebooks separados con nombres claros.
- Código ejecutable de principio a fin.
- Separación explícita entre entrenamiento y prueba.
- Métricas obtenidas para cada ejercicio.
- Matrices de confusión cuando corresponda.
- Comparación entre modelos en los ejercicios 1 y 4, y cuando sea pertinente en el ejercicio 3.
- Interpretación escrita de resultados.
- Una conclusión breve por ejercicio.
- Una sección final con referencias si se usaron fuentes adicionales.

No se evaluará solo que el código ejecute. También se evaluará la calidad de las decisiones y la interpretación.

## Criterios de evaluación

Los criterios principales serán:

- Preparación correcta de datos.
- Uso adecuado de modelos de clasificación.
- Separación clara entre entrenamiento, validación y prueba.
- Evaluación con métricas pertinentes.
- Interpretación correcta de matrices de confusión.
- Comparación razonada entre modelos.
- Claridad de la explicación escrita.
- Reproducibilidad del trabajo.
- Conclusiones coherentes con la evidencia.

## Referencias

Géron, A. (2019). *Hands-on machine learning with Scikit-Learn, Keras, and TensorFlow: Concepts, tools, and techniques to build intelligent systems* (2nd ed.). O'Reilly Media. https://www.oreilly.com/library/view/hands-on-machine-learning/9781492032632/

Apache SpamAssassin Project. (n.d.). *Public corpus*. https://spamassassin.apache.org/old/publiccorpus/

Kaggle. (n.d.). *Titanic: Machine Learning from Disaster*. https://www.kaggle.com/competitions/titanic

Kaggle. (n.d.). *SMS Spam Collection Dataset*. https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset

Kaggle. (n.d.). *Kaggle API*. GitHub. https://github.com/Kaggle/kaggle-api

Pandas development team. (2024). *pandas.read_csv*. Pandas documentation. https://pandas.pydata.org/docs/reference/api/pandas.read_csv.html

Scikit-learn developers. (2024). *sklearn.datasets.load_digits*. Scikit-learn documentation. https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html

Scikit-learn developers. (2024). *sklearn.datasets.load_breast_cancer*. Scikit-learn documentation. https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html

Scikit-learn developers. (2024). *sklearn.neighbors.KNeighborsClassifier*. Scikit-learn documentation. https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html

Scikit-learn developers. (2024). *sklearn.model_selection.GridSearchCV*. Scikit-learn documentation. https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html

Scikit-learn developers. (2024). *sklearn.feature_extraction.text.CountVectorizer*. Scikit-learn documentation. https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html

Scikit-learn developers. (2024). *sklearn.feature_extraction.text.TfidfVectorizer*. Scikit-learn documentation. https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html

Scikit-learn developers. (2024). *sklearn.metrics.classification_report*. Scikit-learn documentation. https://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html
