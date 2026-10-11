# DocFlow AI — Clasificación inteligente de documentos

**DocFlow AI** es una aplicación web que permite identificar diferentes tipos de documentos a partir de imágenes y entregar recomendaciones que ayuden a organizarlos.

Desarrollamos este proyecto para el certamen de **Inteligencia Artificial Generativa**, utilizando modelos preentrenados de visión por computador, Streamlit para la aplicación web y Gemini para generar recomendaciones.

**Aplicación web:** [https://docflow-ai-bsotoh.streamlit.app/](https://docflow-ai-bsotoh.streamlit.app/)

**Repositorio GitHub:** [https://github.com/bsotoh/DocFlow_AI](https://github.com/bsotoh/DocFlow_AI)

## 1. Problema y propósito del proyecto

Actualmente, muchas organizaciones trabajan con grandes cantidades de documentos, como facturas, cartas, formularios e informes. Clasificarlos manualmente puede tomar bastante tiempo, especialmente cuando se manejan muchos archivos diferentes.

Además, si un documento se clasifica incorrectamente, puede ser más difícil encontrarlo posteriormente o utilizarlo cuando se necesita.

A partir de esta problemática desarrollamos **DocFlow AI**, una herramienta que busca hacer más sencilla la identificación y organización de documentos.

Nuestra aplicación permite subir una imagen, identificar a qué categoría pertenece y mostrar qué tan seguro está el modelo de su clasificación. También incorporamos Gemini para entregar recomendaciones que ayuden a gestionar el documento.

La idea es facilitar parte del trabajo manual, manteniendo siempre la posibilidad de que una persona revise los resultados.

## 2. Dataset utilizado

Para entrenar nuestros modelos utilizamos **RVL-CDIP**, un conjunto de datos publicado en 2015 por Adam W. Harley, Alex Ufkes y Konstantinos G. Derpanis.

Este dataset contiene **400.000 imágenes de documentos escaneados**, distribuidas en 16 categorías, con 25.000 imágenes por categoría.

Las categorías son:

1. Carta
2. Formulario
3. Correo electrónico
4. Manuscrito
5. Publicidad
6. Informe científico
7. Publicación científica
8. Especificación
9. Carpeta
10. Noticia
11. Presupuesto
12. Factura
13. Presentación
14. Cuestionario
15. Currículum
16. Memorándum

### Distribución de los datos

El dataset original contiene 320.000 imágenes de entrenamiento, 40.000 de validación y 40.000 de prueba.

Sin embargo, debido al tamaño del conjunto de datos y al tiempo necesario para entrenar los modelos, decidimos utilizar una **muestra aleatoria estratificada del 40%** para entrenamiento y validación.

La distribución que utilizamos fue la siguiente:

| Conjunto      | Dataset original |  DocFlow AI |
| ------------- | ---------------: | ----------: |
| Entrenamiento |          320.000 |     128.000 |
| Validación    |           40.000 |      16.000 |
| Prueba        |           40.000 |      39.999 |
| **Total**     |      **400.000** | **183.999** |

El muestreo estratificado nos permitió mantener la proporción de las 16 categorías. Además, utilizamos las mismas imágenes para entrenar y comparar ambos modelos.

En el caso de la prueba, mantuvimos el conjunto original completo, pero una imagen presentó problemas de lectura. Por eso, la evaluación final se realizó con 39.999 imágenes.

### Origen y licencia

- **Dataset:** [https://huggingface.co/datasets/aharley/rvl_cdip](https://huggingface.co/datasets/aharley/rvl_cdip)
- **Referencia original:** [https://adamharley.com/rvl-cdip/](https://adamharley.com/rvl-cdip/)
- **Autores:** Adam W. Harley, Alex Ufkes y Konstantinos G. Derpanis.
- **Año:** 2015.
- **Origen:** colección IIT-CDIP, relacionada con documentos históricos de la industria tabacalera estadounidense.
- **Licencia:** Hugging Face la identifica como `other`. Por el origen de los documentos, es importante revisar las condiciones de uso de la colección original y no asumir que todas las imágenes tienen una licencia abierta.

### Limitaciones del dataset

Aunque RVL-CDIP contiene una gran cantidad de imágenes, hay algunas limitaciones que debemos considerar.

En primer lugar, los documentos son principalmente archivos históricos de Estados Unidos y están mayoritariamente en inglés. Esto significa que podrían ser diferentes a los documentos utilizados actualmente en una empresa chilena.

También existen imágenes con baja calidad, manchas o problemas de legibilidad, lo que puede dificultar su clasificación.

Por último, aunque el dataset está equilibrado entre sus 16 categorías, en una organización real algunas clases de documentos podrían aparecer con mucha más frecuencia que otras.

Por estas razones, los resultados obtenidos no necesariamente serían los mismos si utilizáramos la aplicación con documentos de otros contextos.

## 3. Modelos utilizados y resultados

Para clasificar los documentos trabajamos con dos modelos preentrenados:

- **EfficientNet-B0**
- **ResNet50**

Ambos habían aprendido a reconocer características visuales a partir de imágenes de ImageNet. Aprovechamos ese aprendizaje y entrenamos nuevas capas para que pudieran identificar las 16 categorías de nuestro dataset.

Este procedimiento se conoce como *transfer learning*.

Para comparar sus resultados de manera justa, utilizamos las mismas muestras de entrenamiento, validación y prueba.

### Comparación de modelos

Los resultados obtenidos sobre las 39.999 imágenes válidas del conjunto de prueba fueron:

| Métrica                 | EfficientNet-B0 |  ResNet50 |
| ----------------------- | --------------: | --------: |
| Accuracy                |          70,93% |    70,14% |
| Precision               |          71,04% |    70,65% |
| Recall                  |          70,85% |    70,03% |
| F1-score                |          70,66% |    70,02% |
| Tiempo de entrenamiento |       60,20 min | 87,97 min |
| Tiempo de evaluación    |        2,76 min |  4,38 min |

*Precision, recall y F1-score corresponden al promedio macro de las 16 categorías.*

### ¿Por qué elegimos EfficientNet-B0?

Después de comparar ambos modelos, decidimos utilizar **EfficientNet-B0**, porque obtuvo resultados ligeramente mejores y necesitó menos tiempo para entrenarse y evaluarse.

Su accuracy fue de **70,93%**, lo que significa que clasificó correctamente aproximadamente 71 de cada 100 imágenes del conjunto de prueba.

Sin embargo, también observamos que algunos documentos resultan más difíciles de distinguir, especialmente cuando comparten características visuales parecidas.

Por ejemplo, una factura clasificada incorrectamente podría terminar en una carpeta equivocada y generar dificultades para encontrarla o procesarla después.

Esto nos permitió entender que no basta con mirar el porcentaje general de aciertos: también es importante considerar qué documentos se confunden y qué consecuencias podrían tener esos errores.

Por eso, DocFlow AI está pensado como una herramienta de apoyo y no como un sistema que tome decisiones definitivas sin revisión humana.

## 4. ¿Cómo funciona DocFlow AI?

Nuestra aplicación combina clasificación de imágenes con recomendaciones generadas por inteligencia artificial.

El proceso es bastante sencillo:

1. El usuario sube una imagen de un documento.
2. EfficientNet-B0 analiza sus características visuales.
3. La aplicación muestra la categoría identificada y su nivel de confianza.
4. El usuario puede revisar las probabilidades de las 16 categorías.
5. Si lo desea, puede solicitar recomendaciones mediante Gemini.

La aplicación está dividida en tres secciones principales.

### Clasificar documento

Aquí el usuario puede cargar una imagen en formato PNG, JPG o TIFF.

Una vez procesada, la aplicación muestra:

- El documento cargado.
- La categoría más probable.
- La confianza de la clasificación.
- Un gráfico con las probabilidades de las 16 categorías.
- Una advertencia cuando la confianza es baja.

### Rendimiento del modelo

En esta sección presentamos los resultados obtenidos durante la evaluación.

Se pueden consultar las principales métricas de EfficientNet-B0 y compararlas con las de ResNet50.

También explicamos algunas limitaciones que identificamos durante el desarrollo.

### Asistente IA

Esta sección utiliza Gemini para generar recomendaciones relacionadas con el documento clasificado.

Por ejemplo, si la aplicación identifica un manuscrito, Gemini puede sugerir formas de digitalizarlo, organizarlo y conservarlo.

Estas recomendaciones sirven como orientación y no reemplazan la revisión de una persona.

## 5. Integración con Gemini

Además de clasificar documentos, queríamos que DocFlow AI pudiera entregar alguna ayuda adicional al usuario.

Por eso incorporamos **Gemini**, un modelo de inteligencia artificial generativa al que nos conectamos mediante una API externa.

Cuando el usuario presiona el botón para generar recomendaciones, Gemini recibe:

- La categoría identificada por EfficientNet-B0.
- El nivel de confianza de la clasificación.
- Instrucciones para entregar recomendaciones de organización y revisión.

Diseñamos el prompt para que las respuestas sean claras, breves y relacionadas con el tipo de documento identificado.

También le indicamos que no invente información específica del documento y que recomiende una revisión manual cuando la confianza sea baja.

**Es importante aclarar que Gemini no recibe la imagen original.** Sus recomendaciones se basan únicamente en la categoría y la confianza entregadas por el modelo de clasificación.

Para proteger la conexión, la clave de Gemini se guarda mediante **Streamlit Secrets** y no se publica en GitHub.

## 6. Aplicación web

DocFlow AI fue desarrollado con Streamlit y está publicado en Streamlit Community Cloud.

**Enlace:** [https://docflow-ai-bsotoh.streamlit.app/](https://docflow-ai-bsotoh.streamlit.app/)

### Cómo utilizar la aplicación

1. Abrir el enlace de DocFlow AI.
2. Entrar a la pestaña **Clasificar documento**.
3. Subir una imagen en formato PNG, JPG o TIFF.
4. Revisar la categoría identificada y su confianza.
5. Consultar el gráfico de probabilidades.
6. Entrar a **Asistente IA**.
7. Presionar **Generar recomendaciones con Gemini** para recibir sugerencias.

La versión publicada funciona directamente desde el navegador, sin necesidad de instalar programas.

## 7. Organización del repositorio

El proyecto está organizado con los siguientes archivos y carpetas:

| Archivo o carpeta        | Contenido                                             |
| ------------------------ | ----------------------------------------------------- |
| `app.py`                 | Código de la aplicación web                           |
| `requirements.txt`       | Librerías necesarias para ejecutarla                  |
| `README.md`              | Información y documentación del proyecto              |
| `notebooks/`             | Notebook con el desarrollo y los resultados obtenidos |
| `modelo/`                | Modelo EfficientNet-B0 entrenado                      |
| `ejemplos/`              | Imágenes para probar la aplicación                    |
| `.streamlit/config.toml` | Configuración visual de Streamlit                     |
| `.gitignore`             | Archivos privados que no deben publicarse             |

El dataset completo no se incluye en GitHub debido a su tamaño. Para consultarlo o descargarlo se puede utilizar el enlace indicado en la sección 2.

## 8. Instalación y ejecución local

Aunque DocFlow AI puede utilizarse desde su enlace público, también es posible ejecutarlo en un computador.

### Requisitos

- Python 3.12.
- Git.
- Librerías indicadas en `requirements.txt`.
- Una clave de Gemini para utilizar el asistente de IA.

### Paso 1. Descargar el repositorio

Clonar el proyecto:

`git clone https://github.com/bsotoh/DocFlow_AI.git`

Entrar a su carpeta:

`cd DocFlow_AI`

### Paso 2. Crear un entorno virtual

Si se utiliza Anaconda, ejecutar:

`conda create -n DocFlow_Web python=3.12 -y`

`conda activate DocFlow_Web`

### Paso 3. Instalar las librerías

Ejecutar:

`python -m pip install -r requirements.txt`

### Paso 4. Configurar Gemini

Crear un archivo llamado `.streamlit/secrets.toml` dentro de la carpeta del proyecto.

En su interior, agregar:

`GEMINI_API_KEY = "TU_CLAVE_PRIVADA"`

La clave puede obtenerse desde Google AI Studio:

[https://aistudio.google.com/apikey](https://aistudio.google.com/apikey)

**Este archivo debe mantenerse privado y no subirse a GitHub.**

### Paso 5. Ejecutar la aplicación

Desde la carpeta principal, ejecutar:

`python -m streamlit run app.py`

Después, abrir la dirección local que indique Streamlit, normalmente:

[http://localhost:8501](http://localhost:8501)

### Sobre el entrenamiento del modelo

El entrenamiento original se realizó utilizando Python 3.9 y TensorFlow 2.10.1.

Para repetirlo desde cero sería necesario descargar RVL-CDIP, preparar las rutas correspondientes y utilizar un entorno compatible.

Sin embargo, **no es necesario volver a entrenar los modelos para utilizar DocFlow AI**, ya que EfficientNet-B0 está guardado y listo para realizar predicciones.

La aplicación publicada utiliza Python 3.12 y TensorFlow 2.19.1, con compatibilidad para Keras anterior.

Las métricas que presentamos corresponden a la evaluación realizada durante el entrenamiento original.

## 9. Limitaciones y posibles mejoras

Durante el desarrollo conseguimos que DocFlow AI clasificara documentos y generara recomendaciones desde una aplicación web. Sin embargo, sabemos que todavía existen aspectos que podríamos mejorar.

Entre ellos se encuentran:

- **Mejorar la clasificación:** seguir ajustando el modelo para distinguir mejor los documentos que actualmente confunde.
- **Trabajar con documentos más variados:** incorporar imágenes actuales y en español para comprobar cómo funciona en otros contextos.
- **Reducir los errores:** agregar mecanismos de revisión, especialmente cuando el modelo tenga dudas.
- **Incorporar nuevas funciones:** permitir que la aplicación también lea el texto de los documentos, para entregar recomendaciones más específicas.

Además, debemos recordar que la confianza mostrada es una estimación del modelo, por lo que una confianza alta no garantiza que la clasificación sea correcta.

Consideramos que DocFlow AI es un buen punto de partida para facilitar la organización documental, pero todavía necesita mejoras antes de utilizarse como una solución completamente automatizada en una empresa.

## 10. Consideraciones éticas y uso responsable

DocFlow AI fue desarrollado con fines académicos.

Aunque buscamos facilitar la clasificación de documentos, reconocemos que el modelo puede equivocarse. Por esta razón, recomendamos revisar los resultados antes de utilizarlos para tomar decisiones importantes.

También es necesario tener cuidado con la información personal o confidencial que puedan contener los documentos. No deberían cargarse archivos sensibles sin la autorización correspondiente.

Por otra parte, el uso del dataset debe respetar las condiciones establecidas por sus fuentes originales.

## 11. Uso de inteligencia artificial durante el desarrollo

Durante el proyecto utilizamos herramientas de inteligencia artificial como apoyo en distintas etapas del trabajo.

**ChatGPT** nos ayudó principalmente a revisar y adaptar códigos de Python, resolver errores, apoyarnos en la redacción del README.md, optimizar el flujo de los datos al entrar en los modelos y desarrollar la interfaz web.

También utilizamos herramientas de inteligencia artificial dentro de la propia aplicación:

- **EfficientNet-B0 y ResNet50:** modelos utilizados para aprender a clasificar los documentos.
- **Gemini:** modelo integrado mediante API para generar recomendaciones según los resultados de la clasificación.

A lo largo del desarrollo fuimos adaptando los códigos a nuestro problema, probando las distintas funcionalidades y comparando los resultados obtenidos.

## 12. Integrantes

Proyecto desarrollado por:

- **Javiera Freire**
- **Isidora Riquelme**
- **Benjamín Soto**

**Asignatura:** Inteligencia Artificial Generativa.

**Evaluación:** Certamen 1 — Aplicación web con modelos preentrenados e IA generativa.

---

**DocFlow AI — Clasificación y organización documental con inteligencia artificial.**
