# Predicción de Licitaciones Ganables - Proyecto de Data Science

Este repositorio contiene un proyecto universitario para el curso de **Data Science**. El objetivo principal es predecir la probabilidad de adjudicación en procesos de compras públicas en Chile, utilizando datos abiertos de **ChileCompra**.

> **Nota:** El proyecto se encuentra en desarrollo. Actualmente contamos con análisis exploratorio de datos (EDA), scripts de preprocesamiento e ingesta, y un modelo base en (`modelo_preliminar.py`), pero **aún falta desarrollar y probar modelos predictivos más robustos y avanzados**.

## Contexto y Definición del Problema

El sistema de compras públicas en Chile (Mercado Público) es fundamental para la contratación entre el Estado y el sector privado. Sin embargo, las empresas suelen participar en licitaciones bajo una alta incertidumbre. Esto ocasiona que incurran en cuantiosos costos operacionales preparando ofertas de baja viabilidad, lo que sumado a la sobreparticipación, merma la probabilidad individual de adjudicación de manera ineficiente y subóptima.

Este problema puede modelarse mediante un enfoque estructurado y cuantitativo. Metodológicamente, se estructura este desafío como una **tarea de clasificación supervisada binaria**, donde se busca lograr una predicción probabilística (un *score* de 0 a 1) que indique el éxito de un par _proveedor-licitación_.

## Objetivos

- **Objetivo General:** Desarrollar un modelo predictivo capaz de estimar la probabilidad de éxito de una empresa en una licitación específica para funcionar como herramienta de apoyo en la toma de decisiones empresariales.
- **Objetivos Específicos:**
  1. Integrar y consolidar las distintas fuentes de datos histórica.
  2. Diseñar variables (features) que capturen la relación y el contexto en la adjudicación.
  3. Desarrollar y evaluar un modelo predictivo con técnicas de Machine Learning para datos tabulares.
  4. Generar un indicador interpretable que estructure un ranking de oportunidades de licitación.

## Arquitectura del Proyecto

Dada la estructura del repositorio de trabajo, las componentes se dividen en los siguientes tópicos:

- **Adquisición y Transformación de Datos** (`Download.py`, `Convert.py`, `Data.py`): Encargados de la obtención masiva en local (ver carpetas `Datos/` y `descargas/`) y preparación inicial del dataset relacional.
- **Limpieza y Preparación** (`limpieza.py`): Filtros de limpieza, consolidación de la variable objetivo y normalización de entidades.
- **Análisis Exploratorio / EDA** (`EDA.py`, `print.py`): Descubrimiento y visualización de un resumen estadístico (con gráficos albergados en el directorio `EDA_Resultados/`).
- **Modelamiento Base** (`modelo_preliminar.py`): Aproximación introductoria para el ensamblaje o testeo del flujo de predicción continuo.

## Próximos Pasos (Trabajo Futuro)

- [ ] **Mejorar los Modelos Predictivos:** Entrenar y optimizar modelos de clasificación avanzados superando el desempeño del modelo preliminar utilizando métricas de evaluación apropiadas (PRECISION, AUC-ROC y log-loss).
- [ ] **Incorporación de Datos Adicionales:** Sumar variables externas del **Servicio de Impuestos Internos (SII)** para comprender el comportamiento por rubro o tamaño de empresas, así como integrar datos geográficos logísticos.
