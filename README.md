# Análisis Exploratorio y Modelo Predictivo — Accidentes de Tráfico en Madrid

Análisis exploratorio sobre los 49.340 siniestros de tráfico registrados por la
Policía Municipal de Madrid durante 2024, junto con un modelo de Regresión
Logística para predecir la positividad en alcohol de las personas implicadas.

## Preguntas de negocio

- ¿En qué franja horaria se producen los accidentes de mayor gravedad?
- ¿Qué condiciones influyen en una alta lesividad?
- ¿Es viable predecir la positividad en alcoholemia a partir de las
  circunstancias del accidente?

## Datos

Registro de accidentes de tráfico de la ciudad de Madrid (2024), publicado como
dato abierto por el Ayuntamiento:
https://datos.madrid.es/portal/site/egob

Los límites de distrito proceden del
[Geoportal del Ayuntamiento de Madrid](https://geoportal.madrid.es/IDEAM_WBGEOPORTAL/dataset.iam?id=541f4ef6-762b-11e9-861d-ecb1d753f6e8).

> **Nota:** el CSV de accidentes no se incluye en el repositorio por su tamaño.
> Descárgalo del portal y colócalo en la raíz del proyecto con el nombre
> `300228-2-accidentes-trafico-detalle-csv.csv`.

## Metodología

**Preparación de los datos**
- Normalización de nombres de columna y conversión de tipos (fechas y horas).
- Tratamiento de nulos, distinguiendo el valor ausente real del no registrado
  (caso de `positiva_droga`, descartada por no ser interpretable).
- Variables derivadas propias: `franja_horaria` y `tipo_via`, esta última
  clasificando texto libre de localización en cinco categorías.
- Depuración de coordenadas UTM para el análisis geográfico.

**Análisis**
- Univariante y bivariante sobre gravedad, franja horaria, tipo de vía,
  vehículo y condiciones meteorológicas.
- Análisis geoespacial por distrito con GeoPandas.

**Modelado**
- Regresión Logística con codificación OneHot para variables nominales y
  mapeo ordinal para la categoría de edad.
- División estratificada train/test: la variable objetivo está fuertemente
  desbalanceada (solo un 2,93 % de positivos).
- Compensación del desequilibrio con `class_weight='balanced'`, validada frente
  a alternativas (`None` y pesos personalizados).
- Evaluación con **recall, F1-score y matriz de confusión**. Se descarta
  *accuracy*: con un 97 % de negativos, un modelo que nunca prediga positivo
  acertaría el 97 % de las veces sin detectar un solo caso.

## Resultados

| Conjunto      | Recall | F1-score |
|---------------|--------|----------|
| Entrenamiento | 0,7504 | 0,1744   |
| Test          | 0,7422 | 0,1713   |

El modelo identifica correctamente **casi 3 de cada 4 positivos** en datos no
vistos. La similitud entre ambos conjuntos indica que ha aprendido patrones
reales y no memorizado el entrenamiento.

El F1-score bajo refleja un número elevado de falsos positivos, consecuencia
directa de priorizar el recall: en seguridad vial, el falso negativo —no
detectar a alguien que ha dado positivo— es el escenario costoso.

**Principales hallazgos**
- La **franja horaria** es el factor más determinante: la madrugada concentra
  el mayor riesgo de positividad.
- Le siguen en peso los **vehículos pesados y de carga** y los **turismos**.
- Entre los tipos de siniestro destacan **choques y vuelcos**.
- El distrito aporta información útil al modelo: eliminarlo empeora las métricas.

## Estructura del repositorio

```
eda_accidentes_trafico.ipynb   Notebook principal: EDA y modelado
models.py                      Clase AlcoholLogisticModel (fit/predict/evaluate/save/load)
testing.py                     Script de prueba de la clase
Distritos.geojson              Límites de los distritos de Madrid
```

## Dependencias

```
pandas · numpy · matplotlib · seaborn · scikit-learn · geopandas
```

## Ejecución

1. Descargar el CSV de accidentes y colocarlo en la raíz del proyecto.
2. Ejecutar el notebook completo con *Restart Kernel and Run All Cells*.
3. Para probar la clase del modelo: `python testing.py`
