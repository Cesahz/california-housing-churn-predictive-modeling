# California Housing & Churn — Modelado Predictivo

Proyecto de Machine Learning supervisado sobre dos dominios distintos: predicción de precios de vivienda en California y predicción de abandono de clientes (churn) en telecomunicaciones. Continuación directa del [análisis exploratorio previo](https://github.com/Cesahz/california-housing-energy-eda).

El desafío plantea dos preguntas centrales: *¿cuánto vale una vivienda dado su contexto?* y *¿qué clientes están en riesgo real de abandonar el servicio?* Este repositorio contiene el pipeline completo — desde feature engineering hasta modelos entrenados, evaluados e interpretados — para ambos problemas.

## Datasets utilizados

| Dataset | Fuente | Descripción |
|---|---|---|
| California Housing Prices | [Kaggle](https://www.kaggle.com/datasets/camnugent/california-housing-prices/data) | Precios de vivienda en California (1990), a nivel de bloque censal |
| Telco Customer Churn | [Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) | Datos de clientes de una empresa de telecomunicaciones y su abandono del servicio |

## Estructura del repositorio

```
├── data/
│   ├── raw/                                        # Datasets originales, sin modificar
│   ├── processed/                                  # Datasets curados y con feature engineering
│   └── recursos/                                    # Material de apoyo y documentacion
├── notebooks/
│   ├── california-housing-feature-engineering.ipynb
│   ├── 2_0-california-housing-regression.ipynb
│   ├── churn-eda.ipynb
│   └── 2_1-churn-classification.ipynb
├── outputs/                                          # Graficos y reportes exportados
├── requirements.txt
└── README.md
```

## Notebook 1 — California Housing (Regresión)

Predicción del precio medio de vivienda a partir de características del bloque censal, construida sobre el dataset ya limpio del EDA previo.

**Proceso:** feature engineering (clustering espacial K-Means, cruces de variables) → selección de features con justificación estadística → train/test split → escalado → regresión lineal (OLS) → evaluación → interpretación de coeficientes.

**Hallazgos principales:**
- `median_income` es el predictor dominante (β = +48,378), consistente con la correlación de 0.65 observada en el EDA.
- Reemplazar latitud/longitud por 10 zonas geográficas obtenidas con K-Means logró un efecto de ubicación comparable al de las coordenadas crudas, con la ventaja de ser directamente interpretable por zona.
- `rooms_per_household` cambia de signo entre su correlación simple (+0.26) y su coeficiente en el modelo multivariado (-15,370) — evidencia de multicolinealidad indirecta vía `median_income`.
- El modelo subestima sistemáticamente las viviendas de alto valor (heterocedasticidad visible en el análisis de residuos), consistente con las limitaciones de un modelo lineal frente a un mercado de lujo con dinámicas no lineales.

**Resultado:** RMSE ≈ $55,000 · R² ≈ 0.66 (el mejor resultado público en Kaggle con modelos no lineales alcanza RMSE $48,000).

## Notebook 2 — Churn de Clientes (Clasificación)

Predicción de abandono de clientes de telecomunicaciones a partir de datos demográficos, de contrato y de uso del servicio.

**Proceso:** EDA propio (limpieza de `TotalCharges`, análisis de balance de clases) → encoding sin redundancia → train/test split estratificado → escalado → regresión logística → evaluación con Accuracy/Precision/Recall → comparación con `class_weight` balanceado.

**Hallazgos principales:**
- `Contract` es el predictor más fuerte: los clientes con contrato mensual tienen 15 veces más probabilidad de abandonar que los de contrato a dos años (42.7% vs 2.8%).
- El dataset está moderadamente desbalanceado (73.46% No churn / 26.54% Yes), lo que hace que Accuracy por sí sola sea insuficiente para evaluar el modelo.
- Se entrenaron dos modelos con el mismo F1 (0.61) pero trade-offs distintos: uno prioriza Precision (0.66), el otro Recall (0.78). La elección del modelo apropiado depende del costo relativo de cada tipo de error para el negocio, no de un único número "mejor".
- El mejor resultado público en Kaggle con modelos más complejos (Random Forest, XGBoost) alcanza F1 de 0.63 — apenas 2 puntos sobre el modelo lineal base, evidencia de que el problema tiene un límite de información inherente (el churn depende de factores no capturados en el dataset).

**Resultado:** Accuracy 0.81 · Precision 0.66 · Recall 0.56 (modelo base) — Accuracy 0.74 · Precision 0.51 · Recall 0.78 (modelo balanceado).

## Herramientas

Python · Pandas · NumPy · Scikit-learn · Matplotlib · Seaborn · Jupyter Notebook

## Instalación

```bash
git clone https://github.com/Cesahz/california-housing-churn-predictive-modeling.git
cd california-housing-churn-predictive-modeling
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
jupyter notebook
```