# Arquitectura del Sistema

## Flujo de datos

```
[datos.gov.co]  [dane.gov.co]  [ideam.gov.co]
      │                │               │
      └────────────────┴───────────────┘
                       │
              src/data/ingestion.py
                       │
              data/raw/*.csv
                       │
           src/data/preprocessing.py
                       │
           data/processed/features.parquet
                       │
              src/ml/model.py
              ┌─────────┴──────────┐
         XGBClassifier         Prophet
         (riesgo)           (pronóstico)
              └─────────┬──────────┘
                        │
             src/backend/app.py  (FastAPI)
                        │
             src/frontend/       (React)
```

## Descripción de componentes

| Componente | Tecnología | Responsabilidad |
|------------|-----------|-----------------|
| Ingesta | Python + sodapy | Descarga desde APIs abiertas |
| ETL | Pandas + GeoPandas | Limpieza, integración, feature engineering |
| Modelo ML | XGBoost + Prophet | Clasificación de riesgo y pronóstico |
| API | FastAPI | Exponer predicciones al frontend |
| Frontend | React + Plotly | Visualización interactiva |

## Variables del modelo

### Features (entrada)
- `week` – Semana epidemiológica (1-52)
- `month` – Mes derivado de la semana
- `is_rainy_season` – Binario: temporada de lluvias
- `cases_lag_2w`, `cases_lag_4w`, `cases_lag_8w` – Casos rezagados
- `cases_ma4` – Media móvil 4 semanas
- `temp_avg` – Temperatura media (°C)
- `precipitation_mm` – Precipitación semanal (mm)
- `humidity_pct` – Humedad relativa (%)
- `precip_x_humidity` – Interacción precipitación × humedad

### Target (salida)
- `risk_label` – Clasificación de riesgo: `bajo` / `medio` / `alto`
  - Bajo:  ≤ 50 casos/semana en el departamento
  - Medio: 51 – 200 casos/semana
  - Alto:  > 200 casos/semana

## Decisiones de diseño

- **XGBoost** sobre Random Forest: mejor manejo de clases desbalanceadas con `scale_pos_weight`.
- **Prophet** sobre ARIMA: captura estacionalidad anual del dengue sin configuración manual de parámetros.
- **Parquet** como formato intermedio: lectura eficiente con tipos correctos y compresión nativa.
- **FastAPI** sobre Flask: validación automática con Pydantic, docs OpenAPI integradas.
