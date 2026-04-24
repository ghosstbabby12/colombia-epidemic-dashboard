# Colombia Epidemic Dashboard

Dashboard epidemiológico interactivo para análisis y predicción de enfermedades transmitidas por mosquitos (dengue, zika, chikungunya) en Colombia, desarrollado para el concurso **Datos al Ecosistema 2026**.

## Objetivo

Brindar a autoridades de salud pública y ciudadanía una herramienta basada en datos abiertos e inteligencia artificial para:

- Visualizar la distribución histórica de casos de dengue por departamento y municipio
- Predecir brotes futuros integrando datos epidemiológicos, climáticos y demográficos
- Clasificar el nivel de riesgo por región (bajo / medio / alto)
- Emitir alertas tempranas ante posibles picos epidémicos

## Fuentes de datos

| Fuente | Descripción | Portal |
|--------|-------------|--------|
| INS – SIVIGILA | Notificaciones semanales de enfermedades | [datos.gov.co](https://www.datos.gov.co) |
| DANE | Proyecciones poblacionales por municipio | [dane.gov.co](https://www.dane.gov.co) |
| IDEAM | Temperatura y precipitación histórica | [ideam.gov.co](https://www.ideam.gov.co) |

## Arquitectura

```
Fuentes abiertas
      │
      ▼
 Ingesta (ingestion.py)
      │
      ▼
 ETL / Limpieza (preprocessing.py)
      │
      ▼
 Feature Engineering + Modelo ML (model.py)
      │
      ▼
 API REST (FastAPI) ──► Dashboard (React)
```

## Tecnologías

**Data & ML**
- Python 3.11+, Pandas, NumPy, GeoPandas
- XGBoost (clasificación de riesgo)
- Prophet (predicción de series temporales)
- Scikit-learn

**Backend**
- FastAPI + Uvicorn

**Frontend**
- React + Vite
- Plotly / Recharts (gráficas)
- Leaflet (mapas coropléticos)

## Estructura del proyecto

```
colombia-epidemic-dashboard/
├── data/
│   ├── raw/            # Datos originales sin modificar
│   └── processed/      # Datos limpios y transformados
├── src/
│   ├── data/           # Ingesta y preprocesamiento
│   ├── ml/             # Modelos de machine learning
│   ├── backend/        # API FastAPI
│   └── frontend/       # Aplicación React
├── notebooks/          # Análisis exploratorio (EDA)
├── docs/               # Documentación técnica
├── requirements.txt
└── README.md
```

## Instalación rápida

```bash
# 1. Clonar el repositorio
git clone https://github.com/ghosstbabby12/colombia-epidemic-dashboard.git
cd colombia-epidemic-dashboard

# 2. Crear entorno virtual
python -m venv venv
source venv/bin/activate   # macOS/Linux
# venv\Scripts\activate    # Windows

# 3. Instalar dependencias
pip install -r requirements.txt

# 4. Descargar datos
python src/data/ingestion.py

# 5. Preprocesar
python src/data/preprocessing.py

# 6. Levantar API
uvicorn src.backend.app:app --reload
```

## Equipo

| Rol | Responsabilidad |
|-----|----------------|
| Ingeniero de Datos | ETL, integración de datasets |
| Analista de Datos | EDA, feature engineering, modelo ML |
| Ingeniera de Software | Backend, Frontend, despliegue |

## Concurso

Proyecto participante en **Datos al Ecosistema 2026** – Convocatoria nacional de soluciones con datos abiertos e inteligencia artificial del Gobierno de Colombia.

## Licencia

MIT © 2026 – Equipo Colombia Epidemic Dashboard




Prueba modelo:

python -m src.ml.model --predict --month 5 --department NARIÑO --city PASTO --disease Dengue

python -m src.ml.model --predict --month 8 --department CHOCO --city QUIBDO --disease Malaria

python -m src.ml.model --predict --month 10 --department VALLE --city CALI --disease Chikungunya

