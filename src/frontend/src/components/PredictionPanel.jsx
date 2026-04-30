import { useEffect, useMemo, useState } from 'react'
import { api } from '../api'

const MESES = [
  { value: 1, label: 'Ene' },
  { value: 2, label: 'Feb' },
  { value: 3, label: 'Mar' },
  { value: 4, label: 'Abr' },
  { value: 5, label: 'May' },
  { value: 6, label: 'Jun' },
  { value: 7, label: 'Jul' },
  { value: 8, label: 'Ago' },
  { value: 9, label: 'Sep' },
  { value: 10, label: 'Oct' },
  { value: 11, label: 'Nov' },
  { value: 12, label: 'Dic' },
]

const ENFERMEDADES = ['Dengue', 'Malaria', 'Chikungunya']

const prettyName = (value) => {
  return String(value || '')
    .toLowerCase()
    .split(' ')
    .filter(Boolean)
    .map((word) => word.charAt(0).toUpperCase() + word.slice(1))
    .join(' ')
}

const getRiskClass = (level) => {
  const value = String(level || '').toLowerCase()

  if (value.includes('alto')) return 'risk-high'
  if (value.includes('medio')) return 'risk-medium'
  if (value.includes('bajo')) return 'risk-low'

  return 'risk-none'
}

export default function PredictionPanel() {
  const [disease, setDisease] = useState('Dengue')
  const [month, setMonth] = useState(5)

  const [departments, setDepartments] = useState([])
  const [cities, setCities] = useState([])

  const [department, setDepartment] = useState('')
  const [city, setCity] = useState('')

  const [prediction, setPrediction] = useState(null)
  const [loadingDepartments, setLoadingDepartments] = useState(false)
  const [loadingCities, setLoadingCities] = useState(false)
  const [loadingPrediction, setLoadingPrediction] = useState(false)
  const [error, setError] = useState('')

  const selectedMonthLabel = useMemo(() => {
    return MESES.find((item) => item.value === Number(month))?.label || month
  }, [month])

  useEffect(() => {
    setLoadingDepartments(true)
    setError('')
    setDepartments([])
    setDepartment('')
    setCities([])
    setCity('')
    setPrediction(null)

    api.departamentosPrediccion({ disease, month })
      .then((response) => {
        const values = response.data || []
        setDepartments(values)

        if (values.length) {
          setDepartment(values[0].value)
        }
      })
      .catch((err) => {
        setError(err.message || 'No fue posible cargar departamentos con predicciones.')
      })
      .finally(() => {
        setLoadingDepartments(false)
      })
  }, [disease, month])

  useEffect(() => {
    if (!department) return

    setLoadingCities(true)
    setCities([])
    setCity('')
    setPrediction(null)

    api.ciudadesPrediccion({ disease, month, department })
      .then((response) => {
        const values = response.data || []
        setCities(values)

        if (values.length) {
          setCity(values[0].value)
        }
      })
      .catch((err) => {
        setError(err.message || 'No fue posible cargar ciudades con predicciones.')
      })
      .finally(() => {
        setLoadingCities(false)
      })
  }, [disease, month, department])

  const handlePredict = async () => {
    if (!disease || !month || !department || !city) {
      setError('Selecciona enfermedad, mes, departamento y ciudad.')
      return
    }

    setError('')
    setPrediction(null)
    setLoadingPrediction(true)

    try {
      const response = await api.prediccion({
        disease,
        month,
        department,
        city,
      })

      setPrediction(response.data)
    } catch (err) {
      setError(err.message || 'No fue posible generar la predicción.')
    } finally {
      setLoadingPrediction(false)
    }
  }

  return (
    <section className="prediction-section">
      <div className="section-heading">
        <span className="eyebrow">Predicción interactiva</span>
        <h2>Estimación de casos por mes, departamento y ciudad</h2>
        <p>
          Selecciona la enfermedad, el mes y la ubicación. Al presionar el botón,
          el sistema consulta la predicción precalculada en Supabase.
        </p>
      </div>

      <div className="prediction-layout">
        <div className="prediction-controls">
          <div className="control-block">
            <p className="control-title">1. Enfermedad</p>
            <div className="button-grid compact">
              {ENFERMEDADES.map((item) => (
                <button
                  key={item}
                  className={`choice-btn ${disease === item ? 'active' : ''}`}
                  onClick={() => setDisease(item)}
                >
                  {item}
                </button>
              ))}
            </div>
          </div>

          <div className="control-block">
            <p className="control-title">2. Mes</p>
            <div className="button-grid months">
              {MESES.map((item) => (
                <button
                  key={item.value}
                  className={`choice-btn ${Number(month) === item.value ? 'active' : ''}`}
                  onClick={() => setMonth(item.value)}
                >
                  {item.label}
                </button>
              ))}
            </div>
          </div>

          <div className="control-block">
            <p className="control-title">
              3. Departamento {loadingDepartments && <span>Cargando...</span>}
            </p>

            <div className="button-grid scroll-list">
              {!loadingDepartments && departments.map((item) => (
                <button
                  key={item.value}
                  className={`choice-btn ${department === item.value ? 'active' : ''}`}
                  onClick={() => setDepartment(item.value)}
                >
                  {item.label}
                </button>
              ))}
            </div>
          </div>

          <div className="control-block">
            <p className="control-title">
              4. Ciudad / municipio {loadingCities && <span>Cargando...</span>}
            </p>

            <div className="button-grid scroll-list city-list">
              {!loadingCities && cities.map((item) => (
                <button
                  key={item.value}
                  className={`choice-btn ${city === item.value ? 'active' : ''}`}
                  onClick={() => setCity(item.value)}
                >
                  {item.label}
                </button>
              ))}
            </div>
          </div>

          {error && <p className="prediction-error">{error}</p>}

          <button
            className="predict-button"
            onClick={handlePredict}
            disabled={loadingPrediction || !department || !city}
          >
            {loadingPrediction ? 'Consultando predicción...' : 'Generar predicción'}
          </button>
        </div>

        <div className="prediction-result">
          {!prediction && (
            <div className="prediction-empty">
              <span>🧠</span>
              <h3>Predicción lista para consultar</h3>
              <p>
                Enfermedad: <strong>{disease}</strong> · Mes: <strong>{selectedMonthLabel}</strong>
              </p>
              <p>
                Departamento: <strong>{department ? prettyName(department) : 'Selecciona uno'}</strong>
              </p>
              <p>
                Ciudad: <strong>{city ? prettyName(city) : 'Selecciona una'}</strong>
              </p>
            </div>
          )}

          {prediction && (
            <>
              <div className="prediction-header">
                <span className={`risk-pill ${getRiskClass(prediction.outbreak_level)}`}>
                  {prediction.outbreak_level}
                </span>
                <span className="model-pill">
                  {prediction.model_used || 'Random Forest'}
                </span>
              </div>

              <div className="prediction-number">
                <strong>{Number(prediction.estimated_cases).toLocaleString('es-CO')}</strong>
                <span>casos estimados</span>
              </div>

              <div className="prediction-summary">
                <div>
                  <span>Enfermedad</span>
                  <strong>{prediction.disease}</strong>
                </div>
                <div>
                  <span>Mes</span>
                  <strong>{selectedMonthLabel}</strong>
                </div>
                <div>
                  <span>Departamento</span>
                  <strong>{prettyName(prediction.department)}</strong>
                </div>
                <div>
                  <span>Ciudad</span>
                  <strong>{prettyName(prediction.city)}</strong>
                </div>
              </div>

              <div className="prediction-details">
                <div>
                  <span>Piso térmico</span>
                  <strong>{prediction.thermal_floor || '—'}</strong>
                </div>
                <div>
                  <span>Temperatura aprox.</span>
                  <strong>{prediction.avg_temp_c} °C</strong>
                </div>
                <div>
                  <span>Humedad aprox.</span>
                  <strong>{prediction.humidity_pct}%</strong>
                </div>
                <div>
                  <span>Precipitación aprox.</span>
                  <strong>{Number(prediction.precipitation_mm).toLocaleString('es-CO')} mm</strong>
                </div>
                <div>
                  <span>Riesgo climático</span>
                  <strong>{prediction.climate_risk_score}</strong>
                </div>
                <div>
                  <span>Ambiente favorable</span>
                  <strong>{prediction.is_vector_favorable}</strong>
                </div>
              </div>

              <div className="model-metrics">
                <p>Métricas del modelo usado</p>
                <span>R²: {prediction.model_validation_metrics?.r2 ?? '—'}</span>
                <span>MAE: {prediction.model_validation_metrics?.mae ?? '—'}</span>
                <span>WAPE: {prediction.model_validation_metrics?.wape_pct ?? '—'}%</span>
              </div>
            </>
          )}
        </div>
      </div>
    </section>
  )
}