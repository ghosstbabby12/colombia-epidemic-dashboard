import { useEffect, useState, useCallback } from 'react'
import { api } from './api'

import KpiCards from './components/KpiCards'
import LineaSemanal from './components/LineaSemanal'
import BarraDepartamentos from './components/BarraDepartamentos'
import TablaAlertas from './components/TablaAlertas'
import PredictionPanel from './components/PredictionPanel'
import PowerBIEmbed from './components/PowerBIEmbed'
import Recomendaciones from './components/Recomendaciones'

const ENFERMEDADES = ['Dengue', 'Malaria', 'Chikungunya']

export default function App() {
  const [activeSection, setActiveSection] = useState('dashboard')

  const [enfermedad, setEnfermedad] = useState('Dengue')
  const [dpto, setDpto] = useState('')
  const [dptos, setDptos] = useState([])

  const [resumen, setResumen] = useState(null)
  const [semana, setSemana] = useState(null)
  const [departamentos, setDepartamentos] = useState(null)
  const [alertas, setAlertas] = useState(null)

  const [apiStatus, setApiStatus] = useState('Conectando...')
  const [dashboardError, setDashboardError] = useState('')

  useEffect(() => {
    api.health()
      .then((response) => {
        setApiStatus(response.data.status === 'ok' ? 'Supabase conectado' : 'Sin conexión')
      })
      .catch(() => {
        setApiStatus('Sin conexión')
      })

    api.resumen()
      .then((response) => {
        setResumen(response.data)
      })
      .catch((error) => {
        console.error(error)
        setResumen({ por_enfermedad: [], pico_epidemico_dengue: {} })
      })

    api.alertas()
      .then((response) => {
        setAlertas(response.data)
      })
      .catch((error) => {
        console.error(error)
        setAlertas({
          total_alertas: 0,
          criterio: 'No fue posible cargar alertas.',
          alertas: [],
        })
      })
  }, [])

  useEffect(() => {
    setDptos([])

    api.casosDepartamento(enfermedad, 40)
      .then((response) => {
        setDptos(response.data.ranking.map((item) => item.departamento))
      })
      .catch((error) => {
        console.error(error)
        setDptos([])
      })
  }, [enfermedad])

  const cargarDatos = useCallback(() => {
    setDashboardError('')
    setSemana(null)
    setDepartamentos(null)

    api.casosSemana(enfermedad, dpto)
      .then((response) => {
        setSemana(response.data)
      })
      .catch((error) => {
        console.error(error)
        setSemana({ serie: [] })
        setDashboardError('Algunas métricas no pudieron cargarse correctamente.')
      })

    api.casosDepartamento(enfermedad)
      .then((response) => {
        setDepartamentos(response.data)
      })
      .catch((error) => {
        console.error(error)
        setDepartamentos({ ranking: [] })
        setDashboardError('Algunas métricas no pudieron cargarse correctamente.')
      })
  }, [enfermedad, dpto])

  useEffect(() => {
    cargarDatos()
  }, [cargarDatos])

  const navItems = [
    { id: 'dashboard', label: 'Dashboard' },
    { id: 'prediccion', label: 'Predicción ML' },
    { id: 'powerbi', label: 'Power BI' },
  ]

  return (
    <div className="dashboard">
      <header className="header">
        <div className="brand">
          <div className="brand-icon">🦟</div>

          <div>
            <h1>Dashboard Epidemiológico Colombia</h1>
            <span>Datos SIVIGILA  · Supabase · Predicción ML</span>
          </div>
        </div>

        <nav className="nav-tabs">
          {navItems.map((item) => (
            <button
              key={item.id}
              className={activeSection === item.id ? 'active' : ''}
              onClick={() => setActiveSection(item.id)}
            >
              {item.label}
            </button>
          ))}
        </nav>

        <span className={apiStatus.includes('conectado') ? 'status-ok' : 'status-error'}>
          ● {apiStatus}
        </span>
      </header>

      <main className="main">
        {activeSection === 'dashboard' && (
          <>
            <section className="hero-mini">
              <span className="eyebrow">Vigilancia epidemiológica</span>
              <h2>Exploración de enfermedades transmitidas por mosquitos</h2>
              <p>
                Visualiza casos, tendencias semanales, distribución por departamento,
                alertas epidemiológicas y recomendaciones de salud pública.
              </p>
            </section>

            <div className="selector-bar">
              {ENFERMEDADES.map((item) => (
                <button
                  key={item}
                  className={`btn-enf ${enfermedad === item ? 'active' : ''}`}
                  onClick={() => {
                    setEnfermedad(item)
                    setDpto('')
                  }}
                >
                  {item}
                </button>
              ))}

              <select
                value={dpto}
                onChange={(event) => setDpto(event.target.value)}
              >
                <option value="">— Todos los departamentos —</option>
                {dptos.map((item) => (
                  <option key={item} value={item}>
                    {item}
                  </option>
                ))}
              </select>
            </div>

            {dashboardError && (
              <div className="dashboard-warning">
                ⚠️ {dashboardError}
              </div>
            )}

            <KpiCards resumen={resumen} enfermedad={enfermedad} />

            <div className="wide-block">
              <LineaSemanal datos={semana} />
            </div>

            <div className="wide-block">
              <BarraDepartamentos datos={departamentos} />
            </div>

            <div className="chart-card info-disease-card" style={{ marginBottom: '1rem' }}>
              <span>🛡️</span>
              <h3>Prevención recomendada</h3>
              <p>
                Elimina aguas estancadas, tapa recipientes, usa repelente en zonas
                de riesgo y consulta oportunamente ante fiebre, dolor muscular o
                síntomas persistentes.
              </p>
            </div>

            <TablaAlertas datos={alertas} />

            <Recomendaciones
              alertas={alertas}
              enfermedad={enfermedad}
              dpto={dpto}
            />
          </>
        )}

        {activeSection === 'prediccion' && <PredictionPanel />}

        {activeSection === 'powerbi' && <PowerBIEmbed />}

        <footer className="footer">
          Fuente: SIVIGILA – INS Colombia · ETL con Scrapy · Supabase · Random Forest · Power BI
        </footer>
      </main>
    </div>
  )
}