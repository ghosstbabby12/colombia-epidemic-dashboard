import { useEffect, useState, useCallback } from 'react'
import { api } from './api'
import KpiCards          from './components/KpiCards'
import LineaSemanal      from './components/LineaSemanal'
import BarraDepartamentos from './components/BarraDepartamentos'
import GraficaEdad       from './components/GraficaEdad'
import GraficaSexo       from './components/GraficaSexo'
import TablaAlertas      from './components/TablaAlertas'

const ENFERMEDADES = ['Dengue', 'Malaria', 'Chikungunya']

export default function App() {
  const [enfermedad, setEnfermedad] = useState('Dengue')
  const [dpto, setDpto]             = useState('')
  const [dptos, setDptos]           = useState([])

  const [resumen,      setResumen]      = useState(null)
  const [semana,       setSemana]       = useState(null)
  const [departamentos,setDepartamentos]= useState(null)
  const [edad,         setEdad]         = useState(null)
  const [sexo,         setSexo]         = useState(null)
  const [alertas,      setAlertas]      = useState(null)

  // Cargar lista de departamentos al inicio
  useEffect(() => {
    api.casosDepartamento('Dengue', 33)
      .then(r => setDptos(r.data.ranking.map(d => d.departamento)))
      .catch(console.error)
    api.resumen().then(r => setResumen(r.data)).catch(console.error)
    api.alertas().then(r => setAlertas(r.data)).catch(console.error)
  }, [])

  // Recargar gráficas cuando cambia enfermedad o departamento
  const cargarDatos = useCallback(() => {
    api.casosSemana(enfermedad, dpto)
      .then(r => setSemana(r.data)).catch(console.error)
    api.casosDepartamento(enfermedad)
      .then(r => setDepartamentos(r.data)).catch(console.error)
    api.grupoEtario(enfermedad, dpto)
      .then(r => setEdad(r.data)).catch(console.error)
    api.sexo(enfermedad, dpto)
      .then(r => setSexo(r.data)).catch(console.error)
  }, [enfermedad, dpto])

  useEffect(() => { cargarDatos() }, [cargarDatos])

  return (
    <div className="dashboard">
      {/* Header */}
      <header className="header">
        <div>
          <h1>Dashboard Epidemiológico Colombia</h1>
          <span>Datos SIVIGILA 2024 — INS Colombia</span>
        </div>
        <span style={{ color: '#22c55e', fontSize: '0.8rem' }}>● API conectada</span>
      </header>

      <main className="main">
        {/* Selector de enfermedad + departamento */}
        <div className="selector-bar">
          {ENFERMEDADES.map(e => (
            <button key={e}
              className={`btn-enf ${enfermedad === e ? 'active' : ''}`}
              onClick={() => setEnfermedad(e)}>
              {e}
            </button>
          ))}
          <select value={dpto} onChange={e => setDpto(e.target.value)} style={{ marginLeft: 'auto' }}>
            <option value="">— Todos los departamentos —</option>
            {dptos.map(d => <option key={d} value={d}>{d}</option>)}
          </select>
        </div>

        {/* KPIs */}
        <KpiCards resumen={resumen} enfermedad={enfermedad} />

        {/* Serie temporal — ancho completo */}
        <div style={{ marginBottom: '1rem' }}>
          <LineaSemanal datos={semana} />
        </div>

        {/* Departamentos + Edad */}
        <div className="charts-grid">
          <BarraDepartamentos datos={departamentos} />
          <GraficaEdad datos={edad} />
        </div>

        {/* Sexo */}
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 2fr', gap: '1rem', marginBottom: '1rem' }}>
          <GraficaSexo datos={sexo} />
          <div className="chart-card" style={{ display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
            <div style={{ textAlign: 'center', color: '#475569' }}>
              <p style={{ fontSize: '2rem' }}>🗺️</p>
              <p style={{ fontSize: '0.85rem', marginTop: '0.5rem' }}>Mapa coroplético</p>
              <p style={{ fontSize: '0.75rem', marginTop: '0.3rem' }}>Próxima versión</p>
            </div>
          </div>
        </div>

        {/* Alertas */}
        <TablaAlertas datos={alertas} />

        {/* Footer */}
        <p style={{ textAlign: 'center', color: '#334155', fontSize: '0.75rem', paddingTop: '1rem' }}>
          Datos al Ecosistema 2026 · Fuente: SIVIGILA – INS Colombia · API: localhost:8000
        </p>
      </main>
    </div>
  )
}
