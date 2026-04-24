export default function KpiCards({ resumen, enfermedad }) {
  const dato = resumen?.por_enfermedad?.find(d => d.enfermedad === enfermedad) || {}
  const pico = resumen?.pico_epidemico_dengue || {}

  const fmt = n => n != null ? Number(n).toLocaleString('es-CO') : '—'

  const cards = [
    { label: 'Total casos 2024',     value: fmt(dato.total_casos),   sub: enfermedad },
    { label: 'Hospitalizados',       value: fmt(dato.hospitalizados), sub: `${dato.total_casos ? Math.round(dato.hospitalizados/dato.total_casos*100) : 0}% del total` },
    { label: 'Edad promedio',        value: dato.edad_promedio ?? '—', sub: 'años' },
    { label: '% Femenino',           value: `${dato.pct_femenino ?? '—'}%`, sub: 'de los casos' },
    { label: 'Semana pico (Dengue)', value: `Sem. ${pico.semana_pico ?? '—'}`, sub: `${fmt(pico.casos_pico)} casos` },
  ]

  return (
    <div className="kpi-grid">
      {cards.map(c => (
        <div key={c.label} className="kpi-card">
          <span className="kpi-label">{c.label}</span>
          <span className="kpi-value">{c.value}</span>
          <span className="kpi-sub">{c.sub}</span>
        </div>
      ))}
    </div>
  )
}
