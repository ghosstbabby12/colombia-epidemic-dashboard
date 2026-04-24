export default function TablaAlertas({ datos }) {
  if (!datos?.alertas) return <p className="loading">Cargando alertas…</p>

  return (
    <div className="alertas-card">
      <p className="alertas-title">
        🚨 Alertas epidémicas activas — {datos.total_alertas} departamentos
      </p>
      <p style={{ fontSize: '0.75rem', color: '#64748b', marginBottom: '0.8rem' }}>
        {datos.criterio}
      </p>
      <table>
        <thead>
          <tr>
            <th>Departamento</th>
            <th>Semana</th>
            <th>Casos</th>
            <th>Promedio hist.</th>
            <th>Ratio</th>
            <th>Nivel</th>
          </tr>
        </thead>
        <tbody>
          {datos.alertas.map((a, i) => (
            <tr key={i}>
              <td style={{ fontWeight: 600 }}>{a.departamento}</td>
              <td>{a.semana}</td>
              <td>{Number(a.casos_semana).toLocaleString('es-CO')}</td>
              <td>{Number(a.promedio_historico).toLocaleString('es-CO')}</td>
              <td style={{ color: a.nivel_alerta === 'ALTO' ? '#f87171' : '#fcd34d' }}>
                {a.ratio}x
              </td>
              <td>
                <span className={`badge ${a.nivel_alerta}`}>{a.nivel_alerta}</span>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}
