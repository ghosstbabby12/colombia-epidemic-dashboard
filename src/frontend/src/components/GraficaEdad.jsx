import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell } from 'recharts'

const COLORES = ['#818cf8', '#6366f1', '#4f46e5', '#4338ca', '#3730a3', '#312e81', '#64748b']

export default function GraficaEdad({ datos }) {
  if (datos === null || datos === undefined) {
    return (
      <div className="chart-card">
        <p className="chart-title">Distribución por grupo etario</p>
        <p className="loading">Cargando distribución por edad…</p>
      </div>
    )
  }

  if (!Array.isArray(datos) || datos.length === 0) {
    return (
      <div className="chart-card">
        <p className="chart-title">Distribución por grupo etario</p>
        <div className="empty-state">
          <span>📊</span>
          <p>No hay datos de grupo etario para el filtro seleccionado.</p>
        </div>
      </div>
    )
  }

  const data = datos
    .map((item) => ({
      grupo_etario: item.grupo_etario || 'Sin dato',
      casos: Number(item.casos || 0),
    }))
    .filter((item) => item.casos >= 0)

  const total = data.reduce((sum, item) => sum + item.casos, 0)

  if (total === 0) {
    return (
      <div className="chart-card">
        <p className="chart-title">Distribución por grupo etario</p>
        <div className="empty-state">
          <span>📊</span>
          <p>No hay casos registrados por grupo etario para el filtro actual.</p>
        </div>
      </div>
    )
  }

  return (
    <div className="chart-card">
      <p className="chart-title">Distribución por grupo etario</p>

      <ResponsiveContainer width="100%" height={220}>
        <BarChart data={data}>
          <XAxis
            dataKey="grupo_etario"
            stroke="#64748b"
            tick={{ fontSize: 11 }}
          />

          <YAxis
            stroke="#64748b"
            tick={{ fontSize: 10 }}
          />

          <Tooltip
            contentStyle={{
              background: '#1e293b',
              border: '1px solid #334155',
              borderRadius: 8,
              color: '#e2e8f0',
            }}
            formatter={(value, name) => [
              Number(value).toLocaleString('es-CO'),
              name === 'casos' ? 'Casos' : name,
            ]}
          />

          <Bar dataKey="casos" radius={[4, 4, 0, 0]} name="casos">
            {data.map((_, index) => (
              <Cell key={index} fill={COLORES[index % COLORES.length]} />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  )
}