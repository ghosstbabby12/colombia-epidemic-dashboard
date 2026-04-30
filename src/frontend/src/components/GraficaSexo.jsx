import { PieChart, Pie, Cell, Tooltip, Legend, ResponsiveContainer } from 'recharts'

const COLORES = {
  M: '#3b82f6',
  F: '#ec4899',
  SIN_DATO: '#64748b',
}

const LABELS = {
  M: 'Masculino',
  F: 'Femenino',
  SIN_DATO: 'Sin dato',
}

export default function GraficaSexo({ datos }) {
  if (datos === null || datos === undefined) {
    return (
      <div className="chart-card">
        <p className="chart-title">Distribución por sexo</p>
        <p className="loading">Cargando distribución por sexo…</p>
      </div>
    )
  }

  if (!Array.isArray(datos) || datos.length === 0) {
    return (
      <div className="chart-card">
        <p className="chart-title">Distribución por sexo</p>
        <div className="empty-state">
          <span>⚧️</span>
          <p>No hay datos de sexo para el filtro seleccionado.</p>
        </div>
      </div>
    )
  }

  const total = datos.reduce((sum, item) => sum + Number(item.casos || 0), 0)

  const data = datos
    .map((item) => {
      const sexo = item.sexo || 'SIN_DATO'
      const casos = Number(item.casos || 0)
      const porcentaje = total > 0 ? Number(((casos / total) * 100).toFixed(2)) : 0

      return {
        name: LABELS[sexo] || sexo,
        value: casos,
        sexo,
        porcentaje,
      }
    })
    .filter((item) => item.value > 0)

  if (data.length === 0 || total === 0) {
    return (
      <div className="chart-card">
        <p className="chart-title">Distribución por sexo</p>
        <div className="empty-state">
          <span>⚧️</span>
          <p>No hay casos registrados por sexo para el filtro actual.</p>
        </div>
      </div>
    )
  }

  return (
    <div className="chart-card">
      <p className="chart-title">Distribución por sexo</p>

      <ResponsiveContainer width="100%" height={220}>
        <PieChart>
          <Pie
            data={data}
            cx="50%"
            cy="50%"
            innerRadius={55}
            outerRadius={85}
            dataKey="value"
            nameKey="name"
            paddingAngle={3}
            label={({ name, payload }) => `${name} ${payload?.porcentaje ?? 0}%`}
          >
            {data.map((item, index) => (
              <Cell
                key={index}
                fill={COLORES[item.sexo] || '#6366f1'}
              />
            ))}
          </Pie>

          <Tooltip
            contentStyle={{
              background: '#1e293b',
              border: '1px solid #334155',
              borderRadius: 8,
              color: '#e2e8f0',
            }}
            formatter={(value, name, props) => [
              `${Number(value).toLocaleString('es-CO')} casos (${props.payload.porcentaje}%)`,
              name,
            ]}
          />

          <Legend wrapperStyle={{ fontSize: 12 }} />
        </PieChart>
      </ResponsiveContainer>
    </div>
  )
}