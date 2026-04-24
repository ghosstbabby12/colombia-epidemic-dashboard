import { PieChart, Pie, Cell, Tooltip, Legend, ResponsiveContainer } from 'recharts'

const COLORES = { M: '#3b82f6', F: '#ec4899' }
const LABELS  = { M: 'Masculino', F: 'Femenino' }

export default function GraficaSexo({ datos }) {
  if (!datos?.length) return <p className="loading">Cargando…</p>

  const data = datos.map(d => ({ name: LABELS[d.sexo] || d.sexo, value: d.casos, sexo: d.sexo }))

  return (
    <div className="chart-card">
      <p className="chart-title">Distribución por sexo</p>
      <ResponsiveContainer width="100%" height={220}>
        <PieChart>
          <Pie data={data} cx="50%" cy="50%" innerRadius={55} outerRadius={85}
               dataKey="value" nameKey="name" paddingAngle={3}
               label={({ name, porcentaje, payload }) =>
                 `${name} ${payload?.porcentaje ?? ''}%`
               }>
            {data.map((d, i) => <Cell key={i} fill={COLORES[d.sexo] || '#6366f1'} />)}
          </Pie>
          <Tooltip
            contentStyle={{ background: '#1e293b', border: '1px solid #334155', borderRadius: 8 }}
            formatter={v => v.toLocaleString('es-CO')}
          />
          <Legend wrapperStyle={{ fontSize: 12 }} />
        </PieChart>
      </ResponsiveContainer>
    </div>
  )
}
