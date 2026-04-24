import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell } from 'recharts'

const COLORES = ['#3b82f6','#60a5fa','#93c5fd','#bfdbfe','#dbeafe',
                 '#2563eb','#1d4ed8','#1e40af','#1e3a8a','#172554']

export default function BarraDepartamentos({ datos }) {
  if (!datos?.ranking?.length) return <p className="loading">Cargando…</p>

  const top10 = datos.ranking.slice(0, 10).map(d => ({
    ...d,
    depto: d.departamento.split(' ').map(w => w[0] + w.slice(1).toLowerCase()).join(' '),
  }))

  return (
    <div className="chart-card">
      <p className="chart-title">Top 10 departamentos</p>
      <ResponsiveContainer width="100%" height={260}>
        <BarChart data={top10} layout="vertical" margin={{ left: 10 }}>
          <XAxis type="number" stroke="#64748b" tick={{ fontSize: 10 }} />
          <YAxis type="category" dataKey="depto" stroke="#64748b" tick={{ fontSize: 10 }} width={90} />
          <Tooltip
            contentStyle={{ background: '#1e293b', border: '1px solid #334155', borderRadius: 8 }}
            formatter={v => v.toLocaleString('es-CO')}
          />
          <Bar dataKey="casos" radius={[0, 4, 4, 0]} name="Casos">
            {top10.map((_, i) => <Cell key={i} fill={COLORES[i % COLORES.length]} />)}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  )
}
