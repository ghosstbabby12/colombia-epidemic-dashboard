import { LineChart, Line, XAxis, YAxis, Tooltip, CartesianGrid, ResponsiveContainer, Legend } from 'recharts'

export default function LineaSemanal({ datos }) {
  if (!datos?.serie?.length) return <p className="loading">Cargando serie temporal…</p>

  return (
    <div className="chart-card" style={{ gridColumn: '1 / -1' }}>
      <p className="chart-title">Casos por semana epidemiológica </p>
      <ResponsiveContainer width="100%" height={240}>
        <LineChart data={datos.serie}>
          <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
          <XAxis dataKey="semana" stroke="#64748b" tick={{ fontSize: 11 }} label={{ value: 'Semana', position: 'insideBottom', offset: -2, fill: '#64748b', fontSize: 11 }} />
          <YAxis stroke="#64748b" tick={{ fontSize: 11 }} />
          <Tooltip
            contentStyle={{ background: '#1e293b', border: '1px solid #334155', borderRadius: 8 }}
            labelFormatter={v => `Semana ${v}`}
          />
          <Legend wrapperStyle={{ fontSize: 12 }} />
          <Line type="monotone" dataKey="casos" stroke="#3b82f6" strokeWidth={2} dot={false} name="Casos" />
          <Line type="monotone" dataKey="hospitalizados" stroke="#f97316" strokeWidth={1.5} dot={false} name="Hospitalizados" />
        </LineChart>
      </ResponsiveContainer>
    </div>
  )
}
