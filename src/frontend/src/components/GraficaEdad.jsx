import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell } from 'recharts'

const COLORES = ['#818cf8','#6366f1','#4f46e5','#4338ca','#3730a3','#312e81']

export default function GraficaEdad({ datos }) {
  if (!datos?.length) return <p className="loading">Cargando…</p>

  return (
    <div className="chart-card">
      <p className="chart-title">Distribución por grupo etario</p>
      <ResponsiveContainer width="100%" height={220}>
        <BarChart data={datos}>
          <XAxis dataKey="grupo_etario" stroke="#64748b" tick={{ fontSize: 11 }} />
          <YAxis stroke="#64748b" tick={{ fontSize: 10 }} />
          <Tooltip
            contentStyle={{ background: '#1e293b', border: '1px solid #334155', borderRadius: 8 }}
            formatter={(v, n) => [v.toLocaleString('es-CO'), n === 'casos' ? 'Casos' : n]}
          />
          <Bar dataKey="casos" radius={[4, 4, 0, 0]} name="casos">
            {datos.map((_, i) => <Cell key={i} fill={COLORES[i % COLORES.length]} />)}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  )
}
