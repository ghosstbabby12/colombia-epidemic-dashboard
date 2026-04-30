const RECOMENDACIONES = {
  Dengue: {
    individual: [
      'Eliminar recipientes con agua estancada como floreros, llantas, baldes o tanques destapados.',
      'Usar repelente con DEET al 30% o Icaridina, especialmente al amanecer y al atardecer.',
      'Instalar toldillos o mallas en puertas y ventanas.',
      'Usar ropa de manga larga y pantalón en zonas de alta transmisión.',
      'Consultar de inmediato ante fiebre alta, dolor de cabeza intenso o erupción cutánea.',
    ],
    comunidad: [
      'Realizar jornadas de descacharrización comunitaria cada 8 días.',
      'Aplicar fumigación focal en focos activos coordinada con autoridades sanitarias.',
      'Fortalecer la vigilancia activa de casos en puestos de salud.',
      'Promover educación puerta a puerta sobre control de criaderos.',
    ],
    sistema: [
      'Activar sala de crisis en secretaría de salud departamental si aumenta el riesgo.',
      'Garantizar disponibilidad de pruebas diagnósticas en IPS primarias.',
      'Reforzar capacidad hospitalaria para manejo de dengue grave.',
      'Realizar notificación inmediata a SIVIGILA con código 210.',
    ],
  },

  Malaria: {
    individual: [
      'Usar toldillos impregnados con insecticida al dormir.',
      'Aplicar repelente en piel expuesta al salir al campo, río o zonas selváticas.',
      'Buscar diagnóstico por gota gruesa ante fiebre con escalofríos.',
      'Completar el tratamiento antimalárico aunque los síntomas mejoren antes.',
      'Evitar actividades al aire libre entre las 18:00 y las 06:00 h en zonas de riesgo.',
    ],
    comunidad: [
      'Realizar rociado residual intradomiciliario con insecticidas aprobados.',
      'Distribuir toldillos impregnados a hogares en riesgo.',
      'Hacer campañas de diagnóstico activo en veredas y comunidades indígenas.',
      'Controlar criaderos de Anopheles en zonas húmedas y cercanas a ríos.',
    ],
    sistema: [
      'Garantizar abastecimiento de antimaláricos en puestos de salud rurales.',
      'Capacitar microscopistas para gota gruesa con control de calidad.',
      'Coordinar planes preventivos adaptados con comunidades étnicas.',
      'Notificar a SIVIGILA con código 490 y hacer seguimiento de casos.',
    ],
  },

  Chikungunya: {
    individual: [
      'Usar repelente y ropa larga durante el día, ya que Aedes aegypti es diurno.',
      'Eliminar agua estancada en recipientes domésticos.',
      'Guardar reposo e hidratarse ante fiebre y dolor articular intenso.',
      'Evitar automedicarse con ibuprofeno o aspirina sin indicación médica.',
      'Consultar si el dolor articular persiste más de una semana.',
    ],
    comunidad: [
      'Controlar Aedes aegypti con larvicidas en depósitos no eliminables.',
      'Realizar jornadas de limpieza en espacios públicos con agua acumulada.',
      'Activar alertas vecinales para notificar casos agrupados.',
      'Reforzar campañas educativas sobre eliminación de criaderos.',
    ],
    sistema: [
      'Realizar diagnóstico diferencial con Dengue y Zika.',
      'Fortalecer manejo del dolor articular persistente.',
      'Notificar a SIVIGILA con código 217 y analizar clústers.',
      'Hacer seguimiento a pacientes con artritis persistente.',
    ],
  },
}

const NIVEL_CONFIG = {
  ALTO: {
    className: 'recommendation-risk high',
    icon: '🔴',
    texto: 'Riesgo ALTO — Se requiere acción inmediata',
  },
  MEDIO: {
    className: 'recommendation-risk medium',
    icon: '🟡',
    texto: 'Riesgo MEDIO — Reforzar medidas preventivas',
  },
  BAJO: {
    className: 'recommendation-risk low',
    icon: '🟢',
    texto: 'Riesgo BAJO — Mantener vigilancia activa',
  },
  NINGUNO: {
    className: 'recommendation-risk none',
    icon: '✅',
    texto: 'Sin alertas activas — Continuar vigilancia rutinaria',
  },
}

export default function Recomendaciones({ alertas, enfermedad, dpto }) {
  if (!alertas) {
    return (
      <div className="recommendations-card">
        <p className="loading">Cargando recomendaciones…</p>
      </div>
    )
  }

  const alertasFiltradas = alertas.alertas?.filter((alerta) => {
    if (!dpto) return true
    return alerta.departamento?.toUpperCase() === dpto?.toUpperCase()
  }) || []

  const nivelGlobal = alertasFiltradas.some((alerta) => alerta.nivel_alerta === 'ALTO')
    ? 'ALTO'
    : alertasFiltradas.some((alerta) => alerta.nivel_alerta === 'MEDIO')
      ? 'MEDIO'
      : alertasFiltradas.length > 0
        ? 'BAJO'
        : 'NINGUNO'

  const config = NIVEL_CONFIG[nivelGlobal]
  const recs = RECOMENDACIONES[enfermedad] || RECOMENDACIONES.Dengue

  const departamentosEnRiesgo = alertas.alertas
    ?.filter((alerta) => alerta.nivel_alerta === 'ALTO')
    .map((alerta) => alerta.departamento)
    .slice(0, 5) || []

  return (
    <section className="recommendations-card">
      <div className="recommendations-header">
        <div>
          <span className="eyebrow">Salud pública</span>
          <h3>🛡️ Recomendaciones — {enfermedad}</h3>
          <p>
            Acciones sugeridas según el nivel de alerta epidemiológica y la enfermedad seleccionada.
          </p>
        </div>

        <div className={config.className}>
          <span>{config.icon}</span>
          <strong>{config.texto}</strong>
        </div>
      </div>

      {departamentosEnRiesgo.length > 0 && (
        <div className="risk-departments">
          <span>Departamentos en alerta alta:</span>
          {departamentosEnRiesgo.map((departamento) => (
            <strong key={departamento}>{departamento}</strong>
          ))}
        </div>
      )}

      <div className="recommendations-grid">
        <RecommendationColumn
          title="👤 Prevención individual"
          color="blue"
          items={recs.individual}
        />

        <RecommendationColumn
          title="🏘️ Acción comunitaria"
          color="green"
          items={recs.comunidad}
        />

        <RecommendationColumn
          title="🏥 Sistema de salud"
          color="orange"
          items={recs.sistema}
        />
      </div>

      <p className="recommendations-note">
        Fuente base: lineamientos de vigilancia en salud pública. Las recomendaciones se presentan como apoyo informativo y deben ajustarse a las decisiones de las autoridades sanitarias locales.
      </p>
    </section>
  )
}

function RecommendationColumn({ title, color, items }) {
  return (
    <div className="recommendation-column">
      <p className={`recommendation-title ${color}`}>{title}</p>

      <ul>
        {items.map((item, index) => (
          <li key={index}>{item}</li>
        ))}
      </ul>
    </div>
  )
}