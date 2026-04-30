import { supabase } from './supabaseClient'

const normalizeName = (value) => {
  return String(value || '')
    .normalize('NFD')
    .replace(/[\u0300-\u036f]/g, '')
    .trim()
    .toUpperCase()
}

const titleText = (value) => {
  return String(value || '')
    .toLowerCase()
    .split(' ')
    .filter(Boolean)
    .map((word) => word.charAt(0).toUpperCase() + word.slice(1))
    .join(' ')
}

const titleDisease = (value) => {
  const text = String(value || '').trim().toLowerCase()

  if (text === 'dengue') return 'Dengue'
  if (text === 'malaria') return 'Malaria'
  if (text === 'chikungunya') return 'Chikungunya'

  return titleText(value)
}

const throwIfError = ({ data, error }) => {
  if (error) throw error
  return data
}

const sumByKey = (rows, keyName, valueFields) => {
  const map = new Map()

  rows.forEach((row) => {
    const key = row[keyName] ?? 'Sin dato'

    if (!map.has(key)) {
      map.set(key, { [keyName]: key })
      valueFields.forEach((field) => {
        map.get(key)[field] = 0
      })
    }

    valueFields.forEach((field) => {
      map.get(key)[field] += Number(row[field] || 0)
    })
  })

  return [...map.values()]
}

export const api = {
  health: async () => {
    const response = await supabase
      .from('etl_load_metadata')
      .select('*')
      .order('loaded_at', { ascending: false })
      .limit(1)

    return {
      data: {
        status: response.error ? 'error' : 'ok',
        last_load: response.data?.[0] || null,
      },
    }
  },

  resumen: async () => {
    const resumenResponse = await supabase
      .from('v_resumen_enfermedad')
      .select('*')
      .order('total_casos', { ascending: false })

    const picoResponse = await supabase
      .from('v_pico_epidemico_dengue')
      .select('*')
      .limit(1)

    const resumen = throwIfError(resumenResponse)
    const pico = throwIfError(picoResponse)

    return {
      data: {
        por_enfermedad: resumen.map((row) => ({
          enfermedad: titleDisease(row.enfermedad),
          total_casos: Number(row.total_casos || 0),
          hospitalizados: Number(row.hospitalizados || 0),
          edad_promedio: Number(row.edad_promedio || 0),
          pct_femenino: Number(row.pct_femenino || 0),
        })),
        pico_epidemico_dengue: pico?.[0] || {},
      },
    }
  },

  enfermedades: async () => {
    const response = await supabase
      .from('v_resumen_enfermedad')
      .select('enfermedad')
      .order('enfermedad', { ascending: true })

    const data = throwIfError(response)

    return {
      data: {
        enfermedades: data.map((row) => titleDisease(row.enfermedad)),
      },
    }
  },

  departamentos: async (enfermedad) => {
    let query = supabase
      .from('v_casos_departamento')
      .select('departamento')
      .order('departamento', { ascending: true })

    if (enfermedad) {
      query = query.eq('enfermedad', enfermedad)
    }

    const data = throwIfError(await query)

    return {
      data: {
        departamentos: [...new Set(data.map((row) => row.departamento))],
      },
    }
  },

  casosSemana: async (enfermedad, departamento) => {
    let query = supabase
      .from('v_casos_semana')
      .select('*')
      .eq('enfermedad', enfermedad)
      .order('semana', { ascending: true })
      .range(0, 5000)

    if (departamento) {
      query = query.eq('departamento', departamento)
    }

    const data = throwIfError(await query)

    const grouped = sumByKey(data, 'semana', ['casos', 'hospitalizados'])
      .map((row) => ({
        semana: Number(row.semana),
        casos: Number(row.casos || 0),
        hospitalizados: Number(row.hospitalizados || 0),
      }))
      .sort((a, b) => a.semana - b.semana)

    return {
      data: {
        serie: grouped,
      },
    }
  },

  casosDepartamento: async (enfermedad, top = 20) => {
    const data = throwIfError(
      await supabase
        .from('v_casos_departamento')
        .select('*')
        .eq('enfermedad', enfermedad)
        .order('casos', { ascending: false })
        .limit(top)
    )

    return {
      data: {
        ranking: data.map((row) => ({
          departamento: row.departamento,
          casos: Number(row.casos || 0),
        })),
      },
    }
  },

  casosMunicipio: async (departamento, enfermedad) => {
    const data = throwIfError(
      await supabase
        .from('v_casos_municipio')
        .select('*')
        .eq('enfermedad', enfermedad)
        .eq('departamento', departamento)
        .order('casos', { ascending: false })
        .range(0, 2000)
    )

    return {
      data: {
        departamento,
        ranking: data.map((row) => ({
          municipio: row.municipio,
          casos: Number(row.casos || 0),
        })),
      },
    }
  },

  grupoEtario: async (enfermedad, departamento) => {
    let query = supabase
      .from('v_grupo_etario')
      .select('*')
      .eq('enfermedad', enfermedad)
      .range(0, 5000)

    if (departamento) {
      query = query.eq('departamento', departamento)
    }

    const data = throwIfError(await query)

    const order = ['0-4', '5-14', '15-29', '30-44', '45-59', '60+', 'Sin dato']

    const grouped = sumByKey(data, 'grupo_etario', ['casos'])
      .map((row) => ({
        grupo_etario: row.grupo_etario,
        casos: Number(row.casos || 0),
      }))
      .sort((a, b) => {
        const aIndex = order.indexOf(a.grupo_etario)
        const bIndex = order.indexOf(b.grupo_etario)

        return (aIndex === -1 ? 99 : aIndex) - (bIndex === -1 ? 99 : bIndex)
      })

    return {
      data: grouped,
    }
  },

  sexo: async (enfermedad, departamento) => {
    let query = supabase
      .from('v_sexo')
      .select('*')
      .eq('enfermedad', enfermedad)
      .range(0, 5000)

    if (departamento) {
      query = query.eq('departamento', departamento)
    }

    const data = throwIfError(await query)

    const grouped = sumByKey(data, 'sexo', ['casos'])
    const total = grouped.reduce((sum, row) => sum + Number(row.casos || 0), 0)

    return {
      data: grouped.map((row) => ({
        sexo: row.sexo,
        casos: Number(row.casos || 0),
        porcentaje: total > 0 ? Number(((Number(row.casos || 0) / total) * 100).toFixed(2)) : 0,
      })),
    }
  },

  alertas: async () => {
    const data = throwIfError(
      await supabase
        .from('v_alertas_dengue')
        .select('*')
        .limit(20)
    )

    return {
      data: {
        total_alertas: data.length,
        criterio: 'Departamentos con incremento superior al 20% frente al promedio semanal histórico disponible.',
        alertas: data.map((row) => ({
          departamento: row.departamento,
          semana: Number(row.semana || 0),
          casos_semana: Number(row.casos_semana || 0),
          promedio_historico: Number(row.promedio_historico || 0),
          ratio: Number(row.ratio || 0),
          nivel_alerta: row.nivel_alerta,
        })),
      },
    }
  },

  departamentosPrediccion: async ({ disease, month }) => {
    const data = throwIfError(
      await supabase
        .from('ml_predictions')
        .select('department')
        .eq('disease', disease)
        .eq('month', Number(month))
        .order('department', { ascending: true })
        .range(0, 5000)
    )

    return {
      data: [...new Set(data.map((row) => row.department))]
        .filter(Boolean)
        .map((item) => ({
          label: titleText(item),
          value: normalizeName(item),
        })),
    }
  },

  ciudadesPrediccion: async ({ disease, month, department }) => {
    const data = throwIfError(
      await supabase
        .from('ml_predictions')
        .select('city')
        .eq('disease', disease)
        .eq('month', Number(month))
        .eq('department', normalizeName(department))
        .order('city', { ascending: true })
        .range(0, 2000)
    )

    return {
      data: [...new Set(data.map((row) => row.city))]
        .filter(Boolean)
        .map((item) => ({
          label: titleText(item),
          value: normalizeName(item),
        })),
    }
  },

  prediccion: async ({ month, department, city, disease }) => {
    const data = throwIfError(
      await supabase
        .from('ml_predictions')
        .select('*')
        .eq('month', Number(month))
        .eq('department', normalizeName(department))
        .eq('city', normalizeName(city))
        .eq('disease', disease)
        .maybeSingle()
    )

    if (!data) {
      throw new Error('No existe una predicción precalculada para esos parámetros.')
    }

    return {
      data: {
        disease: data.disease,
        month: Number(data.month),
        department: data.department,
        city: data.city,
        estimated_cases: Number(data.estimated_cases || 0),
        estimated_outbreak_proxy: Number(data.estimated_outbreak_proxy || 0),
        outbreak_level: data.outbreak_level,
        model_used: data.model_used,
        thermal_floor: data.thermal_floor,
        avg_temp_c: Number(data.avg_temp_c || 0),
        precipitation_mm: Number(data.precipitation_mm || 0),
        humidity_pct: Number(data.humidity_pct || 0),
        tropical_score: Number(data.tropical_score || 0),
        climate_risk_score: Number(data.climate_risk_score || 0),
        is_rainy_season: Number(data.is_rainy_season || 0),
        is_vector_favorable: data.is_vector_favorable,
        cases_previous_month: Number(data.cases_previous_month || 0),
        cases_previous_2_month_mean: Number(data.cases_previous_2_month_mean || 0),
        cases_previous_3_month_mean: Number(data.cases_previous_3_month_mean || 0),
        model_validation_metrics: {
          r2: Number(data.validation_r2 || 0),
          mae: Number(data.validation_mae || 0),
          wape_pct: Number(data.validation_wape_pct || 0),
        },
      },
    }
  },
}