import axios from 'axios'

const BASE = '/api'  // proxy → http://localhost:8000

export const api = {
  resumen:          ()           => axios.get(`${BASE}/resumen`),
  enfermedades:     ()           => axios.get(`${BASE}/enfermedades`),
  departamentos:    (enf)        => axios.get(`${BASE}/departamentos?enfermedad=${enf}`),
  casosSemana:      (enf, dpto)  => axios.get(`${BASE}/casos/semana?enfermedad=${enf}${dpto ? `&departamento=${dpto}` : ''}`),
  casosDepartamento:(enf, top=20)=> axios.get(`${BASE}/casos/departamento?enfermedad=${enf}&top=${top}`),
  casosMunicipio:   (dpto, enf)  => axios.get(`${BASE}/casos/municipio/${dpto}?enfermedad=${enf}`),
  grupoEtario:      (enf, dpto)  => axios.get(`${BASE}/casos/grupo-etario?enfermedad=${enf}${dpto ? `&departamento=${dpto}` : ''}`),
  sexo:             (enf, dpto)  => axios.get(`${BASE}/casos/sexo?enfermedad=${enf}${dpto ? `&departamento=${dpto}` : ''}`),
  alertas:          ()           => axios.get(`${BASE}/alertas`),
}
