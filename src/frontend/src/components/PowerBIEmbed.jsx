const POWERBI_URL =
  import.meta.env.VITE_POWERBI_EMBED_URL ||
  'https://app.powerbi.com/view?r=eyJrIjoiMmE0OTUwMTQtMWU0OC00MmEwLWE5MGItNjQ4Mzc5M2RjOGNlIiwidCI6IjhkMzY4MzZlLTZiNzUtNGRlNi1iYWI5LTVmNGIxNzc1NDI3ZiIsImMiOjR9'

export default function PowerBIEmbed() {
  return (
    <section className="powerbi-section">
      <div className="section-heading center">
        <span className="eyebrow">Power BI</span>
        <h2>Tablero central de análisis epidemiológico</h2>
        <p>
          Reporte interactivo embebido para explorar la información epidemiológica
          procesada del proyecto.
        </p>
      </div>

      <div className="powerbi-card">
        <iframe
          title="Tablero Power BI Epidemiológico Colombia"
          src={POWERBI_URL}
          allowFullScreen
        />
      </div>
    </section>
  )
}