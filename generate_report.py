import os
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    Image,
    PageBreak,
)
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont


def register_fonts():
    # Registrar la fuente Arial del sistema
    arial_path = "/usr/share/fonts/truetype/msttcorefonts/Arial.ttf"
    arial_bold_path = "/usr/share/fonts/truetype/msttcorefonts/Arial_Bold.ttf"
    arial_italic_path = "/usr/share/fonts/truetype/msttcorefonts/Arial_Italic.ttf"

    if os.path.exists(arial_path):
        pdfmetrics.registerFont(TTFont("Arial", arial_path))
        pdfmetrics.registerFont(TTFont("Arial-Bold", arial_bold_path))
        pdfmetrics.registerFont(TTFont("Arial-Italic", arial_italic_path))
        font_regular = "Arial"
        font_bold = "Arial-Bold"
        font_italic = "Arial-Italic"
    else:
        font_regular = "Helvetica"
        font_bold = "Helvetica-Bold"
        font_italic = "Helvetica-Oblique"

    return font_regular, font_bold, font_italic


def create_report(output_pdf_path: str):
    font_reg, font_bold, font_italic = register_fonts()

    doc = SimpleDocTemplate(
        output_pdf_path,
        pagesize=letter,
        leftMargin=54,
        rightMargin=54,
        topMargin=54,
        bottomMargin=54,
    )

    # Estilos tipográficos en estricto blanco y negro, Arial 11 pt, interlineado 1.5 (leading 16.5)
    title_style = ParagraphStyle(
        name="DocTitle",
        fontName=font_bold,
        fontSize=14,
        leading=18,
        textColor=colors.black,
        spaceAfter=6,
    )

    meta_style = ParagraphStyle(
        name="DocMeta",
        fontName=font_reg,
        fontSize=10,
        leading=14,
        textColor=colors.black,
        spaceAfter=14,
    )

    h1_style = ParagraphStyle(
        name="Header1",
        fontName=font_bold,
        fontSize=11,
        leading=16.5,
        textColor=colors.black,
        spaceBefore=10,
        spaceAfter=4,
        keepWithNext=True,
    )

    body_style = ParagraphStyle(
        name="BodyTextCustom",
        fontName=font_reg,
        fontSize=11,
        leading=16.5,  # Interlineado 1.5 respecto a 11 pt
        textColor=colors.black,
        spaceAfter=8,
    )

    code_style = ParagraphStyle(
        name="CodeBlock",
        fontName="Courier",
        fontSize=9,
        leading=13,
        textColor=colors.black,
        backColor=colors.white,  # Estricto blanco y negro, sin sombreados grises
        spaceAfter=8,
        leftIndent=10,
    )

    table_header_style = ParagraphStyle(
        name="TableHeader",
        fontName=font_bold,
        fontSize=7.5,
        leading=9.5,
        textColor=colors.black,
        alignment=1,
    )

    table_cell_style = ParagraphStyle(
        name="TableCell",
        fontName=font_reg,
        fontSize=7.5,
        leading=9.5,
        textColor=colors.black,
        alignment=1,
    )

    caption_style = ParagraphStyle(
        name="Caption",
        fontName=font_italic,
        fontSize=9,
        leading=12,
        textColor=colors.black,
        alignment=1,
        spaceAfter=12,
    )

    story = []

    # Encabezado formal en blanco y negro
    story.append(Paragraph("TC5061 - Experimentación y Seguimiento de Modelos con MLflow", title_style))
    story.append(
        Paragraph(
            "Autor: Oscar Alfonso Valenzuela Muñoz | Materia: MLOps (Semana 4)<br/>"
            "Repositorio: https://github.com/oscartrix/tec-mlops-2026.git",
            meta_style,
        )
    )

    # 1. Comando de ejecución
    story.append(Paragraph("1. Comando de Ejecución", h1_style))
    story.append(
        Paragraph(
            "El script de entrenamiento y seguimiento se ejecuta desde la línea de comandos permitiendo controlar los hiperparámetros y la semilla de aleatoriedad. La configuración con mejor desempeño se reproduce con el siguiente comando:",
            body_style,
        )
    )
    story.append(
        Paragraph(
            "python src/train.py --n-estimators 100 --max-depth 5 --min-samples-split 2 --min-samples-leaf 1 --random-state 42",
            code_style,
        )
    )
    story.append(
        Paragraph(
            "Adicionalmente, la totalidad de los cinco experimentos comparativos se ejecuta de manera automatizada mediante el script run_experiments.sh:",
            body_style,
        )
    )
    story.append(Paragraph("bash run_experiments.sh", code_style))

    # 2. Resumen comparativo de corridas
    story.append(Paragraph("2. Comparación Sistemática de Experimentos", h1_style))
    story.append(
        Paragraph(
            "Se registraron cinco corridas bajo el experimento Paddy_Yield_Random_Forest en el servidor de MLflow Tracking, variando la profundidad máxima, el número de estimadores y las restricciones de bifurcación y hojas. Los resultados obtenidos sobre el conjunto de prueba independiente con semilla fija en 42 se presentan a continuación:",
            body_style,
        )
    )

    # Tabla en estricto blanco y negro, sin líneas horizontales
    table_data = [
        [
            Paragraph("Corrida", table_header_style),
            Paragraph("max_depth", table_header_style),
            Paragraph("n_estimators", table_header_style),
            Paragraph("min_split", table_header_style),
            Paragraph("min_leaf", table_header_style),
            Paragraph("RMSE (kg)", table_header_style),
            Paragraph("MAE (kg)", table_header_style),
            Paragraph("R²", table_header_style),
        ],
        [
            Paragraph("shallow_depth_5", table_cell_style),
            Paragraph("5", table_cell_style),
            Paragraph("100", table_cell_style),
            Paragraph("2", table_cell_style),
            Paragraph("1", table_cell_style),
            Paragraph("910.49", table_cell_style),
            Paragraph("654.73", table_cell_style),
            Paragraph("0.9903", table_cell_style),
        ],
        [
            Paragraph("regularized_split10_leaf4", table_cell_style),
            Paragraph("None", table_cell_style),
            Paragraph("200", table_cell_style),
            Paragraph("10", table_cell_style),
            Paragraph("4", table_cell_style),
            Paragraph("931.82", table_cell_style),
            Paragraph("673.95", table_cell_style),
            Paragraph("0.9899", table_cell_style),
        ],
        [
            Paragraph("intermediate_depth_10_leaf_2", table_cell_style),
            Paragraph("10", table_cell_style),
            Paragraph("150", table_cell_style),
            Paragraph("5", table_cell_style),
            Paragraph("2", table_cell_style),
            Paragraph("952.39", table_cell_style),
            Paragraph("685.91", table_cell_style),
            Paragraph("0.9894", table_cell_style),
        ],
        [
            Paragraph("deep_trees_depth_15_n250", table_cell_style),
            Paragraph("15", table_cell_style),
            Paragraph("250", table_cell_style),
            Paragraph("2", table_cell_style),
            Paragraph("1", table_cell_style),
            Paragraph("965.75", table_cell_style),
            Paragraph("693.32", table_cell_style),
            Paragraph("0.9891", table_cell_style),
        ],
        [
            Paragraph("baseline_default", table_cell_style),
            Paragraph("None", table_cell_style),
            Paragraph("100", table_cell_style),
            Paragraph("2", table_cell_style),
            Paragraph("1", table_cell_style),
            Paragraph("968.59", table_cell_style),
            Paragraph("695.62", table_cell_style),
            Paragraph("0.9890", table_cell_style),
        ],
    ]

    # Tabla sin líneas horizontales y en blanco y negro puro
    comp_table = Table(table_data, colWidths=[120, 52, 60, 46, 44, 48, 48, 38])
    comp_table.setStyle(
        TableStyle(
            [
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("TOPPADDING", (0, 0), (-1, -1), 3),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ]
        )
    )
    story.append(comp_table)
    story.append(Spacer(1, 10))

    # 3. Análisis de Resultados en prosa continua y directa
    story.append(Paragraph("3. Análisis de Resultados", h1_style))
    story.append(
        Paragraph(
            "Entre las configuraciones evaluadas, el ensamble con profundidad acotada a cinco niveles (shallow_depth_5) obtuvo el desempeño más favorable, alcanzando un RMSE de 910.49 kg, un MAE de 654.73 kg y un coeficiente de determinación R² de 0.9903. En comparación con la línea base no restringida, este ajuste representó una reducción de 58.10 kg en el error cuadrático medio y de 40.89 kg en el error absoluto medio, lo que confirma que limitar el crecimiento vertical de los árboles reduce la sobreespecialización en las fluctuaciones puntuales de la muestra de entrenamiento.",
            body_style,
        )
    )
    story.append(
        Paragraph(
            "El hiperparámetro con mayor influencia en la capacidad predictiva fue la profundidad máxima de los árboles, junto con el tamaño mínimo de muestra en las hojas. Cuando se permitió un crecimiento libre o excesivamente profundo, el modelo tendió a memorizar patrones locales y ruido en las variables agronómicas continuas. Por el contrario, restringir la profundidad o exigir un mínimo de cuatro muestras por hoja aportó una regularización efectiva. Incrementar el número de árboles de 100 a 250 aportó beneficios marginales despreciables frente al costo computacional adicional, sin corregir la sobreestimación cuando la profundidad permaneció sin límites.",
            body_style,
        )
    )
    story.append(
        Paragraph(
            "Es fundamental señalar que el rendimiento agrícola en este conjunto de datos presenta una correlación lineal superior a 0.99 con la superficie de la parcela en hectáreas, debido a que la producción total en kilogramos está dominada por el tamaño del terreno. Por esta razón, modelos lineales elementales ya explican cerca del 98.7% de la varianza total, y las variaciones entre configuraciones de Random Forest corresponden a ajustes finos en torno a desviaciones locales.",
            body_style,
        )
    )
    story.append(
        Paragraph(
            "Entre las limitaciones principales se encuentra que la partición aleatoria convencional entre entrenamiento y prueba no contempla correlaciones espaciales ni temporales entre ciclos de cosecha, lo que puede sobrestimar el desempeño ante variaciones climáticas imprevistas. Asimismo, la codificación de variables categóricas aumentó la dimensionalidad a 63 columnas, favoreciendo la dispersión en las decisiones de corte de los árboles. Por tanto, la validación del modelo requiere evaluaciones adicionales con datos de cosechas subsecuentes.",
            body_style,
        )
    )

    # Anexo 1: Capturas de la interfaz de MLflow
    story.append(PageBreak())
    story.append(Paragraph("Anexo: Evidencias de Seguimiento en MLflow", h1_style))
    story.append(
        Paragraph(
            "Las siguientes capturas corresponden a la interfaz de MLflow Tracking en el servidor local (http://127.0.0.1:5000), donde se verifica el registro centralizado del experimento, las cinco corridas seleccionadas y la comparativa de hiperparámetros y métricas.",
            body_style,
        )
    )
    story.append(Spacer(1, 8))

    img_runs_table = "reports/figures/mlflow_runs_table.png"
    if os.path.exists(img_runs_table):
        story.append(Image(img_runs_table, width=470, height=220))
        story.append(
            Paragraph("Figura 1. Tabla de corridas de entrenamiento en MLflow con las cinco configuraciones y métricas registradas.", caption_style)
        )

    story.append(Spacer(1, 6))

    img_compare_view = "reports/figures/mlflow_comparison_view.png"
    if os.path.exists(img_compare_view):
        story.append(Image(img_compare_view, width=470, height=220))
        story.append(
            Paragraph("Figura 2. Vista de comparación en MLflow contrastando las métricas e hiperparámetros de las cinco ejecuciones.", caption_style)
        )

    # Anexo 2: Detalle y artefactos de la corrida ganadora
    story.append(PageBreak())
    story.append(Paragraph("Anexo: Detalle y Artefactos de la Corrida Seleccionada", h1_style))
    story.append(
        Paragraph(
            "En la vista de detalle de la corrida shallow_depth_5 se constató el almacenamiento de métricas, parámetros y artefactos, incluyendo la visualización de valores reales frente a predichos y la importancia de características calculada para el ensamble.",
            body_style,
        )
    )
    story.append(Spacer(1, 6))

    img_artifacts_preview = "reports/figures/mlflow_run_artifacts_preview.png"
    if os.path.exists(img_artifacts_preview):
        story.append(Image(img_artifacts_preview, width=470, height=220))
        story.append(
            Paragraph("Figura 3. Explorador de artefactos en MLflow mostrando la gráfica de valores reales frente a predichos.", caption_style)
        )

    story.append(Spacer(1, 6))

    img_fi = "reports/figures/feature_importance.png"
    if os.path.exists(img_fi):
        story.append(Image(img_fi, width=340, height=205))
        story.append(
            Paragraph("Figura 4. Gráfica de importancia de variables registrada como artefacto del modelo.", caption_style)
        )

    doc.build(story)
    print(f"Reporte generado exitosamente en: {output_pdf_path}")


if __name__ == "__main__":
    out_path = "reports/TC5061_Experimentacion_MLFlow_ValenzuelaOscar.pdf"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    create_report(out_path)
