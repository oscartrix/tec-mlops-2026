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


def create_report(output_pdf_path: str):
    doc = SimpleDocTemplate(
        output_pdf_path,
        pagesize=letter,
        leftMargin=54,
        rightMargin=54,
        topMargin=54,
        bottomMargin=54,
    )

    styles = getSampleStyleSheet()

    # Tipografía solicitada: Arial (Helvetica en ReportLab), 11 pt, interlineado 1.5 (leading = 16.5)
    # Estricto blanco y negro para todos los títulos y textos
    title_style = ParagraphStyle(
        name="DocTitle",
        fontName="Helvetica-Bold",
        fontSize=15,
        leading=20,
        textColor=colors.black,
        spaceAfter=6,
    )

    subtitle_style = ParagraphStyle(
        name="DocSubtitle",
        fontName="Helvetica",
        fontSize=10,
        leading=14,
        textColor=colors.black,
        spaceAfter=14,
    )

    h1_style = ParagraphStyle(
        name="Header1",
        fontName="Helvetica-Bold",
        fontSize=12,
        leading=16.5,
        textColor=colors.black,
        spaceBefore=10,
        spaceAfter=6,
        keepWithNext=True,
    )

    body_style = ParagraphStyle(
        name="BodyTextCustom",
        fontName="Helvetica",
        fontSize=11,
        leading=16.5,  # Interlineado exacto de 1.5 a 11 pt
        textColor=colors.black,
        spaceAfter=8,
    )

    code_style = ParagraphStyle(
        name="CodeBlock",
        fontName="Courier",
        fontSize=9.5,
        leading=13.5,
        textColor=colors.black,
        backColor=colors.HexColor("#F3F4F6"),
        borderPadding=6,
        spaceAfter=8,
    )

    table_header_style = ParagraphStyle(
        name="TableHeader",
        fontName="Helvetica-Bold",
        fontSize=9,
        leading=12,
        textColor=colors.black,
        alignment=1,
    )

    table_cell_style = ParagraphStyle(
        name="TableCell",
        fontName="Helvetica",
        fontSize=8.5,
        leading=11.5,
        textColor=colors.black,
        alignment=1,
    )

    caption_style = ParagraphStyle(
        name="Caption",
        fontName="Helvetica-Oblique",
        fontSize=9,
        leading=12,
        textColor=colors.black,
        alignment=1,
        spaceAfter=12,
    )

    story = []

    # Encabezado del documento
    story.append(Paragraph("TC5061 - Experimentación y Seguimiento de Modelos con MLflow", title_style))
    story.append(
        Paragraph(
            "<b>Autor:</b> Oscar Alfonso Valenzuela Muñoz &nbsp;|&nbsp; "
            "<b>Materia:</b> MLOps (Semana 4)<br/>"
            "<b>Repositorio GitHub:</b> https://github.com/oscartrix/tec-mlops-2026.git",
            subtitle_style,
        )
    )

    # 1. Comando exacto de ejecución
    story.append(Paragraph("1. Comando de Ejecución", h1_style))
    story.append(
        Paragraph(
            "La ejecución del script de entrenamiento parametrizado se realiza directamente desde la terminal. Para reproducir la mejor configuración encontrada se ejecuta el siguiente comando:",
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
            "Para ejecutar la batería completa de las cinco corridas comparativas de forma desatendida, se incluye el script de automatización:",
            body_style,
        )
    )
    story.append(Paragraph("bash run_experiments.sh", code_style))

    # 2. Resumen comparativo de corridas
    story.append(Paragraph("2. Comparación Sistemática de Experimentos", h1_style))
    story.append(
        Paragraph(
            "Se registraron cinco corridas bajo el experimento <i>Paddy_Yield_Random_Forest</i> en el servidor de MLflow Tracking, variando sistemáticamente la profundidad máxima, el número de estimadores y las restricciones de división en nodos y hojas. La tabla siguiente presenta los parámetros evaluados y las métricas obtenidas sobre el conjunto de prueba independiente (20% de los datos con semilla fija en 42):",
            body_style,
        )
    )

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
            Paragraph("shallow_depth_5 (Mejor)", table_cell_style),
            Paragraph("5", table_cell_style),
            Paragraph("100", table_cell_style),
            Paragraph("2", table_cell_style),
            Paragraph("1", table_cell_style),
            Paragraph("<b>910.49</b>", table_cell_style),
            Paragraph("<b>654.73</b>", table_cell_style),
            Paragraph("<b>0.9903</b>", table_cell_style),
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

    comp_table = Table(table_data, colWidths=[120, 52, 58, 48, 48, 58, 58, 46])
    comp_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E5E7EB")),
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#9CA3AF")),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F9FAFB")]),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    story.append(comp_table)
    story.append(Spacer(1, 10))

    # 3. Análisis de Resultados
    story.append(Paragraph("3. Análisis de Resultados", h1_style))
    story.append(
        Paragraph(
            "<b>Configuración con mejor desempeño:</b> La corrida identificada como <i>shallow_depth_5</i> obtuvo el mejor resultado global entre todas las variantes evaluadas, registrando un RMSE de 910.49 kg, un MAE de 654.73 kg y un coeficiente de determinación R² de 0.9903. Al contrastarla con la línea base no restringida (RMSE de 968.59 kg y MAE de 695.62 kg), se aprecia una reducción del error cuadrático medio de 58.10 kg y de 40.89 kg en el error absoluto. Esto evidencia que fijar una cota superior a la profundidad previene la sobreespecialización del ensamble en fluctuaciones particulares del conjunto de entrenamiento.",
            body_style,
        )
    )
    story.append(
        Paragraph(
            "<b>Hiperparámetro más influyente:</b> El hiperparámetro de mayor impacto sobre la capacidad predictiva fue la profundidad máxima del árbol (<i>max_depth</i>), seguido de la restricción del tamaño de nodo hoja (<i>min_samples_leaf</i>). Los árboles con profundidad libre (<i>max_depth=None</i>) o elevada (<i>max_depth=15</i>) tendieron a memorizar ruido local en las variables agronómicas continuas. En contraste, restringir la profundidad a 5 niveles o imponer un mínimo de 4 observaciones por hoja (<i>regularized_split10_leaf4</i>, RMSE 931.82 kg) aportó una regularización estructural decisiva. Asimismo, incrementar el número de árboles de 100 a 250 generó retornos marginales decrecientes y elevó el tiempo de cómputo sin mitigar el sobreajuste cuando la profundidad permaneció sin control.",
            body_style,
        )
    )
    story.append(
        Paragraph(
            "<b>Limitaciones observadas:</b> En primer lugar, la partición aleatoria simple no modela posibles agrupaciones geoespaciales o temporales inherentes a los ciclos agrícolas del cultivo de arroz, lo que puede sobrestimar el desempeño en regiones no contempladas. En segundo lugar, la codificación por variables ficticias (One-Hot Encoding) aumentó la dimensionalidad a 63 columnas, incrementando la dispersión en las decisiones de corte de los árboles. Por último, aunque el coeficiente R² supera 0.98 debido a dependencias funcionales entre variables de entrada y rendimiento, la validez del modelo requiere corroboración en temporadas posteriores con condiciones climáticas atípicas.",
            body_style,
        )
    )

    # Anexo de Evidencias de MLflow
    story.append(PageBreak())
    story.append(Paragraph("Anexo: Evidencias de Seguimiento en MLflow", h1_style))
    story.append(
        Paragraph(
            "A continuación se presentan las capturas de pantalla obtenidas directamente desde la interfaz web de MLflow Tracking (alojada en http://127.0.0.1:5000), donde se comprueba el registro centralizado del experimento, las cinco corridas y los artefactos generados.",
            body_style,
        )
    )
    story.append(Spacer(1, 8))

    img_runs_table = "reports/figures/mlflow_runs_table.png"
    if os.path.exists(img_runs_table):
        story.append(Image(img_runs_table, width=460, height=225))
        story.append(
            Paragraph("Figura 1. Tabla de corridas en MLflow con las cinco configuraciones registradas y seleccionadas para comparación.", caption_style)
        )

    story.append(Spacer(1, 6))

    img_compare_view = "reports/figures/mlflow_comparison_view.png"
    if os.path.exists(img_compare_view):
        story.append(Image(img_compare_view, width=460, height=225))
        story.append(
            Paragraph(
                "Figura 2. Vista de comparación de MLflow (Comparing 5 Runs) con gráfica de coordenadas paralelas y contraste de métricas.",
                caption_style,
            )
        )

    # Segunda página de anexos: artefactos y modelo
    story.append(PageBreak())
    story.append(Paragraph("Anexo: Detalle y Artefactos de la Corrida Ganadora", h1_style))
    story.append(
        Paragraph(
            "Se verificó el registro de artefactos dentro del almacén de MLflow, incluyendo gráficas de dispersión (valores reales vs. predichos), distribución de residuales, importancia de características, reporte JSON de métricas y el modelo empaquetado con firma de entrada y salida.",
            body_style,
        )
    )
    story.append(Spacer(1, 6))

    img_artifacts_preview = "reports/figures/mlflow_run_artifacts_preview.png"
    if os.path.exists(img_artifacts_preview):
        story.append(Image(img_artifacts_preview, width=460, height=225))
        story.append(
            Paragraph("Figura 3. Explorador de artefactos en MLflow mostrando la previsualización directa de la gráfica de dispersión para shallow_depth_5.", caption_style)
        )

    story.append(Spacer(1, 6))

    img_fi = "reports/figures/feature_importance.png"
    if os.path.exists(img_fi):
        story.append(Image(img_fi, width=330, height=205))
        story.append(
            Paragraph("Figura 4. Gráfica de importancia de características registrada como artefacto en MLflow.", caption_style)
        )

    doc.build(story)
    print(f"Reporte generado exitosamente en: {output_pdf_path}")


if __name__ == "__main__":
    out_path = "reports/TC5061_Experimentacion_MLFlow_ValenzuelaOscar.pdf"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    create_report(out_path)
