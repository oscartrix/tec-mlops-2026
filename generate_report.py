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
        fontName=font_reg,
        fontSize=11,
        leading=16.5,
        textColor=colors.black,
        backColor=colors.white,  # Estricto blanco y negro, sin sombreados grises
        spaceAfter=8,
        leftIndent=10,
    )

    table_header_style = ParagraphStyle(
        name="TableHeader",
        fontName=font_bold,
        fontSize=11,
        leading=13,
        textColor=colors.black,
        alignment=1,
    )

    table_cell_style = ParagraphStyle(
        name="TableCell",
        fontName=font_reg,
        fontSize=11,
        leading=13,
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
    story.append(Paragraph("1. Comando de ejecución", h1_style))
    story.append(
        Paragraph(
            "El script se ejecuta desde la línea de comandos y recibe los hiperparámetros y la semilla como argumentos. La mejor configuración se reproduce con este comando:",
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
            "Las cinco corridas del experimento se ejecutan en secuencia con run_experiments.sh:",
            body_style,
        )
    )
    story.append(Paragraph("bash run_experiments.sh", code_style))

    # 2. Resumen comparativo de corridas
    story.append(Paragraph("2. Comparación de corridas", h1_style))
    story.append(
        Paragraph(
            "Se registraron cinco corridas en el experimento Paddy_Yield_Random_Forest. Entre ellas cambian la profundidad máxima, el número de árboles y el mínimo de muestras para dividir un nodo y por hoja. La semilla fue 42 en todas. Estos son los resultados sobre el conjunto de prueba:",
            body_style,
        )
    )

    # Tabla en estricto blanco y negro, sin líneas horizontales
    table_data = [
        [
            Paragraph("Corrida", table_header_style),
            Paragraph("Prof. máx.", table_header_style),
            Paragraph("Árboles", table_header_style),
            Paragraph("Min. split", table_header_style),
            Paragraph("Min. hoja", table_header_style),
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
    comp_table = Table(table_data, colWidths=[174, 44, 54, 38, 38, 52, 52, 46])
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
    story.append(Paragraph("3. Análisis de resultados", h1_style))
    analysis = [
        "La configuración con mejor desempeño fue shallow_depth_5, con RMSE de 910.49 kg, MAE de 654.73 kg y R² de 0.9903. Frente a baseline_default, el RMSE bajó 58.10 kg y el MAE 40.89 kg.",
        "El hiperparámetro más influyente fue la profundidad máxima. Las corridas baseline_default y shallow_depth_5 solo difieren en ese valor (sin límite contra cinco niveles), así que la mejora se le puede atribuir directamente. Limitar la profundidad obliga a cada árbol a promediar grupos más grandes de parcelas en lugar de ajustarse a diferencias pequeñas del conjunto de entrenamiento. En cambio, subir de 100 a 250 árboles con profundidad 15 dejó el error casi igual que la línea base. La corrida regularized_split10_leaf4 también mejoró, pero en ella cambiaron tres parámetros a la vez y no es posible separar el efecto de cada uno.",
        "El ejercicio tiene varias limitaciones. La producción en kilos depende casi por completo de la superficie sembrada: la correlación con las hectáreas es de 0.99 y una regresión lineal ya explica el 98.7 % de la varianza. Por eso las diferencias entre configuraciones son pequeñas. Además, cada configuración se evaluó con una sola partición. Al repetir la comparación con las semillas 1, 7 y 123, la profundidad cinco siguió siendo mejor, pero el RMSE cambió hasta 100 kg solo por la partición, una magnitud parecida a la diferencia entre configuraciones. Una validación cruzada daría una comparación más confiable. Por último, la partición aleatoria no separa las cosechas por temporada ni por región, por lo que el desempeño con datos de ciclos futuros podría ser menor.",
    ]
    for text in analysis:
        story.append(Paragraph(text, body_style))

    # Anexo 1: Capturas de la interfaz de MLflow
    story.append(PageBreak())
    story.append(Paragraph("Anexo: evidencias en MLflow", h1_style))
    story.append(
        Paragraph(
            "Las capturas se tomaron de la interfaz de MLflow en el servidor local (http://127.0.0.1:5000). Muestran las cinco corridas del experimento con sus parámetros y métricas, y la vista de comparación entre ellas.",
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
        story.append(Image(img_compare_view, width=470, height=248))
        story.append(
            Paragraph("Figura 2. Vista de comparación de MLflow con los hiperparámetros y el RMSE de las cinco corridas.", caption_style)
        )

    # Anexo 2: Detalle y artefactos de la corrida ganadora
    story.append(PageBreak())
    story.append(Paragraph("Anexo: artefactos de la mejor corrida", h1_style))
    story.append(
        Paragraph(
            "La pestaña de artefactos de la corrida shallow_depth_5 muestra los archivos registrados: las tres gráficas de evaluación en la carpeta plots y el archivo metrics.json en la carpeta metrics. El modelo se registró con mlflow.sklearn.log_model y aparece en la columna Models de la tabla de corridas.",
            body_style,
        )
    )
    story.append(Spacer(1, 6))

    img_artifacts_preview = "reports/figures/mlflow_run_artifacts_preview.png"
    if os.path.exists(img_artifacts_preview):
        story.append(Image(img_artifacts_preview, width=470, height=248))
        story.append(
            Paragraph("Figura 3. Explorador de artefactos en MLflow mostrando la gráfica de valores reales frente a predichos.", caption_style)
        )

    story.append(Spacer(1, 6))

    img_fi = "reports/figures/feature_importance.png"
    if os.path.exists(img_fi):
        story.append(Image(img_fi, width=340, height=227))
        story.append(
            Paragraph("Figura 4. Gráfica de importancia de variables registrada como artefacto del modelo.", caption_style)
        )

    doc.build(story)
    print(f"Reporte generado exitosamente en: {output_pdf_path}")


if __name__ == "__main__":
    out_path = "reports/TC5061_Experimentacion_MLFlow_ValenzuelaOscar.pdf"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    create_report(out_path)
