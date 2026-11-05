"""
Generates the two-column IEEE Literature Review Word document (.docx)
using STRICTLY the 12 PDF papers located in the 'Research Papers' directory.
"""
import os
import docx
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.oxml import parse_xml
from docx.oxml.ns import nsdecls

def create_two_column_literature_review():
    doc = Document()

    # 1. Page Margins (IEEE standard: 0.75 in top/bottom, 0.65 in left/right)
    for section in doc.sections:
        section.top_margin = Inches(0.75)
        section.bottom_margin = Inches(0.75)
        section.left_margin = Inches(0.65)
        section.right_margin = Inches(0.65)
        
        # 2-column layout (IEEE style: 2 columns, 0.25 in spacing)
        sectPr = section._sectPr
        for c in sectPr.xpath('./w:cols'):
            sectPr.remove(c)
        cols_xml = f'<w:cols {nsdecls("w")} w:num="2" w:space="360"/>'
        sectPr.append(parse_xml(cols_xml))

    # Base Font: Times New Roman 10pt (IEEE standard)
    style = doc.styles['Normal']
    font = style.font
    font.name = 'Times New Roman'
    font.size = Pt(10)
    font.color.rgb = RGBColor(0, 0, 0)

    # Section Heading: II. LITERATURE REVIEW
    h2 = doc.add_paragraph()
    h2.paragraph_format.space_before = Pt(0)
    h2.paragraph_format.space_after = Pt(4)
    h2.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r_h2 = h2.add_run('II.  LITERATURE REVIEW')
    r_h2.font.bold = True
    r_h2.font.size = Pt(10.5)

    # Intro sentence
    p_intro = doc.add_paragraph()
    p_intro.paragraph_format.space_after = Pt(6)
    p_intro.paragraph_format.line_spacing = 1.05
    p_intro.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    p_intro.add_run(
        'The pre-existing literature studies summarized in Table I provides strong evidence for the '
        'rapid development of deep learning techniques for pulmonary disease detection using chest X-rays. '
        'The literature is organized in reverse chronological order from 2026 down to 2024.'
    )

    # Table Title
    p_tab = doc.add_paragraph()
    p_tab.paragraph_format.space_before = Pt(4)
    p_tab.paragraph_format.space_after = Pt(4)
    p_tab.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.LEFT
    r_tab = p_tab.add_run('TABLE I: Related Work on Pulmonary Disease Detection using Deep Learning Techniques')
    r_tab.font.bold = True
    r_tab.font.size = Pt(8.5)

    # Table Data: STRICTLY the 12 papers from the Research Papers/ folder
    # Ordered: 2026 (6 papers) -> 2025 (5 papers) -> 2024 (1 paper)
    table_data = [
        ('Year', 'Author(s)', 'Title of the Paper', 'Methods Used', 'Assessment'),
        ('2026', 'Ba et al. [1]', 'A monotonic multi-expert Vision Transformer for clinical chest X-ray diagnosis', 'Multi-expert Vision Transformer with monotonic attention', 'Specialized attention experts improved discrimination of subtle radiographic patterns in ambiguous clinical cases.'),
        ('2026', 'Shi et al. [2]', 'Graph guided multiscale cross attention for multilabel chest X ray classification', 'ViT + DenseNet-121 dual visual encoder + GCN label reasoning', 'Proved modeling inter-disease pathological dependencies alongside visual cross-attention improves multi-label diagnosis.'),
        ('2026', 'Aperstein et al. [3]', 'Selective multipathology chest X-ray classification via rejection mechanisms', 'Multi-pathology classification + uncertainty rejection triage', 'Demonstrated routing high-uncertainty scans to senior radiologists drastically prevents clinical misdiagnosis.'),
        ('2026', 'Mažonienė & Šešok [4]', 'Explainable Multi-Label Chest X-Ray Disease Classification Using DualPool-DenseNet121', 'Dual-pooling DenseNet-121 + Grad-CAM heatmaps', 'Dual pooling preserved localized anomaly peaks while retaining global context across complex radiographs.'),
        ('2026', 'Shao et al. [5]', 'TRCGL-Net: A Long-Tailed Multi-Label Chest X-Ray Classification Framework', 'Generative data augmentation + label co-occurrence modeling', 'Mitigated long-tailed class imbalance and improved sensitivity on rare tail thoracic abnormalities.'),
        ('2026', 'Pan [6]', 'CDO-VIB DenseNet: An Improved DenseNet-121 Network for Multi-Label CXR', 'DenseNet-121 + Contextual Dropout + Information Bottleneck', 'Eliminated redundant background features, enhancing classification robustness on noisy chest radiographs.'),
        ('2025', 'Agarwal & Verma [7]', 'Advanced Chest X-Ray Analysis via Transformer-Based Descriptors & Cross-Model Attention', 'Transformer image descriptors + cross-model attention', 'Enhanced feature saliency across radiographs by aligning transformer descriptors with local CNN representations.'),
        ('2025', 'Bahram et al. [8]', 'CheX-DFuseNet: Multi-Label Chest Disease Classification via Hybrid CNN-ViT', 'EfficientNetV2-M + Vision Transformer + Optimization Learning', 'Demonstrated combining EfficientNet with Vision Transformer yields superior feature richness and diagnostic accuracy.'),
        ('2025', 'Fu et al. [9]', 'Explainable hybrid transformer for multi-classification of lung disease using CXR', 'Hybrid CNN-transformer architecture + visual explanation', 'Provided localized visual explanations and improved separation across diverse lung disease categories.'),
        ('2025', 'Vo & Do [10]', 'Incorporating self-attention into DenseNet for multi-label chest X-ray classification', 'DenseNet-121 integrated with spatial & channel self-attention', 'Self-attention effectively captured bilateral thoracic spatial correlations beyond localized convolutions.'),
        ('2025', 'Zhao & Wang [11]', 'Multi-label CXR image classification based on long-range dependencies & label relations', 'CNN feature extractor + non-local attention + GCN label graph', 'Jointly captured non-local contextual cues and pathology co-occurrence dependencies, improving multi-label F1 scores.'),
        ('2024', 'Xu & Rajpurkar [12]', 'Calibration and Uncertainty Estimation in Self-Supervised Chest X-ray Models', 'CheXzero model, Temperature Scaling, MC Dropout', 'Demonstrated severe miscalibration on external clinical datasets (ECE 0.12–0.41), proving uncertainty estimation is vital.')
    ]

    # Create table fitting the 3.45-inch column width
    table = doc.add_table(rows=len(table_data), cols=5)
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = False

    col_widths = [Inches(0.38), Inches(0.68), Inches(0.88), Inches(0.73), Inches(0.78)]

    # Add full black grid borders matching user screenshot
    tblPr = table._tbl.tblPr
    borders_xml = (
        f'<w:tblBorders {nsdecls("w")}>\n'
        f'  <w:top w:val="single" w:sz="4" w:space="0" w:color="000000"/>\n'
        f'  <w:bottom w:val="single" w:sz="4" w:space="0" w:color="000000"/>\n'
        f'  <w:left w:val="single" w:sz="4" w:space="0" w:color="000000"/>\n'
        f'  <w:right w:val="single" w:sz="4" w:space="0" w:color="000000"/>\n'
        f'  <w:insideH w:val="single" w:sz="4" w:space="0" w:color="000000"/>\n'
        f'  <w:insideV w:val="single" w:sz="4" w:space="0" w:color="000000"/>\n'
        f'</w:tblBorders>'
    )
    tblPr.append(parse_xml(borders_xml))

    for row_idx, row in enumerate(table.rows):
        trPr = row._tr.get_or_add_trPr()
        trPr.append(parse_xml(f'<w:cantSplit {nsdecls("w")}/>'))
        if row_idx == 0:
            trPr.append(parse_xml(f'<w:tblHeader {nsdecls("w")}/>'))

        for col_idx, text in enumerate(table_data[row_idx]):
            cell = row.cells[col_idx]
            cell.width = col_widths[col_idx]
            cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER

            # Set cell padding
            tcPr = cell._tc.get_or_add_tcPr()
            mar_xml = (
                f'<w:tcMar {nsdecls("w")}>\n'
                f'  <w:top w:w="20" w:type="dxa"/>\n'
                f'  <w:bottom w:w="20" w:type="dxa"/>\n'
                f'  <w:left w:w="40" w:type="dxa"/>\n'
                f'  <w:right w:w="40" w:type="dxa"/>\n'
                f'</w:tcMar>'
            )
            tcPr.append(parse_xml(mar_xml))

            # Header row shading
            if row_idx == 0:
                shd = parse_xml(f'<w:shd {nsdecls("w")} w:fill="F5F5F5"/>')
                tcPr.append(shd)

            p = cell.paragraphs[0]
            p.paragraph_format.space_before = Pt(1)
            p.paragraph_format.space_after = Pt(1)
            p.paragraph_format.line_spacing = 0.95

            if col_idx == 0:
                p.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER
            else:
                p.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.LEFT

            r = p.add_run(text)
            r.font.name = 'Times New Roman'
            r.font.size = Pt(7.0)
            if row_idx == 0 or (col_idx == 0 and row_idx > 0):
                r.font.bold = True

    # Narrative text following the table
    p_text1 = doc.add_paragraph()
    p_text1.paragraph_format.space_before = Pt(8)
    p_text1.paragraph_format.space_after = Pt(6)
    p_text1.paragraph_format.line_spacing = 1.05
    p_text1.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    p_text1.add_run(
        'The foundational evaluation by Xu and Rajpurkar [12] revealed that while deep radiograph classification '
        'models achieve high AUCs, they exhibit severe miscalibration on external clinical distributions (ECE 0.12–0.41), '
        'proving that uncalibrated model outputs present severe clinical risks.'
    )

    p_text2 = doc.add_paragraph()
    p_text2.paragraph_format.space_after = Pt(6)
    p_text2.paragraph_format.line_spacing = 1.05
    p_text2.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    p_text2.add_run(
        'Throughout 2025, investigations focused on hybrid architectures and attention enhancements. '
        'Bahram et al. [8] developed CheX-DFuseNet combining EfficientNetV2-M with Vision Transformers to maximize '
        'representation richness. Agarwal and Verma [7] demonstrated cross-model attention to align transformer image descriptors '
        'with convolutional maps, while Fu et al. [9] developed explainable hybrid transformers for multi-class separation. '
        'In parallel, Vo and Do [10] integrated self-attention into DenseNet, and Zhao and Wang [11] combined non-local '
        'dependencies with Graph Convolutional Networks (GCN) to capture clinical co-occurrences.'
    )

    p_text3 = doc.add_paragraph()
    p_text3.paragraph_format.space_after = Pt(6)
    p_text3.paragraph_format.line_spacing = 1.05
    p_text3.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    p_text3.add_run(
        'Most recently in 2026, research prioritized holistic diagnostic frameworks. Ba et al. [1] formulated multi-expert '
        'Vision Transformers with monotonic attention to resolve subtle pathological ambiguities. Shi et al. [2] combined '
        'dual visual encoders with graph-guided cross-attention. Aperstein et al. [3] implemented uncertainty-based rejection triage '
        'to refer ambiguous radiographs to senior radiologists, and Mažonienė and Šešok [4] integrated dual pooling with Grad-CAM. '
        'Shao et al. [5] (TRCGL-Net) addressed long-tailed multi-label imbalance via generative augmentation, and Pan [6] formulated '
        'CDO-VIB DenseNet to eliminate background noise. Despite these advances, pre-existing models rarely operate at high spatial '
        'resolution (512×512) or incorporate paired quantitative explainability audits across dual CNN-ViT backbones.'
    )

    # Transition to Methodology
    h3 = doc.add_paragraph()
    h3.paragraph_format.space_before = Pt(12)
    h3.paragraph_format.space_after = Pt(4)
    h3.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r_h3 = h3.add_run('III.  METHODOLOGY')
    r_h3.font.bold = True
    r_h3.font.size = Pt(10.5)

    p_m_intro = doc.add_paragraph()
    p_m_intro.paragraph_format.space_after = Pt(6)
    p_m_intro.paragraph_format.line_spacing = 1.05
    p_m_intro.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    p_m_intro.add_run(
        'The proposed RESPIRA framework integrates high-resolution radiograph representation learning, '
        'a dual-backbone hybrid vision system coupling EfficientNet-B0 and ViT-B/16, multi-modal probability '
        'and latent feature fusion, calibrated uncertainty triage, and paired visual explainability. '
        'The target classification space comprises six thoracic conditions: Atelectasis, Bacterial Pneumonia, '
        'Normal (Healthy), Pulmonary Edema, Tuberculosis, and Viral Pneumonia.'
    )

    # Subsection A
    h3_a = doc.add_paragraph()
    h3_a.paragraph_format.space_before = Pt(6)
    h3_a.paragraph_format.space_after = Pt(2)
    r_h3_a = h3_a.add_run('A. Dataset Curation and Preprocessing')
    r_h3_a.font.italic = True
    r_h3_a.font.bold = True
    r_h3_a.font.size = Pt(10)

    p_m_a = doc.add_paragraph()
    p_m_a.paragraph_format.space_after = Pt(6)
    p_m_a.paragraph_format.line_spacing = 1.05
    p_m_a.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    p_m_a.add_run(
        'Raw radiographs undergo deterministic preprocessing comprising Contrast Limited Adaptive Histogram '
        'Equalization (CLAHE, 8×8 grid, clip limit 2.0), followed by morphological lung field segmentation. '
        'Otsu thresholding and morphological filtering (9×9 elliptical structuring element) isolate the bilateral '
        'pulmonary lobes, with an anatomical ellipse fallback to safeguard against severe apical consolidations.'
    )

    # Subsection B
    h3_b = doc.add_paragraph()
    h3_b.paragraph_format.space_before = Pt(6)
    h3_b.paragraph_format.space_after = Pt(2)
    r_h3_b = h3_b.add_run('B. Dual-Backbone Architecture & Multi-Modal Fusion')
    r_h3_b.font.italic = True
    r_h3_b.font.bold = True
    r_h3_b.font.size = Pt(10)

    p_m_b = doc.add_paragraph()
    p_m_b.paragraph_format.space_after = Pt(6)
    p_m_b.paragraph_format.line_spacing = 1.05
    p_m_b.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    p_m_b.add_run(
        'High-resolution (512×512) inputs are processed in parallel through EfficientNet-B0 (capturing local textural '
        'opacities) and ViT-B/16 (capturing global thoracic contextual reasoning). Classification outputs are unified via '
        'an optimal weighted probability fusion rule (0.35 EfficientNet + 0.65 ViT), achieving 92.45% test accuracy, '
        '93.85% Macro F1, and 0.9910 Macro ROC-AUC across 1,947 independent held-out radiographs.'
    )

    # Subsection C
    h3_c = doc.add_paragraph()
    h3_c.paragraph_format.space_before = Pt(6)
    h3_c.paragraph_format.space_after = Pt(2)
    r_h3_c = h3_c.add_run('C. Calibrated Uncertainty Triage & Dual XAI Auditing')
    r_h3_c.font.italic = True
    r_h3_c.font.bold = True
    r_h3_c.font.size = Pt(10)

    p_m_c = doc.add_paragraph()
    p_m_c.paragraph_format.space_after = Pt(6)
    p_m_c.paragraph_format.line_spacing = 1.05
    p_m_c.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    p_m_c.add_run(
        'Dual uncertainty quantification combining normalized binary entropy and decision margin uncertainty stratifies '
        'predictions into a 3-tier clinical triage rule (Low Risk, Moderate Review, Mandatory Senior Audit). Concurrently, '
        'a paired explainability audit comparing Grad-CAM and ViT Cross-Attention Mapping achieves a 93.73% cross-modality '
        'explanation agreement rate, providing clinicians with verified, transparent visual rationales.'
    )

    # Section References: strictly the 12 papers
    h_ref = doc.add_paragraph()
    h_ref.paragraph_format.space_before = Pt(12)
    h_ref.paragraph_format.space_after = Pt(4)
    h_ref.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r_href = h_ref.add_run('REFERENCES')
    r_href.font.bold = True
    r_href.font.size = Pt(10.5)

    references_list = [
        '[1] T. Ba, C. Bi, J. Yu, M. Z. Jhandir, H. M. Raza ur Rehman, and G. S. Choi, "A monotonic multi-expert Vision Transformer for clinical chest X-ray diagnosis," Front. Med., vol. 13, Art. no. 1847086, 2026.',
        '[2] G. Shi, Z. Wang, Y. Shi, J. Pan, L. Sun, F. Fang, and L. Jin, "Graph guided multiscale cross attention for multilabel chest X ray classification," Sci. Rep., vol. 16, Art. no. 22761, 2026.',
        '[3] Y. Aperstein, A. Tzahar, A. Gottlib, T. Verber, R. S. Damti, and A. Apartsin, "Selective multipathology chest X-ray classification via rejection mechanisms," Sci. Rep., 2026, doi: 10.1038/s41598-026-66294-7.',
        '[4] E. Mažonienė and D. Šešok, "Explainable multi-label chest X-ray disease classification using DualPool-DenseNet121 and Grad-CAM visualization," Symmetry, vol. 18, no. 7, Art. no. 1294, 2026.',
        '[5] T. Shao, H. Ling, L. Zhang, J. Wu, J. Wang, Y. Gao, and F. Wang, "TRCGL-Net: A Long-Tailed Multi-Label Chest X-Ray Classification Framework with Generative Data Augmentation and Label Co-Occurrence Modeling," 2026.',
        '[6] C. Pan, "CDO-VIB DenseNet: An Improved DenseNet-121 Network for Multi-Label Chest X-ray Disease Classification," Health, Med. Therapeutics, vol. 65, 2026.',
        '[7] L. Agarwal and B. Verma, "Advanced Chest X-Ray Analysis via Transformer-Based Image Descriptors and Cross-Model Attention Mechanism," 2025.',
        '[8] A. M. Bahram, S. M. Omer, H. M. Mohammed, and S. A. Aula, "Enhanced Chest Disease Classification Using an Improved CheXNet Framework with EfficientNetV2-M and Optimization-Driven Learning," 2025.',
        '[9] X. Fu, R. Lin, W. Du, A. Tavares, et al., "Explainable hybrid transformer for multi-classification of lung disease using chest X-rays," 2025.',
        '[10] T.-T. Vo and T.-N. Do, "Incorporating self-attention into DenseNet for multi-label chest X-ray image classification," CTU J. Innov. Sustain. Dev., vol. 17, pp. 20–28, 2025.',
        '[11] X. Zhao and X. Wang, "Multi-label chest X-ray image classification based on long-range dependencies capture and label relationships learning," 2025.',
        '[12] J. Xu and P. Rajpurkar, "Calibration and uncertainty estimation challenges in self-supervised chest X-ray pathology classification models," in Proc. Med. Imaging Deep Learn. (MIDL), 2024, pp. 1–6.'
    ]

    for ref in references_list:
        p_r = doc.add_paragraph()
        p_r.paragraph_format.space_after = Pt(2)
        p_r.paragraph_format.line_spacing = 0.95
        p_r.paragraph_format.left_indent = Inches(0.2)
        p_r.paragraph_format.first_line_indent = Inches(-0.2)
        r = p_r.add_run(ref)
        r.font.name = 'Times New Roman'
        r.font.size = Pt(8.0)

    out_path = os.path.join('docs', 'LITERATURE_REVIEW_FINAL.docx')
    doc.save(out_path)
    print(f'Successfully generated: {out_path}')
    try:
        doc.save(os.path.join('docs', 'LITERATURE_REVIEW_TWO_COLUMN.docx'))
        print('Also updated LITERATURE_REVIEW_TWO_COLUMN.docx')
    except Exception:
        pass

if __name__ == '__main__':
    create_two_column_literature_review()
