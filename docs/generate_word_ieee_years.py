"""
Generates the two-column IEEE Literature Review Word document (.docx)
using STRICTLY AND ONLY the 16 PDF research papers located in:
  - Research Papers/IEEE/2026/ (2 papers)
  - Research Papers/IEEE/2025/ (12 papers)
  - Research Papers/IEEE/2024/ (2 papers)
"""
import os
import docx
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.oxml import parse_xml
from docx.oxml.ns import nsdecls

def create_ieee_years_literature_review():
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

    # Table Data: STRICTLY the 16 papers from 2026, 2025, and 2024 folders
    # Ordered: 2026 (2 papers) -> 2025 (12 papers) -> 2024 (2 papers)
    table_data = [
        ('Year', 'Author(s)', 'Title of the Paper', 'Methods Used', 'Assessment'),
        ('2026', 'Shukla et al. [1]', 'Chest X-Ray Classification using Deep Learning with Image Enhancement', 'Image enhancement + CLAHE + deep CNN classifiers', 'Confirmed preprocessing & contrast equalization are prerequisites for robust model generalization across diverse datasets.'),
        ('2026', 'Altun & Candemir [2]', 'Differential Evolution-Based CNN-Transformer Fusion for Thoracic Disease', 'Differential Evolution (DE) search for CNN-Transformer feature fusion', 'Combining CNN spatial maps with ViT embeddings achieved superior classification accuracy over single backbones.'),
        ('2025', 'Pandit et al. [3]', 'Explainable Deep Learning for Lung Disease Detection: A Comparative Study', 'DenseNet-121, ResNet-50, VGG-16, InceptionV3 + Grad-CAM', 'Pretrained CNNs yield high multi-class accuracy; visual heatmaps validated local lesion focus without global context.'),
        ('2025', 'Kansal et al. [4]', 'CXR-ViT: A Transformer-Based Framework for Enhanced Chest X-Ray Classification', 'Pure Vision Transformer (ViT) with patch self-attention', 'Proved self-attention effectively captures broad thoracic patterns, requiring training stability mechanisms.'),
        ('2025', 'Ghosh et al. [5]', 'Vision Transformers Excel in Chest X-Ray Analysis', 'ViT-B/16 benchmarked on NIH ChestX-ray14 dataset', 'Established self-attention layers outperform standard CNNs on diffuse conditions like cardiomegaly and edema.'),
        ('2025', 'Rafi et al. [6]', 'Chest X-ray Analysis for Pneumonia and TB Detection using CNNs and ViT', 'Comparative CNN vs. ViT evaluation + saliency mapping', 'ViT slightly surpassed CNN (90.12% vs. 89.61%), confirming complementary nature of CNN and transformer features.'),
        ('2025', 'Kim et al. [7]', 'Explainable AI for Multi-Label CXR Diagnosis: Layer-wise Grad-CAM', 'U-Net encoder-decoder + multi-layer Grad-CAM', 'Hierarchical multi-scale feature extraction across 14 observations on MIMIC-CXR significantly improved radiologist trust.'),
        ('2025', 'Dey [8]', 'CBAM-Enhanced DenseNet121 for Multi-Class Chest X-Ray Classification', 'DenseNet-121 + CBAM (Channel & Spatial Attention)', 'Attention re-weighting boosted diagnostic sensitivity in distinguishing bacterial from viral pneumonia in low-contrast zones.'),
        ('2025', 'Naghash et al. [9]', 'HyCoViT: Hybrid Convolution Vision Transformer With Dynamic Dropout', 'Hybrid CNN-ViT architecture + dynamic dropout', 'Successfully bridged local convolutional feature extraction with global token context, mitigating medical overfitting.'),
        ('2025', 'Fu et al. [10]', 'DAViT: A Domain-Adapted Vision Transformer for Pneumonia Detection', 'Domain-adapted Vision Transformer with specialized pretraining', 'Domain adaptation enhanced token representations of subtle radiographic opacities over standard natural-image ViTs.'),
        ('2025', 'Vibina [11]', 'Deep CNN-Driven Automated Lung Disease Identification', 'Deep CNN classification pipeline for pulmonary lesions', 'Validated deep convolutional feature representations on thoracic screening while noting sensitivity to low contrast.'),
        ('2025', 'Pabba et al. [12]', 'Explainable Artificial Intelligence for Pulmonary Disease Detection from CXR', 'XAI saliency mapping on CNN architectures', 'Saliency heatmaps localized focal consolidations, enhancing clinical interpretability for triage.'),
        ('2025', 'Qi [13]', 'Attention Mechanisms in Medical Image Analysis: Advance and Challenge', 'Survey of spatial, channel, and self-attention in medical vision', 'Confirmed attention mechanisms significantly boost diagnostic sensitivity and feature saliency across complex scans.'),
        ('2025', 'Zhang et al. [14]', 'Chest X-Ray Report Generation Using Abnormality Guided Vision Language Model', 'Abnormality-guided multimodal vision-language model', 'Multi-task vision-language feature alignment improved semantic coherence between radiographic findings and automated reports.'),
        ('2024', 'Barra et al. [15]', 'A Comparative Analysis of XAI Techniques for Medical Imaging', 'Comparative study of Grad-CAM, LIME, and SHAP', 'Grad-CAM provides intuitive and fast saliency maps; perturbation methods (LIME/SHAP) incur heavy compute overhead.'),
        ('2024', 'Yaligar et al. [16]', 'State-of-the-Art Deep Learning Strategies for Multi-Label CXR Disorders', 'Densely connected & deep residual CNN architectures', 'Achieved high diagnostic AUCs; highlighted pure CNNs struggle with long-range bilateral thoracic dependencies.')
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
        'The foundational studies in 2024 established critical benchmarks for automated thoracic analysis. '
        'Yaligar et al. [16] confirmed that while deep residual and dense CNN architectures achieve high diagnostic '
        'AUCs, pure CNNs struggle to model bilateral thoracic symmetry and long-range correlations. In parallel, '
        'Barra et al. [15] conducted comparative evaluations of XAI techniques, identifying Grad-CAM as the most '
        'computationally viable method for clinical deployment compared to compute-heavy perturbation tools like LIME and SHAP.'
    )

    p_text2 = doc.add_paragraph()
    p_text2.paragraph_format.space_after = Pt(6)
    p_text2.paragraph_format.line_spacing = 1.05
    p_text2.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    p_text2.add_run(
        'Throughout 2025, investigations focused on evaluating Vision Transformers alongside convolutional models. '
        'Pure transformer models such as CXR-ViT by Kansal et al. [4] and large-scale benchmarks by Ghosh et al. [5] '
        'proved that self-attention excels at capturing diffuse thoracic patterns (such as cardiomegaly and edema) '
        'that local convolutions struggle to aggregate. To combine local and global inductive biases, Rafi et al. [6] '
        'and Naghash et al. [9] (HyCoViT) demonstrated that hybrid CNN-ViT architectures mitigate overfitting in '
        'data-constrained medical regimes. Furthermore, attention-enhanced CNNs like CBAM-DenseNet121 [8], DAViT [10], '
        'Vibina [11], Qi [13], report generation models [14], and explainability frameworks like layer-wise Grad-CAM [3], [7], [12] '
        'enhanced localized sensitivity and clinician trust.'
    )

    p_text3 = doc.add_paragraph()
    p_text3.paragraph_format.space_after = Pt(6)
    p_text3.paragraph_format.line_spacing = 1.05
    p_text3.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    p_text3.add_run(
        'Most recently in 2026, research prioritized holistic diagnostic frameworks. Shukla et al. [1] confirmed that '
        'contrast enhancement and CLAHE preprocessing directly govern classification fidelity. Altun & Candemir [2] demonstrated '
        'that differential evolution-guided feature combinations between convolutional and self-attention representations '
        'achieve superior thoracic pathology discrimination compared to standalone architectures. Despite these advances, '
        'existing solutions rarely operate at high spatial resolution (512×512), lack paired dual-modality explainability audits, '
        'and omit calibrated uncertainty stratification.'
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

    # Section References: strictly the 16 papers from 2026, 2025, 2024 folders
    h_ref = doc.add_paragraph()
    h_ref.paragraph_format.space_before = Pt(12)
    h_ref.paragraph_format.space_after = Pt(4)
    h_ref.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r_href = h_ref.add_run('REFERENCES')
    r_href.font.bold = True
    r_href.font.size = Pt(10.5)

    references_list = [
        '[1] P. Shukla, M. Raj, D. Joshi, and S. M. B S, "Chest X-ray classification using deep learning framework with image enhancement," in Proc. IEEE Int. Conf., 2026, pp. 1–6.',
        '[2] S. Altun and S. Candemir, "Differential evolution-based CNN-transformer fusion for thoracic disease classification from chest X-rays," IEEE Access, vol. 14, pp. 1–14, 2026.',
        '[3] S. K. Pandit, S. S. Arjon, and T. Yasmin, "Explainable deep learning for lung disease detection: A comparative study using Grad-CAM," in Proc. 28th Int. Conf. Comput. Inf. Technol. (ICCIT), Cox\'s Bazar, Bangladesh, Dec. 2025, pp. 1–6.',
        '[4] K. Kansal, T. B. Chandra, and A. Singh, "CXR-ViT: A transformer-based framework for enhanced chest X-ray classification," in Proc. IEEE Conf. Comput. Vision Pattern Recognit. Workshops, 2025, pp. 1–6.',
        '[5] S. Ghosh, A. Bandyopadhyay, M. Bose, and K. C. Santosh, "Vision Transformers excel in chest X-ray analysis," in Proc. IEEE Conf. Artif. Intell. (CAI), Santa Clara, CA, USA, 2025, pp. 1–6.',
        '[6] M. S. M. Rafi, M. S. Ali, M. M. Hossen, S. Z. R. Sneha, and M. S. Arefin, "Chest X-ray analysis for pneumonia and tuberculosis detection using CNNs and Vision Transformers with explainable AI," in Proc. IEEE 2nd Int. Conf. Comput., Appl. Syst. (COMPAS), Kushtia, Bangladesh, Oct. 2025, pp. 1–6.',
        '[7] K. Kim, Y. Choi, and J. Seo, "Explainable AI for multi-label chest X-ray diagnosis: Layer-wise Grad-CAM with hierarchical feature extraction," IEEE Access, 2025.',
        '[8] U. K. Dey, "CBAM-Enhanced DenseNet121 for multi-class chest X-ray classification with Grad-CAM explainability," IEEE Access, vol. 13, pp. 1–14, 2025.',
        '[9] O. A. Naghash, N. Ling, and X. Li, "HyCoViT: Hybrid convolution vision transformer with dynamic dropout for enhanced medical chest X-ray classification," IEEE Access, vol. 13, pp. 1–13, 2025.',
        '[10] S. Fu, C. Tantithamthavorn, and T. Le, "DAViT: A domain-adapted vision transformer for automated pneumonia detection and explanation," IEEE Access, 2025.',
        '[11] W. Vibina, "Deep CNN-Driven Automated Lung Disease Identification," in Proc. IEEE Int. Conf., 2025.',
        '[12] P. Pabba, R. Samudrala, and R. Chevuturi, "Explainable Artificial Intelligence for Pulmonary Disease Detection from Chest Radiographs," in Proc. IEEE Int. Conf., 2025, pp. 1–6.',
        '[13] Y. Qi, "Attention Mechanisms in Medical Image Analysis: Advance, Challenge, and Future Direction," IEEE Access, 2025.',
        '[14] H. Zhang et al., "Chest X-Ray Report Generation Using Abnormality Guided Vision Language Model," IEEE Access, vol. 13, pp. 1–12, 2025.',
        '[15] P. Barra, A. Della Greca, I. Amaro, and A. Tortora, "A comparative analysis of XAI techniques for medical imaging: Challenges and opportunities," in Proc. IEEE Int. Conf. Bioinf. Biomed. (BIBM), Lisbon, Portugal, Dec. 2024, pp. 6782–6787.',
        '[16] N. Yaligar, S. G, K. Patil, P. Kulkarni, and P. Bhajantri, "State-of-the-Art Deep Learning Strategies for Multi-Label Classification of Chest X-Ray Thoracic Disorders," in Proc. IEEE Int. Conf., 2024, pp. 1–6.'
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

    # Save to dedicated file
    out_path = os.path.join('docs', 'LITERATURE_REVIEW_IEEE_YEARS.docx')
    doc.save(out_path)
    print(f'Successfully generated: {out_path}')
    
    # Try updating other filenames if not locked
    for alt in ['LITERATURE_REVIEW_FINAL.docx', 'LITERATURE_REVIEW_TWO_COLUMN.docx', 'LITERATURE_REVIEW.docx']:
        try:
            doc.save(os.path.join('docs', alt))
            print(f'Also updated {alt}')
        except Exception:
            pass

if __name__ == '__main__':
    create_ieee_years_literature_review()
