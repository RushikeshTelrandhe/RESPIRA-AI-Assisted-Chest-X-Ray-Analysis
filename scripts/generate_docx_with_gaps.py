"""
Generates the two-column IEEE Literature Review Word document (.docx)
using STRICTLY AND ONLY the 16 PDF research papers located in:
  - Research Papers/IEEE/2026/ (2 papers)
  - Research Papers/IEEE/2025/ (12 papers)
  - Research Papers/IEEE/2024/ (2 papers)

Table I columns:
  Year | Author(s) | Title of the Paper | Methods Used | Research Gap
(Assessment column replaced with Research Gap based on rigorous per-paper analysis)
"""
import os
import docx
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.oxml import parse_xml
from docx.oxml.ns import nsdecls

def create_ieee_literature_review_docx(output_filename):
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
        'The pre-existing literature studies summarized in Table I provide strong evidence for the '
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
    # Columns: Year | Author(s) | Title of the Paper | Methods Used | Research Gap
    table_data = [
        ('Year', 'Author(s)', 'Title of the Paper', 'Methods Used', 'Research Gap'),
        ('2026', 'Shukla et al. [1]', 'Chest X-Ray Classification using Deep Learning with Image Enhancement', 'Contrast enhancement + CLAHE + deep CNN classifiers (EfficientNet-B0)', 'Restricted to 3-class classification without multi-label support; relies on single CNN lacking global context and predictive uncertainty.'),
        ('2026', 'Altun & Candemir [2]', 'Differential Evolution-Based CNN-Transformer Fusion for Thoracic Disease', 'Differential Evolution (DE) search for CNN-Transformer feature fusion', 'Lacks external multi-center validation, relies on noisy NLP-extracted reports (NIH CXR), and omits calibrated uncertainty bounds.'),
        ('2025', 'Pandit et al. [3]', 'Explainable Deep Learning for Lung Disease Detection: A Comparative Study', 'DenseNet-121, ResNet-50, VGG-16, InceptionV3 + Grad-CAM', 'Restricted to standalone CNN backbones lacking global attention; visual explanations rely only on qualitative Grad-CAM without uncertainty triage.'),
        ('2025', 'Kansal et al. [4]', 'CXR-ViT: A Transformer-Based Framework for Enhanced Chest X-Ray Classification', 'Pure Vision Transformer (ViT) with patch self-attention', 'Pure Vision Transformer struggles to capture fine-grained localized parenchymal textures and micro-opacities without CNN inductive bias.'),
        ('2025', 'Ghosh et al. [5]', 'Vision Transformers Excel in Chest X-Ray Analysis', 'ViT-B/16 benchmarked on NIH ChestX-ray14 dataset', 'Evaluates ViT and CNN architectures purely as isolated standalone models without hybrid fusion; suffers from poor F1 on tail classes.'),
        ('2025', 'Rafi et al. [6]', 'Chest X-ray Analysis for Pneumonia and TB Detection using CNNs and ViT', 'Comparative CNN vs. ViT evaluation + saliency mapping', 'Compares CNN and ViT models as separate uncoupled streams rather than fusing features; confined to 3-class data without multi-label support.'),
        ('2025', 'Kim et al. [7]', 'Explainable AI for Multi-Label CXR Diagnosis: Layer-wise Grad-CAM', 'U-Net encoder-decoder + multi-layer Grad-CAM', 'Relies strictly on convolutional representations without self-attention for global thoracic dependencies, lacking quantitative concordance auditing.'),
        ('2025', 'Dey [8]', 'CBAM-Enhanced DenseNet121 for Multi-Class Chest X-Ray Classification', 'DenseNet-121 + CBAM (Channel & Spatial Attention)', 'Restricted to single-institution pediatric dataset with potential label noise from metadata filenames; lacks external multi-disease cohort validation.'),
        ('2025', 'Naghash et al. [9]', 'HyCoViT: Hybrid Convolution Vision Transformer With Dynamic Dropout', 'Hybrid CNN-ViT architecture + dynamic dropout', 'Combines CNN and ViT features but lacks multi-stage cross-attention alignment and disease-conditioned query modeling; omits uncertainty quantification.'),
        ('2025', 'Fu et al. [10]', 'DAViT: A Domain-Adapted Vision Transformer for Pneumonia Detection', 'Domain-adapted Vision Transformer with specialized pretraining', 'Restricted to binary/ternary pneumonia detection; fails to generalize to comprehensive thoracic pathologies and lacks uncertainty-based triage.'),
        ('2025', 'Vibina [11]', 'Deep CNN-Driven Automated Lung Disease Identification', 'Deep CNN classification pipeline for pulmonary lesions', 'Dependent on single-center datasets and pure CNN extractors with limited receptive fields; lacks self-attention and confidence calibration.'),
        ('2025', 'Pabba et al. [12]', 'Explainable Artificial Intelligence for Pulmonary Disease Detection from CXR', 'XAI saliency mapping on CNN architectures', 'Highlights that existing literature predominantly focuses on post-hoc visual heatmaps without quantitative validation or clinical trust metrics.'),
        ('2025', 'Qi [13]', 'Attention Mechanisms in Medical Image Analysis: Advance and Challenge', 'Survey of spatial, channel, and self-attention in medical vision', 'Identifies high computational complexity, lack of cross-modal adaptability, and scarcity of lightweight hybrid architectures with uncertainty feedback.'),
        ('2025', 'Edirisinghe et al. [14]', 'Chest X-Ray Report Generation Using Abnormality Guided Vision Language Model', 'Abnormality-guided multimodal vision-language model (META-CXR)', 'High computational footprint and complex vision-language decoding cause heavy latency, making lightweight offline desktop deployment impractical.'),
        ('2024', 'Barra et al. [15]', 'A Comparative Analysis of XAI Techniques for Medical Imaging', 'Comparative study of Grad-CAM, LIME, and SHAP', 'Perturbation methods (LIME/SHAP) are computationally prohibitive for clinical workflows, while Grad-CAM suffers from coarse spatial granularity.'),
        ('2024', 'Yaligar et al. [16]', 'State-of-the-Art Deep Learning Strategies for Multi-Label CXR Disorders', 'Densely connected & deep residual CNN architectures', 'Pure CNN models struggle with long-range bilateral thoracic dependencies; exhibits poor sensitivity on tail pathologies without uncertainty estimation.')
    ]

    # Create table fitting the 3.45-inch column width
    table = doc.add_table(rows=len(table_data), cols=5)
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = False

    col_widths = [Inches(0.35), Inches(0.65), Inches(0.85), Inches(0.72), Inches(0.88)]

    # Add full black grid borders matching user screenshot
    tblPr = table._tbl.tblPr
    borders_xml = f'''
    <w:tblBorders {nsdecls("w")}>
        <w:top w:val="single" w:sz="6" w:space="0" w:color="000000"/>
        <w:left w:val="single" w:sz="6" w:space="0" w:color="000000"/>
        <w:bottom w:val="single" w:sz="6" w:space="0" w:color="000000"/>
        <w:right w:val="single" w:sz="6" w:space="0" w:color="000000"/>
        <w:insideH w:val="single" w:sz="4" w:space="0" w:color="000000"/>
        <w:insideV w:val="single" w:sz="4" w:space="0" w:color="000000"/>
    </w:tblBorders>
    '''
    tblPr.append(parse_xml(borders_xml))

    # Format Header Row
    hdr_cells = table.rows[0].cells
    for i, cell in enumerate(hdr_cells):
        cell.width = col_widths[i]
        tcPr = cell._tc.get_or_add_tcPr()
        # Light gray header background
        shd_xml = f'<w:shd {nsdecls("w")} w:fill="EDEDED"/>'
        tcPr.append(parse_xml(shd_xml))
        # Margins: top/bottom 25, left/right 25
        tcMar_xml = f'''
        <w:tcMar {nsdecls("w")}>
            <w:top w:w="30" w:type="dxa"/>
            <w:bottom w:w="30" w:type="dxa"/>
            <w:left w:w="25" w:type="dxa"/>
            <w:right w:w="25" w:type="dxa"/>
        </w:tcMar>
        '''
        tcPr.append(parse_xml(tcMar_xml))
        
        p = cell.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.space_before = Pt(1)
        p.paragraph_format.space_after = Pt(1)
        p.paragraph_format.line_spacing = 0.95
        run = p.add_run(table_data[0][i])
        run.font.bold = True
        run.font.name = 'Times New Roman'
        run.font.size = Pt(6)
        run.font.color.rgb = RGBColor(0, 0, 0)
        cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER

    # Format Data Rows
    for row_idx, row_values in enumerate(table_data[1:], start=1):
        row = table.rows[row_idx]
        for col_idx, text in enumerate(row_values):
            cell = row.cells[col_idx]
            cell.width = col_widths[col_idx]
            tcPr = cell._tc.get_or_add_tcPr()
            
            # Subtle zebra shading
            if row_idx % 2 == 0:
                shd_xml = f'<w:shd {nsdecls("w")} w:fill="F9F9F9"/>'
                tcPr.append(parse_xml(shd_xml))

            tcMar_xml = f'''
            <w:tcMar {nsdecls("w")}>
                <w:top w:w="20" w:type="dxa"/>
                <w:bottom w:w="20" w:type="dxa"/>
                <w:left w:w="20" w:type="dxa"/>
                <w:right w:w="20" w:type="dxa"/>
            </w:tcMar>
            '''
            tcPr.append(parse_xml(tcMar_xml))

            p = cell.paragraphs[0]
            p.paragraph_format.space_before = Pt(0.5)
            p.paragraph_format.space_after = Pt(0.5)
            p.paragraph_format.line_spacing = 0.95

            # Alignment
            if col_idx == 0:
                p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            elif col_idx == 1:
                p.alignment = WD_ALIGN_PARAGRAPH.LEFT
            else:
                p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY

            run = p.add_run(text)
            run.font.name = 'Times New Roman'
            run.font.size = Pt(5.5)
            run.font.color.rgb = RGBColor(0, 0, 0)
            
            if col_idx == 0:
                run.font.bold = True
            if col_idx == 2:
                run.font.italic = True
                
            cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER

    # Add synthesis text after table
    p_spacer = doc.add_paragraph()
    p_spacer.paragraph_format.space_before = Pt(6)
    p_spacer.paragraph_format.space_after = Pt(2)

    # Synthesis Paragraphs
    p_synth1 = doc.add_paragraph()
    p_synth1.paragraph_format.space_after = Pt(4)
    p_synth1.paragraph_format.line_spacing = 1.05
    p_synth1.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    p_synth1.add_run(
        'Foundational studies conducted in 2024 established core benchmarks and architectural trade-offs in '
        'computer-aided thoracic diagnosis. Yaligar et al. [16] benchmarked dense residual convolutional architectures '
        'across multi-label disorders, revealing that while deep CNNs achieve high diagnostic AUCs, localized convolutional '
        'kernels struggle to model long-range spatial correlations across bilateral lung fields. Concurrently, Barra et al. [15] '
        'evaluated explainable AI techniques across clinical radiography, demonstrating that gradient-weighted saliency methods '
        '(Grad-CAM) provide computationally viable heatmaps compared to computationally prohibitive perturbation frameworks '
        'such as LIME and SHAP.'
    )

    p_synth2 = doc.add_paragraph()
    p_synth2.paragraph_format.space_after = Pt(4)
    p_synth2.paragraph_format.line_spacing = 1.05
    p_synth2.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    p_synth2.add_run(
        'Throughout 2025, investigations focused on evaluating Vision Transformers alongside convolutional models. '
        'Pure transformer models such as CXR-ViT by Kansal et al. [4] and large-scale benchmarks by Ghosh et al. [5] '
        'proved that patch-level self-attention excels at capturing diffuse thoracic patterns (such as cardiomegaly and '
        'pulmonary edema) that local convolutions struggle to aggregate. To combine local textural inductive bias with global '
        'contextual embeddings, Rafi et al. [6] and Naghash et al. [9] (HyCoViT) demonstrated that hybrid CNN-ViT architectures '
        'mitigate overfitting in data-constrained medical regimes. Furthermore, attention-enhanced CNNs like CBAM-DenseNet121 [8], '
        'DAViT [10], Vibina [11], Qi [13], abnormality-guided report generation [14], and explainability frameworks like '
        'layer-wise Grad-CAM [3], [7], [12] enhanced localized sensitivity and clinician trust.'
    )

    p_synth3 = doc.add_paragraph()
    p_synth3.paragraph_format.space_after = Pt(4)
    p_synth3.paragraph_format.line_spacing = 1.05
    p_synth3.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    p_synth3.add_run(
        'Most recently in 2026, research prioritized holistic diagnostic frameworks. Shukla et al. [1] confirmed that '
        'contrast enhancement and CLAHE preprocessing directly govern classification fidelity across diverse scanner sources. '
        'Altun and Candemir [2] demonstrated that differential evolution-guided feature combinations between convolutional '
        'and self-attention representations achieve superior thoracic pathology discrimination compared to standalone backbones.'
    )

    # Identified Research Gaps Heading
    h_gaps = doc.add_paragraph()
    h_gaps.paragraph_format.space_before = Pt(4)
    h_gaps.paragraph_format.space_after = Pt(2)
    r_hg = h_gaps.add_run('Identified Research Gaps:')
    r_hg.font.bold = True
    r_hg.font.size = Pt(9.5)

    gaps = [
        ('1. Resolution and High-Frequency Loss: ', 'Most existing models downsample radiographs to 224x224, losing subtle apical nodules and fine reticular opacities.'),
        ('2. Isolated Backbones vs. Deep Relational Conditioning: ', 'Fusion strategies often rely on late heuristic concatenation without modeling cross-attention and inter-disease relational dependencies.'),
        ('3. Absence of Calibrated Uncertainty in Diagnostic Workflows: ', 'High-accuracy models routinely omit predictive uncertainty bounds, treating confident and ambiguous predictions identically.'),
        ('4. Lack of Dual-Modality Quantitative XAI Auditing: ', 'Visual explanations are rarely audited quantitatively across paired CNN and ViT modalities on identical test cohorts.')
    ]
    for title, desc in gaps:
        p_g = doc.add_paragraph()
        p_g.paragraph_format.space_after = Pt(2)
        p_g.paragraph_format.line_spacing = 1.05
        p_g.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        r_t = p_g.add_run(title)
        r_t.font.bold = True
        r_t.font.size = Pt(9)
        r_d = p_g.add_run(desc)
        r_d.font.size = Pt(9)

    # Resolution Heading
    h_res = doc.add_paragraph()
    h_res.paragraph_format.space_before = Pt(4)
    h_res.paragraph_format.space_after = Pt(2)
    r_hr = h_res.add_run('How the Proposed RESPIRA Framework Resolves These Gaps:')
    r_hr.font.bold = True
    r_hr.font.size = Pt(9.5)

    resolutions = [
        ('1. Deterministic Preprocessing Pipeline: ', 'CLAHE contrast equalization alongside Otsu morphological lung field masking and anatomical fallback heuristics.'),
        ('2. Dual-Backbone Hybrid Architecture: ', 'EfficientNet-B0 (local textural inductive bias) coupled with Vision Transformer ViT-B/16 (global thoracic context).'),
        ('3. High-Resolution Multi-Modal Fusion: ', 'High-resolution (512x512) weighted probability fusion (0.35 EfficientNet + 0.65 ViT) alongside a 5-stage latent feature fusion pipeline with disease query attention.'),
        ('4. Calibrated Uncertainty Quantification: ', 'Normalized binary entropy and decision margin uncertainty combined into a 3-tier clinical risk triage rule (Low Risk, Moderate Review, Mandatory Senior Audit).'),
        ('5. Paired Dual-Modality Explainability Audit: ', 'Systematic quantitative concordance audit comparing Grad-CAM and ViT Attention Mapping across 1,947 held-out test radiographs (93.73% cross-modality agreement rate).'),
        ('6. Full Clinical Platform Implementation: ', 'Authenticated asynchronous FastAPI backend, SQLAlchemy ORM, automated ReportLab PDF diagnostic reporting, and a React 19 / Electron desktop suite.')
    ]
    for title, desc in resolutions:
        p_r = doc.add_paragraph()
        p_r.paragraph_format.space_after = Pt(2)
        p_r.paragraph_format.line_spacing = 1.05
        p_r.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        r_t = p_r.add_run(title)
        r_t.font.bold = True
        r_t.font.size = Pt(9)
        r_d = p_r.add_run(desc)
        r_d.font.size = Pt(9)

    # References Section
    p_ref_h = doc.add_paragraph()
    p_ref_h.paragraph_format.space_before = Pt(6)
    p_ref_h.paragraph_format.space_after = Pt(3)
    p_ref_h.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r_ref = p_ref_h.add_run('REFERENCES')
    r_ref.font.bold = True
    r_ref.font.size = Pt(10)

    references = [
        '[1] P. Shukla, M. Raj, D. Joshi, and S. M. B S, "Chest X-ray classification using deep learning framework with image enhancement," in Proc. IEEE Int. Conf., 2026, pp. 1–6.',
        '[2] S. Altun and S. Candemir, "Differential evolution-based CNN-transformer fusion for thoracic disease classification from chest X-rays," IEEE Access, vol. 14, pp. 1–14, 2026.',
        '[3] S. K. Pandit, S. S. Arjon, and T. Yasmin, "Explainable deep learning for lung disease detection: A comparative study using Grad-CAM," in Proc. 28th Int. Conf. Comput. Inf. Technol. (ICCIT), Cox\'s Bazar, Bangladesh, Dec. 2025, pp. 1–6.',
        '[4] K. Kansal, T. B. Chandra, and A. Singh, "CXR-ViT: A transformer-based framework for enhanced chest X-ray classification," in Proc. IEEE Conf. Comput. Vision Pattern Recognit. Workshops, 2025, pp. 1–6.',
        '[5] S. Ghosh, A. Bandyopadhyay, M. Bose, and K. Santosh, "Vision Transformers excel in chest X-ray analysis," in Proc. IEEE EMBS Int. Conf. Biomed. Health Inform. (BHI), 2025, pp. 1–6.',
        '[6] M. S. M. Rafi, M. S. Ali, M. M. Hossen, S. Z. R. Sneha, and M. S. Arefin, "Chest X-ray analysis for pneumonia and tuberculosis detection using CNNs and Vision Transformers with Explainable AI," in Proc. IEEE 2nd Int. Conf. Comput. Appl. Syst. (COMPAS), Kushtia, Bangladesh, Oct. 2025, pp. 1–6.',
        '[7] K. Kim, Y. Choi, and J. Seo, "Explainable AI for multi-label chest X-ray diagnosis: Layer-wise Grad-CAM with hierarchical feature extraction," IEEE Access, vol. 13, pp. 1–10, 2025.',
        '[8] U. K. Dey, "CBAM-enhanced DenseNet121 for multi-class chest X-ray classification with Grad-CAM explainability," IEEE Access, vol. 13, pp. 1–11, 2025.',
        '[9] O. A. Naghash, N. Ling, and X. Li, "HyCoViT: Hybrid convolution Vision Transformer with dynamic dropout for enhanced medical chest X-ray classification," IEEE Access, vol. 13, pp. 112623–112638, July 2025.',
        '[10] M. Fu, C. Tantithamthavorn, and T. Le, "DAViT: A domain-adapted Vision Transformer for automated pneumonia detection and explanation using chest X-ray images," IEEE Access, vol. 13, pp. 1–12, 2025.',
        '[11] W. Vibina, D. P. Isravel, and J. P. M. Dhas, "Deep CNN-driven automated lung disease identification," in Proc. IEEE 3rd Int. Conf. Circuit Power Comput. Technol. (ICCPCT), 2025, pp. 1–6.',
        '[12] P. Pabba, U. Samudrala, V. S. S. Chevuturi, and S. Babu, "Explainable artificial intelligence for pulmonary disease detection from chest radiographs," in Proc. IEEE Int. Conf. Signal Process. Commun. Eng. Syst. (SPACES), 2025, pp. 1–6.',
        '[13] Y. Qi, R. Chen, and J. Cai, "Attention mechanisms in medical image analysis: Advance, challenge, and future direction," in Proc. IEEE 4th Int. Conf. Comput. Netw. Commun. Technol. (CNCT), 2025, pp. 1–6.',
        '[14] D. Edirisinghe, W. Nimalsiri, M. Hennayake, D. Meedeniya, and G. Lim, "Chest X-ray report generation using abnormality guided vision language model," IEEE Access, vol. 13, pp. 1–15, Sept. 2025.',
        '[15] P. Barra, A. Della Greca, I. Amaro, C. Marrocco, and M. Molinara, "A comparative analysis of XAI techniques for medical imaging: Challenges and opportunities," in Proc. IEEE Int. Conf. Bioinform. Biomed. (BIBM), Lisbon, Portugal, Dec. 2024, pp. 6782–6787.',
        '[16] N. Yaligar, S. G, K. Patil, P. Kulkarni, and V. K, "State-of-the-art deep learning strategies for multi-label classification of chest X-ray thoracic disorders," in Proc. IEEE 9th Int. Conf. Commun. Electron. Syst. (ICCES), Coimbatore, India, 2024, pp. 1–6.'
    ]

    for ref in references:
        p_ref = doc.add_paragraph()
        p_ref.paragraph_format.space_before = Pt(1)
        p_ref.paragraph_format.space_after = Pt(2)
        p_ref.paragraph_format.line_spacing = 1.0
        p_ref.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        r = p_ref.add_run(ref)
        r.font.name = 'Times New Roman'
        r.font.size = Pt(7.5)

    doc.save(output_filename)
    print(f"Successfully generated {output_filename}")

if __name__ == '__main__':
    create_ieee_literature_review_docx('docs/LITERATURE_REVIEW_IEEE_YEARS.docx')
    create_ieee_literature_review_docx('docs/LITERATURE_REVIEW_TWO_COLUMN.docx')
