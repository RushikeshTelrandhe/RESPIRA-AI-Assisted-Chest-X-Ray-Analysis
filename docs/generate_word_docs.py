"""
Generates high-quality Microsoft Word (.docx) files for the Literature Review and full paper.
"""
import os
import docx
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.oxml import parse_xml
from docx.oxml.ns import nsdecls

def create_literature_review_docx():
    doc = Document()

    # Standard 0.75-inch margins for table fit
    for section in doc.sections:
        section.top_margin = Inches(0.75)
        section.bottom_margin = Inches(0.75)
        section.left_margin = Inches(0.75)
        section.right_margin = Inches(0.75)

    # Base font: Times New Roman 11pt
    style = doc.styles['Normal']
    font = style.font
    font.name = 'Times New Roman'
    font.size = Pt(11)
    font.color.rgb = RGBColor(0, 0, 0)

    # Document Header / Section Title
    title = doc.add_paragraph()
    title.paragraph_format.space_before = Pt(0)
    title.paragraph_format.space_after = Pt(4)
    title.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = title.add_run('SECTION II: LITERATURE REVIEW')
    run.font.bold = True
    run.font.size = Pt(16)
    run.font.name = 'Times New Roman'

    sub = doc.add_paragraph()
    sub.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER
    sub.paragraph_format.space_after = Pt(16)
    r_sub = sub.add_run('RESPIRA: An Explainable and Uncertainty-Aware High-Resolution Hybrid Deep Learning Framework for Multi-Class Thoracic Disease Classification')
    r_sub.font.italic = True
    r_sub.font.size = Pt(10.5)

    # Introductory text
    p1 = doc.add_paragraph()
    p1.paragraph_format.space_after = Pt(10)
    p1.paragraph_format.line_spacing = 1.15
    p1.add_run(
        'The pre-existing literature studies summarized in Table I provide strong evidence for the '
        'rapid evolution of deep learning, Vision Transformers, and hybrid fusion architectures for '
        'pulmonary disease detection using chest radiographs (CXRs). The literature is organized in '
        'reverse chronological order from recent 2026 breakthroughs down to 2024 foundational benchmarks.'
    )

    # Table Title
    p_tab = doc.add_paragraph()
    p_tab.paragraph_format.space_before = Pt(10)
    p_tab.paragraph_format.space_after = Pt(6)
    p_tab.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r_tab = p_tab.add_run('TABLE I: Related Work on Pulmonary Disease Detection Using Deep Learning Techniques (2026 – 2024)')
    r_tab.font.bold = True
    r_tab.font.size = Pt(10.5)

    table_data = [
        ('Year', 'Author(s)', 'Title of the Paper', 'Methods Used', 'Key Findings / Assessment'),
        ('2026', 'Shukla et al. [1]', 'Chest X-Ray Classification using Deep Learning Framework with Image Enhancement', 'Input image contrast enhancement + CLAHE + deep CNN classifiers', 'Confirmed that input preprocessing and contrast equalization are fundamental prerequisites for robust model generalization across diverse datasets.'),
        ('2026', 'Altun & Candemir [2]', 'Differential Evolution-Based CNN-Transformer Fusion for Thoracic Disease Classification from Chest X-Rays', 'Metaheuristic Differential Evolution (DE) search for CNN-Transformer feature fusion', 'Combining CNN spatial representations with ViT global embeddings produced superior classification accuracy over single backbones.'),
        ('2026', 'Shi et al. [3]', 'Graph guided multiscale cross attention for multilabel chest X ray classification', 'ViT + DenseNet-121 dual visual encoder with GCN label co-occurrence reasoning', 'Proved that modeling inter-disease pathological dependencies alongside multiscale visual cross-attention significantly improves multi-label diagnosis.'),
        ('2026', 'Aperstein et al. [4]', 'Selective multipathology chest X-ray classification via rejection mechanisms', 'Multi-pathology classification combined with uncertainty-based rejection triage', 'Demonstrated that rejecting and routing ambiguous, high-uncertainty scans to senior radiologists drastically prevents clinical misdiagnosis.'),
        ('2026', 'Mažonienė & Šešok [5]', 'Explainable Multi-Label Chest X-Ray Disease Classification Using DualPool-DenseNet121 and Grad-CAM Visualization', 'Dual-pooling DenseNet-121 architecture with Grad-CAM visual heatmaps', 'Dual pooling preserved localized anomaly peaks while retaining global anatomical context across complex multi-label radiographs.'),
        ('2026', 'Ba et al. [6]', 'A monotonic multi-expert Vision Transformer for clinical chest X-ray diagnosis', 'Multi-expert Vision Transformer with monotonic attention modules', 'Multiple specialized self-attention experts improved discrimination of subtle radiographic patterns in ambiguous clinical cases.'),
        ('2025', 'Pandit et al. [7]', 'Explainable Deep Learning for Lung Disease Detection: A Comparative Study Using Grad-CAM', 'DenseNet-121, ResNet-50, VGG-16, InceptionV3, Xception + Grad-CAM', 'Pretrained CNNs yield high multi-class accuracy; visual explanations validated local lesion focus, but lacked global attention modeling.'),
        ('2025', 'Kansal et al. [8]', 'CXR-ViT: A Transformer-Based Framework for Enhanced Chest X-Ray Classification', 'Pure Vision Transformer (ViT) with patch-level self-attention', 'Proved self-attention effectively captures broad thoracic patterns, though requiring substantial training stability mechanisms.'),
        ('2025', 'Ghosh et al. [9]', 'Vision Transformers Excel in Chest X-Ray Analysis', 'ViT-B/16 benchmarked on large-scale NIH ChestX-ray14 dataset', 'Established that self-attention layers outperform standard CNNs on diffuse conditions like cardiomegaly and edema.'),
        ('2025', 'Rafi et al. [10]', 'Chest X-ray Analysis for Pneumonia and Tuberculosis Detection using CNNs and Vision Transformers with Explainable AI', 'Comparative CNN vs. ViT evaluation with gradient-based saliency mapping', 'ViT slightly surpassed CNN (90.12% vs. 89.61% accuracy), confirming the complementary nature of convolutional and transformer features.'),
        ('2025', 'Kim et al. [11]', 'Explainable AI for Multi-Label Chest X-ray Diagnosis: Layer-wise Grad-CAM with Hierarchical Feature Extraction', 'U-Net encoder-decoder concatenation with multi-layer Grad-CAM', 'Hierarchical multi-scale feature extraction across 14 observations on MIMIC-CXR significantly improved radiologist trust.'),
        ('2025', 'Dey [12]', 'CBAM-Enhanced DenseNet121 for Multi-Class Chest X-Ray Classification with Grad-CAM Explainability', 'DenseNet-121 integrated with CBAM (Channel & Spatial Attention)', 'Attention re-weighting boosted diagnostic sensitivity in distinguishing bacterial pneumonia from viral pneumonia in low-contrast zones.'),
        ('2025', 'Naghash et al. [13]', 'HyCoViT: Hybrid Convolution Vision Transformer With Dynamic Dropout for Enhanced Medical Chest X-Ray Classification', 'Hybrid CNN-ViT architecture with dynamic regularization dropout', 'Successfully bridged local convolutional feature extraction with global token context, mitigating overfitting in medical imaging.'),
        ('2024', 'Barra et al. [14]', 'A Comparative Analysis of XAI Techniques for Medical Imaging: Challenges and Opportunities', 'Comparative study of Grad-CAM, LIME, and SHAP on medical images', 'Grad-CAM provides intuitive and computationally efficient saliency maps; perturbation methods (LIME/SHAP) incur heavy compute overhead.'),
        ('2024', 'Yaligar et al. [15]', 'State-of-the-Art Deep Learning Strategies for Multi-Label Classification of Chest X-Ray Thoracic Disorders', 'Densely connected and deep residual CNN architectures', 'Achieved high diagnostic AUCs on multi-label CXR datasets; highlighted that pure CNNs struggle with long-range bilateral thoracic dependencies.'),
        ('2024', 'Xu & Rajpurkar [16]', 'Calibration and Uncertainty Estimation Challenges in Self-Supervised Chest X-ray Pathology Classification Models', 'CheXzero self-supervised model, Temperature Scaling, MC Dropout', 'Demonstrated severe miscalibration on external clinical datasets (ECE 0.12–0.41), proving uncertainty quantification is vital for clinical safety.')
    ]

    table = doc.add_table(rows=len(table_data), cols=5)
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = False

    col_widths = [Inches(0.65), Inches(1.35), Inches(1.85), Inches(1.55), Inches(1.60)]

    # Clean borders XML
    tblPr = table._tbl.tblPr
    borders_xml = (
        f'<w:tblBorders {nsdecls("w")}>\n'
        f'  <w:top w:val="single" w:sz="6" w:space="0" w:color="000000"/>\n'
        f'  <w:bottom w:val="single" w:sz="6" w:space="0" w:color="000000"/>\n'
        f'  <w:insideH w:val="single" w:sz="4" w:space="0" w:color="CCCCCC"/>\n'
        f'  <w:insideV w:val="none"/>\n'
        f'  <w:left w:val="none"/>\n'
        f'  <w:right w:val="none"/>\n'
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

            # Header shading
            if row_idx == 0:
                tcPr = cell._tc.get_or_add_tcPr()
                shd = parse_xml(f'<w:shd {nsdecls("w")} w:fill="F2F4F7"/>')
                tcPr.append(shd)

            p = cell.paragraphs[0]
            p.paragraph_format.space_before = Pt(3)
            p.paragraph_format.space_after = Pt(3)
            p.paragraph_format.line_spacing = 1.05

            if col_idx == 0:
                p.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER
            else:
                p.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.LEFT

            r = p.add_run(text)
            r.font.name = 'Times New Roman'
            r.font.size = Pt(9.5)
            if row_idx == 0 or (col_idx == 0 and row_idx > 0):
                r.font.bold = True

    doc.add_paragraph().paragraph_format.space_after = Pt(10)

    # Narrative Synthesis
    p_nar1 = doc.add_paragraph()
    p_nar1.paragraph_format.space_after = Pt(8)
    p_nar1.paragraph_format.line_spacing = 1.15
    p_nar1.add_run(
        'Most recently in 2026, researchers have increasingly prioritized holistic frameworks that combine '
        'advanced image preprocessing, hybrid architectures, inter-disease relational dependencies, and '
        'selective classification triage. Shukla et al. [1] demonstrated that high-fidelity image enhancement '
        'and CLAHE preprocessing directly dictate downstream classification performance. To exploit '
        'complementary representations, Altun and Candemir [2] established that combining CNN spatial feature '
        'maps with ViT embeddings via differential evolution metaheuristics yields superior diagnostic power over '
        'standalone backbones. Concurrently, Shi et al. [3] showed that coupling dual CNN-ViT visual encoders with '
        'graph-guided label reasoning resolves clinical co-occurrence dependencies between thoracic abnormalities. '
        'Addressing clinical safety, Aperstein et al. [4] formulated automated rejection mechanisms to route '
        'high-uncertainty scans to senior radiologists, while Mažonienė and Šešok [5] combined dual-pooling '
        'mechanisms with Grad-CAM visualization. Furthermore, Ba et al. [6] proposed multi-expert Vision '
        'Transformers to capture subtle radiographic variations.'
    )

    p_nar2 = doc.add_paragraph()
    p_nar2.paragraph_format.space_after = Pt(8)
    p_nar2.paragraph_format.line_spacing = 1.15
    p_nar2.add_run(
        'Throughout 2025, the research community focused heavily on evaluating Vision Transformers alongside '
        'convolutional models. Pure transformer models such as CXR-ViT by Kansal et al. [8] and large-scale '
        'benchmarks by Ghosh et al. [9] proved that self-attention excels at capturing diffuse thoracic patterns '
        '(such as cardiomegaly and edema) that local convolutions struggle to aggregate. To combine local and '
        'global inductive biases, Rafi et al. [10] and Naghash et al. [13] (HyCoViT) demonstrated that hybrid '
        'CNN-ViT architectures mitigate overfitting in data-constrained medical regimes. Furthermore, '
        'attention-enhanced CNNs like CBAM-DenseNet121 [12] and explainability frameworks like layer-wise Grad-CAM '
        '[7], [11] enhanced localized sensitivity and clinician trust.'
    )

    p_nar3 = doc.add_paragraph()
    p_nar3.paragraph_format.space_after = Pt(8)
    p_nar3.paragraph_format.line_spacing = 1.15
    p_nar3.add_run(
        'Earlier foundational studies in 2024 established critical benchmarks and architectural trade-offs. '
        'Yaligar et al. [15] confirmed that while deep residual and dense CNNs achieve high diagnostic AUCs, '
        'pure CNNs struggle to model bilateral thoracic symmetry. In parallel, Barra et al. [14] conducted '
        'comparative evaluations of XAI techniques, identifying Grad-CAM as the most computationally viable '
        'method for clinical deployment compared to compute-heavy perturbation tools like LIME and SHAP. '
        'Crucially, Xu and Rajpurkar [16] revealed that high-performing chest X-ray models exhibit severe '
        'miscalibration on external clinical cohorts (ECE 0.12–0.41), establishing that uncalibrated model '
        'outputs present grave clinical risks.'
    )

    # Research Gaps
    p_gap_h = doc.add_paragraph()
    p_gap_h.paragraph_format.space_before = Pt(8)
    p_gap_h.paragraph_format.space_after = Pt(4)
    r_gap_h = p_gap_h.add_run('Identified Research Gaps in the Pre-Existing Literature:')
    r_gap_h.font.bold = True

    gaps = [
        'Resolution and Inductive Bias Disconnect: Most transformer pipelines downsample radiographs to low resolutions (224 × 224), discarding subtle apical lesions and fine reticular consolidations.',
        'Heuristic Fusion vs. Relational Conditioning: Fusion strategies predominantly rely on unweighted concatenation without explicitly modeling inter-disease relational dependencies.',
        'Lack of Dual-Modality Quantitative XAI Auditing: Visual explanations are rarely audited quantitatively across paired CNN and ViT modalities on identical test cohorts.',
        'Absence of Calibrated Uncertainty in Diagnostic Workflows: Deep learning models routinely lack calibrated uncertainty estimation to flag borderline predictions for clinical risk triage.',
        'Separation of ML Models from Deployable Clinical Software: Research models are frequently evaluated in isolation as research scripts, lacking authenticated, end-to-end integration into clinical backends with automated diagnostic reporting.'
    ]

    for g in gaps:
        p_g = doc.add_paragraph(style='List Bullet')
        p_g.paragraph_format.space_after = Pt(3)
        p_g.paragraph_format.line_spacing = 1.15
        colon_idx = g.find(':')
        if colon_idx != -1:
            r_b = p_g.add_run(g[:colon_idx+1])
            r_b.font.bold = True
            p_g.add_run(g[colon_idx+1:])
        else:
            p_g.add_run(g)

    # How RESPIRA Resolves Gaps
    p_resp_h = doc.add_paragraph()
    p_resp_h.paragraph_format.space_before = Pt(10)
    p_resp_h.paragraph_format.space_after = Pt(4)
    r_resp_h = p_resp_h.add_run('How the Proposed RESPIRA Framework Resolves These Gaps:')
    r_resp_h.font.bold = True

    resp_items = [
        'Deterministic Preprocessing Pipeline: Incorporates Contrast Limited Adaptive Histogram Equalization (CLAHE) alongside Otsu morphological lung field masking and anatomical fallback heuristics.',
        'Dual-Backbone Hybrid Architecture: Couples EfficientNet-B0 (localized textural inductive bias) with Vision Transformer ViT-B/16 (global thoracic context).',
        'Dual Multi-Modal Fusion Engines: High-resolution (512 × 512) weighted probability fusion alongside an advanced 5-stage latent feature fusion pipeline featuring bidirectional cross-attention, 6 disease-conditioned query embeddings, and inter-disease relational self-attention modeling.',
        'Calibrated Uncertainty Quantification: Normalized binary entropy and decision margin uncertainty combined into a 3-tier clinical risk triage stratification protocol (Low Risk, Moderate Review, High Risk Audit).',
        'Paired Dual-Modality Explainability Audit: Systematic quantitative concordance audit comparing Grad-CAM and ViT Cross-Attention Mapping across 1,947 held-out test radiographs.',
        'Full Clinical Platform Implementation: Authenticated asynchronous FastAPI backend, SQLAlchemy ORM with row-level physician ownership validation, automated ReportLab PDF diagnostic reporting, and a React 19 / Electron desktop suite.'
    ]

    for r in resp_items:
        p_r = doc.add_paragraph(style='List Bullet')
        p_r.paragraph_format.space_after = Pt(3)
        p_r.paragraph_format.line_spacing = 1.15
        colon_idx = r.find(':')
        if colon_idx != -1:
            r_b = p_r.add_run(r[:colon_idx+1])
            r_b.font.bold = True
            p_r.add_run(r[colon_idx+1:])
        else:
            p_r.add_run(r)

    # References
    p_ref_h = doc.add_paragraph()
    p_ref_h.paragraph_format.space_before = Pt(14)
    p_ref_h.paragraph_format.space_after = Pt(6)
    r_ref_h = p_ref_h.add_run('REFERENCES')
    r_ref_h.font.bold = True
    r_ref_h.font.size = Pt(12)

    refs = [
        '[1] P. Shukla, M. Raj, D. Joshi, and S. M. B S, "Chest X-ray classification using deep learning framework with image enhancement," in Proc. IEEE Int. Conf., 2026, pp. 1–6.',
        '[2] S. Altun and S. Candemir, "Differential evolution-based CNN-transformer fusion for thoracic disease classification from chest X-rays," IEEE Access, vol. 14, pp. 1–14, 2026.',
        '[3] G. Shi, Z. Wang, Y. Shi, J. Pan, L. Sun, F. Fang, and L. Jin, "Graph guided multiscale cross attention for multilabel chest X ray classification," Sci. Rep., vol. 16, Art. no. 22761, 2026.',
        '[4] Y. Aperstein, A. Tzahar, A. Gottlib, T. Verber, R. S. Damti, and A. Apartsin, "Selective multipathology chest X-ray classification via rejection mechanisms," Sci. Rep., 2026, doi: 10.1038/s41598-026-66294-7.',
        '[5] E. Mažonienė and D. Šešok, "Explainable multi-label chest X-ray disease classification using DualPool-DenseNet121 and Grad-CAM visualization," Symmetry, vol. 18, no. 7, Art. no. 1294, 2026.',
        '[6] T. Ba, C. Bi, J. Yu, M. Z. Jhandir, H. M. Raza ur Rehman, and G. S. Choi, "A monotonic multi-expert Vision Transformer for clinical chest X-ray diagnosis," Front. Med., vol. 13, Art. no. 1847086, 2026.',
        '[7] S. K. Pandit, S. S. Arjon, and T. Yasmin, "Explainable deep learning for lung disease detection: A comparative study using Grad-CAM," in Proc. 28th Int. Conf. Comput. Inf. Technol. (ICCIT), Cox\'s Bazar, Bangladesh, Dec. 2025, pp. 1–6.',
        '[8] K. Kansal, T. B. Chandra, and A. Singh, "CXR-ViT: A transformer-based framework for enhanced chest X-ray classification," in Proc. IEEE Conf. Comput. Vision Pattern Recognit. Workshops, 2025, pp. 1–6.',
        '[9] S. Ghosh, A. Bandyopadhyay, M. Bose, and K. C. Santosh, "Vision Transformers excel in chest X-ray analysis," in Proc. IEEE Conf. Artif. Intell. (CAI), Santa Clara, CA, USA, 2025, pp. 1–6.',
        '[10] M. S. M. Rafi, M. S. Ali, M. M. Hossen, S. Z. R. Sneha, and M. S. Arefin, "Chest X-ray analysis for pneumonia and tuberculosis detection using CNNs and Vision Transformers with explainable AI," in Proc. IEEE 2nd Int. Conf. Comput., Appl. Syst. (COMPAS), Kushtia, Bangladesh, Oct. 2025, pp. 1–6.',
        '[11] K. Kim, Y. Choi, and J. Seo, "Explainable AI for multi-label chest X-ray diagnosis: Layer-wise Grad-CAM with hierarchical feature extraction," IEEE Access, 2025.',
        '[12] U. K. Dey, "CBAM-Enhanced DenseNet121 for multi-class chest X-ray classification with Grad-CAM explainability," IEEE Access, vol. 13, pp. 1–14, 2025.',
        '[13] O. A. Naghash, N. Ling, and X. Li, "HyCoViT: Hybrid convolution vision transformer with dynamic dropout for enhanced medical chest X-ray classification," IEEE Access, vol. 13, pp. 1–13, 2025.',
        '[14] P. Barra, A. Della Greca, I. Amaro, and A. Tortora, "A comparative analysis of XAI techniques for medical imaging: Challenges and opportunities," in Proc. IEEE Int. Conf. Bioinf. Biomed. (BIBM), Lisbon, Portugal, Dec. 2024, pp. 6782–6787.',
        '[15] N. Yaligar, S. G, K. Patil, P. Kulkarni, and P. Bhajantri, "State-of-the-art deep learning strategies for multi-label classification of chest X-ray thoracic disorders," in Proc. IEEE Int. Conf., 2024, pp. 1–6.',
        '[16] J. Xu and P. Rajpurkar, "Calibration and uncertainty estimation challenges in self-supervised chest X-ray pathology classification models," in Proc. Med. Imaging Deep Learn. (MIDL), Paris, France, Jul. 2024, pp. 1–6.'
    ]

    for ref in refs:
        p_ref = doc.add_paragraph()
        p_ref.paragraph_format.space_after = Pt(4)
        p_ref.paragraph_format.line_spacing = 1.15
        p_ref.paragraph_format.left_indent = Inches(0.3)
        p_ref.paragraph_format.first_line_indent = Inches(-0.3)
        r = p_ref.add_run(ref)
        r.font.name = 'Times New Roman'
        r.font.size = Pt(10)

    out_path = os.path.join('docs', 'LITERATURE_REVIEW.docx')
    doc.save(out_path)
    print(f'Successfully generated: {out_path}')

if __name__ == '__main__':
    create_literature_review_docx()
