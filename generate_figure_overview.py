import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
import os

def create_overview_figure(save_dirs=['figures', 'scratch']):
    # Publication-grade monochrome / grayscale setup
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['DejaVu Sans', 'Arial', 'Helvetica'],
        'font.size': 10,
        'figure.dpi': 300,
        'savefig.dpi': 300
    })

    fig = plt.figure(figsize=(16.8, 6.8), facecolor='white')
    ax = fig.add_subplot(111)
    ax.set_xlim(0, 16.8)
    ax.set_ylim(0, 6.8)
    ax.axis('off')

    # Monochrome / Grayscale Scientific Palette (Slate tones)
    c_black = '#0F172A'       # Slate 900
    c_dark_gray = '#334155'   # Slate 700
    c_mid_gray = '#64748B'    # Slate 500
    c_light_gray = '#CBD5E1'  # Slate 300
    c_border = '#94A3B8'      # Slate 400
    c_bg_panel = '#F8FAFC'    # Slate 50
    c_bg_card = '#F1F5F9'     # Slate 100
    c_white = '#FFFFFF'

    def draw_card(x, y, w, h, bg_color=c_white, border_color=c_border, border_w=1.2, radius=0.08, zorder=2):
        box = FancyBboxPatch((x, y), w, h,
                             boxstyle=f"round,pad=0.0,rounding_size={radius}",
                             facecolor=bg_color, edgecolor=border_color,
                             linewidth=border_w, zorder=zorder)
        ax.add_patch(box)

    def draw_badge(x, y, w, h, text, bg_color=c_black, text_color=c_white, fontsize=8.2, zorder=5):
        box = FancyBboxPatch((x, y), w, h,
                             boxstyle="round,pad=0.0,rounding_size=0.06",
                             facecolor=bg_color, edgecolor='none', zorder=zorder)
        ax.add_patch(box)
        ax.text(x + w/2, y + h/2, text, color=text_color, fontsize=fontsize, fontweight='bold',
                ha='center', va='center', zorder=zorder+1)

    def draw_arrow(x1, y1, x2, y2, color=c_black, lw=1.6, zorder=5):
        arrow = FancyArrowPatch((x1, y1), (x2, y2),
                                arrowstyle='->', color=color,
                                mutation_scale=15, linewidth=lw, zorder=zorder)
        ax.add_patch(arrow)

    def draw_magnifier(cx, cy, r=0.13, zorder=7):
        lens = Circle((cx, cy), r, edgecolor=c_black, facecolor='white', linewidth=1.6, zorder=zorder)
        ax.add_patch(lens)
        ax.plot([cx + r*0.7, cx + r*1.6], [cy - r*0.7, cy - r*1.6],
                color=c_black, linewidth=2.4, solid_capstyle='round', zorder=zorder)

    # =========================================================================
    # MASTER SCIENTIFIC HEADER
    # =========================================================================
    draw_card(0.4, 6.10, 16.0, 0.48, bg_color=c_black, border_color=c_black, border_w=1.0, radius=0.06)
    ax.text(8.4, 6.34, "EXPLAINABLE SYSTEM 1 CLINICAL TRIAGE",
            color=c_white, fontsize=10.2, fontweight='bold', ha='center', va='center', zorder=5)

    # =========================================================================
    # STAGE 1 (LEFT): RAW PATIENT NARRATIVE (SHOWN NORMALLY, UNCLUTTERED)
    # =========================================================================
    p1_x, p1_y, p1_w, p1_h = 0.50, 0.45, 4.50, 5.40
    draw_card(p1_x, p1_y, p1_w, p1_h, bg_color=c_bg_panel, border_color=c_dark_gray, border_w=1.4, radius=0.12)
    draw_badge(p1_x + 0.25, p1_y + p1_h - 0.46, p1_w - 0.50, 0.36,
               "1. Raw Patient Narrative", bg_color=c_dark_gray, text_color=c_white, fontsize=8.6)

    ax.text(p1_x + p1_w/2, p1_y + p1_h - 0.72, "Colloquial First-Person Medical Complaint",
            fontsize=7.4, color=c_mid_gray, ha='center', zorder=4)

    # Clinical Record Document Card
    doc_x, doc_y, doc_w, doc_h = 0.70, 0.75, 4.10, 3.95
    draw_card(doc_x, doc_y, doc_w, doc_h, bg_color=c_white, border_color=c_border, border_w=1.2, radius=0.08)

    # Document Header Strip
    draw_card(doc_x, doc_y + doc_h - 0.44, doc_w, 0.44, bg_color=c_bg_card, border_color=c_border, border_w=1.0, radius=0.06)
    ax.text(doc_x + 0.20, doc_y + doc_h - 0.22, "PATIENT CLINICAL RECORD",
            fontsize=7.4, fontweight='bold', color=c_dark_gray, va='center', zorder=4)
    ax.text(doc_x + doc_w - 0.20, doc_y + doc_h - 0.22, "Symptom2Disease Intake",
            fontsize=6.8, color=c_mid_gray, ha='right', va='center', zorder=4)

    # Chief Complaint Header
    ax.text(doc_x + 0.25, doc_y + doc_h - 0.65, "Chief Complaint Narrative:",
            fontsize=7.8, fontweight='bold', color=c_black, va='top', zorder=4)

    # Raw Text formatted completely normally as plain, unstyled conversational clinical intake
    raw_text = (
        "\"I have an itchy red rash on my arms with\n"
        "dry silvery scales peeling off. My skin is\n"
        "cracked and bleeds occasionally.\""
    )
    ax.text(doc_x + 0.25, doc_y + doc_h - 1.05, raw_text,
            fontsize=9.4, color=c_dark_gray, va='top', linespacing=1.85, zorder=4)

    # Clean document note at bottom
    draw_card(doc_x + 0.22, doc_y + 0.35, doc_w - 0.44, 0.75,
              bg_color=c_bg_panel, border_color=c_light_gray, border_w=0.8, radius=0.05)
    ax.text(doc_x + doc_w/2, doc_y + 0.85, "Unprocessed Conversational Text",
            fontsize=7.3, fontweight='bold', color=c_dark_gray, ha='center', va='center', zorder=4)
    ax.text(doc_x + doc_w/2, doc_y + 0.58, "Direct input to non-autoregressive triage pipeline",
            fontsize=6.8, color=c_mid_gray, ha='center', va='center', zorder=4)

    # Arrow from Stage 1 to Stage 2 (Gap: 5.00 to 5.90)
    draw_arrow(5.15, 3.00, 5.75, 3.00, color=c_black, lw=1.8)
    ax.text(5.45, 3.24, "Triage", fontsize=7.4, fontweight='bold', color=c_black, ha='center', zorder=5)

    # =========================================================================
    # STAGE 2 (MIDDLE): EXPLAINABLE SYSTEM 1 MODEL ARCHITECTURE
    # =========================================================================
    p2_x, p2_y, p2_w, p2_h = 5.90, 0.45, 5.00, 5.40
    draw_card(p2_x, p2_y, p2_w, p2_h, bg_color=c_bg_panel, border_color=c_dark_gray, border_w=1.4, radius=0.12)
    draw_badge(p2_x + 0.25, p2_y + p2_h - 0.46, p2_w - 0.50, 0.36,
               "2. Explainable System 1 Model Architecture", bg_color=c_dark_gray, text_color=c_white, fontsize=8.6)

    ax.text(p2_x + p2_w/2, p2_y + p2_h - 0.72, "Direct Non-Autoregressive Confidence Scoring",
            fontsize=7.4, color=c_mid_gray, ha='center', zorder=4)

    # Architecture Pipeline Diagram
    arch_x, arch_y, arch_w, arch_h = 6.10, 3.85, 4.60, 0.95
    draw_card(arch_x, arch_y, arch_w, arch_h, bg_color=c_white, border_color=c_border, border_w=1.1, radius=0.08)

    # Network stages
    draw_badge(arch_x + 0.15, arch_y + 0.46, 1.30, 0.38, "Encoder Backbone", bg_color=c_bg_card, text_color=c_black, fontsize=6.8)
    draw_arrow(arch_x + 1.50, arch_y + 0.65, arch_x + 1.75, arch_y + 0.65, color=c_dark_gray, lw=1.2)

    draw_badge(arch_x + 1.80, arch_y + 0.46, 1.15, 0.38, "Pooling Head", bg_color=c_bg_card, text_color=c_black, fontsize=6.8)
    draw_arrow(arch_x + 3.00, arch_y + 0.65, arch_x + 3.25, arch_y + 0.65, color=c_dark_gray, lw=1.2)

    draw_badge(arch_x + 3.30, arch_y + 0.46, 1.15, 0.38, "Linear Classifier", bg_color=c_bg_card, text_color=c_black, fontsize=6.8)

    ax.text(arch_x + arch_w/2, arch_y + 0.20, "Single forward pass  |  <80 ms Latency  |  Zero Token Generation",
            fontsize=6.8, fontweight='bold', color=c_dark_gray, ha='center', zorder=4)

    draw_arrow(p2_x + p2_w/2, 3.85, p2_x + p2_w/2, 3.55, color=c_black, lw=1.3)

    # Horizontal Probability Bar Chart
    chart_x, chart_y, chart_w, chart_h = 6.10, 0.75, 4.60, 2.75
    draw_card(chart_x, chart_y, chart_w, chart_h, bg_color=c_white, border_color=c_border, border_w=1.2, radius=0.08)

    ax.text(chart_x + 0.20, chart_y + chart_h - 0.26, "Disease Confidence Distribution (N = 24):",
            fontsize=7.4, fontweight='bold', color=c_black, zorder=4)

    diseases = [
        ("Psoriasis", "99.88%", 0.9988, True),
        ("Impetigo (Differential)", "0.10%", 0.05, False),
        ("Fungal Infection", "0.01%", 0.02, False),
        ("Acne / Dermatitis", "0.01%", 0.02, False),
        ("Bronchial Asthma", "< 0.01%", 0.01, False),
        ("Remaining 19 Conditions", "< 0.01%", 0.01, False),
    ]

    by = chart_y + chart_h - 0.60
    max_bar_w = 1.90
    bar_start_x = chart_x + 1.62

    for d_name, d_prob, d_ratio, is_target in diseases:
        lbl_weight = 'bold' if is_target else 'normal'
        lbl_color = c_black if is_target else c_dark_gray
        ax.text(bar_start_x - 0.08, by + 0.07, d_name,
                fontsize=6.8, fontweight=lbl_weight, color=lbl_color, ha='right', va='center', zorder=4)

        # Bar Background Track
        track = FancyBboxPatch((bar_start_x, by), max_bar_w, 0.19,
                               boxstyle="round,pad=0.0,rounding_size=0.03",
                               facecolor=c_bg_card, edgecolor=c_light_gray, linewidth=0.6, zorder=3)
        ax.add_patch(track)

        # Actual Probability Bar
        fill_w = max(max_bar_w * d_ratio, 0.06 if not is_target else max_bar_w * d_ratio)
        bar_color = c_black if is_target else c_mid_gray
        bar = FancyBboxPatch((bar_start_x, by), fill_w, 0.19,
                             boxstyle="round,pad=0.0,rounding_size=0.03",
                             facecolor=bar_color, edgecolor='none', zorder=4)
        ax.add_patch(bar)

        # Percentage Text - strictly contained within the chart card
        if is_target:
            ax.text(bar_start_x + fill_w - 0.08, by + 0.09, f"{d_prob} [Predicted]",
                    fontsize=6.6, fontweight='bold', color=c_white, ha='right', va='center', zorder=5)
        else:
            ax.text(bar_start_x + max_bar_w + 0.08, by + 0.09, d_prob,
                    fontsize=6.6, color=c_mid_gray, ha='left', va='center', zorder=4)

        by -= 0.31

    # Bottom Diagnosis Box
    draw_card(chart_x + 0.20, chart_y + 0.14, chart_w - 0.40, 0.32,
              bg_color=c_bg_panel, border_color=c_dark_gray, border_w=0.9, radius=0.04)
    ax.text(chart_x + chart_w/2, chart_y + 0.30, "Primary Diagnosis: Psoriasis (Confidence: 99.88%)",
            fontsize=7.1, fontweight='bold', color=c_black, ha='center', va='center', zorder=4)

    # Arrow from Stage 2 to Stage 3 (Gap: 10.90 to 11.80, width 0.90)
    draw_arrow(11.05, 3.00, 11.65, 3.00, color=c_black, lw=1.8)
    ax.text(11.35, 3.24, "Attribution", fontsize=7.4, fontweight='bold', color=c_black, ha='center', zorder=5)

    # =========================================================================
    # STAGE 3 (RIGHT): FEATURE ATTRIBUTION (ZOOM INTO TEXT, PERFECT SPACING)
    # =========================================================================
    p3_x, p3_y, p3_w, p3_h = 11.80, 0.45, 4.50, 5.40
    draw_card(p3_x, p3_y, p3_w, p3_h, bg_color=c_bg_panel, border_color=c_dark_gray, border_w=1.4, radius=0.12)
    draw_badge(p3_x + 0.25, p3_y + p3_h - 0.46, p3_w - 0.50, 0.36,
               "3. Feature Attribution: Zoom on Text", bg_color=c_dark_gray, text_color=c_white, fontsize=8.6)

    ax.text(p3_x + p3_w/2, p3_y + p3_h - 0.72, "Isolating Key Contributing Clinical Words",
            fontsize=7.4, color=c_mid_gray, ha='center', zorder=4)

    # Magnified Feature Zoom Card
    zoom_x, zoom_y, zoom_w, zoom_h = 12.00, 3.55, 4.10, 1.35
    draw_card(zoom_x, zoom_y, zoom_w, zoom_h, bg_color=c_white, border_color=c_black, border_w=1.5, radius=0.10)

    # Header with Magnifier icon
    draw_magnifier(zoom_x + 0.35, zoom_y + zoom_h - 0.28, r=0.11)
    ax.text(zoom_x + 0.65, zoom_y + zoom_h - 0.28, "Zoom: Salient Diagnostic Feature",
            fontsize=7.3, fontweight='bold', color=c_black, va='center', zorder=4)

    # Magnified Phrase Box
    draw_card(zoom_x + 0.20, zoom_y + 0.16, zoom_w - 0.40, 0.72,
              bg_color=c_bg_card, border_color=c_dark_gray, border_w=1.2, radius=0.06)
    ax.text(zoom_x + zoom_w/2, zoom_y + 0.60, "\"dry silvery scales peeling off\"",
            fontsize=9.2, fontweight='bold', color=c_black, ha='center', va='center', zorder=5)
    ax.text(zoom_x + zoom_w/2, zoom_y + 0.34, "Cardinal Hallmark Symptom of Psoriasis",
            fontsize=7.2, fontweight='bold', color=c_dark_gray, ha='center', va='center', zorder=5)

    # Word-Level Saliency Bar Chart
    sal_x, sal_y, sal_w, sal_h = 12.00, 0.75, 4.10, 2.65
    draw_card(sal_x, sal_y, sal_w, sal_h, bg_color=c_white, border_color=c_border, border_w=1.2, radius=0.08)

    ax.text(sal_x + 0.20, sal_y + sal_h - 0.26, "Word-Level Attribution Saliency:",
            fontsize=7.4, fontweight='bold', color=c_black, zorder=4)

    tokens = [
        ("dry", 0.72),
        ("silvery", 0.94),
        ("scales", 0.98),
        ("peeling", 0.80),
        ("off", 0.30)
    ]

    col_w = 0.54
    spacing = 0.18
    start_tx = sal_x + 0.36
    base_ty = sal_y + 0.95
    max_h = 0.92

    for tok_word, score in tokens:
        bar_h = max_h * score
        if score > 0.90:
            tok_color = c_black
        elif score > 0.70:
            tok_color = c_dark_gray
        else:
            tok_color = c_light_gray

        # Saliency Bar
        s_bar = FancyBboxPatch((start_tx, base_ty), col_w, bar_h,
                               boxstyle="round,pad=0.0,rounding_size=0.03",
                               facecolor=tok_color, edgecolor='none', zorder=4)
        ax.add_patch(s_bar)

        # Score text above bar
        ax.text(start_tx + col_w/2, base_ty + bar_h + 0.06, f"{score:.2f}",
                fontsize=6.6, fontweight='bold', color=c_black, ha='center', va='bottom', zorder=4)

        # Token box below
        draw_card(start_tx - 0.02, base_ty - 0.30, col_w + 0.04, 0.24,
                  bg_color=c_bg_card, border_color=c_border, border_w=0.8, radius=0.03)
        ax.text(start_tx + col_w/2, base_ty - 0.18, tok_word,
                fontsize=6.9, fontweight='bold', color=c_black, ha='center', va='center', zorder=4)

        start_tx += col_w + spacing

    # Diagnostic Conclusion Note
    draw_card(sal_x + 0.20, sal_y + 0.12, sal_w - 0.40, 0.42,
              bg_color=c_bg_panel, border_color=c_border, border_w=0.8, radius=0.04)
    ax.text(sal_x + sal_w/2, sal_y + 0.35, "Decisive Diagnostic Drivers:",
            fontsize=6.8, fontweight='bold', color=c_dark_gray, ha='center', va='center', zorder=4)
    ax.text(sal_x + sal_w/2, sal_y + 0.20, "\"scales\" & \"silvery\" definitively establish Psoriasis",
            fontsize=6.7, style='italic', color=c_black, ha='center', va='center', zorder=4)

    # Save
    for sdir in save_dirs:
        abs_sdir = os.path.abspath(sdir)
        os.makedirs(abs_sdir, exist_ok=True)
        pdf_path = os.path.join(abs_sdir, 'figure_overview.pdf')
        png_path = os.path.join(abs_sdir, 'figure_overview.png')
        if os.path.exists(pdf_path):
            try: os.remove(pdf_path)
            except Exception: pass
        if os.path.exists(png_path):
            try: os.remove(png_path)
            except Exception: pass
        plt.savefig(pdf_path, bbox_inches='tight', dpi=300)
        plt.savefig(png_path, bbox_inches='tight', dpi=300)
        print(f"Saved: {pdf_path} and {png_path}")

    plt.close()

if __name__ == '__main__':
    create_overview_figure()
    print("Symmetric, perfectly aligned overview figure generated successfully!")
