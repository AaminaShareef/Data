"""
report_generator.py
────────────────────────────────────────────────────────────────────────────
Auralis — PDF Report Generator
Builds a styled multi-page PDF from AnalysisResult data.
Uses ReportLab (layout) + Matplotlib (charts).

Usage:
    from kpi_engine.services.report_generator import generate_pdf_report
    pdf_bytes = generate_pdf_report(payload)
"""

import io
import math
import base64
from datetime import datetime

import matplotlib
matplotlib.use('Agg')          # headless — no display needed
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

from reportlab.lib.pagesizes  import A4
from reportlab.lib.units       import mm, cm
from reportlab.lib             import colors
from reportlab.lib.styles      import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums       import TA_LEFT, TA_CENTER, TA_RIGHT
from reportlab.platypus        import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    Image, PageBreak, HRFlowable, KeepTogether,
)
from reportlab.platypus.flowables import Flowable

# ── Auralis colour palette ────────────────────────────────────────────────────
NAVY        = colors.HexColor('#0B3B4B')
NAVY_MID    = colors.HexColor('#1D5A6D')
NAVY_LIGHT  = colors.HexColor('#2C6F84')
TEAL        = colors.HexColor('#4A9B8D')
TEAL_LIGHT  = colors.HexColor('#7BBDB3')
TEAL_PALE   = colors.HexColor('#E0F2F0')
NAVY_PALE   = colors.HexColor('#E3F0F5')
BG_GREY     = colors.HexColor('#F4F7FA')
TEXT        = colors.HexColor('#1A2B3C')
TEXT_SOFT   = colors.HexColor('#3A4A5E')
MUTED       = colors.HexColor('#6B7A90')
RED         = colors.HexColor('#DC2626')
AMBER       = colors.HexColor('#D97706')
GREEN       = colors.HexColor('#059669')
WHITE       = colors.white
BLACK       = colors.black

# Matplotlib palette
MPL_COLORS = ['#0B3B4B', '#1D5A6D', '#4A9B8D', '#7BBDB3',
              '#2C6F84', '#B2DFDB', '#E3F0F5', '#0d5266']

PAGE_W, PAGE_H = A4
MARGIN = 20 * mm


# ── Custom flowable: coloured rule ───────────────────────────────────────────
class ColorRule(Flowable):
    def __init__(self, width, height=2, color=TEAL):
        super().__init__()
        self.width  = width
        self.height = height
        self.color  = color

    def draw(self):
        self.canv.setFillColor(self.color)
        self.canv.rect(0, 0, self.width, self.height, fill=1, stroke=0)


# ── Style factory ─────────────────────────────────────────────────────────────
def _styles():
    base = getSampleStyleSheet()

    def ps(name, **kw):
        return ParagraphStyle(name, **kw)

    return {
        'cover_title': ps('cover_title',
            fontName='Helvetica-Bold', fontSize=28, textColor=WHITE,
            leading=34, alignment=TA_LEFT, spaceAfter=6),

        'cover_sub': ps('cover_sub',
            fontName='Helvetica', fontSize=13, textColor=colors.HexColor('#B2DFDB'),
            leading=18, alignment=TA_LEFT, spaceAfter=4),

        'cover_meta': ps('cover_meta',
            fontName='Helvetica', fontSize=10, textColor=colors.HexColor('#9ECDD8'),
            leading=14, alignment=TA_LEFT),

        'section_heading': ps('section_heading',
            fontName='Helvetica-Bold', fontSize=14, textColor=NAVY,
            leading=18, spaceBefore=14, spaceAfter=6),

        'subsection': ps('subsection',
            fontName='Helvetica-Bold', fontSize=11, textColor=NAVY_MID,
            leading=14, spaceBefore=8, spaceAfter=4),

        'body': ps('body',
            fontName='Helvetica', fontSize=9, textColor=TEXT_SOFT,
            leading=14, spaceAfter=4),

        'body_italic': ps('body_italic',
            fontName='Helvetica-Oblique', fontSize=10, textColor=TEXT_SOFT,
            leading=15, spaceAfter=6),

        'caption': ps('caption',
            fontName='Helvetica', fontSize=8, textColor=MUTED,
            leading=11, alignment=TA_CENTER, spaceAfter=8),

        'kpi_name': ps('kpi_name',
            fontName='Helvetica-Bold', fontSize=9, textColor=NAVY,
            leading=12),

        'kpi_val': ps('kpi_val',
            fontName='Helvetica-Bold', fontSize=13, textColor=NAVY_MID,
            leading=16),

        'insight_title': ps('insight_title',
            fontName='Helvetica-Bold', fontSize=10, textColor=NAVY,
            leading=13, spaceBefore=6, spaceAfter=2),

        'insight_body': ps('insight_body',
            fontName='Helvetica', fontSize=9, textColor=TEXT_SOFT,
            leading=13, spaceAfter=4),

        'filter_pill': ps('filter_pill',
            fontName='Helvetica-Bold', fontSize=8, textColor=TEAL,
            leading=11),

        'footer': ps('footer',
            fontName='Helvetica', fontSize=7, textColor=MUTED,
            leading=10, alignment=TA_CENTER),

        'table_header': ps('table_header',
            fontName='Helvetica-Bold', fontSize=8, textColor=WHITE,
            leading=11),

        'table_cell': ps('table_cell',
            fontName='Helvetica', fontSize=8, textColor=TEXT_SOFT,
            leading=11),
    }


# ── Matplotlib chart helpers ──────────────────────────────────────────────────
def _fig_to_image(fig, width_mm=80, height_mm=55):
    """Convert matplotlib figure to ReportLab Image flowable."""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight',
                facecolor='#F4F7FA', edgecolor='none')
    plt.close(fig)
    buf.seek(0)
    return Image(buf, width=width_mm*mm, height=height_mm*mm)


def _apply_mpl_style(ax, title=''):
    ax.set_facecolor('#F4F7FA')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#CBD5E0')
    ax.spines['bottom'].set_color('#CBD5E0')
    ax.tick_params(colors='#6B7A90', labelsize=7)
    ax.yaxis.grid(True, color='rgba(11,59,75,0.07)', linewidth=0.5, linestyle='--')
    ax.set_axisbelow(True)
    if title:
        ax.set_title(title, fontsize=9, fontweight='bold', color='#0B3B4B', pad=6)


def make_bar_chart(data, col, columns_meta):
    fig, ax = plt.subplots(figsize=(5.5, 3.2))
    fig.patch.set_facecolor('#F4F7FA')

    col_type = columns_meta.get(col, 'categorical')
    if col_type == 'categorical':
        counts = {}
        for row in data:
            k = str(row.get(col, '(empty)'))
            counts[k] = counts.get(k, 0) + 1
        sorted_items = sorted(counts.items(), key=lambda x: x[1], reverse=True)[:12]
        labels = [x[0][:15] for x in sorted_items]
        values = [x[1] for x in sorted_items]
        bars = ax.bar(labels, values, color=MPL_COLORS[:len(labels)], edgecolor='white', linewidth=0.5)
        ax.set_xlabel(col.replace('_', ' '), fontsize=7, color='#6B7A90')
        ax.set_ylabel('Count', fontsize=7, color='#6B7A90')
        plt.xticks(rotation=35, ha='right', fontsize=6)
    else:
        vals = [float(r[col]) for r in data if r.get(col) is not None
                and _is_num(r[col])]
        ax.hist(vals, bins=20, color=MPL_COLORS[2], edgecolor='white', linewidth=0.5, alpha=0.85)
        ax.set_xlabel(col.replace('_', ' '), fontsize=7, color='#6B7A90')
        ax.set_ylabel('Frequency', fontsize=7, color='#6B7A90')

    _apply_mpl_style(ax, f'Distribution — {col.replace("_", " ")}')
    fig.tight_layout(pad=0.8)
    return _fig_to_image(fig, 85, 55)


def make_line_chart(data, col):
    fig, ax = plt.subplots(figsize=(5.5, 3.2))
    fig.patch.set_facecolor('#F4F7FA')
    vals = [float(r[col]) if _is_num(r.get(col)) else None for r in data]
    vals = [v for v in vals if v is not None]
    ax.plot(range(len(vals)), vals, color=MPL_COLORS[0], linewidth=1.5, alpha=0.9)
    ax.fill_between(range(len(vals)), vals, alpha=0.1, color=MPL_COLORS[0])
    ax.set_xlabel('Record Index', fontsize=7, color='#6B7A90')
    ax.set_ylabel(col.replace('_', ' '), fontsize=7, color='#6B7A90')
    _apply_mpl_style(ax, f'Trend — {col.replace("_", " ")}')
    fig.tight_layout(pad=0.8)
    return _fig_to_image(fig, 85, 55)


def make_scatter_chart(data, col_x, col_y):
    fig, ax = plt.subplots(figsize=(5.5, 3.2))
    fig.patch.set_facecolor('#F4F7FA')
    xs = [float(r[col_x]) for r in data if _is_num(r.get(col_x)) and _is_num(r.get(col_y))]
    ys = [float(r[col_y]) for r in data if _is_num(r.get(col_x)) and _is_num(r.get(col_y))]
    ax.scatter(xs, ys, color=MPL_COLORS[2], alpha=0.55, s=12, edgecolors='none')
    if len(xs) >= 2:
        z = np.polyfit(xs, ys, 1)
        p = np.poly1d(z)
        xline = np.linspace(min(xs), max(xs), 100)
        ax.plot(xline, p(xline), color=MPL_COLORS[0], linewidth=1.5, linestyle='--', alpha=0.8)
    ax.set_xlabel(col_x.replace('_', ' '), fontsize=7, color='#6B7A90')
    ax.set_ylabel(col_y.replace('_', ' '), fontsize=7, color='#6B7A90')
    _apply_mpl_style(ax, f'{col_x.replace("_"," ")} vs {col_y.replace("_"," ")}')
    fig.tight_layout(pad=0.8)
    return _fig_to_image(fig, 85, 55)


def make_heatmap(data, num_cols):
    cols = num_cols[:10]
    if len(cols) < 2:
        return None
    matrix = []
    for c1 in cols:
        row = []
        for c2 in cols:
            v1 = [float(r[c1]) for r in data if _is_num(r.get(c1)) and _is_num(r.get(c2))]
            v2 = [float(r[c2]) for r in data if _is_num(r.get(c1)) and _is_num(r.get(c2))]
            n  = min(len(v1), len(v2))
            row.append(_pearson(v1[:n], v2[:n]))
        matrix.append(row)

    n = len(cols)
    fig, ax = plt.subplots(figsize=(max(4, n*0.65), max(3.5, n*0.55)))
    fig.patch.set_facecolor('#F4F7FA')
    im = ax.imshow(matrix, cmap='RdYlGn', vmin=-1, vmax=1, aspect='auto')
    labels = [c.replace('_', '\n') for c in cols]
    ax.set_xticks(range(n)); ax.set_xticklabels(labels, fontsize=6, rotation=45, ha='right')
    ax.set_yticks(range(n)); ax.set_yticklabels(labels, fontsize=6)
    for i in range(n):
        for j in range(n):
            ax.text(j, i, f'{matrix[i][j]:.2f}', ha='center', va='center',
                    fontsize=5.5, color='white' if abs(matrix[i][j]) > 0.5 else '#1A2B3C')
    plt.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    _apply_mpl_style(ax, 'Correlation Heatmap')
    fig.tight_layout(pad=0.8)
    w = max(85, n * 10)
    h = max(55, n * 8)
    return _fig_to_image(fig, w, h)


def make_box_chart(data, num_cols):
    cols = num_cols[:8]
    if not cols:
        return None
    vals_list = []
    for c in cols:
        v = [float(r[c]) for r in data if _is_num(r.get(c))]
        vals_list.append(v if v else [0])

    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    fig.patch.set_facecolor('#F4F7FA')
    bp = ax.boxplot(vals_list, patch_artist=True, notch=False,
                    medianprops=dict(color='white', linewidth=2))
    for i, patch in enumerate(bp['boxes']):
        patch.set_facecolor(MPL_COLORS[i % len(MPL_COLORS)])
        patch.set_alpha(0.8)
    for element in ['whiskers', 'fliers', 'caps']:
        for item in bp[element]:
            item.set(color='#6B7A90', linewidth=1)

    ax.set_xticklabels([c.replace('_', '\n') for c in cols], fontsize=6)
    ax.set_ylabel('Value', fontsize=7, color='#6B7A90')
    _apply_mpl_style(ax, 'Box Plot — Numeric Columns')
    fig.tight_layout(pad=0.8)
    return _fig_to_image(fig, 85, 58)


def make_histogram(data, col):
    vals = [float(r[col]) for r in data if _is_num(r.get(col))]
    if not vals:
        return None
    fig, ax = plt.subplots(figsize=(5.5, 3.2))
    fig.patch.set_facecolor('#F4F7FA')
    n, bins, patches = ax.hist(vals, bins=25, edgecolor='white', linewidth=0.4)
    # colour gradient
    norm = plt.Normalize(n.min(), n.max())
    for cnt, patch in zip(n, patches):
        patch.set_facecolor(plt.cm.Blues(0.3 + 0.6 * norm(cnt)))
    ax.axvline(np.mean(vals), color=MPL_COLORS[0], linewidth=1.5,
               linestyle='--', label=f'Mean: {np.mean(vals):.2f}')
    ax.legend(fontsize=7)
    ax.set_xlabel(col.replace('_', ' '), fontsize=7, color='#6B7A90')
    ax.set_ylabel('Frequency', fontsize=7, color='#6B7A90')
    _apply_mpl_style(ax, f'Histogram — {col.replace("_", " ")}')
    fig.tight_layout(pad=0.8)
    return _fig_to_image(fig, 85, 55)


def make_donut_chart(data, cat_col):
    if not cat_col:
        return None
    counts = {}
    for row in data:
        k = str(row.get(cat_col, '(empty)'))
        counts[k] = counts.get(k, 0) + 1
    sorted_items = sorted(counts.items(), key=lambda x: x[1], reverse=True)[:10]
    labels = [x[0][:18] for x in sorted_items]
    values = [x[1] for x in sorted_items]

    fig, ax = plt.subplots(figsize=(4.5, 3.5))
    fig.patch.set_facecolor('#F4F7FA')
    wedges, texts, autotexts = ax.pie(
        values, labels=None, autopct='%1.1f%%',
        colors=MPL_COLORS[:len(values)], startangle=90,
        wedgeprops=dict(width=0.55, edgecolor='white', linewidth=1.5),
        pctdistance=0.75,
    )
    for at in autotexts:
        at.set_fontsize(6)
        at.set_color('white')
    ax.legend(wedges, labels, loc='center left', bbox_to_anchor=(1, 0.5),
              fontsize=6, frameon=False)
    ax.set_title(f'{cat_col.replace("_", " ")} Breakdown',
                 fontsize=9, fontweight='bold', color='#0B3B4B', pad=6)
    fig.tight_layout(pad=0.5)
    return _fig_to_image(fig, 80, 55)


# ── Utility functions ─────────────────────────────────────────────────────────
def _is_num(v):
    if v is None:
        return False
    try:
        f = float(v)
        return not math.isnan(f) and not math.isinf(f)
    except (TypeError, ValueError):
        return False


def _pearson(xs, ys):
    n = min(len(xs), len(ys))
    if n < 2:
        return 0.0
    mx = sum(xs[:n]) / n
    my = sum(ys[:n]) / n
    num = sum((xs[i]-mx)*(ys[i]-my) for i in range(n))
    dx  = sum((xs[i]-mx)**2 for i in range(n))
    dy  = sum((ys[i]-my)**2 for i in range(n))
    den = (dx * dy) ** 0.5
    return round(num / den, 4) if den else 0.0


def _trend_arrow(trend):
    return {'up': '▲', 'down': '▼', 'neutral': '→'}.get(str(trend).lower(), '→')


def _trend_color(trend):
    return {'up': GREEN, 'down': RED, 'neutral': MUTED}.get(str(trend).lower(), MUTED)


def _safe_str(v, max_len=60):
    s = str(v) if v is not None else ''
    return s[:max_len] + '…' if len(s) > max_len else s


# ── Page template (header + footer) ─────────────────────────────────────────
class AuralisDocTemplate(SimpleDocTemplate):
    def __init__(self, buffer, dataset_name='', domain='', **kw):
        super().__init__(buffer, **kw)
        self.dataset_name = dataset_name
        self.domain       = domain
        self._page_count  = 0

    def handle_pageEnd(self):
        self._page_count += 1
        super().handle_pageEnd()

    def afterPage(self):
        c = self.canv
        w, h = A4

        # ── top rule
        c.setFillColor(NAVY)
        c.rect(MARGIN, h - 10*mm, w - 2*MARGIN, 1.5, fill=1, stroke=0)

        # ── footer rule
        c.setFillColor(TEAL_LIGHT)
        c.rect(MARGIN, 8*mm, w - 2*MARGIN, 1, fill=1, stroke=0)

        # ── footer text
        c.setFont('Helvetica', 7)
        c.setFillColor(MUTED)
        c.drawString(MARGIN, 5*mm, f'Auralis Insights  ·  {self.dataset_name}  ·  {self.domain}')
        c.drawRightString(w - MARGIN, 5*mm,
                          f'Generated {datetime.now().strftime("%d %b %Y")}  ·  Page {self._page_count}')


# ── COVER PAGE ─────────────────────────────────────────────────────────────────
def _build_cover(story, S, dataset_name, domain_display, total_rows,
                 filtered_rows, active_filters, generated_at):
    W = PAGE_W - 2 * MARGIN

    # dark navy background block drawn via table trick
    cover_data = [['']]
    cover_tbl = Table(cover_data, colWidths=[W], rowHeights=[120*mm])
    cover_tbl.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), NAVY),
        ('ROWPADDING', (0,0), (-1,-1), 0),
    ]))
    story.append(cover_tbl)
    story.append(Spacer(1, -120*mm))   # overlay content on top

    story.append(Spacer(1, 20*mm))
    story.append(ColorRule(W, 4, TEAL))
    story.append(Spacer(1, 8*mm))

    story.append(Paragraph('Auralis Insights', ParagraphStyle(
        'al', fontName='Helvetica', fontSize=10,
        textColor=colors.HexColor('#7BBDB3'), leading=14)))
    story.append(Spacer(1, 3*mm))
    story.append(Paragraph(dataset_name or 'Analysis Report', S['cover_title']))
    story.append(Spacer(1, 2*mm))
    story.append(Paragraph(f'Domain: {domain_display}', S['cover_sub']))
    story.append(Spacer(1, 6*mm))

    meta_lines = [
        f'Generated:  {generated_at}',
        f'Total records:  {total_rows:,}',
        f'Filtered records:  {filtered_rows:,}',
        f'Active filters:  {len(active_filters)} applied' if active_filters else 'Active filters:  None',
    ]
    for line in meta_lines:
        story.append(Paragraph(line, S['cover_meta']))
        story.append(Spacer(1, 1*mm))

    story.append(Spacer(1, 10*mm))
    story.append(ColorRule(W, 2, TEAL_LIGHT))
    story.append(Spacer(1, 100*mm))   # push next content to next page

    # ── Active filters block (if any)
    if active_filters:
        story.append(PageBreak())
        story.append(Paragraph('Applied Filters', S['section_heading']))
        story.append(ColorRule(W, 2, TEAL))
        story.append(Spacer(1, 4*mm))

        filter_rows = [
            [Paragraph('<b>Column</b>', S['table_header']),
             Paragraph('<b>Filter Type</b>', S['table_header']),
             Paragraph('<b>Value</b>', S['table_header'])],
        ]
        for col, f in active_filters.items():
            if f.get('type') == 'numeric':
                val = f'{f.get("min","?")} → {f.get("max","?")}'
                ftype = 'Numeric Range'
            else:
                vals = list(f.get('values', []))[:8]
                val  = ', '.join(str(v) for v in vals)
                if len(f.get('values', [])) > 8:
                    val += f'  (+{len(f["values"])-8} more)'
                ftype = 'Category'
            filter_rows.append([
                Paragraph(_safe_str(col), S['table_cell']),
                Paragraph(ftype, S['table_cell']),
                Paragraph(_safe_str(val, 80), S['table_cell']),
            ])

        tbl = Table(filter_rows, colWidths=[W*0.35, W*0.20, W*0.45])
        tbl.setStyle(TableStyle([
            ('BACKGROUND',  (0,0), (-1,0),  NAVY),
            ('BACKGROUND',  (0,1), (-1,-1), NAVY_PALE),
            ('ROWBACKGROUNDS', (0,1), (-1,-1), [NAVY_PALE, WHITE]),
            ('GRID',        (0,0), (-1,-1), 0.3, colors.HexColor('#CBD5E0')),
            ('TOPPADDING',  (0,0), (-1,-1), 5),
            ('BOTTOMPADDING',(0,0),(-1,-1), 5),
            ('LEFTPADDING', (0,0), (-1,-1), 7),
            ('VALIGN',      (0,0), (-1,-1), 'MIDDLE'),
        ]))
        story.append(tbl)


# ── KPI SECTION ───────────────────────────────────────────────────────────────
def _build_kpis(story, S, kpis, filtered_data, columns_meta):
    W = PAGE_W - 2 * MARGIN
    story.append(PageBreak())
    story.append(Paragraph('Key Performance Indicators', S['section_heading']))
    story.append(ColorRule(W, 2, TEAL))
    story.append(Spacer(1, 4*mm))

    if not kpis:
        story.append(Paragraph('No KPI data available.', S['body']))
        return

    # Recalculate KPIs from filtered data if provided
    kpi_rows = [[
        Paragraph('<b>KPI</b>', S['table_header']),
        Paragraph('<b>Value</b>', S['table_header']),
        Paragraph('<b>Trend</b>', S['table_header']),
        Paragraph('<b>Description</b>', S['table_header']),
    ]]
    for kpi in kpis:
        trend = kpi.get('trend', 'neutral')
        arrow = _trend_arrow(trend)
        kpi_rows.append([
            Paragraph(f"{kpi.get('icon','')} {_safe_str(kpi.get('name',''), 35)}",
                      S['kpi_name']),
            Paragraph(str(kpi.get('formatted_value', kpi.get('value', '—'))),
                      S['kpi_val']),
            Paragraph(arrow, ParagraphStyle('tr', fontName='Helvetica-Bold',
                      fontSize=12, textColor=_trend_color(trend), leading=14)),
            Paragraph(_safe_str(kpi.get('explanation', ''), 80), S['body']),
        ])

    tbl = Table(kpi_rows, colWidths=[W*0.28, W*0.17, W*0.08, W*0.47])
    tbl.setStyle(TableStyle([
        ('BACKGROUND',     (0,0), (-1,0),  NAVY),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [NAVY_PALE, WHITE]),
        ('GRID',           (0,0), (-1,-1), 0.3, colors.HexColor('#CBD5E0')),
        ('TOPPADDING',     (0,0), (-1,-1), 6),
        ('BOTTOMPADDING',  (0,0), (-1,-1), 6),
        ('LEFTPADDING',    (0,0), (-1,-1), 7),
        ('VALIGN',         (0,0), (-1,-1), 'MIDDLE'),
        ('ROWPADDING',     (0,1), (-1,-1), 3),
    ]))
    story.append(tbl)


# ── CHARTS SECTION ────────────────────────────────────────────────────────────
def _build_charts(story, S, filtered_data, columns_meta):
    W     = PAGE_W - 2 * MARGIN
    num_cols = [c for c, t in columns_meta.items() if t == 'numeric']
    cat_cols = [c for c, t in columns_meta.items() if t == 'categorical']

    story.append(PageBreak())
    story.append(Paragraph('Data Visualisations', S['section_heading']))
    story.append(ColorRule(W, 2, TEAL))
    story.append(Spacer(1, 4*mm))

    if not filtered_data:
        story.append(Paragraph('No data available for charts.', S['body']))
        return

    # ── Row 1: Bar + Line ─────────────────────────────────────────────────────
    charts_row1 = []

    bar_col = cat_cols[0] if cat_cols else (num_cols[0] if num_cols else None)
    if bar_col:
        try:
            charts_row1.append(make_bar_chart(filtered_data, bar_col, columns_meta))
        except Exception:
            charts_row1.append(Paragraph('Chart unavailable', S['caption']))

    line_col = num_cols[0] if num_cols else None
    if line_col:
        try:
            charts_row1.append(make_line_chart(filtered_data, line_col))
        except Exception:
            charts_row1.append(Paragraph('Chart unavailable', S['caption']))

    if charts_row1:
        if len(charts_row1) == 2:
            tbl = Table([charts_row1], colWidths=[W/2, W/2])
            tbl.setStyle(TableStyle([
                ('ALIGN',  (0,0), (-1,-1), 'CENTER'),
                ('VALIGN', (0,0), (-1,-1), 'TOP'),
                ('LEFTPADDING',  (0,0), (-1,-1), 3),
                ('RIGHTPADDING', (0,0), (-1,-1), 3),
            ]))
            story.append(tbl)
        else:
            story.append(charts_row1[0])
        story.append(Spacer(1, 3*mm))

    # ── Row 2: Scatter + Heatmap ──────────────────────────────────────────────
    charts_row2 = []

    if len(num_cols) >= 2:
        try:
            charts_row2.append(make_scatter_chart(filtered_data, num_cols[0], num_cols[1]))
        except Exception:
            charts_row2.append(Paragraph('Chart unavailable', S['caption']))

    if len(num_cols) >= 2:
        try:
            hm = make_heatmap(filtered_data, num_cols)
            if hm:
                charts_row2.append(hm)
        except Exception:
            charts_row2.append(Paragraph('Chart unavailable', S['caption']))

    if charts_row2:
        if len(charts_row2) == 2:
            tbl = Table([charts_row2], colWidths=[W/2, W/2])
            tbl.setStyle(TableStyle([
                ('ALIGN',  (0,0), (-1,-1), 'CENTER'),
                ('VALIGN', (0,0), (-1,-1), 'TOP'),
                ('LEFTPADDING',  (0,0), (-1,-1), 3),
                ('RIGHTPADDING', (0,0), (-1,-1), 3),
            ]))
            story.append(tbl)
        elif charts_row2:
            story.append(charts_row2[0])
        story.append(Spacer(1, 3*mm))

    # ── Row 3: Box + Histogram + Donut ────────────────────────────────────────
    story.append(PageBreak())
    story.append(Paragraph('Distribution Analysis', S['section_heading']))
    story.append(ColorRule(W, 2, TEAL))
    story.append(Spacer(1, 4*mm))

    charts_row3 = []

    if num_cols:
        try:
            bp = make_box_chart(filtered_data, num_cols)
            if bp:
                charts_row3.append(bp)
        except Exception:
            pass

    hist_col = num_cols[1] if len(num_cols) > 1 else (num_cols[0] if num_cols else None)
    if hist_col:
        try:
            charts_row3.append(make_histogram(filtered_data, hist_col))
        except Exception:
            pass

    if charts_row3:
        tbl = Table([charts_row3],
                    colWidths=[W/len(charts_row3)]*len(charts_row3))
        tbl.setStyle(TableStyle([
            ('ALIGN',  (0,0), (-1,-1), 'CENTER'),
            ('VALIGN', (0,0), (-1,-1), 'TOP'),
            ('LEFTPADDING',  (0,0), (-1,-1), 3),
            ('RIGHTPADDING', (0,0), (-1,-1), 3),
        ]))
        story.append(tbl)
        story.append(Spacer(1, 4*mm))

    if cat_cols:
        try:
            donut = make_donut_chart(filtered_data, cat_cols[0])
            if donut:
                story.append(Paragraph('Category Breakdown', S['subsection']))
                story.append(donut)
        except Exception:
            pass


# ── AI NARRATIVE SECTION ──────────────────────────────────────────────────────
def _build_narrative(story, S, narrative):
    W = PAGE_W - 2 * MARGIN
    if not narrative:
        return
    story.append(PageBreak())
    story.append(Paragraph('AI Data Story', S['section_heading']))
    story.append(ColorRule(W, 2, TEAL))
    story.append(Spacer(1, 4*mm))

    # styled box
    box_data = [[Paragraph(narrative, S['body_italic'])]]
    box_tbl  = Table(box_data, colWidths=[W])
    box_tbl.setStyle(TableStyle([
        ('BACKGROUND',  (0,0), (-1,-1), TEAL_PALE),
        ('LEFTPADDING', (0,0), (-1,-1), 14),
        ('RIGHTPADDING',(0,0), (-1,-1), 14),
        ('TOPPADDING',  (0,0), (-1,-1), 12),
        ('BOTTOMPADDING',(0,0),(-1,-1), 12),
        ('ROUNDEDCORNERS', (0,0), (-1,-1), 6),
    ]))
    story.append(box_tbl)
    story.append(Spacer(1, 4*mm))
    story.append(Paragraph(
        'Powered by Arcee Trinity via OpenRouter · Auralis Insights',
        ParagraphStyle('src', fontName='Helvetica', fontSize=7,
                       textColor=MUTED, leading=10)))


# ── INSIGHTS SECTION ──────────────────────────────────────────────────────────
def _build_insights(story, S, insights):
    W = PAGE_W - 2 * MARGIN
    if not insights:
        return
    story.append(PageBreak())
    story.append(Paragraph('Insights & Correlations', S['section_heading']))
    story.append(ColorRule(W, 2, TEAL))
    story.append(Spacer(1, 4*mm))

    STRENGTH_COLORS = {
        'strong':   TEAL,
        'moderate': AMBER,
        'weak':     MUTED,
        'outlier':  RED,
        'skew':     colors.HexColor('#7C3AED'),
    }

    for ins in insights:
        title  = ins.get('title', '')
        desc   = ins.get('description', '')
        itype  = ins.get('type', 'general')
        strength = ins.get('strength', 'moderate')
        bar_color = STRENGTH_COLORS.get(strength, STRENGTH_COLORS.get(itype, MUTED))

        # left-bar + content
        content_data = [[
            '',  # coloured left bar — drawn via background
            Paragraph(f'<b>{_safe_str(title, 60)}</b>', S['insight_title']),
        ]]
        inner = Table(content_data, colWidths=[3, W - 3])
        inner.setStyle(TableStyle([
            ('BACKGROUND',  (0,0), (0,-1), bar_color),
            ('LEFTPADDING', (1,0), (1,-1), 10),
            ('TOPPADDING',  (0,0), (-1,-1), 0),
            ('BOTTOMPADDING',(0,0),(-1,-1), 0),
            ('VALIGN',      (0,0), (-1,-1), 'MIDDLE'),
        ]))

        story.append(inner)
        story.append(Paragraph(desc, S['insight_body']))
        story.append(Spacer(1, 3*mm))


# ── DATA TABLE SECTION ────────────────────────────────────────────────────────
def _build_data_table(story, S, filtered_data, columns_meta, max_rows=150):
    W    = PAGE_W - 2 * MARGIN
    cols = list(columns_meta.keys())
    if not cols or not filtered_data:
        return

    story.append(PageBreak())
    total = len(filtered_data)
    shown = min(total, max_rows)
    story.append(Paragraph(
        f'Filtered Data Table  ({shown:,} of {total:,} records shown)',
        S['section_heading']))
    story.append(ColorRule(W, 2, TEAL))
    story.append(Spacer(1, 4*mm))

    # Cap columns at 12 for readability
    display_cols = cols[:12]
    col_w = W / len(display_cols)

    header = [Paragraph(c.replace('_', ' '), S['table_header']) for c in display_cols]
    rows   = [header]
    for row in filtered_data[:shown]:
        rows.append([
            Paragraph(_safe_str(row.get(c, ''), 20), S['table_cell'])
            for c in display_cols
        ])

    tbl = Table(rows, colWidths=[col_w]*len(display_cols), repeatRows=1)
    tbl.setStyle(TableStyle([
        ('BACKGROUND',     (0,0), (-1,0),  NAVY),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [NAVY_PALE, WHITE]),
        ('GRID',           (0,0), (-1,-1), 0.25, colors.HexColor('#CBD5E0')),
        ('TOPPADDING',     (0,0), (-1,-1), 4),
        ('BOTTOMPADDING',  (0,0), (-1,-1), 4),
        ('LEFTPADDING',    (0,0), (-1,-1), 5),
        ('FONTSIZE',       (0,0), (-1,-1), 7),
        ('VALIGN',         (0,0), (-1,-1), 'MIDDLE'),
        ('ROWPADDING',     (0,0), (-1,-1), 2),
    ]))
    story.append(tbl)

    if total > max_rows:
        story.append(Spacer(1, 3*mm))
        story.append(Paragraph(
            f'Note: {total - max_rows:,} additional records not shown. '
            f'Export filtered CSV from the Interactive Dashboard to get all rows.',
            S['body']))


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════
def generate_pdf_report(payload: dict) -> bytes:
    """
    Build and return a PDF as bytes.

    payload keys:
        dataset_name    str
        domain_display  str
        total_rows      int
        filtered_rows   int
        active_filters  dict
        kpis            list
        insights        list
        narrative       str
        filtered_data   list[dict]   (up to 2000 rows)
        columns_meta    dict         {col: 'numeric'|'categorical'|'datetime'}
        generated_at    str
    """
    buf          = io.BytesIO()
    dataset_name = payload.get('dataset_name', 'Dataset')
    domain       = payload.get('domain_display', 'Analysis')
    total_rows   = int(payload.get('total_rows', 0))
    filtered_rows= int(payload.get('filtered_rows', 0))
    active_filters = payload.get('active_filters', {})
    kpis         = payload.get('kpis', [])
    insights     = payload.get('insights', [])
    narrative    = payload.get('narrative', '')
    filtered_data= payload.get('filtered_data', [])
    columns_meta = payload.get('columns_meta', {})
    generated_at = payload.get('generated_at',
                               datetime.now().strftime('%d %b %Y, %H:%M'))

    doc = AuralisDocTemplate(
        buf,
        dataset_name=dataset_name,
        domain=domain,
        pagesize=A4,
        leftMargin=MARGIN, rightMargin=MARGIN,
        topMargin=14*mm, bottomMargin=14*mm,
    )

    S     = _styles()
    story = []

    _build_cover(story, S, dataset_name, domain, total_rows,
                 filtered_rows, active_filters, generated_at)

    _build_kpis(story, S, kpis, filtered_data, columns_meta)

    _build_charts(story, S, filtered_data, columns_meta)

    if narrative:
        _build_narrative(story, S, narrative)

    if insights:
        _build_insights(story, S, insights)

    _build_data_table(story, S, filtered_data, columns_meta)

    doc.build(story)
    return buf.getvalue()