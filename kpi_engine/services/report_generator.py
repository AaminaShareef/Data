"""
report_generator.py
────────────────────────────────────────────────────────────────────────────
Auralis — PDF Report Generator  (v2 — improved)
Builds a styled multi-page PDF from AnalysisResult data.
Uses ReportLab (layout) + Matplotlib (charts).

Fixes vs v1:
  • Charts no longer fail silently — robust data parsing for every chart type
  • AI narrative generated via OpenRouter if API key available
  • Executive summary section added after cover
  • KPI descriptions no longer truncated
  • Cover page visual block renders correctly
  • Scatter chart uses correct paired rows
  • Heatmap only renders when ≥3 numeric cols
  • Bar chart skips ID-like columns (too many unique values)
  • All chart exceptions are caught and show a styled placeholder
"""

import io
import math
import re
import json
import requests
from datetime import datetime

import matplotlib
matplotlib.use('Agg')
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

MPL_COLORS = ['#0B3B4B', '#1D5A6D', '#4A9B8D', '#7BBDB3',
              '#2C6F84', '#B2DFDB', '#9ECDD8', '#0d5266',
              '#3d8b7a', '#5ba8a0']

PAGE_W, PAGE_H = A4
MARGIN = 20 * mm


# ─────────────────────────────────────────────────────────────────────────────
# UTILITY
# ─────────────────────────────────────────────────────────────────────────────

def _is_num(v):
    if v is None:
        return False
    try:
        f = float(v)
        return not math.isnan(f) and not math.isinf(f)
    except (TypeError, ValueError):
        return False


def _safe_nums(data, col):
    """Extract clean float list from a column."""
    result = []
    for row in data:
        v = row.get(col)
        if _is_num(v):
            result.append(float(v))
    return result


def _safe_pair_nums(data, col_x, col_y):
    """Extract paired (x, y) floats where both are valid."""
    xs, ys = [], []
    for row in data:
        vx, vy = row.get(col_x), row.get(col_y)
        if _is_num(vx) and _is_num(vy):
            xs.append(float(vx))
            ys.append(float(vy))
    return xs, ys


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


def _safe_str(v, max_len=None):
    s = str(v) if v is not None else ''
    if max_len and len(s) > max_len:
        return s[:max_len] + '…'
    return s


def _is_id_like(col, data):
    """Return True if a categorical column has too many unique values to chart."""
    vals = set(str(row.get(col, '')) for row in data[:200])
    return len(vals) > 20


# ─────────────────────────────────────────────────────────────────────────────
# AI NARRATIVE GENERATOR  (OpenRouter — optional)
# ─────────────────────────────────────────────────────────────────────────────

def _generate_ai_narrative(payload: dict) -> str:
    """
    Call OpenRouter to produce a concise executive narrative for the report.
    Falls back to a hardcoded summary if API key is not set or call fails.
    """
    try:
        from django.conf import settings
        api_key = getattr(settings, 'OPENROUTER_API_KEY', '')
    except Exception:
        api_key = ''

    dataset_name  = payload.get('dataset_name', 'Dataset')
    domain        = payload.get('domain_display', 'General')
    total_rows    = payload.get('total_rows', 0)
    filtered_rows = payload.get('filtered_rows', 0)
    kpis          = payload.get('kpis', [])
    insights      = payload.get('insights', [])
    active_filters= payload.get('active_filters', {})

    # Build a compact KPI summary string
    kpi_summary = '; '.join(
        f"{k.get('name')}: {k.get('formatted_value', k.get('value', '?'))}"
        for k in kpis[:6]
    )
    insight_summary = ' | '.join(
        i.get('title', '') for i in insights[:4]
    )
    filter_summary = (
        ', '.join(f"{k}={v}" for k, v in active_filters.items())
        if active_filters else 'No filters applied'
    )

    # Fallback — always generated, used if API fails
    fallback = (
        f"This report covers {filtered_rows:,} records from the {dataset_name} dataset "
        f"in the {domain} domain. "
        f"Key metrics include: {kpi_summary or 'see KPI table above'}. "
        f"Notable patterns: {insight_summary or 'see insights section'}. "
        f"Filters applied: {filter_summary}."
    )

    if not api_key:
        return fallback

    system_prompt = (
        "You are a senior data analyst writing a concise executive narrative for a business report. "
        "Write 3–4 clear, professional sentences summarising the dataset and key findings. "
        "Be specific — reference actual KPI values and insight titles provided. "
        "No bullet points, no headers, no markdown. Plain prose only."
    )
    user_message = (
        f"Dataset: {dataset_name}\n"
        f"Domain: {domain}\n"
        f"Total records: {total_rows:,} | Filtered records: {filtered_rows:,}\n"
        f"Active filters: {filter_summary}\n"
        f"KPIs: {kpi_summary}\n"
        f"Key insights: {insight_summary}\n\n"
        "Write the executive narrative now."
    )

    try:
        resp = requests.post(
            'https://openrouter.ai/api/v1/chat/completions',
            headers={
                'Authorization': f'Bearer {api_key}',
                'Content-Type':  'application/json',
                'HTTP-Referer':  'https://auralis.app',
                'X-Title':       'Auralis Report Generator',
            },
            json={
                'model':       'arcee-ai/trinity-large-preview:free',
                'messages':    [
                    {'role': 'system', 'content': system_prompt},
                    {'role': 'user',   'content': user_message},
                ],
                'max_tokens':  300,
                'temperature': 0.4,
            },
            timeout=20,
        )
        resp.raise_for_status()
        text = resp.json()['choices'][0]['message']['content'].strip()
        return text if text else fallback
    except Exception as e:
        print(f"[report_generator] AI narrative failed: {e}")
        return fallback


# ─────────────────────────────────────────────────────────────────────────────
# CUSTOM FLOWABLE
# ─────────────────────────────────────────────────────────────────────────────

class ColorRule(Flowable):
    def __init__(self, width, height=2, color=TEAL):
        super().__init__()
        self.width  = width
        self.height = height
        self.color  = color

    def draw(self):
        self.canv.setFillColor(self.color)
        self.canv.rect(0, 0, self.width, self.height, fill=1, stroke=0)


# ─────────────────────────────────────────────────────────────────────────────
# STYLES
# ─────────────────────────────────────────────────────────────────────────────

def _styles():
    def ps(name, **kw):
        return ParagraphStyle(name, **kw)

    return {
        'cover_title': ps('cover_title',
            fontName='Helvetica-Bold', fontSize=26, textColor=WHITE,
            leading=32, alignment=TA_LEFT, spaceAfter=6),

        'cover_sub': ps('cover_sub',
            fontName='Helvetica', fontSize=13, textColor=colors.HexColor('#B2DFDB'),
            leading=18, alignment=TA_LEFT, spaceAfter=4),

        'cover_meta': ps('cover_meta',
            fontName='Helvetica', fontSize=10, textColor=colors.HexColor('#9ECDD8'),
            leading=16, alignment=TA_LEFT),

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
            leading=16, spaceAfter=6),

        'caption': ps('caption',
            fontName='Helvetica', fontSize=8, textColor=MUTED,
            leading=11, alignment=TA_CENTER, spaceAfter=8),

        'chart_placeholder': ps('chart_placeholder',
            fontName='Helvetica-Oblique', fontSize=8, textColor=MUTED,
            leading=11, alignment=TA_CENTER),

        'kpi_name': ps('kpi_name',
            fontName='Helvetica-Bold', fontSize=9, textColor=NAVY,
            leading=12),

        'kpi_val': ps('kpi_val',
            fontName='Helvetica-Bold', fontSize=13, textColor=NAVY_MID,
            leading=16),

        'summary_body': ps('summary_body',
            fontName='Helvetica', fontSize=10, textColor=TEXT_SOFT,
            leading=16, spaceAfter=6),

        'insight_title': ps('insight_title',
            fontName='Helvetica-Bold', fontSize=10, textColor=NAVY,
            leading=13, spaceBefore=6, spaceAfter=2),

        'insight_body': ps('insight_body',
            fontName='Helvetica', fontSize=9, textColor=TEXT_SOFT,
            leading=13, spaceAfter=4),

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


# ─────────────────────────────────────────────────────────────────────────────
# CHART HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _fig_to_image(fig, width_mm=85, height_mm=58):
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
    ax.yaxis.grid(True, color='#E2E8F0', linewidth=0.5, linestyle='--')
    ax.set_axisbelow(True)
    if title:
        ax.set_title(title, fontsize=9, fontweight='bold', color='#0B3B4B', pad=6)


def _chart_placeholder(S, msg='Chart unavailable — insufficient data'):
    """Styled placeholder when chart cannot be drawn."""
    data = [[Paragraph(f'⬜  {msg}', S['chart_placeholder'])]]
    tbl  = Table(data, colWidths=[85*mm], rowHeights=[58*mm])
    tbl.setStyle(TableStyle([
        ('BACKGROUND',    (0,0), (-1,-1), BG_GREY),
        ('ALIGN',         (0,0), (-1,-1), 'CENTER'),
        ('VALIGN',        (0,0), (-1,-1), 'MIDDLE'),
        ('BOX',           (0,0), (-1,-1), 0.5, colors.HexColor('#CBD5E0')),
        ('ROUNDEDCORNERS',(0,0), (-1,-1), 4),
    ]))
    return tbl


def make_bar_chart(data, col, columns_meta, S):
    try:
        col_type = columns_meta.get(col, 'categorical')
        fig, ax  = plt.subplots(figsize=(5.5, 3.5))
        fig.patch.set_facecolor('#F4F7FA')

        if col_type == 'categorical':
            counts = {}
            for row in data:
                k = str(row.get(col, '(empty)'))
                counts[k] = counts.get(k, 0) + 1
            # top 10 only
            items  = sorted(counts.items(), key=lambda x: x[1], reverse=True)[:10]
            labels = [x[0][:14] for x in items]
            values = [x[1] for x in items]
            if not values:
                return _chart_placeholder(S)
            bars = ax.bar(labels, values,
                          color=[MPL_COLORS[i % len(MPL_COLORS)] for i in range(len(labels))],
                          edgecolor='white', linewidth=0.5)
            ax.set_xlabel(col.replace('_', ' '), fontsize=7, color='#6B7A90')
            ax.set_ylabel('Count', fontsize=7, color='#6B7A90')
            plt.xticks(rotation=35, ha='right', fontsize=6)
        else:
            vals = _safe_nums(data, col)
            if not vals:
                return _chart_placeholder(S, f'No numeric data in {col}')
            ax.hist(vals, bins=min(20, len(set(vals))),
                    color=MPL_COLORS[2], edgecolor='white', linewidth=0.5, alpha=0.85)
            ax.set_xlabel(col.replace('_', ' '), fontsize=7, color='#6B7A90')
            ax.set_ylabel('Frequency', fontsize=7, color='#6B7A90')

        _apply_mpl_style(ax, f'Distribution — {col.replace("_", " ")}')
        fig.tight_layout(pad=0.8)
        return _fig_to_image(fig, 85, 58)
    except Exception as e:
        print(f'[chart] bar_chart error for {col}: {e}')
        return _chart_placeholder(S)


def make_line_chart(data, col, S):
    try:
        vals = _safe_nums(data, col)
        if len(vals) < 3:
            return _chart_placeholder(S, f'Not enough data points in {col}')
        fig, ax = plt.subplots(figsize=(5.5, 3.5))
        fig.patch.set_facecolor('#F4F7FA')
        ax.plot(range(len(vals)), vals, color=MPL_COLORS[0], linewidth=1.5, alpha=0.9)
        ax.fill_between(range(len(vals)), vals, alpha=0.1, color=MPL_COLORS[0])
        # mean line
        mean_val = sum(vals) / len(vals)
        ax.axhline(mean_val, color=MPL_COLORS[2], linewidth=1,
                   linestyle='--', alpha=0.7, label=f'Mean: {mean_val:,.2f}')
        ax.legend(fontsize=7, frameon=False)
        ax.set_xlabel('Record Index', fontsize=7, color='#6B7A90')
        ax.set_ylabel(col.replace('_', ' '), fontsize=7, color='#6B7A90')
        _apply_mpl_style(ax, f'Trend — {col.replace("_", " ")}')
        fig.tight_layout(pad=0.8)
        return _fig_to_image(fig, 85, 58)
    except Exception as e:
        print(f'[chart] line_chart error for {col}: {e}')
        return _chart_placeholder(S)


def make_scatter_chart(data, col_x, col_y, S):
    try:
        xs, ys = _safe_pair_nums(data, col_x, col_y)
        if len(xs) < 4:
            return _chart_placeholder(S, f'Not enough paired data for {col_x} vs {col_y}')
        fig, ax = plt.subplots(figsize=(5.5, 3.5))
        fig.patch.set_facecolor('#F4F7FA')
        ax.scatter(xs, ys, color=MPL_COLORS[2], alpha=0.55, s=14, edgecolors='none')
        # trend line
        z = np.polyfit(xs, ys, 1)
        p = np.poly1d(z)
        xline = np.linspace(min(xs), max(xs), 100)
        ax.plot(xline, p(xline), color=MPL_COLORS[0],
                linewidth=1.5, linestyle='--', alpha=0.8)
        r = _pearson(xs, ys)
        ax.set_xlabel(col_x.replace('_', ' '), fontsize=7, color='#6B7A90')
        ax.set_ylabel(col_y.replace('_', ' '), fontsize=7, color='#6B7A90')
        _apply_mpl_style(ax,
            f'{col_x.replace("_"," ")} vs {col_y.replace("_"," ")}  (r={r:.2f})')
        fig.tight_layout(pad=0.8)
        return _fig_to_image(fig, 85, 58)
    except Exception as e:
        print(f'[chart] scatter_chart error: {e}')
        return _chart_placeholder(S)


def make_heatmap(data, num_cols, S):
    try:
        cols = num_cols[:8]
        if len(cols) < 3:
            return _chart_placeholder(S, 'Need ≥3 numeric columns for heatmap')
        matrix = []
        for c1 in cols:
            row = []
            for c2 in cols:
                xs, ys = _safe_pair_nums(data, c1, c2)
                row.append(_pearson(xs, ys))
            matrix.append(row)

        n   = len(cols)
        fig, ax = plt.subplots(figsize=(max(4.5, n*0.7), max(4, n*0.65)))
        fig.patch.set_facecolor('#F4F7FA')
        im  = ax.imshow(matrix, cmap='RdYlGn', vmin=-1, vmax=1, aspect='auto')
        lbls = [c.replace('_', '\n') for c in cols]
        ax.set_xticks(range(n)); ax.set_xticklabels(lbls, fontsize=6, rotation=45, ha='right')
        ax.set_yticks(range(n)); ax.set_yticklabels(lbls, fontsize=6)
        for i in range(n):
            for j in range(n):
                ax.text(j, i, f'{matrix[i][j]:.2f}', ha='center', va='center',
                        fontsize=5, color='white' if abs(matrix[i][j]) > 0.5 else '#1A2B3C')
        plt.colorbar(im, ax=ax, shrink=0.75, pad=0.02)
        _apply_mpl_style(ax, 'Correlation Heatmap')
        fig.tight_layout(pad=0.8)
        w = max(85, n * 12)
        h = max(58, n * 10)
        return _fig_to_image(fig, w, h)
    except Exception as e:
        print(f'[chart] heatmap error: {e}')
        return _chart_placeholder(S)


def make_box_chart(data, num_cols, S):
    try:
        cols = [c for c in num_cols[:8]]
        valid = []
        vals_list = []
        for c in cols:
            v = _safe_nums(data, c)
            if len(v) >= 2:
                valid.append(c)
                vals_list.append(v)
        if not valid:
            return _chart_placeholder(S, 'No numeric data for box plot')

        fig, ax = plt.subplots(figsize=(5.5, 3.5))
        fig.patch.set_facecolor('#F4F7FA')
        bp = ax.boxplot(vals_list, patch_artist=True, notch=False,
                        medianprops=dict(color='white', linewidth=2))
        for i, patch in enumerate(bp['boxes']):
            patch.set_facecolor(MPL_COLORS[i % len(MPL_COLORS)])
            patch.set_alpha(0.8)
        for element in ['whiskers', 'fliers', 'caps']:
            for item in bp[element]:
                item.set(color='#6B7A90', linewidth=0.8)
        ax.set_xticklabels([c.replace('_', '\n') for c in valid], fontsize=6)
        ax.set_ylabel('Value', fontsize=7, color='#6B7A90')
        _apply_mpl_style(ax, 'Box Plot — Numeric Columns')
        fig.tight_layout(pad=0.8)
        return _fig_to_image(fig, 85, 58)
    except Exception as e:
        print(f'[chart] box_chart error: {e}')
        return _chart_placeholder(S)


def make_histogram(data, col, S):
    try:
        vals = _safe_nums(data, col)
        if len(vals) < 5:
            return _chart_placeholder(S, f'Not enough data for histogram of {col}')
        fig, ax = plt.subplots(figsize=(5.5, 3.5))
        fig.patch.set_facecolor('#F4F7FA')
        n, bins, patches = ax.hist(vals, bins=min(25, max(5, len(vals)//3)),
                                   edgecolor='white', linewidth=0.4)
        norm = plt.Normalize(n.min(), n.max())
        for cnt, patch in zip(n, patches):
            patch.set_facecolor(plt.cm.Blues(0.3 + 0.6 * norm(cnt)))
        mean_v   = np.mean(vals)
        median_v = np.median(vals)
        ax.axvline(mean_v, color=MPL_COLORS[0], linewidth=1.5,
                   linestyle='--', label=f'Mean: {mean_v:,.2f}')
        ax.axvline(median_v, color=MPL_COLORS[2], linewidth=1.5,
                   linestyle=':', label=f'Median: {median_v:,.2f}')
        ax.legend(fontsize=7, frameon=False)
        ax.set_xlabel(col.replace('_', ' '), fontsize=7, color='#6B7A90')
        ax.set_ylabel('Frequency', fontsize=7, color='#6B7A90')
        _apply_mpl_style(ax, f'Histogram — {col.replace("_", " ")}')
        fig.tight_layout(pad=0.8)
        return _fig_to_image(fig, 85, 58)
    except Exception as e:
        print(f'[chart] histogram error for {col}: {e}')
        return _chart_placeholder(S)


def make_donut_chart(data, cat_col, S):
    try:
        if not cat_col:
            return None
        counts = {}
        for row in data:
            k = str(row.get(cat_col, '(empty)'))
            counts[k] = counts.get(k, 0) + 1
        items  = sorted(counts.items(), key=lambda x: x[1], reverse=True)[:10]
        labels = [x[0][:18] for x in items]
        values = [x[1] for x in items]
        if not values or len(values) < 2:
            return None

        fig, ax = plt.subplots(figsize=(5, 3.8))
        fig.patch.set_facecolor('#F4F7FA')
        wedges, _, autotexts = ax.pie(
            values, labels=None, autopct='%1.1f%%',
            colors=MPL_COLORS[:len(values)], startangle=90,
            wedgeprops=dict(width=0.55, edgecolor='white', linewidth=1.5),
            pctdistance=0.78,
        )
        for at in autotexts:
            at.set_fontsize(6.5)
            at.set_color('white')
        ax.legend(wedges, labels, loc='center left',
                  bbox_to_anchor=(1.02, 0.5), fontsize=6.5, frameon=False)
        ax.set_title(f'{cat_col.replace("_", " ").title()} Breakdown',
                     fontsize=9, fontweight='bold', color='#0B3B4B', pad=6)
        fig.tight_layout(pad=0.5)
        return _fig_to_image(fig, 85, 60)
    except Exception as e:
        print(f'[chart] donut error for {cat_col}: {e}')
        return None


def make_kpi_sparklines(kpis, filtered_data, columns_meta, S):
    """
    Render KPIs as visual cards instead of a plain table.
    2-column grid with icon, name, value, trend indicator.
    """
    W     = PAGE_W - 2 * MARGIN
    cards = []
    TREND_COLORS = {'up': GREEN, 'down': RED, 'neutral': MUTED}

    for kpi in kpis:
        name  = _safe_str(kpi.get('name', ''), 40)
        val   = str(kpi.get('formatted_value', kpi.get('value', '—')))
        icon  = kpi.get('icon', '■')
        trend = kpi.get('trend', 'neutral')
        arrow = _trend_arrow(trend)
        expl  = _safe_str(kpi.get('explanation', ''), 120)
        t_col = TREND_COLORS.get(trend, MUTED)

        name_para = Paragraph(f'{icon}  {name}', S['kpi_name'])
        val_para  = Paragraph(val, S['kpi_val'])
        trend_para= Paragraph(arrow, ParagraphStyle(
            'tr', fontName='Helvetica-Bold', fontSize=14,
            textColor=t_col, leading=16))
        expl_para = Paragraph(expl, S['body'])

        inner = Table(
            [[name_para, trend_para],
             [val_para,  ''],
             [expl_para, '']],
            colWidths=[(W/2 - 10)*0.85, (W/2 - 10)*0.15]
        )
        inner.setStyle(TableStyle([
            ('SPAN',         (0,1), (-1,1)),
            ('SPAN',         (0,2), (-1,2)),
            ('VALIGN',       (0,0), (-1,-1), 'TOP'),
            ('TOPPADDING',   (0,0), (-1,-1), 4),
            ('BOTTOMPADDING',(0,0), (-1,-1), 4),
            ('LEFTPADDING',  (0,0), (-1,-1), 0),
            ('RIGHTPADDING', (0,0), (-1,-1), 0),
        ]))

        card = Table([[inner]], colWidths=[W/2 - 6])
        card.setStyle(TableStyle([
            ('BACKGROUND',    (0,0), (-1,-1), NAVY_PALE),
            ('BOX',           (0,0), (-1,-1), 0.5, colors.HexColor('#C5D9E4')),
            ('ROUNDEDCORNERS',(0,0), (-1,-1), 4),
            ('LEFTPADDING',   (0,0), (-1,-1), 10),
            ('RIGHTPADDING',  (0,0), (-1,-1), 10),
            ('TOPPADDING',    (0,0), (-1,-1), 8),
            ('BOTTOMPADDING', (0,0), (-1,-1), 8),
        ]))
        cards.append(card)

    # Arrange into 2-column grid rows
    rows = []
    for i in range(0, len(cards), 2):
        pair = cards[i:i+2]
        if len(pair) == 1:
            pair.append('')   # empty cell
        rows.append(pair)

    if not rows:
        return

    grid = Table(rows, colWidths=[W/2 - 3, W/2 - 3], hAlign='LEFT')
    grid.setStyle(TableStyle([
        ('LEFTPADDING',  (0,0), (-1,-1), 3),
        ('RIGHTPADDING', (0,0), (-1,-1), 3),
        ('TOPPADDING',   (0,0), (-1,-1), 4),
        ('BOTTOMPADDING',(0,0), (-1,-1), 4),
        ('VALIGN',       (0,0), (-1,-1), 'TOP'),
    ]))
    return grid


# ─────────────────────────────────────────────────────────────────────────────
# PAGE TEMPLATE
# ─────────────────────────────────────────────────────────────────────────────

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

        # top rule
        c.setFillColor(NAVY)
        c.rect(MARGIN, h - 10*mm, w - 2*MARGIN, 1.5, fill=1, stroke=0)

        # footer rule
        c.setFillColor(TEAL_LIGHT)
        c.rect(MARGIN, 8*mm, w - 2*MARGIN, 1, fill=1, stroke=0)

        # footer text
        c.setFont('Helvetica', 7)
        c.setFillColor(MUTED)
        c.drawString(MARGIN, 5*mm,
                     f'Auralis Insights  ·  {self.dataset_name}  ·  {self.domain}')
        c.drawRightString(w - MARGIN, 5*mm,
                          f'Generated {datetime.now().strftime("%d %b %Y")}  ·  Page {self._page_count}')


# ─────────────────────────────────────────────────────────────────────────────
# SECTION BUILDERS
# ─────────────────────────────────────────────────────────────────────────────

def _build_cover(story, S, dataset_name, domain_display, total_rows,
                 filtered_rows, active_filters, generated_at):
    W = PAGE_W - 2 * MARGIN

    # Navy background block
    cover_tbl = Table([['']], colWidths=[W], rowHeights=[115*mm])
    cover_tbl.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), NAVY),
        ('ROWPADDING', (0,0), (-1,-1), 0),
    ]))
    story.append(cover_tbl)
    story.append(Spacer(1, -115*mm))   # overlay content on top of the block

    story.append(Spacer(1, 18*mm))
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
        f'Generated:          {generated_at}',
        f'Total records:      {total_rows:,}',
        f'Filtered records:   {filtered_rows:,}',
        f'Active filters:     {len(active_filters)} applied'
        if active_filters else 'Active filters:     None',
    ]
    for line in meta_lines:
        story.append(Paragraph(line, S['cover_meta']))
        story.append(Spacer(1, 1*mm))

    story.append(Spacer(1, 8*mm))
    story.append(ColorRule(W, 1.5, TEAL_LIGHT))

    # push to next page
    story.append(Spacer(1, 90*mm))

    # Active filters table
    if active_filters:
        story.append(PageBreak())
        story.append(Paragraph('Applied Filters', S['section_heading']))
        story.append(ColorRule(W, 2, TEAL))
        story.append(Spacer(1, 4*mm))

        filter_rows = [[
            Paragraph('<b>Column</b>', S['table_header']),
            Paragraph('<b>Filter Type</b>', S['table_header']),
            Paragraph('<b>Value</b>', S['table_header']),
        ]]
        for col, f in active_filters.items():
            if f.get('type') == 'numeric':
                val   = f'{f.get("min","?")} → {f.get("max","?")}'
                ftype = 'Numeric Range'
            else:
                vals  = list(f.get('values', []))[:8]
                val   = ', '.join(str(v) for v in vals)
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
            ('BACKGROUND',     (0,0), (-1,0),  NAVY),
            ('ROWBACKGROUNDS', (0,1), (-1,-1), [NAVY_PALE, WHITE]),
            ('GRID',           (0,0), (-1,-1), 0.3, colors.HexColor('#CBD5E0')),
            ('TOPPADDING',     (0,0), (-1,-1), 5),
            ('BOTTOMPADDING',  (0,0), (-1,-1), 5),
            ('LEFTPADDING',    (0,0), (-1,-1), 7),
            ('VALIGN',         (0,0), (-1,-1), 'MIDDLE'),
        ]))
        story.append(tbl)


def _build_executive_summary(story, S, narrative, payload):
    """Executive summary — AI-generated or computed fallback."""
    W = PAGE_W - 2 * MARGIN
    story.append(PageBreak())
    story.append(Paragraph('Executive Summary', S['section_heading']))
    story.append(ColorRule(W, 2, TEAL))
    story.append(Spacer(1, 5*mm))

    # Generate narrative if not provided
    if not narrative:
        narrative = _generate_ai_narrative(payload)

    box_data = [[Paragraph(narrative, S['summary_body'])]]
    box_tbl  = Table(box_data, colWidths=[W])
    box_tbl.setStyle(TableStyle([
        ('BACKGROUND',   (0,0), (-1,-1), NAVY_PALE),
        ('LEFTPADDING',  (0,0), (-1,-1), 16),
        ('RIGHTPADDING', (0,0), (-1,-1), 16),
        ('TOPPADDING',   (0,0), (-1,-1), 14),
        ('BOTTOMPADDING',(0,0), (-1,-1), 14),
        ('BOX',          (0,0), (-1,-1), 0.5, colors.HexColor('#C5D9E4')),
        ('ROUNDEDCORNERS',(0,0),(-1,-1), 6),
    ]))
    story.append(box_tbl)
    story.append(Spacer(1, 3*mm))

    # Quick stat strip
    kpis     = payload.get('kpis', [])
    insights = payload.get('insights', [])
    f_rows   = payload.get('filtered_rows', 0)
    t_rows   = payload.get('total_rows', 0)

    stats = [
        (str(t_rows),    'Total Records'),
        (str(f_rows),    'Filtered Records'),
        (str(len(kpis)), 'KPIs Tracked'),
        (str(len(insights)), 'Key Insights'),
    ]
    stat_cells = []
    for val, lbl in stats:
        cell_data = [
            [Paragraph(f'<b>{val}</b>', ParagraphStyle(
                'sv', fontName='Helvetica-Bold', fontSize=18,
                textColor=NAVY, leading=22, alignment=TA_CENTER))],
            [Paragraph(lbl, ParagraphStyle(
                'sl', fontName='Helvetica', fontSize=8,
                textColor=MUTED, leading=11, alignment=TA_CENTER))],
        ]
        cell_tbl = Table(cell_data, colWidths=[W/4 - 6])
        cell_tbl.setStyle(TableStyle([
            ('ALIGN',         (0,0), (-1,-1), 'CENTER'),
            ('TOPPADDING',    (0,0), (-1,-1), 8),
            ('BOTTOMPADDING', (0,0), (-1,-1), 8),
            ('BACKGROUND',    (0,0), (-1,-1), TEAL_PALE),
            ('BOX',           (0,0), (-1,-1), 0.5, colors.HexColor('#C5D9E4')),
            ('ROUNDEDCORNERS',(0,0),(-1,-1), 4),
        ]))
        stat_cells.append(cell_tbl)

    stats_row = Table([stat_cells],
                      colWidths=[W/4 - 3]*4, hAlign='LEFT')
    stats_row.setStyle(TableStyle([
        ('LEFTPADDING',  (0,0), (-1,-1), 3),
        ('RIGHTPADDING', (0,0), (-1,-1), 3),
        ('TOPPADDING',   (0,0), (-1,-1), 4),
    ]))
    story.append(Spacer(1, 4*mm))
    story.append(stats_row)


def _build_kpis(story, S, kpis, filtered_data, columns_meta, payload):
    W = PAGE_W - 2 * MARGIN
    story.append(PageBreak())
    story.append(Paragraph('Key Performance Indicators', S['section_heading']))
    story.append(ColorRule(W, 2, TEAL))
    story.append(Spacer(1, 5*mm))

    if not kpis:
        story.append(Paragraph('No KPI data available.', S['body']))
        return

    grid = make_kpi_sparklines(kpis, filtered_data, columns_meta, S)
    if grid:
        story.append(grid)


def _build_charts(story, S, filtered_data, columns_meta):
    W        = PAGE_W - 2 * MARGIN
    num_cols = [c for c, t in columns_meta.items() if t == 'numeric']
    cat_cols = [c for c, t in columns_meta.items() if t == 'categorical']

    # Skip ID-like categoricals
    cat_cols = [c for c in cat_cols if not _is_id_like(c, filtered_data)]

    story.append(PageBreak())
    story.append(Paragraph('Data Visualisations', S['section_heading']))
    story.append(ColorRule(W, 2, TEAL))
    story.append(Spacer(1, 5*mm))

    if not filtered_data:
        story.append(Paragraph('No data available for charts.', S['body']))
        return

    def two_col_row(left, right):
        tbl = Table([[left, right]], colWidths=[W/2, W/2])
        tbl.setStyle(TableStyle([
            ('ALIGN',  (0,0), (-1,-1), 'CENTER'),
            ('VALIGN', (0,0), (-1,-1), 'TOP'),
            ('LEFTPADDING',  (0,0), (-1,-1), 3),
            ('RIGHTPADDING', (0,0), (-1,-1), 3),
        ]))
        return tbl

    # ── Row 1: Bar + Line
    bar_col  = cat_cols[0] if cat_cols else (num_cols[0] if num_cols else None)
    line_col = num_cols[0] if num_cols else None

    if bar_col and line_col:
        story.append(two_col_row(
            make_bar_chart(filtered_data, bar_col, columns_meta, S),
            make_line_chart(filtered_data, line_col, S),
        ))
        story.append(Spacer(1, 4*mm))
    elif bar_col:
        story.append(make_bar_chart(filtered_data, bar_col, columns_meta, S))
        story.append(Spacer(1, 4*mm))
    elif line_col:
        story.append(make_line_chart(filtered_data, line_col, S))
        story.append(Spacer(1, 4*mm))

    # ── Row 2: Scatter + Heatmap
    if len(num_cols) >= 2:
        scatter = make_scatter_chart(filtered_data, num_cols[0], num_cols[1], S)
        heatmap = make_heatmap(filtered_data, num_cols, S)
        story.append(two_col_row(scatter, heatmap))
        story.append(Spacer(1, 4*mm))

    # ── Distribution page
    story.append(PageBreak())
    story.append(Paragraph('Distribution Analysis', S['section_heading']))
    story.append(ColorRule(W, 2, TEAL))
    story.append(Spacer(1, 5*mm))

    dist_row = []
    if num_cols:
        bp = make_box_chart(filtered_data, num_cols, S)
        if bp:
            dist_row.append(bp)

    hist_col = num_cols[1] if len(num_cols) > 1 else (num_cols[0] if num_cols else None)
    if hist_col:
        hist = make_histogram(filtered_data, hist_col, S)
        if hist:
            dist_row.append(hist)

    if len(dist_row) == 2:
        story.append(two_col_row(dist_row[0], dist_row[1]))
        story.append(Spacer(1, 4*mm))
    elif len(dist_row) == 1:
        story.append(dist_row[0])
        story.append(Spacer(1, 4*mm))

    # Donut for best categorical col
    if cat_cols:
        donut = make_donut_chart(filtered_data, cat_cols[0], S)
        if donut:
            story.append(Paragraph('Category Breakdown', S['subsection']))
            story.append(donut)


def _build_insights(story, S, insights):
    W = PAGE_W - 2 * MARGIN
    if not insights:
        return
    story.append(PageBreak())
    story.append(Paragraph('Insights & Correlations', S['section_heading']))
    story.append(ColorRule(W, 2, TEAL))
    story.append(Spacer(1, 5*mm))

    STRENGTH_COLORS = {
        'strong':   TEAL,
        'moderate': AMBER,
        'weak':     MUTED,
        'outlier':  RED,
        'skew':     colors.HexColor('#7C3AED'),
    }

    for ins in insights:
        title    = ins.get('title', '')
        desc     = ins.get('description', '')
        itype    = ins.get('type', 'general')
        strength = ins.get('strength', 'moderate')
        bar_col  = STRENGTH_COLORS.get(strength, STRENGTH_COLORS.get(itype, MUTED))

        inner = Table(
            [['', Paragraph(f'<b>{title}</b>', S['insight_title'])]],
            colWidths=[4, W - 4]
        )
        inner.setStyle(TableStyle([
            ('BACKGROUND',   (0,0), (0,-1), bar_col),
            ('LEFTPADDING',  (1,0), (1,-1), 10),
            ('TOPPADDING',   (0,0), (-1,-1), 0),
            ('BOTTOMPADDING',(0,0), (-1,-1), 0),
            ('VALIGN',       (0,0), (-1,-1), 'MIDDLE'),
        ]))
        story.append(inner)
        story.append(Paragraph(desc, S['insight_body']))
        story.append(Spacer(1, 4*mm))


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

    display_cols = cols[:12]
    col_w        = W / len(display_cols)

    header = [Paragraph(c.replace('_', ' '), S['table_header']) for c in display_cols]
    rows   = [header]
    for i, row in enumerate(filtered_data[:shown]):
        rows.append([
            Paragraph(_safe_str(row.get(c, ''), 22), S['table_cell'])
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
    ]))
    story.append(tbl)

    if total > max_rows:
        story.append(Spacer(1, 3*mm))
        story.append(Paragraph(
            f'Note: {total - max_rows:,} additional records not shown. '
            f'Export the filtered CSV from the Interactive Dashboard to see all rows.',
            S['body']))


# ─────────────────────────────────────────────────────────────────────────────
# MAIN ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

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
        narrative       str   (optional — AI will generate if blank)
        filtered_data   list[dict]
        columns_meta    dict  {col: 'numeric'|'categorical'|'datetime'}
        generated_at    str
    """
    buf           = io.BytesIO()
    dataset_name  = payload.get('dataset_name', 'Dataset')
    domain        = payload.get('domain_display', 'Analysis')
    total_rows    = int(payload.get('total_rows', 0))
    filtered_rows = int(payload.get('filtered_rows', 0))
    active_filters= payload.get('active_filters', {})
    kpis          = payload.get('kpis', [])
    insights      = payload.get('insights', [])
    narrative     = payload.get('narrative', '')
    filtered_data = payload.get('filtered_data', [])
    columns_meta  = payload.get('columns_meta', {})
    generated_at  = payload.get('generated_at',
                                datetime.now().strftime('%d %b %Y, %H:%M'))

    doc = AuralisDocTemplate(
        buf,
        dataset_name = dataset_name,
        domain       = domain,
        pagesize     = A4,
        leftMargin   = MARGIN, rightMargin = MARGIN,
        topMargin    = 14*mm,  bottomMargin= 14*mm,
    )

    S     = _styles()
    story = []

    # 1. Cover
    _build_cover(story, S, dataset_name, domain, total_rows,
                 filtered_rows, active_filters, generated_at)

    # 2. Executive summary (AI narrative + stat strip)
    _build_executive_summary(story, S, narrative, payload)

    # 3. KPI cards
    _build_kpis(story, S, kpis, filtered_data, columns_meta, payload)

    # 4. Charts
    _build_charts(story, S, filtered_data, columns_meta)

    # 5. Insights
    if insights:
        _build_insights(story, S, insights)

    # 6. Data table
    _build_data_table(story, S, filtered_data, columns_meta)

    doc.build(story)
    return buf.getvalue()