"""
agents/pdf_generation_agent.py

QuoteFlow AI v3.0 — PDF Quote Generator & Alibaba Cloud OSS Uploader

Deterministic tool node — NO Qwen / LLM call happens here. Renders an
enterprise-grade quote PDF (branded header, QR code linking to the live
audit trail, visual confidence/risk badges, and a footer carrying the
document fingerprint), then attempts to upload it to Alibaba Cloud OSS.

If OSS is unavailable (e.g. pending Alibaba Cloud account billing/identity
verification, missing credentials, or network restrictions), this module
falls back to a 'local://' reference instead of raising — so the LangGraph
pipeline still completes end-to-end. backend/main.py exposes a single
unified endpoint (/api/v1/rfq/{task_id}/quote-pdf) that transparently
serves either storage backend, so the dashboard's "View Quote" link works
identically regardless of where the file actually lives.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import uuid
from datetime import datetime, timezone
from typing import Any

import oss2
import qrcode
import yaml
from reportlab.lib import colors
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    Image,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

# ============================================================
# Configuration
# ============================================================

_CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "config",
    "settings.yaml",
)

with open(_CONFIG_PATH, "r", encoding="utf-8") as _handle:
    _SETTINGS = yaml.safe_load(_handle)

_PDF_CFG = _SETTINGS["pdf_generation"]
_OSS_CFG = _SETTINGS["alibaba_cloud"]["oss"]
_ALIBABA_CFG = _SETTINGS["alibaba_cloud"]
_THRESHOLDS = _SETTINGS["thresholds"]

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_OUTPUT_DIR = os.path.join(_PROJECT_ROOT, _PDF_CFG["output_dir"])
_BACKEND_API_BASE_URL = os.environ.get("QUOTEFLOW_API_BASE_URL", "http://localhost:8000")

# Brand palette — matches dashboard.py's design system
_INK = colors.HexColor("#0F172A")
_MUTED = colors.HexColor("#64748B")
_ACCENT_BLUE = colors.HexColor("#3B82F6")
_GREEN = colors.HexColor("#16A34A")
_AMBER = colors.HexColor("#D97706")
_RED = colors.HexColor("#DC2626")
_TRACK_GRAY = colors.HexColor("#E2E8F0")
_PANEL_BG = colors.HexColor("#F8FAFC")
_BORDER = colors.HexColor("#E2E8F0")


class PDFGenerationError(Exception):
    """Raised when PDF rendering fails."""


class OSSUploadError(Exception):
    """Raised when uploading the generated PDF to Alibaba Cloud OSS fails."""


# ============================================================
# Styles
# ============================================================

def _build_pdf_styles() -> dict[str, ParagraphStyle]:
    base_styles = getSampleStyleSheet()
    return {
        "wordmark": ParagraphStyle(
            "Wordmark", parent=base_styles["Normal"],
            fontName="Helvetica-Bold", fontSize=20, leading=24,
        ),
        "tagline": ParagraphStyle(
            "Tagline", parent=base_styles["Normal"],
            fontName="Helvetica", fontSize=8.5, textColor=_MUTED, leading=11,
        ),
        "doc_title": ParagraphStyle(
            "DocTitle", parent=base_styles["Normal"],
            fontName="Helvetica-Bold", fontSize=13, textColor=_INK, leading=16,
        ),
        "qr_caption": ParagraphStyle(
            "QRCaption", parent=base_styles["Normal"],
            fontName="Helvetica", fontSize=6.5, textColor=_MUTED,
            leading=8, alignment=1,
        ),
        "section_header": ParagraphStyle(
            "SectionHeader", parent=base_styles["Heading2"],
            fontName="Helvetica-Bold", fontSize=12, textColor=_INK,
            spaceBefore=16, spaceAfter=8,
        ),
        "body": ParagraphStyle(
            "Body", parent=base_styles["Normal"],
            fontName="Helvetica", fontSize=9.5, textColor=colors.HexColor("#1E293B"),
            leading=14,
        ),
        "footer_label": ParagraphStyle(
            "FooterLabel", parent=base_styles["Normal"],
            fontName="Helvetica-Bold", fontSize=6.5, textColor=_MUTED, leading=9,
        ),
        "footer_value": ParagraphStyle(
            "FooterValue", parent=base_styles["Normal"],
            fontName="Courier", fontSize=6.5, textColor=colors.HexColor("#334155"), leading=9,
        ),
        "footer_brand": ParagraphStyle(
            "FooterBrand", parent=base_styles["Normal"],
            fontName="Helvetica-Oblique", fontSize=7.5, textColor=_MUTED,
            alignment=1, leading=10,
        ),
    }


# ============================================================
# Visual helpers — QR code, progress bars, tier badges
# ============================================================

def _generate_qr_image(task_id: str) -> Image:
    """
    Generate an in-memory QR code pointing to the task's live status/audit
    trail endpoint, and return it as a reportlab flowable Image.
    """
    target_url = f"{_BACKEND_API_BASE_URL}/api/v1/rfq/{task_id}/status"
    qr = qrcode.QRCode(box_size=6, border=1)
    qr.add_data(target_url)
    qr.make(fit=True)
    pil_image = qr.make_image(fill_color="#0F172A", back_color="white")

    buffer = io.BytesIO()
    pil_image.save(buffer, format="PNG")
    buffer.seek(0)
    return Image(buffer, width=0.85 * inch, height=0.85 * inch)


def _risk_tier(risk_score: float | None) -> tuple[str, colors.Color]:
    if risk_score is None:
        return "N/A", _MUTED
    if risk_score >= _THRESHOLDS["risk_threshold"]:
        return "HIGH RISK", _RED
    if risk_score >= _THRESHOLDS["risk_threshold"] * 0.6:
        return "MODERATE RISK", _AMBER
    return "LOW RISK", _GREEN


def _confidence_tier(confidence_score: float | None) -> tuple[str, colors.Color]:
    if confidence_score is None:
        return "N/A", _MUTED
    if confidence_score >= _THRESHOLDS["confidence_threshold"]:
        return "HIGH CONFIDENCE", _GREEN
    if confidence_score >= _THRESHOLDS["confidence_threshold"] * 0.7:
        return "MODERATE CONFIDENCE", _AMBER
    return "LOW CONFIDENCE", _RED


def _build_score_row(label: str, score: float | None, tier_label: str, tier_color: colors.Color) -> Table:
    """
    Build a single visual score row: label · mini progress bar · percentage
    · colored tier badge — used for both confidence and risk display.
    """
    bar_total_width = 1.8 * inch
    pct = max(0.0, min(1.0, score)) if score is not None else 0.0
    filled_width = bar_total_width * pct
    empty_width = bar_total_width - filled_width

    if filled_width <= 0:
        bar_row = [[""]]
        bar_widths = [bar_total_width]
        bar_colors = [_TRACK_GRAY]
    elif empty_width <= 0:
        bar_row = [[""]]
        bar_widths = [bar_total_width]
        bar_colors = [tier_color]
    else:
        bar_row = [["", ""]]
        bar_widths = [filled_width, empty_width]
        bar_colors = [tier_color, _TRACK_GRAY]

    bar_table = Table(bar_row, colWidths=bar_widths, rowHeights=[7])
    bar_style_cmds = [("TOPPADDING", (0, 0), (-1, -1), 0), ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
                       ("LEFTPADDING", (0, 0), (-1, -1), 0), ("RIGHTPADDING", (0, 0), (-1, -1), 0)]
    for idx, bar_color in enumerate(bar_colors):
        bar_style_cmds.append(("BACKGROUND", (idx, 0), (idx, 0), bar_color))
    bar_table.setStyle(TableStyle(bar_style_cmds))

    score_text = f"{score:.0%}" if score is not None else "N/A"
    badge_cell = Table(
        [[tier_label]], colWidths=[1.3 * inch], rowHeights=[16],
    )
    badge_cell.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (0, 0), tier_color),
        ("TEXTCOLOR", (0, 0), (0, 0), colors.white),
        ("FONTNAME", (0, 0), (0, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (0, 0), 6.5),
        ("ALIGN", (0, 0), (0, 0), "CENTER"),
        ("VALIGN", (0, 0), (0, 0), "MIDDLE"),
    ]))

    row_table = Table(
        [[label, bar_table, score_text, badge_cell]],
        colWidths=[1.5 * inch, bar_total_width + 0.1 * inch, 0.55 * inch, 1.4 * inch],
    )
    row_table.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (0, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (0, 0), 9.5),
        ("TEXTCOLOR", (0, 0), (0, 0), _INK),
        ("FONTNAME", (2, 0), (2, 0), "Helvetica-Bold"),
        ("FONTSIZE", (2, 0), (2, 0), 9.5),
        ("TEXTCOLOR", (2, 0), (2, 0), _INK),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    return row_table


def _compute_document_hash(state: dict[str, Any]) -> str:
    """
    Compute a SHA-256 content fingerprint over the quote's core decision
    fields. This is not a hash of the PDF bytes (which would require a
    two-pass render) — it is a tamper-evident fingerprint of the underlying
    decision data, suitable for cross-referencing against audit_log.
    """
    fingerprint_payload = {
        "task_id": state.get("task_id"),
        "quote_amount": state.get("quote_amount"),
        "margin_pct": state.get("margin_pct"),
        "risk_score": state.get("risk_score"),
        "confidence_score": state.get("confidence_score"),
        "extracted_items": state.get("extracted_items"),
    }
    canonical = json.dumps(fingerprint_payload, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


# ============================================================
# PDF rendering
# ============================================================

def _render_quote_pdf(state: dict[str, Any], local_pdf_path: str, document_hash: str) -> None:
    """
    Render the final, enterprise-styled quote document to local_pdf_path.

    Raises:
        PDFGenerationError: if rendering fails for any reason.
    """
    try:
        styles = _build_pdf_styles()
        doc = SimpleDocTemplate(
            local_pdf_path, pagesize=LETTER,
            topMargin=0.6 * inch, bottomMargin=0.6 * inch,
            leftMargin=0.65 * inch, rightMargin=0.65 * inch,
        )

        elements: list[Any] = []
        currency = _PDF_CFG["currency_symbol"]
        task_id = state.get("task_id", "N/A")

        # ---------- Header: wordmark + tagline (left) / QR code (right) ----------
        wordmark_html = (
            f'<font color="{_INK.hexval()}">Quote</font>'
            f'<font color="{_ACCENT_BLUE.hexval()}">Flow</font>'
            f'<font color="{_INK.hexval()}"> AI</font>'
        )
        header_left = [
            Paragraph(wordmark_html, styles["wordmark"]),
            Spacer(1, 2),
            Paragraph("Autonomous B2B Quoting Platform · Official Price Quotation", styles["tagline"]),
        ]
        header_right = [
            _generate_qr_image(task_id),
            Spacer(1, 2),
            Paragraph("Scan for live<br/>audit trail", styles["qr_caption"]),
        ]
        header_table = Table([[header_left, header_right]], colWidths=[4.7 * inch, 1.35 * inch])
        header_table.setStyle(TableStyle([
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("ALIGN", (1, 0), (1, 0), "CENTER"),
            ("LEFTPADDING", (0, 0), (-1, -1), 0),
            ("RIGHTPADDING", (0, 0), (-1, -1), 0),
            ("TOPPADDING", (0, 0), (-1, -1), 0),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
        ]))
        elements.append(header_table)
        elements.append(Spacer(1, 8))

        accent_bar = Table([[""]], colWidths=[6.05 * inch], rowHeights=[3])
        accent_bar.setStyle(TableStyle([("BACKGROUND", (0, 0), (0, 0), _ACCENT_BLUE)]))
        elements.append(accent_bar)
        elements.append(Spacer(1, 18))

        # ---------- Quote metadata ----------
        generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        client_name = state.get("client_name") or "Valued Client"
        client_email = state.get("client_email") or "N/A"

        meta_table = Table(
            [
                ["Quote Reference", task_id],
                ["Client", client_name],
                ["Client Email", client_email],
                ["Generated On", generated_at],
            ],
            colWidths=[1.5 * inch, 4.55 * inch],
        )
        meta_table.setStyle(TableStyle([
            ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
            ("FONTNAME", (1, 0), (1, -1), "Helvetica"),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
            ("TEXTCOLOR", (0, 0), (0, -1), _MUTED),
            ("TEXTCOLOR", (1, 0), (1, -1), _INK),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ("TOPPADDING", (0, 0), (-1, -1), 5),
        ]))
        elements.append(meta_table)
        elements.append(Spacer(1, 6))

        # ---------- Line items ----------
        elements.append(Paragraph("Requested Line Items", styles["section_header"]))
        extracted_items = state.get("extracted_items", [])
        item_rows = [["SKU", "Description", "Qty"]]
        for item in extracted_items:
            item_rows.append([
                str(item.get("item_sku", "N/A")),
                str(item.get("description", "N/A")),
                str(item.get("quantity", "N/A")),
            ])

        items_table = Table(item_rows, colWidths=[1.75 * inch, 3.5 * inch, 0.8 * inch])
        items_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), _INK),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
            ("ALIGN", (2, 0), (2, -1), "CENTER"),
            ("GRID", (0, 0), (-1, -1), 0.5, _BORDER),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, _PANEL_BG]),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("TOPPADDING", (0, 0), (-1, -1), 8),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
            ("LEFTPADDING", (0, 0), (-1, -1), 10),
        ]))
        elements.append(items_table)
        elements.append(Spacer(1, 6))

        # ---------- Pricing summary ----------
        elements.append(Paragraph("Pricing Summary", styles["section_header"]))
        quote_amount = state.get("quote_amount", 0.0)
        margin_pct = state.get("margin_pct", 0.0)

        amount_table = Table(
            [["Total Quote Amount", f"{currency}{quote_amount:,.2f}"],
             ["Blended Margin", f"{margin_pct * 100:.2f}%"]],
            colWidths=[2.3 * inch, 3.75 * inch],
        )
        amount_table.setStyle(TableStyle([
            ("FONTNAME", (0, 0), (0, -1), "Helvetica"),
            ("TEXTCOLOR", (0, 0), (0, -1), _MUTED),
            ("FONTSIZE", (0, 0), (-1, -1), 9.5),
            ("FONTNAME", (1, 0), (1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (1, 0), (1, 0), 15),
            ("TEXTCOLOR", (1, 0), (1, 0), _INK),
            ("FONTNAME", (1, 1), (1, 1), "Helvetica-Bold"),
            ("TEXTCOLOR", (1, 1), (1, 1), _INK),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ("TOPPADDING", (0, 0), (-1, -1), 6),
        ]))
        elements.append(amount_table)
        elements.append(Spacer(1, 10))

        # ---------- Visual confidence / risk indicators ----------
        risk_score = state.get("risk_score")
        confidence_score = state.get("confidence_score")
        risk_tier_label, risk_tier_color = _risk_tier(risk_score)
        confidence_tier_label, confidence_tier_color = _confidence_tier(confidence_score)

        elements.append(_build_score_row("AI Confidence", confidence_score, confidence_tier_label, confidence_tier_color))
        elements.append(Spacer(1, 4))
        elements.append(_build_score_row("Risk Assessment", risk_score, risk_tier_label, risk_tier_color))
        elements.append(Spacer(1, 14))

        # ---------- Reasoning ----------
        reasoning_summary = state.get("reasoning_summary")
        if reasoning_summary:
            elements.append(Paragraph("Assessment Notes", styles["section_header"]))
            elements.append(Paragraph(reasoning_summary, styles["body"]))
            elements.append(Spacer(1, 12))

        # ---------- Footer ----------
        footer_rule = Table([[""]], colWidths=[6.05 * inch], rowHeights=[1])
        footer_rule.setStyle(TableStyle([("BACKGROUND", (0, 0), (0, 0), _BORDER)]))
        elements.append(footer_rule)
        elements.append(Spacer(1, 8))

        footer_meta = Table(
            [
                [Paragraph("GENERATED", styles["footer_label"]), Paragraph("AUDIT ID", styles["footer_label"])],
                [Paragraph(generated_at, styles["footer_value"]), Paragraph(task_id, styles["footer_value"])],
                [Paragraph("DOCUMENT HASH (SHA-256)", styles["footer_label"]), ""],
                [Paragraph(document_hash, styles["footer_value"]), ""],
            ],
            colWidths=[3.6 * inch, 2.45 * inch],
        )
        footer_meta.setStyle(TableStyle([
            ("SPAN", (0, 2), (1, 2)),
            ("SPAN", (0, 3), (1, 3)),
            ("TOPPADDING", (0, 0), (-1, -1), 1),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 1),
        ]))
        elements.append(footer_meta)
        elements.append(Spacer(1, 8))
        elements.append(Paragraph(
            "Powered by <b>QuoteFlow AI</b> · LangGraph Orchestration · Qwen2.5 Reasoning · Alibaba Cloud",
            styles["footer_brand"],
        ))

        doc.build(elements)
    except Exception as exc:  # noqa: BLE001 — surface as a clean domain error
        raise PDFGenerationError(f"[pdf_generation_agent] Failed to render PDF: {exc}") from exc


def _upload_to_oss(local_pdf_path: str, object_key: str) -> str:
    """Upload the generated PDF to Alibaba Cloud OSS and return a presigned URL."""
    access_key_id = os.environ.get(_ALIBABA_CFG["access_key_id_env"])
    access_key_secret = os.environ.get(_ALIBABA_CFG["access_key_secret_env"])

    if not access_key_id or not access_key_secret:
        raise OSSUploadError(
            "[pdf_generation_agent] Alibaba Cloud credentials are not set in the environment "
            f"({_ALIBABA_CFG['access_key_id_env']} / {_ALIBABA_CFG['access_key_secret_env']})"
        )

    try:
        auth = oss2.Auth(access_key_id, access_key_secret)
        bucket = oss2.Bucket(auth, f"https://{_OSS_CFG['endpoint']}", _OSS_CFG["bucket_name"])

        with open(local_pdf_path, "rb") as file_handle:
            bucket.put_object(object_key, file_handle, headers={"Content-Type": "application/pdf"})

        return bucket.sign_url("GET", object_key, _OSS_CFG["presigned_url_expiry_seconds"], slash_safe=True)
    except oss2.exceptions.OssError as exc:
        raise OSSUploadError(f"[pdf_generation_agent] OSS upload failed: {exc}") from exc
    except OSError as exc:
        raise OSSUploadError(f"[pdf_generation_agent] Could not read local PDF file for upload: {exc}") from exc


def generate_and_upload_quote(state: dict[str, Any]) -> str:
    """
    Render the final, enterprise-styled quote PDF and upload it to Alibaba
    Cloud OSS, returning a presigned URL. Falls back to a 'local://<path>'
    reference if OSS is unavailable, so the pipeline still completes.
    backend/main.py's /quote-pdf endpoint transparently serves either form.

    Args:
        state: The current AgentState dict. Must contain 'task_id', 'quote_amount'.

    Returns:
        A presigned HTTPS URL if OSS upload succeeds, otherwise a
        'local://<absolute_path>' reference.

    Raises:
        PDFGenerationError: if PDF rendering itself fails.
    """
    task_id = state.get("task_id")
    if not task_id:
        raise PDFGenerationError("[pdf_generation_agent] 'task_id' is required to generate a quote PDF")

    os.makedirs(_OUTPUT_DIR, exist_ok=True)

    file_name = f"quote_{task_id}_{uuid.uuid4().hex[:8]}.pdf"
    local_pdf_path = os.path.join(_OUTPUT_DIR, file_name)

    document_hash = _compute_document_hash(state)
    _render_quote_pdf(state, local_pdf_path, document_hash)

    object_key = f"{_OSS_CFG['quote_prefix']}{file_name}"

    try:
        return _upload_to_oss(local_pdf_path, object_key)
    except OSSUploadError as exc:
        print(f"[pdf_generation_agent] OSS upload unavailable, falling back to local reference: {exc}")
        return f"local://{os.path.abspath(local_pdf_path)}"