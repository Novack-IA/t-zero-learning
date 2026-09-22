#!/usr/bin/env python3
"""Render the assignment report (report.md + figures) into a PDF.

Plain layout: serif body text, one figure per question, A4, narrow margins.
The prose lives in scripts/dqn_assignment/report.md so it can be edited
without touching the layout code.
"""
import sys
from pathlib import Path

from reportlab.lib.enums import TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import Image, Paragraph, SimpleDocTemplate, Spacer

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC = Path(__file__).resolve().parent / "report.md"
FIG_DIR = REPO_ROOT / "runs" / "dqn_assignment" / "figures"
OUT = REPO_ROOT / "runs" / "dqn_assignment" / "relatorio_dqn.pdf"

BODY = ParagraphStyle("body", fontName="Times-Roman", fontSize=8.5, leading=9.8,
                      alignment=TA_JUSTIFY, spaceAfter=3)
H1 = ParagraphStyle("h1", fontName="Times-Bold", fontSize=13, leading=15, spaceAfter=4)
H2 = ParagraphStyle("h2", fontName="Times-Bold", fontSize=9.8, leading=12,
                    spaceBefore=6, spaceAfter=2)
SUB = ParagraphStyle("sub", parent=BODY, fontSize=8.0, leading=9.6, spaceAfter=1)
CAP = ParagraphStyle("cap", parent=BODY, fontSize=7.4, leading=9, alignment=0,
                     spaceBefore=1, spaceAfter=5)
MONO = ParagraphStyle("mono", parent=BODY, fontName="Courier", fontSize=7.2, leading=8.8,
                      alignment=0, spaceAfter=3)


def inline(text: str) -> str:
    """Minimal markdown inline conversion: **bold**, *italic*, `code`."""
    import re
    text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"`(.+?)`", r'<font face="Courier" size="7.6">\1</font>', text)
    text = re.sub(r"(?<!\*)\*([^*]+?)\*(?!\*)", r"<i>\1</i>", text)
    return text


def build():
    story = []
    for raw in SRC.read_text().split("\n"):
        line = raw.rstrip()
        if not line.strip():
            continue
        if line.startswith("!["):  # ![caption](figure-name)
            cap, _, rest = line[2:].partition("](")
            name = rest.rstrip(")")
            path = FIG_DIR / f"{name}.png"
            from reportlab.lib.utils import ImageReader
            iw, ih = ImageReader(str(path)).getSize()
            width = 16.3 * cm
            story.append(Image(str(path), width=width, height=width * ih / iw))
            if cap:
                story.append(Paragraph(inline(cap), CAP))
        elif line.startswith("# "):
            story.append(Paragraph(inline(line[2:]), H1))
        elif line.startswith("## "):
            story.append(Paragraph(inline(line[3:]), H2))
        elif line.startswith("> "):
            story.append(Paragraph(inline(line[2:]), MONO))
        elif line.startswith("_"):
            story.append(Paragraph(inline(line.strip("_")), SUB))
        else:
            story.append(Paragraph(inline(line), BODY))

    doc = SimpleDocTemplate(
        str(OUT), pagesize=A4,
        leftMargin=1.9 * cm, rightMargin=1.9 * cm,
        topMargin=1.3 * cm, bottomMargin=1.2 * cm,
        title="Relatorio DQN", author="Gustavo Novack",
    )
    doc.build(story)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    if not SRC.exists():
        sys.exit(f"missing {SRC}")
    build()
