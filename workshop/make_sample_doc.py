"""
Generates sample_document.pdf — a synthetic, fictional student handbook for the
AI Society (AIS) at a fictional university. Nothing here is a real institution's
document, so it is safe to redistribute publicly.

Run:  python make_sample_doc.py
"""
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.enums import TA_LEFT

SECTIONS = [
    ("AI Society Member Handbook", None),
    ("About the AI Society",
     "The AI Society (AIS) is a student-run organization dedicated to helping "
     "members learn, build, and ship artificial intelligence projects. "
     "Membership is open to all students regardless of major or prior experience. "
     "The Society runs weekly workshops, a semester-long project track, and a "
     "mentorship program pairing newer members with experienced builders. "
     "Our mission is to make hands-on AI education accessible and practical."),
    ("Membership and Dues",
     "Standard membership costs 15 dollars per semester and includes access to "
     "all workshops, the project track, and the members-only Discord server. "
     "Dues are waived for members who volunteer as workshop assistants for at "
     "least three sessions in a semester. Membership renews each semester and "
     "does not carry over automatically; members must re-register at the start "
     "of the fall and spring terms."),
    ("Workshops",
     "Workshops run every Tuesday from 6 to 8 PM in the Engineering building, "
     "room 210. Each workshop is self-contained, so members can attend any "
     "session without having attended previous ones. Topics rotate across "
     "machine learning fundamentals, retrieval-augmented generation, model "
     "deployment, and applied projects. Workshop recordings are posted to the "
     "Discord server within 48 hours for members who cannot attend live."),
    ("The Project Track",
     "The project track is a semester-long program where teams of three to five "
     "members build and ship a working AI application. Teams meet weekly and "
     "present a demo at the end-of-semester showcase. Members who complete the "
     "project track receive a certificate and are eligible to apply for the "
     "Society's summer research stipend. Project ideas must be approved by a "
     "faculty advisor before the fourth week of the semester."),
    ("Code of Conduct",
     "All members are expected to treat one another with respect. Harassment, "
     "discrimination, and plagiarism are not tolerated and may result in removal "
     "from the Society. Members must credit external code, datasets, and models "
     "used in their projects. Violations should be reported to any officer or to "
     "the faculty advisor. Reports are handled confidentially."),
    ("Officer Elections",
     "Officer elections are held once per year during the last workshop of the "
     "spring semester. Any member in good standing who has attended at least six "
     "workshops during the academic year is eligible to run. Officers serve a "
     "one-year term and may be re-elected once. Open positions include President, "
     "Vice President, Treasurer, and Workshop Coordinator."),
    ("Contact and Resources",
     "Members can reach officers through the Discord server or by email at "
     "ais-officers@example.edu. The Society maintains a shared resource library "
     "of tutorials, datasets, and past project repositories, accessible from the "
     "members-only section of the Discord. New members should start with the "
     "onboarding channel, which links to setup guides and the current semester "
     "schedule."),
]

def build(path="sample_document.pdf"):
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle("HandbookTitle", parent=styles["Title"], fontSize=22, spaceAfter=18)
    h_style = ParagraphStyle("HandbookH", parent=styles["Heading2"], fontSize=14, spaceBefore=14, spaceAfter=6, alignment=TA_LEFT)
    body_style = ParagraphStyle("HandbookBody", parent=styles["BodyText"], fontSize=11, leading=16, spaceAfter=8)
    doc = SimpleDocTemplate(path, pagesize=LETTER, topMargin=1*inch, bottomMargin=1*inch, leftMargin=1*inch, rightMargin=1*inch, title="AI Society Member Handbook")
    story = []
    for i, (heading, body) in enumerate(SECTIONS):
        if body is None:
            story.append(Paragraph(heading, title_style))
            story.append(Paragraph("A fictional handbook used for the Modern RAG in Practice workshop.", body_style))
            story.append(Spacer(1, 0.2*inch))
        else:
            story.append(Paragraph(heading, h_style))
            story.append(Paragraph(body, body_style))
    doc.build(story)
    print("wrote", path)

if __name__ == "__main__":
    build()
