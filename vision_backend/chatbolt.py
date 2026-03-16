from pypdf import PdfReader

pdf_text = ""

def load_pdf(file):
    global pdf_text
    reader = PdfReader(file)
    pdf_text = ""

    for page in reader.pages:
        pdf_text += page.extract_text()

    return "PDF loaded successfully"


def ask_question(question):
    global pdf_text

    if question.lower() in pdf_text.lower():
        return "The report contains information related to your question."

    else:
        return "Sorry, I could not find that information in the report."