from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

vectorstore = None


def load_pdf(file_path):

    loader = PyPDFLoader(file_path)
    documents = loader.load()

    text_splitter = CharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    docs = text_splitter.split_documents(documents)

    if len(docs) == 0:
        return "No readable text found in the PDF."

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    global vectorstore
    vectorstore = FAISS.from_documents(docs, embeddings)

    return "PDF processed successfully."

def ask_question(question):

    global vectorstore

    if vectorstore is None:
        return "Please upload a PDF first."

    docs = vectorstore.similarity_search(question, k=1)

    context = docs[0].page_content.lower()

    # detect disease
    disease = None
    if "glaucoma" in context:
        disease = "glaucoma"
    elif "cataract" in context:
        disease = "cataract"
    elif "retina" in context:
        disease = "retina disease"
    elif "normal" in context:
        disease = "normal"

    # detect risk
    risk = None
    if "high risk" in context:
        risk = "high"
    elif "low risk" in context:
        risk = "low"

    question = question.lower()

    # explanation request
    if "explain" in question or "problem" in question:

        if disease == "glaucoma":
            explanation = (
                "The report indicates that you may have glaucoma. "
                "Glaucoma is an eye disease that damages the optic nerve and can lead to vision loss if not treated early."
            )

        elif disease == "cataract":
            explanation = (
                "The report suggests cataract, which means the lens of the eye becomes cloudy and affects vision."
            )

        elif disease == "retina disease":
            explanation = (
                "The report indicates a possible retinal disease, which affects the retina at the back of the eye."
            )

        else:
            explanation = "The report indicates that your eyes appear normal."

        # add recommendation
        if risk == "high":
            explanation += " The analysis shows a high risk level, so it is strongly recommended that you consult an ophthalmologist as soon as possible."

        return explanation

    # normal questions
    if "risk" in question:
        if risk == "high":
            return "Risk Level: High Risk"
        elif risk == "low":
            return "Risk Level: Low Risk"

    if "confidence" in question:
        import re
        match = re.search(r"\d+\.\d+%", context)
        if match:
            return f"Confidence Level: {match.group()}"

    if "disease" in question or "condition" in question:
        return f"Detected Condition: {disease.capitalize()}"

    if "recommend" in question or "treatment" in question:
        return "It is recommended to consult an ophthalmologist for a proper medical examination."

    return "The information is available in the report but could not be extracted."
