import os
from PyPDF2 import PdfReader


def extract_file_text(filepath, filename):
    ext = os.path.splitext(filename)[1].lower()

    if ext == '.txt':
        with open(filepath, 'r', encoding='utf-8') as f:
            return f.read()

    elif ext == '.pdf':
        reader = PdfReader(filepath)
        return "\n".join(page.extract_text() for page in reader.pages if page.extract_text())


    elif ext == '.md' or ext == '.tex':
        with open(filepath, 'r', encoding='utf-8') as f:
            return f.read()


    else:
        raise ValueError(f"Unsupported file extension: {ext}")
