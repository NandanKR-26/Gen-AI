import PyPDF2
import re
import sys

def extract_text_from_pdf(pdf_path):
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = ""
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + " "
        return text
    
def preprocess_text(text):
    if not text:
        return ""
    #GPT2 regex
    reg = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\S\p{L}\p{N}]+|\s+(?!\S)|\s+""")
    return re.findall(reg,text) 


def main():
    raw_text = extract_text_from_pdf("Data/CUDA_C_Programming_Guide.pdf")
    print(raw_text[:500])
    cleaned_text = preprocess_text(raw_text)
    print(cleaned_text[:500])


if __name__ == "__main__":
    main()