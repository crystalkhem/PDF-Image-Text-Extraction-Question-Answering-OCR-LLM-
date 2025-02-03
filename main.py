import cv2
import pytesseract
import fitz  # PyMuPDF
from transformers import pipeline
import os
import argparse
import re

# Set Tesseract path (Windows only)
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Argument parser
parser = argparse.ArgumentParser(description="Extract text from a multi-page PDF or images using Tesseract OCR and Hugging Face.")
parser.add_argument("question", type=str, help="Question to ask the extracted text.")
parser.add_argument("--pdf_file", type=str, help="Path to the PDF file if specified in --pdf.")
parser.add_argument("--img_folder", type=str, default="./", help="Folder containing images (PNG, JPG). Will process all images in folder.")
parser.add_argument("--pdf", type=bool, default=False, help="Specify if input is a PDF. If yes, it will be converted to JPG for OCR.")
parser.add_argument("--output_folder", type=str, default="output_folder", help="Folder to save extracted images and text.")
parser.add_argument("--dpi", type=int, default=300, help="DPI for converting PDF to images.")
args = parser.parse_args()

# Create output folder if it doesn't exist
os.makedirs(args.output_folder, exist_ok=True)

# Hold extracted text from all pages
full_text = ""

# Function to extract numerical values from filenames for correct sorting
def extract_number(filename):
    match = re.search(r"(\d+)", filename)
    return int(match.group()) if match else float("inf")  # Return a large number if no digits are found

# Function to process images to text
def image_to_text(page_path, page_num):
    img = cv2.imread(page_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    text = pytesseract.image_to_string(gray)

    # Save text for each page
    text_file_path = os.path.join(args.output_folder, f"page_{page_num}.txt")
    with open(text_file_path, "w", encoding="utf-8") as f:
        f.write(text)

    return f"\n--- Page {page_num} ---\n{text}"

# Process PDF and ensure correct page order
if args.pdf:
    pdf = fitz.open(args.pdf_file)  
    for page_num in range(len(pdf)):  # Ensuring correct page order
        page = pdf.load_page(page_num)  # Load page by index
        pix = page.get_pixmap(matrix=fitz.Matrix(args.dpi / 72, args.dpi / 72))
        
        # Save image with proper page numbering
        page_path = os.path.join(args.output_folder, f'page_{page_num + 1}.jpg')
        pix.pil_save(page_path)

        # Extract text and append
        full_text += image_to_text(page_path, page_num + 1)

else:
    # Get all image files, sort them numerically
    image_files = sorted(
        [f for f in os.listdir(args.img_folder) if f.lower().endswith((".png", ".jpg", ".jpeg", ".tiff"))],
        key=extract_number
    )  

    # Ensure sequential page numbering
    for page_num, filename in enumerate(image_files, start=1):  
        page_path = os.path.join(args.img_folder, filename)
        full_text += image_to_text(page_path, page_num)

# Save full extracted text to a file
full_text_file = os.path.join(args.output_folder, "full_extracted_text.txt")
with open(full_text_file, "w", encoding="utf-8") as f:
    f.write(full_text)

print("\nExtracted text saved in:", args.output_folder)

# Load Hugging Face QA model
qa_model = pipeline("question-answering", model="deepset/roberta-large-squad2")

# Ask a question using the extracted text
response = qa_model(question=args.question, context=full_text)

print("\nAnswer:", response["answer"])
