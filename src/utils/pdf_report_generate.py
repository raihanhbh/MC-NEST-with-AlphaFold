import streamlit as st
import openai
from fpdf import FPDF
from io import BytesIO
from PIL import Image
import base64
import os
import re
import unicodedata
from utils.constants import (
    VALID_TOKEN_KEY,
    OPENAI_TOKEN_KEY,
    SEQUENCE_INPUT_KEY,
    PDB_STRUCTURES_KEY,
    BEST_HYPOTHESIS_KEY,
    EVALUATION_RESULTS_KEY,
    FOXM1,
    MYC,
    RMSD
)

# -------------------- GPT Section Generators -------------------- #
def generate_section(openai_key, hypothesis, section_title):
    

    prompt = f"""
    You are a scientific researcher. Based on the following hypothesis, write the "{section_title}" section of a scientific research paper:

    Hypothesis: {hypothesis}

    Please structure the content appropriately, use formal scientific language, and keep the content focused.
    """

    client = openai.OpenAI(api_key=openai_key)
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7
    )

    return response.choices[0].message.content.strip()

# Section-specific wrappers
def generate_introduction(openai_key, hypothesis):
    return generate_section(openai_key, hypothesis, "Introduction")

def generate_background(openai_key, hypothesis):
    return generate_section(openai_key, hypothesis, "Background")

def generate_methodology(openai_key, hypothesis):
    return generate_section(openai_key, hypothesis, "Methodology")

def generate_results_analysis(openai_key, hypothesis):
    return generate_section(openai_key, hypothesis, "Result and Analysis")

def generate_conclusion(openai_key, hypothesis):
    return generate_section(openai_key, hypothesis, "Conclusion")

# -------------------- PDF Generation -------------------- #
def sanitize_text(text):
    return unicodedata.normalize("NFKD", text).encode("latin-1", "ignore").decode("latin-1")

class JournalPDF(FPDF):
    pdf_sections = [
        "Introduction",
        "Background",
        "Methodology",
        "Results and Analysis",
        "Conclusion"
    ]
    def __init__(self):
        super().__init__()
        self.set_margins(25.4, 15, 25.4)  # 1 inch margins on both sides

    def header(self):
        self.set_font("Times", 'B', 16)
        self.cell(0, 10, "AI as Co-Scientist: Collaborative AI Agents", ln=True, align='C')
        self.ln(5)

    def footer(self):
        self.set_y(-15)
        self.set_font("Times", 'I', 8)
        self.cell(0, 10, f"Page {self.page_no()}", align='C')

    def write_with_bold_markers(self, text):
        pattern = r'(\*\*[^*]+\*\*)'
        parts = re.split(pattern, text)
        for part in parts:
            if part.startswith('**') and part.endswith('**') and part[2:-2] in self.pdf_sections:
                pass 
            elif (part.startswith('#') or part.endswith('##')) and part[2:-2] in self.pdf_sections:
                pass 
            elif part.startswith('**') and part.endswith('**'):
                self.set_font("Times", 'B', 12)
                self.write(8, sanitize_text(part[2:-2]))
                self.set_font("Times", '', 12)
            else:
                self.write(8, sanitize_text(part))

    def chapter_body(self, body):
        self.set_font("Times", '', 12)
        for line in sanitize_text(body).split("\n"):
            self.set_x(self.l_margin)
            self.write_with_bold_markers(line)
            self.ln(8)
        self.ln()

# Insert structure images with captions
def insert_image_with_caption(pdf_obj, image_path, caption):
    if os.path.exists(image_path):
        pdf_obj.set_font("Times", 'I', 10)
        pdf_obj.ln(2)
        pdf_obj.cell(0, 6, caption, ln=True, align="C")
        pdf_obj.image(image_path, w=100)
        pdf_obj.ln(8)
    else:
        pdf_obj.set_font("Times", 'I', 10)
        pdf_obj.cell(0, 6, f"[Missing image: {caption}]", ln=True)

def insert_side_by_side_images(pdf_obj, img1_path, img2_path, caption1, caption2, image_width=80, spacing=10):
    if not (os.path.exists(img1_path) and os.path.exists(img2_path)):
        pdf_obj.set_font("Times", 'I', 10)
        pdf_obj.cell(0, 10, "[One or both images missing]", ln=True)
        return

    # Calculate x start position to center the combined width
    total_width = (2 * image_width) + spacing
    x_start = (pdf_obj.w - total_width) / 2
    y_start = pdf_obj.get_y()

    # First image
    pdf_obj.image(img1_path, x=x_start, y=y_start, w=image_width)

    # Second image
    pdf_obj.image(img2_path, x=x_start + image_width + spacing, y=y_start, w=image_width)

    # Move cursor below images
    pdf_obj.set_y(y_start + image_width + 5)

    # Add captions centered below each image
    pdf_obj.set_font("Times", 'I', 10)
    pdf_obj.set_x(x_start)
    pdf_obj.cell(image_width, 5, caption1, border=0, ln=0, align="C")
    pdf_obj.set_x(x_start + image_width + spacing)
    pdf_obj.cell(image_width, 5, caption2, border=0, ln=1, align="C")
    pdf_obj.ln(10)
        
def generate_pdf_documentation(intro, background, methodology, result_analysis, conclusion):
    section_titles = [
        ("Introduction", intro),
        ("Background", background),
        ("Methodology", methodology),
        ("Results and Analysis", result_analysis),
        ("Conclusion", conclusion)
    ]
    pdf = JournalPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    for title, content in section_titles:
        pdf.set_font("Times", 'B', 14)
        pdf.cell(0, 10, title, ln=True)
        pdf.ln(2)
        
        if title == "Methodology":
            pass
        elif title == "Results and Analysis":
            image_path1 = "/tmp/user_input_structure.png"
            image_path2 = "/tmp/synthetic_structure.png"
            insert_side_by_side_images(
                pdf,
                img1_path=image_path1,
                img2_path=image_path2,
                caption1="Figure 1: User Input Structure",
                caption2="Figure 2: Synthetic Structure"
            )
            pdf.chapter_body(content)
        else:
            pdf.chapter_body(content)
            # # Assuming the image is saved in a specific path
            # image_path = "/tmp/user_input_structure.png"
            # insert_image_with_caption(pdf, image_path, "Figure 1: User Input Structure")
            # image_path = "/tmp/synthetic_structure.png"
            # insert_image_with_caption(pdf, image_path, "Figure 2: Synthetic Structure")
            

    temp_pdf_path = "/tmp/protein_documentation.pdf"
    pdf.output(temp_pdf_path)
    with open(temp_pdf_path, "rb") as f:
        pdf_buffer = BytesIO(f.read())
    os.remove(temp_pdf_path)
    return pdf_buffer

# -------------------- Streamlit Integration -------------------- #
def display_pdf_download_link(pdf_buffer):
    b64 = base64.b64encode(pdf_buffer.read()).decode()
    href = f'<a href="data:application/pdf;base64,{b64}" download="Protein_Research_Documentation.pdf">📄 Download Research PDF</a>'
    st.markdown(href, unsafe_allow_html=True)
    pdf_buffer.seek(0)

# -------------------- Streamlit Button-Based UI -------------------- #
def run_full_pdf_generation_ui(openai_key, hypothesis):
    prompt_input_intro = st.text_input("Enter text for Ehancing Introduction prompt", value="", help="This text will Ehancing Introduction prompt")
    prompt_input_background = st.text_input("Enter text for Ehancing Background prompt", value="", help="This text will Ehancing Background prompt")
    prompt_input_methodology = st.text_input("Enter text for Ehancing Methodology prompt", value="", help="This text will Ehancing Methodology prompt")
    prompt_input_result_analysis = st.text_input("Enter text for Ehancing Result and Analysis prompt", value="", help="This text will Ehancing Result and Analysis prompt")
    prompt_input_conclusion = st.text_input("Enter text for Ehancing Conclusion prompt", value="", help="This text will Ehancing Conclusion prompt")
    
    if st.button("📄 Generate PDF Documentation"):
        with st.spinner("Generating all sections..."):
            intro = generate_introduction(openai_key, hypothesis + prompt_input_intro)
            background = generate_background(openai_key, hypothesis + prompt_input_background)
            methodology = generate_methodology(openai_key, hypothesis + prompt_input_methodology)
            result_analysis = generate_results_analysis(openai_key, hypothesis + prompt_input_result_analysis)
            conclusion = generate_conclusion(openai_key, hypothesis + prompt_input_conclusion)
            pdf_buffer = generate_pdf_documentation(
                intro, background, methodology,
                result_analysis, conclusion
            )
        st.success("📄 PDF Ready!")
        display_pdf_download_link(pdf_buffer)

# -------------------- Sample Usage in Streamlit -------------------- #
if __name__ == "__main__":
    st.title("Protein Hypothesis Documentation Generator")

    if "openai_key" not in st.session_state:
        st.session_state.openai_key = ""
    if "hypothesis" not in st.session_state:
        st.session_state.hypothesis = ""
    if "metrics_text" not in st.session_state:
        st.session_state.metrics_text = ""

    st.session_state.openai_key = st.text_input("Enter your OpenAI API Key", type="password", value=st.session_state.openai_key)
    st.session_state.hypothesis = st.text_area("Paste your generated hypothesis", value=st.session_state.hypothesis)
    st.session_state.metrics_text = st.text_area("Enter evaluation metrics summary", value=st.session_state.metrics_text)

    if all([st.session_state.openai_key, st.session_state.hypothesis, st.session_state.metrics_text]):
        run_full_pdf_generation_ui(
            st.session_state.openai_key,
            st.session_state.hypothesis
        )
    else:
        st.info("Please fill in all inputs to enable PDF generation.")
