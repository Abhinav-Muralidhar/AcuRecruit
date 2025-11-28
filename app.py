import streamlit as st
import json
import os
import re
from google import genai
from google.genai import types
from pydantic import BaseModel, Field
from typing import List, Optional
from dotenv import load_dotenv
import io
import pandas as pd
from datetime import datetime

# ReportLab imports for professional PDF generation
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_JUSTIFY, TA_CENTER, TA_LEFT

load_dotenv()

# --- Pydantic Data Model ---
class ExtractedResumeData(BaseModel):
    """Schema for extracting core information from a candidate's resume."""
    hard_skills: List[str] = Field(description="A list of 5-10 most relevant technical or hard skills.")
    key_projects: List[str] = Field(description="A list of 3-5 most impactful project titles or brief descriptions.")
    experience_summary: str = Field(description="A brief, 1-2 sentence summary of the candidate's professional background.")

# --- Configuration Constants ---
MAX_QUESTIONS = 10 

# --- Styling / CSS ---
st.set_page_config(page_title="AI Interview Agent", layout="wide")

CUSTOM_CSS = """
<style>
/* Make primary buttons blue and secondary green for clarity */
div.stButton > button[kind='primary']{ background-color:#0b6efd; color: white; border-radius:8px; padding:8px 14px;}
div.stButton > button[kind='secondary']{ background-color:#10b981; color: white; border-radius:8px; padding:8px 14px;}
/* Slightly nicer card-like container */
.reportcard { background:#ffffff; padding:16px; border-radius:10px; box-shadow:0 6px 18px rgba(15,23,42,0.06); }
/* Remove default sidebar styling for cleaner look */
section[data-testid='stSidebar'] { background: #f8fafc; }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# --- Utility Functions for Report Display ---
SECTION_METRIC_MAP = {
    1: 'technical',
    2: 'communication',
    3: 'cultural_fit'
}

def get_color_from_score(score_str):
    try:
        score = float(score_str.split('/')[0].strip())
        if score >= 9.0: return "#1e90ff"  # Blue (Excellent)
        elif score >= 7.0: return "#3cb371"  # Good
        elif score >= 5.0: return "#FFD700"  # Medium
        else: return "#dc143c"  # Low
    except:
        return "#6c757d"

def parse_report_and_extract_sw(full_response_text):
    """
    Parses the LLM response to separate the JSON data (Strengths/Weaknesses)
    from the readable Markdown report.
    """
    # 1. Extract JSON block for S&W
    json_match = re.search(r'\{.*"strengths":.*"weaknesses":.*\}', full_response_text, re.DOTALL | re.IGNORECASE)
    sw_data = {"strengths": [], "weaknesses": []}
    
    clean_report_text = full_response_text
    
    if json_match:
        json_str = json_match.group(0)
        try:
            sw_data = json.loads(json_str)
            clean_report_text = full_response_text.replace(json_str, "").strip()
        except:
            pass
            
    # 2. Extract Scores
    metrics = {}
    score_pattern = r"(Technical Score|Communication Score|Cultural Fit Score):\s*([\d\.]+\s*/\s*10)"
    for match in re.finditer(score_pattern, clean_report_text, re.IGNORECASE):
        key = match.group(1).replace(' Score', '').replace(' ', '_').lower()
        metrics[key] = match.group(2).strip()
    
    # Remove raw score lines from text display to avoid duplication
    clean_report_text = re.sub(score_pattern + r'\n?', '', clean_report_text, flags=re.IGNORECASE).strip()
    
    return metrics, clean_report_text, sw_data

def format_report_html(report_text, metrics):
    """Styles the markdown report with colors based on scores."""
    def colorize_section(section_index, section_title, content, metrics):
        metric_key = SECTION_METRIC_MAP.get(section_index, 'unknown')
        score = metrics.get(metric_key, 'N/A')
        color = get_color_from_score(score)
        num_score = score.split('/')[0].strip() if score != 'N/A' else '?'
        styled_heading = f"## <span style='color: {color};'>{section_title} (Rating: {num_score}/10)</span>"
        
        # Ensure bullet points are formatted cleanly
        styled_content = content
        if not content.lstrip().startswith(('*', '-')):
             styled_content = "\n* " + "\n* ".join([s.strip() for s in content.split('. ') if s.strip()])
             
        return f"{styled_heading}\n{styled_content}"

    sections = re.split(r'(\d\.\s*[A-Za-z\s]+)', report_text)
    final_output = []
    
    if sections[0].strip():
        final_output.append(sections[0])
        
    for i in range(1, len(sections), 2):
        section_heading = sections[i].strip()
        section_content = sections[i+1].strip() if i+1 < len(sections) else ""
        
        section_index_match = re.match(r'(\d)\.', section_heading)
        if section_index_match:
            section_index = int(section_index_match.group(1))
            pure_title = re.sub(r'^\d\.\s*', '', section_heading)
            pure_title = re.sub(r'\s*\(Score:.*?\)', '', pure_title).strip()
            final_output.append(colorize_section(section_index, pure_title, section_content, metrics))
        else:
            final_output.append(section_heading + '\n' + section_content)
            
    return "\n---\n".join(final_output)

# --- API and Utility Functions ---

def get_gemini_client():
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        st.error("Error: GEMINI_API_KEY not found.")
        return None
    return genai.Client(api_key=api_key)

def convert_file_to_parts(uploaded_file):
    if uploaded_file is None: return None
    return types.Part.from_bytes(data=uploaded_file.getvalue(), mime_type=uploaded_file.type)

def extract_resume_data(client, uploaded_file, role):
    with st.spinner("Analyzing resume..."):
        try:
            file_part = convert_file_to_parts(uploaded_file)
            prompt_text = "Extract core skills and experience for this job role."
            response = client.models.generate_content(
                model='gemini-2.5-flash',
                contents=[file_part, prompt_text],
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=ExtractedResumeData,
                    temperature=0.0
                )
            )
            parsed_data = json.loads(response.text)
            return ExtractedResumeData.model_validate(parsed_data)
        except Exception as e:
            st.error(f"Extraction error: {e}")
            return None

def generate_analysis_report(client, chat_history, role, resume_context_json, recruiter_personality=None):
    with st.spinner("Generating final analysis report..."):
        transcript_messages = chat_history[1:]
        transcript = "\n".join([f"{msg['role'].upper()}: {msg['content']}" for msg in transcript_messages])
        
        system_instruction = (
            "You are a Senior HR Analyst. Analyze the interview transcript.\n"
            f"Role: {role}. Resume Data: {json.dumps(resume_context_json)}.\n"
            "1. Output 3 lines EXACTLY like this:\n"
            "Technical Score: [X.X / 10]\nCommunication Score: [Y.Y / 10]\nCultural Fit Score: [Z.Z / 10]\n"
            "2. Provide 3 Detailed Sections (Markdown):\n"
            "1. Technical Assessment\n2. Behavioral Assessment\n3. Final Recommendation\n"
            "3. CRITICAL: At the very end, append a JSON block strictly for strengths/weaknesses:\n"
            '{"strengths": ["Item 1", "Item 2", "Item 3"], "weaknesses": ["Item 1", "Item 2", "Item 3"]}'
        )
        
        try:
            response = client.models.generate_content(
                model='gemini-2.5-flash',
                contents=[f"Transcript:\n{transcript}"],
                config=types.GenerateContentConfig(system_instruction=system_instruction, temperature=0.2)
            )
            return response.text
        except Exception as e:
            return f"Error: {e}"

# --- Professional PDF Generation (ReportLab) ---

def create_formal_pdf(report_text, metrics, sw_data, role, filename='interview_report.pdf'):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, rightMargin=72, leftMargin=72, topMargin=72, bottomMargin=18)
    
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name='Justify', alignment=TA_JUSTIFY))
    styles.add(ParagraphStyle(name='CenterTitle', parent=styles['Heading1'], alignment=TA_CENTER, spaceAfter=20))
    styles.add(ParagraphStyle(name='SubTitle', parent=styles['Heading2'], spaceAfter=10, textColor=colors.darkblue))
    
    story = []
    
    # 1. Header
    story.append(Paragraph("CONFIDENTIAL INTERVIEW EVALUATION", styles['CenterTitle']))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.black, spaceAfter=10))
    
    # 2. Meta Data
    date_str = datetime.now().strftime("%Y-%m-%d")
    meta_data = [
        [f"Role Applied: {role}", f"Date: {date_str}"],
        [f"Technical Score: {metrics.get('technical', 'N/A')}", f"Communication Score: {metrics.get('communication', 'N/A')}"],
    ]
    t = Table(meta_data, colWidths=[300, 200])
    t.setStyle(TableStyle([
        ('FONTNAME', (0,0), (-1,-1), 'Helvetica-Bold'),
        ('BOTTOMPADDING', (0,0), (-1,-1), 12),
    ]))
    story.append(t)
    story.append(Spacer(1, 20))
    
    # 3. Strengths & Weaknesses Table
    if sw_data and (sw_data.get('strengths') or sw_data.get('weaknesses')):
        story.append(Paragraph("Candidate Assessment Summary", styles['SubTitle']))
        
        # Balance lists to equal length for table display
        s_list = sw_data.get('strengths', [])
        w_list = sw_data.get('weaknesses', [])
        max_len = max(len(s_list), len(w_list))
        s_list += [''] * (max_len - len(s_list))
        w_list += [''] * (max_len - len(w_list))
        
        data = [['Strengths', 'Areas for Improvement']]
        for s, w in zip(s_list, w_list):
            data.append([Paragraph(f"• {s}", styles["BodyText"]) if s else "", 
                         Paragraph(f"• {w}", styles["BodyText"]) if w else ""])
            
        sw_table = Table(data, colWidths=[230, 230])
        sw_table.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#f1f5f9")),
            ('TEXTCOLOR', (0,0), (-1,0), colors.black),
            ('ALIGN', (0,0), (-1,-1), 'LEFT'),
            ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
            ('BOTTOMPADDING', (0,0), (-1,0), 12),
            ('GRID', (0,0), (-1,-1), 1, colors.lightgrey),
            ('VALIGN', (0,0), (-1,-1), 'TOP'),
        ]))
        story.append(sw_table)
        story.append(Spacer(1, 20))

    # 4. Main Narrative Report
    story.append(Paragraph("Detailed Evaluation", styles['SubTitle']))
    
    # Process text specifically to remove markdown bolding for PDF
    clean_text = report_text.replace('**', '').replace('##', '')
    
    for line in clean_text.split('\n'):
        if line.strip():
            # Check if it looks like a header (numbered list or short caps)
            if re.match(r'^\d\.', line) or (len(line) < 40 and line.isupper()):
                 story.append(Paragraph(line, styles['Heading3']))
            else:
                 story.append(Paragraph(line, styles['BodyText']))
            story.append(Spacer(1, 6))

    # 5. Footer
    story.append(Spacer(1, 30))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.lightgrey))
    story.append(Paragraph("Generated by AI Interview Agent - Internal Use Only", 
                           ParagraphStyle('Footer', parent=styles['Normal'], fontSize=8, textColor=colors.grey, alignment=TA_CENTER)))

    doc.build(story)
    buffer.seek(0)
    return buffer.read()

# --- Session / Reset Utilities ---

def reset_session():
    keys_to_delete = [
        "interview_started", "interview_finished", "messages", "client", 
        "current_question_count", "resume_context_json", "job_role", "analysis_complete",
        "uploaded_resume"
    ]
    for key in keys_to_delete:
        if key in st.session_state:
            del st.session_state[key]
    st.rerun()

# --- Start / Interview Logic ---

def start_analysis_and_store_data(role, uploaded_file):
    st.session_state.client = get_gemini_client()
    if not st.session_state.client: return
    
    st.session_state.job_role = role
    resume_data = extract_resume_data(st.session_state.client, uploaded_file, role)
    if resume_data:
        st.session_state.resume_context_json = resume_data.model_dump()
        st.session_state.analysis_complete = True

def start_interview(recruiter_personality=None):
    st.session_state.interview_started = True
    st.session_state.interview_finished = False
    st.session_state.current_question_count = 0
    
    system_instruction = (
        f"You are a professional interviewer for the {st.session_state.job_role} role. "
        f"Ask {MAX_QUESTIONS} questions total. One at a time. "
        "Do not output scores. Do not output 'Question X of Y'."
    )
    if recruiter_personality:
        system_instruction += f" Style: {recruiter_personality}."

    chat_config = types.GenerateContentConfig(system_instruction=system_instruction)
    st.session_state.chat = st.session_state.client.chats.create(model="gemini-2.5-flash", config=chat_config)
    
    initial_msg = f"Hello. I've reviewed your resume. Let's begin the interview for the {st.session_state.job_role} position. Please introduce yourself."
    st.session_state.messages = [{"role": "assistant", "content": initial_msg}]

# --- UI Layout ---

import streamlit as st
from PIL import Image

# Load the logo from the same folder
logo = Image.open("images/acuRecruit1.png")
favicon=Image.open("images/acuRecruit.png")

st.set_page_config(
    page_title="AcuRecruit",  # This is the tab title
    page_icon=favicon,             # Path to your favicon/logo
    layout="wide"
)

# Display it
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.image(logo, width=850)

st.divider()

st.title("🤖 AI Contextual Interview Agent")
st.caption("Automated Technical & Behavioral Evaluation")

# Session State Initialization
if "interview_started" not in st.session_state: st.session_state.interview_started = False
if "interview_finished" not in st.session_state: st.session_state.interview_finished = False
if "messages" not in st.session_state: st.session_state.messages = []
if "client" not in st.session_state: st.session_state.client = None
if "current_question_count" not in st.session_state: st.session_state.current_question_count = 0
if "analysis_complete" not in st.session_state: st.session_state.analysis_complete = False

# 1. SETUP VIEW
if not st.session_state.interview_started:
    st.header("1. Configuration")
    if st.button("↩️ Reset"): reset_session()

    with st.form("setup_form"):
        ROLE_OPTIONS = ["Software Engineer",
    "DevOps Developer",
    "Frontend Developer",
    "Backend Developer",
    "Full Stack Developer",
    "Mobile App Developer",
    "Data Scientist",
    "Machine Learning Engineer",
    "AI Researcher",
    "Data Analyst",
    "Data Engineer",
    "Cloud Engineer",
    "DevOps Engineer",
    "Cybersecurity Analyst",
    "IT Support Specialist",
    "Network Engineer",
    "Product Manager",
    "Project Manager",
    "Program Manager",
    "Business Analyst",
    "Scrum Master",
    "UI/UX Designer",
    "Graphic Designer",
    "Product Designer",
    "Marketing Manager",
    "Digital Marketing Specialist",
    "SEO Specialist",
    "Content Writer",
    "Copywriter",
    "Sales Executive",
    "Account Manager",
    "Customer Success Manager",
    "HR Specialist",
    "Recruiter",
    "Finance Analyst",
    "Accountant",
    "Operations Manager",
    "Supply Chain Manager",
    "Quality Assurance Engineer",
    "Test Automation Engineer",
    "Technical Support Engineer",
    "Solutions Architect",
    "Systems Administrator",
    "Database Administrator",
    "Research Analyst",
    "Clinical Research Associate",
    "Mechanical Engineer",
    "Electrical Engineer",
    "Civil Engineer",
    "Biomedical Engineer",
    "Teacher",
    "Trainer",
    "Consultant",
    "Legal Advisor",
    "Administrative Assistant",
    "Executive Assistant" ]
        selected_role = st.selectbox("Select Role", [""] + ROLE_OPTIONS)
        jd_text = st.text_area("Or Paste Job Description", height=100)
        uploaded_file = st.file_uploader("Upload Resume (PDF)", type=["pdf"])
        recruiter_style = st.selectbox("Interviewer Style", ["Default", "Strict", "Friendly"])
        
        if st.form_submit_button("Analyze Resume"):
            if (selected_role or jd_text) and uploaded_file:
                role = selected_role if selected_role else jd_text.split('\n')[0]
                start_analysis_and_store_data(role, uploaded_file)
            else:
                st.error("Missing Role or Resume.")

    if st.session_state.analysis_complete:
        st.success("Ready.")
        if st.button("🚀 Start Interview", type="primary"):
            start_interview(recruiter_style if recruiter_style != 'Default' else None)
            st.rerun()

# 2. FINAL REPORT VIEW
elif st.session_state.interview_finished:
    st.header(f"📋 Candidate Evaluation: {st.session_state.job_role}")
    if st.button("↩️ New Session"): reset_session()

    if st.session_state.client:
        # Generate raw response
        raw_response = generate_analysis_report(
            st.session_state.client, st.session_state.messages, 
            st.session_state.job_role, st.session_state.resume_context_json
        )
        
        # Parse response into metrics, text, and S&W table data
        metrics, clean_text, sw_data = parse_report_and_extract_sw(raw_response)
        
        # Display Top-Level Metrics
        col1, col2, col3 = st.columns(3)
        col1.metric("Technical Score", metrics.get('technical', 'N/A'))
        col2.metric("Communication Score", metrics.get('communication', 'N/A'))
        col3.metric("Cultural Fit Score", metrics.get('cultural_fit', 'N/A'))

        st.markdown("---")
        
        # Display Strengths & Weaknesses Table (Pandas)
        st.subheader("Key Findings")
        if sw_data and (sw_data.get('strengths') or sw_data.get('weaknesses')):
            # Normalize list lengths for dataframe
            s = sw_data.get('strengths', [])
            w = sw_data.get('weaknesses', [])
            max_len = max(len(s), len(w))
            s += [''] * (max_len - len(s))
            w += [''] * (max_len - len(w))
            
            df_sw = pd.DataFrame({'Strengths': s, 'Areas for Improvement': w})
            st.table(df_sw)
        
        # Display Formatted Text Report
        st.subheader("Detailed Analysis")
        formatted_html = format_report_html(clean_text, metrics)
        st.markdown(formatted_html, unsafe_allow_html=True)
        
        # Professional PDF Export
        st.markdown("---")
        pdf_bytes = create_formal_pdf(clean_text, metrics, sw_data, st.session_state.job_role)
        st.download_button("Download Official Report (PDF)", data=pdf_bytes, file_name="candidate_report.pdf", mime='application/pdf')
    
    st.balloons()

# 3. INTERVIEW VIEW
else:
    st.header(f"Interviewing for: {st.session_state.job_role}")
    
    chat_col, info_col = st.columns([0.7, 0.3])
    
    # --- RIGHT COLUMN: Candidate Info ---
    with info_col:
        st.markdown("### 👤 Candidate Profile")
        if 'resume_context_json' in st.session_state:
            data = st.session_state.resume_context_json
            with st.expander("Experience Summary", expanded=False):
                st.info(data.get('experience_summary', 'N/A'))
            with st.expander("Top Skills", expanded=False):
                skills = data.get('hard_skills', [])
                st.write(", ".join(skills) if skills else "No skills extracted.")
            with st.expander("Key Projects", expanded=False):
                projects = data.get('key_projects', [])
                if projects:
                    for p in projects:
                        st.markdown(f"- {p}")
                else:
                    st.write("No projects listed.")
        
        st.divider()
        display_count = min(st.session_state.current_question_count, MAX_QUESTIONS)
        st.metric("Questions Asked", f"{display_count} / {MAX_QUESTIONS}")

    # --- LEFT COLUMN: Chat Interface ---
    with chat_col:
        chat_container = st.container(height=340)
        
        # 1. Render Chat History
        with chat_container:
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])

        # 2. Check if Interview is FINISHED (Count > 10)
        # We use > (greater than) so the user can still answer when Count == 10
        if st.session_state.current_question_count > MAX_QUESTIONS:
            st.success("✅ Interview Complete.")
            if st.button("Generate Final Report", type="primary", use_container_width=True):
                st.session_state.interview_finished = True
                st.rerun()
        
        # 3. Chat Input Logic
        else:
            if prompt := st.chat_input("Type your answer here..."):
                st.session_state.messages.append({"role": "user", "content": prompt})
                with chat_container:
                    st.chat_message("user").markdown(prompt)

                # LOGIC CHECK: Is this the answer to the LAST question?
                if st.session_state.current_question_count == MAX_QUESTIONS:
                    # STOP AI GENERATION. Do not ask Question 11.
                    with st.spinner("Wrapping up interview..."):
                        final_msg = "Thank you! That was the final question. The interview is complete. Please click 'Generate Final Report' to view your results."
                        st.session_state.messages.append({"role": "assistant", "content": final_msg})
                        with chat_container:
                            st.chat_message("assistant").markdown(final_msg)
                        
                        st.session_state.current_question_count += 1
                        st.rerun()
                
                else:
                    with st.spinner("Interviewer is thinking..."):
                        try:
                            response = st.session_state.chat.send_message(prompt)
                            st.session_state.messages.append({"role": "assistant", "content": response.text})
                            with chat_container:
                                st.chat_message("assistant").markdown(response.text)
                            
                            st.session_state.current_question_count += 1
                            st.rerun()
                        except Exception as e:
                            st.error(f"API Error: {e}")

# Disclaimer
st.markdown("---")
st.caption("AI-Generated Content. Not a final hiring decision.")