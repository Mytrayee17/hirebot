import streamlit as st
import google.generativeai as genai
from dotenv import load_dotenv
import os
import re
import json
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import streamlit.components.v1 as components
from streamlit_lottie import st_lottie
import requests  # For fetching Lottie animation

# Download VADER lexicon for sentiment analysis (run once)
try:
    nltk.data.find('sentiment/vader_lexicon')
except LookupError:
    nltk.download('vader_lexicon', quiet=True)

# Initialize VADER sentiment analyzer
analyzer = SentimentIntensityAnalyzer()

# --- Configuration and Initialization ---

load_dotenv()

try:
    gemini_api_key = os.getenv("GOOGLE_API_KEY")
    if not gemini_api_key:
        st.error("Google API Key not found. Please set GOOGLE_API_KEY in your .env file.")
        st.stop()
    genai.configure(api_key=gemini_api_key)
except Exception as e:
    st.error(f"Error configuring Gemini API: {e}")
    st.stop()

model = genai.GenerativeModel('gemini-2.0-flash')


# --- Helper to load Lottie animation ---
def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()


# --- Custom CSS for UI Enhancements ---
# Update CSS to match Figma design
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    html, body, [class*="st-emotion-"] {
        font-family: 'Inter', sans-serif;
    }
    .stApp {
        background: linear-gradient(to right, #fdfbfb, #ebedee); /* Default gradient */
        padding: 1.5rem;
    }

    /* Welcome page specific background */
    body[data-st-page="welcome_page"] .stApp {
        background: url("data:image/jpeg;base64,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") no-repeat center center fixed;
        background-size: cover;
    }

    /* General Container Styling */
    .st-emotion-cache-1c7y2gy { /* Target for main container */
        padding: 2.5rem;
        border-radius: 16px;
        box-shadow: 0 8px 20px rgba(0, 0, 0, 0.15);
        background-color: white;
        max-width: 900px; /* Increased max-width for better 3-column layout */
        margin: auto;
    }

    /* Adjusted chat message bubbles */
    .st-emotion-cache-h4y6a0 { /* Chat message container */
        border-radius: 10px;
    }
    .st-emotion-cache-user-message { /* User message bubble */
        background-color: #e0f2f7; /* Light blue */
        border-radius: 15px 15px 3px 15px;
        padding: 12px 18px;
        margin-bottom: 10px;
        box-shadow: 0 2px 5px rgba(0, 0, 0, 0.08);
        max-width: 85%;
        margin-left: auto;
        text-align: right;
    }
    .st-emotion-cache-assistant-message { /* Assistant message bubble */
        background-color: #f0f3f6; /* Lighter blue/gray */
        border-radius: 15px 15px 15px 3px;
        padding: 12px 18px;
        margin-bottom: 10px;
        box-shadow: 0 2px 5px rgba(0, 0, 0, 0.08);
        max-width: 85%;
        text-align: left;
    }
    .stButton > button {
        border-radius: 10px;
        border: none;
        color: white;
        background-color: #2250F4; /* TalentScout Blue */
        padding: 0.85rem 1.5rem;
        font-size: 1.05rem;
        font-weight: 600;
        transition: all 0.3s ease-in-out;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
    }
    .stButton > button:hover {
        background-color: #88B1F9; /* Accent blue */
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.15);
        transform: translateY(-2px);
    }
    .st-emotion-cache-1v0mb9k { /* chat input container */
        border-radius: 12px;
        box-shadow: 0 3px 10px rgba(0, 0, 0, 0.1);
    }
    .st-emotion-cache-1rs6k2d { /* selectbox */
        border-radius: 10px;
        border: 1px solid #ced4da;
        box-shadow: inset 0 1px 3px rgba(0, 0, 0, 0.05);
    }
    .st-emotion-cache-1wmy9hp { /* text input */
        border-radius: 10px;
        border: 1px solid #EAEAEA; /* Light gray for forms */
    }


    /* Welcome Page Styles */
    .welcome-container {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        min-height: 80vh;
        text-align: center;
        background: linear-gradient(to bottom right, #fdfbfb, #ebedee); /* Light gradient */
        border-radius: 16px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
        padding: 3rem;
    }
    .welcome-container h1 {
        font-size: 3rem;
        color: #151A29; /* Dark text */
        margin-bottom: 0.5rem;
    }
    .welcome-container h3 {
        font-size: 1.5rem;
        color: #2250F4; /* TalentScout Blue */
        margin-bottom: 2rem;
    }

    /* Candidate Info Form Container */
    .candidate-form-container {
        background-color: #f8f9fa;
        border-radius: 12px;
        padding: 2.5rem;
        margin-bottom: 20px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
    }
    .candidate-form-container .stTextInput label,
    .candidate-form-container .stSelectbox label,
    .candidate-form-container .stFileUploader label {
        color: #151A29;
        font-weight: 600;
        margin-bottom: 0.5rem;
    }
    .candidate-form-container .stTextInput div input,
    .candidate-form-container .stSelectbox div[data-baseweb="select"] {
        border-radius: 8px;
        border: 1px solid #EAEAEA;
        padding: 0.75rem 1rem;
    }
    .candidate-form-container .stFileUploader button {
        background-color: #EAEAEA;
        color: #151A29;
        border: 1px solid #ced4da;
    }

    /* Progress Tracker (Info Gathering Page) */
    .progress-tracker-info-page {
        background-color: #F3F6FF;
        border: 2px solid #88B1F9;
        border-radius: 12px;
        padding: 15px 20px;
        margin-bottom: 20px; /* Space between tracker and form */
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
        text-align: center;
        font-family: 'Inter', sans-serif;
    }
    .progress-tracker-info-page h3 {
        color: #2250F4;
        margin-bottom: 5px;
        font-size: 1.3rem;
    }
    .progress-tracker-info-page p {
        margin: 0;
        font-size: 1rem;
        color: #151A29;
    }


    /* Chat Interface Layout (3 columns) */
    .chat-main-container {
        display: flex;
        flex-direction: row; /* Default for desktop */
        gap: 20px;
        margin-top: 20px;
    }
    .insights-panel, .candidate-summary-panel {
        flex: 1; /* Takes 1/3 of the space */
        background-color: #f8f9fa;
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 2px 10px rgba(0,0,0,0.05);
        min-height: 70vh; /* Match chat window height */
    }
    .chat-window-panel {
        flex: 2; /* Takes 2/3 of the space (middle column) */
        background-color: #ffffff;
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 2px 10px rgba(0,0,0,0.05);
        min-height: 70vh; /* Ensure it takes up vertical space */
        display: flex;
        flex-direction: column;
        justify-content: space-between;
    }
    .chat-messages-area {
        flex-grow: 1; /* Allow messages to fill space */
        overflow-y: auto; /* Scrollable chat history */
        padding-right: 10px; /* Space for scrollbar */
    }
    .chat-window-panel h3 {
        color: #151A29;
        margin-top: 0;
        margin-bottom: 1rem;
    }
    /* Specific styling for the interview progress container */
    .interview-panel {
        background-color: #F3F6FF; /* Accent color */
        border: 2px solid #88B1F9; /* Lighter accent */
        border-radius: 12px;
        padding: 20px;
        margin: 0px 0 20px 0;
        text-align: center;
        font-family: 'Inter', sans-serif;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
        animation: fadeIn 1s ease-out;
    }
    .interview-panel h3 {
        color: #2250F4; /* TalentScout Blue */
        margin-bottom: 10px;
        font-size: 1.5rem;
    }
    .interview-panel .status-indicator {
        display: inline-block;
        width: 18px;
        height: 18px;
        border-radius: 50%;
        margin-left: 10px;
        vertical-align: middle;
        animation: pulse 1.5s infinite ease-in-out;
    }
    .status-ready { background-color: #28a745; } /* Green */
    .status-inprogress { background-color: #ffc107; } /* Yellow */
    .status-completed { background-color: #007bff; } /* Blue */

    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    @keyframes pulse {
        0% { transform: scale(1); opacity: 0.8; }
        50% { transform: scale(1.1); opacity: 1; }
        100% { transform: scale(1); opacity: 0.8; }
    }
    /* Responsive adjustments */
    @media (max-width: 768px) {
        .st-emotion-cache-1c7y2gy {
            padding: 1.5rem;
            border-radius: 10px;
            box-shadow: 0 4px 10px rgba(0, 0, 0, 0.1);
        }
        .stApp {
            padding: 0.5rem;
        }
        .st-emotion-cache-user-message, .st-emotion-cache-assistant-message {
            max-width: 95%;
            padding: 10px 12px;
        }
        .welcome-container {
            padding: 2rem;
            min-height: 70vh;
        }
        .welcome-container h1 {
            font-size: 2.2rem;
        }
        .welcome-container h3 {
            font-size: 1.2rem;
        }
        .progress-tracker-info-page, .candidate-form-container {
            padding: 1rem;
        }
        .chat-main-container {
            flex-direction: column; /* Stack columns on mobile */
        }
        .chat-window-panel, .insights-panel, .candidate-summary-panel {
            min-height: auto; /* Allow height to adjust */
        }
        .chat-history-display {
            height: 40vh; /* Adjust for smaller screens */
        }
    }
    </style>
    """,
    unsafe_allow_html=True
)

# --- Session State Management (Crucial for Streamlit) ---

if "messages" not in st.session_state:
    st.session_state.messages = []

if "candidate_info" not in st.session_state:
    st.session_state.candidate_info = {
        "full_name": None,
        "email": None,
        "phone_number": None,
        "country_code": None,
        "years_experience": None,
        "desired_positions": None,
        "current_location": None,
        "tech_stack": [],
        "technical_questions_generated": [],
        "current_question_index": 0,
        "technical_Youtubes": {},
        "technical_answer_ai_detection": {},
        "technical_answer_sentiment": {},
        "tech_stack_to_question": {},
        "preferred_language": "English",
        "resume_uploaded": False,  # New: Track resume upload status
        "linkedin_profile": None,  # New: LinkedIn profile
        "current_company": None  # New: Current company
    }

if "page" not in st.session_state:
    st.session_state.page = "welcome"  # Controls which page is displayed

if "conversation_stage" not in st.session_state:
    st.session_state.conversation_stage = "greeting"  # Used within chatbot_interface for flow control

if "awaiting_elaboration" not in st.session_state:
    st.session_state.awaiting_elaboration = False
if "last_question_for_elaboration" not in st.session_state:
    st.session_state.last_question_for_elaboration = None

COUNTRY_CODES = [
    "+1 (USA/Canada)", "+44 (UK)", "+91 (India)", "+61 (Australia)",
    "+49 (Germany)", "+33 (France)", "+81 (Japan)", "+86 (China)",
    "+55 (Brazil)", "+7 (Russia)", "+27 (South Africa)", "+34 (Spain)"
]


# --- Helper function to generate the custom interview panel HTML ---
def get_interview_panel_html(status_text, stage_text, status_class):
    return f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Interview Panel</title>
        <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap" rel="stylesheet">
        <style>
            /* This style block is for the specific HTML component, not global */
            .interview-panel {{
                background-color: #F3F6FF; /* Accent color */
                border: 2px solid #88B1F9; /* Lighter accent */
                border-radius: 12px;
                padding: 20px;
                margin: 0px 0 20px 0;
                text-align: center;
                font-family: 'Inter', sans-serif;
                box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
                animation: fadeIn 1s ease-out;
            }}
            .interview-panel h3 {{
                color: #2250F4; /* TalentScout Blue */
                margin-bottom: 10px;
                font-size: 1.5rem;
            }}
            .interview-panel .status-indicator {{
                display: inline-block;
                width: 18px;
                height: 18px;
                border-radius: 50%;
                margin-left: 10px;
                vertical-align: middle;
                animation: pulse 1.5s infinite ease-in-out;
            }}
            .status-ready {{ background-color: #28a745; }} /* Green */
            .status-inprogress {{ background-color: #ffc107; }} /* Yellow */
            .status-completed {{ background-color: #007bff; }} /* Blue */

            @keyframes fadeIn {{
                from {{ opacity: 0; transform: translateY(20px); }}
                to {{ opacity: 1; transform: translateY(0); }}
            }}
            @keyframes pulse {{
                0% {{ transform: scale(1); opacity: 0.8; }}
                50% {{ transform: scale(1.1); opacity: 1; }}
                100% {{ transform: scale(1); opacity: 0.8; }}
            }}
            @media (max-width: 768px) {{
                .interview-panel {{
                    padding: 15px;
                    margin: 0px 0 15px 0;
                }}
                .interview-panel h3 {{
                    font-size: 1.2rem;
                }}
                .interview-panel .status-indicator {{
                    width: 14px;
                    height: 14px;
                }}
            }}
        </style>
    </head>
    <body>
        <div class="interview-panel">
            <h3>Interview Status: <span id="current-status">{status_text}</span></h3>
            <span class="status-indicator {status_class}" id="status-dot"></span>
            <p>Current Stage: <span id="current-stage-text">{stage_text}</span></p>
        </div>
    </body>
    </html>
    """


# --- Chatbot Logic (modified to accept preferred language) ---

def get_gemini_response(prompt_or_history, is_history=True, generation_config=None, response_schema=None,
                        preferred_language="English"):
    language_instruction = f"Respond concisely and professionally, in {preferred_language}. "

    if is_history:
        formatted_history = [{"role": m["role"], "parts": [m["content"]]} for m in prompt_or_history]
        if formatted_history and formatted_history[-1]["role"] == "user":
            formatted_history[-1]["parts"][0] = language_instruction + formatted_history[-1]["parts"][0]
        else:
            formatted_history.append({"role": "user", "parts": [language_instruction]})
    else:
        formatted_history = [{"role": "user", "parts": [language_instruction + prompt_or_history]}]

    payload = {"contents": formatted_history}
    if response_schema:
        payload["generationConfig"] = {"responseMimeType": "application/json", "responseSchema": response_schema}
    if generation_config:
        if "generationConfig" in payload:
            payload["generationConfig"].update(generation_config)
        else:
            payload["generationConfig"] = generation_config

    try:
        response = model.generate_content(formatted_history, generation_config=payload.get("generationConfig"))

        if response.candidates:
            text_content = response.candidates[0].content.parts[0].text
            if response_schema:
                try:
                    return json.loads(text_content)
                except json.JSONDecodeError:
                    print(f"Warning: Expected JSON, but received non-JSON: {text_content}")
                    return text_content
            return text_content
        else:
            print(f"No candidates found in response: {response}")
            return "I apologize, I couldn't generate a response. Please try again."
    except Exception as e:
        print(f"Error calling Gemini API: {e}")
        return f"An error occurred while processing. Please try again. (Error: {e})"


# --- Helper Functions for Validation and Sentiment Analysis ---

def is_valid_email(email):
    return re.match(r"[^@]+@[^@]+\.[^@]+", email)


def is_valid_mobile_number(number):
    return re.match(r"^\d{7,15}$", number.replace(" ", "").replace("-", ""))


def is_valid_years_experience(years):
    try:
        y = int(years)
        return y >= 0
    except ValueError:
        return False


def analyze_sentiment(text):
    scores = analyzer.polarity_scores(text)
    if scores['compound'] >= 0.05:
        return "Positive 😊"
    elif scores['compound'] <= -0.05:
        return "Negative 😞"
    else:
        return "Neutral 😐"


# --- Page Rendering Functions ---

def welcome_page():
    lottie_robot = load_lottieurl("https://lottie.host/80e90924-f7b6-455b-b997-3d906151f158/xV2xM95m0S.json")

    st.markdown("<div class='welcome-container'>", unsafe_allow_html=True)
    st.markdown("<h1>TalentScout – Your Virtual Hiring Assistant</h1>", unsafe_allow_html=True)
    st.markdown("<h3>Where tech talent meets opportunity</h3>", unsafe_allow_html=True)

    if lottie_robot:
        st_lottie(lottie_robot, height=250, key="robot_lottie")

    if st.button("🚀 Start Application", key="start_application_button"):
        st.session_state.page = "candidate_info_collection"
        # Reset info for new application
        st.session_state.candidate_info = {
            "full_name": None, "email": None, "phone_number": None, "country_code": None,
            "years_experience": None, "desired_positions": None, "current_location": None,
            "tech_stack": [], "technical_questions_generated": [], "current_question_index": 0,
            "technical_Youtubes": {}, "technical_answer_ai_detection": {},
            "technical_answer_sentiment": {}, "tech_stack_to_question": {}, "preferred_language": "English",
            "resume_uploaded": False, "linkedin_profile": None, "current_company": None
        }
        st.session_state.messages = []  # Clear messages for new conversation
        st.session_state.conversation_stage = "greeting"  # Reset stage for new conversation
        st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)


def candidate_info_collection_page():
    # Progress Tracker (Top Left)
    st.markdown("""
    <div class="progress-tracker-info-page">
        <h3>📌 Progress Tracker</h3>
        <p><b>Step 1 of 3</b> – Info Gathering</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div class='candidate-form-container'>", unsafe_allow_html=True)
    st.subheader("Let’s get to know you")

    # Use a form for input collection
    with st.form("candidate_info_form"):
        col1, col2 = st.columns(2)

        with col1:
            full_name = st.text_input("Full Name", value=st.session_state.candidate_info["full_name"] or "",
                                      key="full_name_input")
            email = st.text_input("Email Address", value=st.session_state.candidate_info["email"] or "",
                                  key="email_input")

            # Phone Number with Country Code Selectbox
            phone_col1, phone_col2 = st.columns([0.4, 0.6])
            with phone_col1:
                # Find the index of the current country code, default to 0 if not found
                current_country_code_index = 0
                if st.session_state.candidate_info["country_code"] in COUNTRY_CODES:
                    current_country_code_index = COUNTRY_CODES.index(
                        st.session_state.candidate_info["country_code"]) + 1  # +1 because of "" option

                selected_country_code = st.selectbox(
                    "Country Code",
                    options=[""] + COUNTRY_CODES,
                    index=current_country_code_index,
                    key="country_code_selector_form"
                )
            with phone_col2:
                # Extract digits-only part of phone number if present
                current_phone_number_digits = ""
                if st.session_state.candidate_info["phone_number"] and " " in st.session_state.candidate_info[
                    "phone_number"]:
                    current_phone_number_digits = st.session_state.candidate_info["phone_number"].split(" ")[-1]
                elif st.session_state.candidate_info["phone_number"]:  # If no space, assume it's just digits
                    current_phone_number_digits = st.session_state.candidate_info["phone_number"]

                phone_number = st.text_input("Phone Number (digits only)", value=current_phone_number_digits,
                                             key="phone_number_input")

            years_experience = st.number_input("Years of Experience", min_value=0, max_value=50,
                                               value=st.session_state.candidate_info["years_experience"] if
                                               st.session_state.candidate_info["years_experience"] is not None else 0,
                                               key="years_experience_input")

        with col2:
            current_company = st.text_input("Current Company (Type 'N/A' if fresher)",
                                            value=st.session_state.candidate_info["current_company"] or "",
                                            key="current_company_input")
            desired_positions = st.text_input("Desired Position(s)",
                                              value=st.session_state.candidate_info["desired_positions"] or "",
                                              help="e.g., Software Engineer, Data Scientist",
                                              key="desired_positions_input")
            current_location = st.text_input("Current Location (City, Country)",
                                             value=st.session_state.candidate_info["current_location"] or "",
                                             key="current_location_input")
            tech_stack_input = st.text_input("Primary Tech Stack (comma-separated)",
                                             value=", ".join(st.session_state.candidate_info["tech_stack"]) or "",
                                             help="e.g., Python, React, AWS", key="tech_stack_input")
            linkedin_profile = st.text_input("LinkedIn Profile URL (Optional)",
                                             value=st.session_state.candidate_info["linkedin_profile"] or "",
                                             key="linkedin_profile_input")

            # Resume Upload placeholder
            uploaded_resume = st.file_uploader("Upload Resume", type=["pdf", "docx"], key="resume_uploader")
            if uploaded_resume:
                st.session_state.candidate_info["resume_uploaded"] = True
                st.info("Resume uploaded successfully! (Note: Actual resume parsing is not implemented in this demo.)")
            # If a resume was previously uploaded but no new file is selected, maintain the uploaded status
            elif st.session_state.candidate_info["resume_uploaded"]:
                st.success("Resume previously uploaded.")

        st.markdown("---")  # Separator before buttons
        submit_button = st.form_submit_button("Continue to Smart Screening →")

        if submit_button:
            # Basic validation
            validation_passed = True
            if not full_name:
                st.error("Please enter your full name.")
                validation_passed = False
            if not is_valid_email(email):
                st.error("Please enter a valid email address.")
                validation_passed = False
            if not selected_country_code:
                st.error("Please select your country code.")
                validation_passed = False
            if not is_valid_mobile_number(phone_number):
                st.error("Please enter a valid phone number (digits only).")
                validation_passed = False
            if years_experience is None or years_experience < 0:
                st.error("Please enter a valid number of years of experience.")
                validation_passed = False
            if not desired_positions:
                st.error("Please enter your desired position(s).")
                validation_passed = False
            else:  # Validate desired positions with LLM
                validation_prompt_position = f"""
                You are an AI assistant tasked with validating user input for the "Desired Position" field.
                Given the user's input, determine if it appears to be a reasonable and relevant job title or type of position.
                Respond only with "Valid" if the input is reasonable, or "Invalid" if it seems irrelevant, nonsensical, or clearly not a valid job title.
                Respond in {st.session_state.candidate_info["preferred_language"]}.
                Input: "{desired_positions}"
                Output:
                """
                validation_result_position = get_gemini_response(validation_prompt_position, is_history=False,
                                                                 preferred_language=st.session_state.candidate_info[
                                                                     "preferred_language"]).strip()
                if validation_result_position == "Invalid":
                    st.error(
                        "Please enter a valid desired job title or type of position (e.g., 'Software Engineer', 'Data Scientist').")
                    validation_passed = False

            if not current_location:
                st.error("Please enter your current location.")
                validation_passed = False

            # Validate tech stack with LLM
            parsed_tech_stack = []
            if tech_stack_input:
                tech_stack_prompt = f"""
                You are an expert AI assistant tasked with identifying and extracting all distinct technologies from a given text.
                A technology can be a programming language, framework, library, database, tool, or a specific concept/domain within tech.
                Parse the following text and return a *comma-separated list of ONLY the identified technologies*.
                Ensure that if a technology is mentioned, it is included. Do not include any conversational filler or extra sentences.
                If no clear technologies are identified, respond with 'None'.
                Respond in {st.session_state.candidate_info["preferred_language"]}.
                Text: {tech_stack_input.strip()}
                """
                parsed_tech_stack_raw = get_gemini_response(tech_stack_prompt, is_history=False,
                                                            preferred_language=st.session_state.candidate_info[
                                                                "preferred_language"]).strip()
                if parsed_tech_stack_raw and parsed_tech_stack_raw.lower() != 'none':
                    parsed_tech_stack = [t.strip() for t in parsed_tech_stack_raw.split(',') if t.strip()]

            if not parsed_tech_stack:  # If parsed_tech_stack is empty after LLM processing
                st.error("Please enter a valid list of technologies (e.g., Python, React, AWS).")
                validation_passed = False

            if validation_passed:
                st.session_state.candidate_info["full_name"] = full_name
                st.session_state.candidate_info["email"] = email
                st.session_state.candidate_info["country_code"] = selected_country_code
                st.session_state.candidate_info["phone_number"] = f"{selected_country_code} {phone_number}"
                st.session_state.candidate_info["years_experience"] = years_experience
                st.session_state.candidate_info["desired_positions"] = desired_positions
                st.session_state.candidate_info["current_location"] = current_location
                st.session_state.candidate_info["tech_stack"] = parsed_tech_stack  # Use parsed technologies
                st.session_state.candidate_info["linkedin_profile"] = linkedin_profile if linkedin_profile else None
                st.session_state.candidate_info[
                    "current_company"] = current_company if current_company and current_company.lower() != 'n/a' else 'N/A (Fresher)'

                # Transition to chat interface
                st.session_state.page = "chatbot_interface"
                st.session_state.conversation_stage = "start_screening"  # New stage to kick off chat
                st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)


def chatbot_interface():
    # Main layout for chat and candidate summary
    # 3-column layout: Answer Insights | Chat Window | Candidate Summary
    insights_col, chat_col, summary_col = st.columns([1, 2, 1])

    with insights_col:
        st.markdown("<div class='insights-panel'>", unsafe_allow_html=True)  # Start insights-panel
        # Answer Insights
        if st.session_state.candidate_info["technical_Youtubes"]:
            st.subheader("Answer Insights")
            st.markdown("---")
            num_questions_answered = len(st.session_state.candidate_info["technical_Youtubes"])
            if num_questions_answered > 0:
                for q_text, answer in st.session_state.candidate_info["technical_Youtubes"].items():
                    ai_detect = st.session_state.candidate_info["technical_answer_ai_detection"].get(q_text, 'N/A')
                    sentiment = st.session_state.candidate_info["technical_answer_sentiment"].get(q_text, 'N/A')

                    # Truncate question for display if too long
                    display_q = q_text.split(' - ')[-1]  # Get just the question part
                    if len(display_q) > 50:
                        display_q = display_q[:50] + "..."

                    st.markdown(f"*Q:* {display_q}")
                    st.markdown(f"*AI Detection:* {ai_detect}, *Sentiment:* {sentiment}")
                    st.markdown("---")
            else:
                st.markdown("No answer insights available yet.")
        else:
            st.subheader("Answer Insights")
            st.markdown("---")
            st.markdown("No answer insights available yet.")
        st.markdown("</div>", unsafe_allow_html=True)  # End insights-panel

    with chat_col:
        st.markdown("<div class='chat-window-panel'>", unsafe_allow_html=True)

        # Interview Status (no heading above it)
        current_status_text = "In Progress"
        current_stage_text = "Technical Assessment"
        current_status_class = "status-inprogress"
        components.html(get_interview_panel_html(current_status_text, current_stage_text, current_status_class),
                        height=180)

        # Technical Questions Progress
        st.subheader("Technical Questions Progress")
        num_questions_asked = len(st.session_state.candidate_info["technical_questions_generated"])
        num_questions_answered = len(st.session_state.candidate_info["technical_Youtubes"])
        st.markdown(f"*Answered:* {num_questions_answered} / {num_questions_asked}")

        # Progress bar
        if num_questions_asked > 0:
            progress_percentage = (num_questions_answered / num_questions_asked) * 100
            st.progress(progress_percentage / 100)
        else:
            st.progress(0)  # 0% if no questions asked yet
        st.markdown("---")  # Separator

        st.write("AI avatar + name “TalentBot”")  # Placeholder for AI avatar

        st.markdown("<div class='chat-messages-area'>", unsafe_allow_html=True)
        # Initial greeting from TalentBot if starting screening
        if st.session_state.conversation_stage == "start_screening":
            greeting_message = f"👋 Hi {st.session_state.candidate_info['full_name']}, thanks for applying! Let's dive into your tech expertise. I'll now ask you some technical questions based on your skills."
            st.session_state.messages.append({"role": "assistant", "content": greeting_message})
            st.session_state.conversation_stage = "generate_technical_questions"
            st.rerun()  # Rerun to display greeting and move to question generation

        # Display existing chat messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        st.markdown("</div>", unsafe_allow_html=True)  # End chat-messages-area

        # Input Box at the bottom of the chat window
        if st.session_state.conversation_stage != "conclude_interview" and st.session_state.conversation_stage != "ended":
            prompt_input = st.chat_input("Type your answer here...", key="chat_input")
            if prompt_input:
                st.session_state.messages.append({"role": "user", "content": prompt_input})
                with st.chat_message("user"):
                    st.markdown(prompt_input)

                sentiment_result = analyze_sentiment(prompt_input)  # Still perform sentiment analysis

                conversation_ending_keywords = ["bye", "exit", "quit", "thank you", "end conversation", "done",
                                                "finish", "stop"]
                if any(keyword in prompt_input.lower() for keyword in conversation_ending_keywords):
                    response_text = "Acknowledged. Transitioning to conclusion."
                    st.session_state.conversation_stage = "conclude_interview"

                with st.chat_message("assistant"):
                    with st.spinner("Thinking..."):
                        current_stage = st.session_state.conversation_stage
                        response_text = ""
                        lang = st.session_state.candidate_info["preferred_language"]

                        if current_stage == "generate_technical_questions":
                            st.session_state.candidate_info["technical_questions_generated"] = []
                            st.session_state.candidate_info["current_question_index"] = 0
                            st.session_state.candidate_info["tech_stack_to_question"] = {}

                            all_techs = st.session_state.candidate_info["tech_stack"]
                            years_exp = st.session_state.candidate_info["years_experience"]

                            if not all_techs:
                                response_text = get_gemini_response(
                                    "Inform user that no technologies were found for questions and suggest they go back to the info page to provide a tech stack, then restart the screening.",
                                    is_history=False, preferred_language=lang
                                )
                                st.session_state.conversation_stage = "ended"
                            else:
                                techs_to_process = all_techs[:5]  # Limit to 5 technologies for questions
                                for tech in techs_to_process:
                                    question_gen_prompt = f"""
                                    You are an AI Hiring Assistant for a tech recruitment agency.
                                    The candidate has {years_exp} years of experience.
                                    Based on the technology or concept "{tech}", generate 2-3 distinct, varied, and concise technical interview questions suitable for a candidate with {years_exp} years of experience.
                                    Ensure a good mix of conceptual, practical/scenario-based, and best-practice questions.
                                    Present them as a numbered list, one question per line. Do NOT include any introductory or concluding sentences or conversational filler.
                                    Respond in {lang}.

                                    Example for Python and 3 years experience:
                                    1. Explain decorator patterns in Python and provide a use case.
                                    2. How would you optimize a Python application for memory efficiency?
                                    """
                                    tech_questions_raw = get_gemini_response(question_gen_prompt, is_history=False,
                                                                             preferred_language=lang)
                                    cleaned_questions = []
                                    for q in tech_questions_raw.split('\n'):
                                        cleaned_q = re.sub(r"^\d+\.\s*", "", q).strip()
                                        if cleaned_q:
                                            cleaned_questions.append(cleaned_q)

                                    # Ensure at least 2 questions are generated
                                    while len(cleaned_questions) < 2:
                                        cleaned_questions.append(get_gemini_response(
                                            f"Generate a general question about {tech} for someone with {years_exp} years of experience.",
                                            is_history=False, preferred_language=lang))

                                    st.session_state.candidate_info["tech_stack_to_question"][tech] = cleaned_questions
                                    st.session_state.candidate_info["technical_questions_generated"].extend(
                                        [f"{tech}** - {q}" for q in cleaned_questions])

                                if st.session_state.candidate_info["technical_questions_generated"]:
                                    first_question = st.session_state.candidate_info["technical_questions_generated"][0]
                                    response_text = f"Great! Let's start with the technical questions.\n\nQuestion {st.session_state.candidate_info['current_question_index'] + 1}: {first_question}"
                                    st.session_state.conversation_stage = "ask_technical_questions"
                                else:
                                    response_text = get_gemini_response(
                                        "Inform user that technical questions could not be generated and to try again later.",
                                        is_history=False, preferred_language=lang
                                    )
                                    st.session_state.conversation_stage = "ended"

                        elif current_stage == "ask_technical_questions":
                            q_index = st.session_state.candidate_info["current_question_index"]
                            question_text = st.session_state.candidate_info["technical_questions_generated"][q_index]
                            candidate_answer = prompt_input.strip()

                            if st.session_state.awaiting_elaboration:
                                last_q_for_elaboration = st.session_state.last_question_for_elaboration
                                if last_q_for_elaboration and last_q_for_elaboration in st.session_state.candidate_info[
                                    "technical_Youtubes"]:
                                    st.session_state.candidate_info["technical_Youtubes"][
                                        last_q_for_elaboration] += "\n\n(Elaboration): " + candidate_answer
                                else:
                                    print(
                                        f"Warning: Elaboration received but last_question_for_elaboration was not found: {last_q_for_elaboration}")
                                    st.session_state.candidate_info["technical_Youtubes"][
                                        question_text] = candidate_answer  # Fallback to current question

                                st.session_state.awaiting_elaboration = False
                                st.session_state.last_question_for_elaboration = None

                                st.session_state.candidate_info["current_question_index"] += 1
                                next_q_index = st.session_state.candidate_info["current_question_index"]
                                total_questions = len(st.session_state.candidate_info["technical_questions_generated"])

                                if next_q_index < total_questions:
                                    next_question = st.session_state.candidate_info["technical_questions_generated"][
                                        next_q_index]
                                    response_text = get_gemini_response(
                                        "Acknowledge additional details and present the next question.",
                                        is_history=False, preferred_language=lang
                                    ) + f"\n\nQuestion {next_q_index + 1}: {next_question}"
                                else:
                                    response_text = get_gemini_response(
                                        "Acknowledge additional details and inform user that all technical questions are collected.",
                                        is_history=False, preferred_language=lang
                                    )
                                    st.session_state.conversation_stage = "conclude_interview"

                            else:  # Normal question answering flow
                                st.session_state.candidate_info["technical_Youtubes"][question_text] = candidate_answer
                                st.session_state.candidate_info["technical_answer_sentiment"][
                                    question_text] = analyze_sentiment(candidate_answer)

                                ai_detection_prompt = f"""
                                Analyze the following candidate's answer to a technical question. Determine if the answer appears to be generated by an AI (e.g., overly formal, generic, comprehensive without natural pauses/hesitations, sounds like a textbook definition) or if it exhibits human-like characteristics (e.g., conversational, potentially less structured, specific examples from experience, some natural imperfection).
                                Respond only with "AI-generated" or "Human-like".
                                Respond in {lang}.

                                Question: {question_text}
                                Candidate Answer: {candidate_answer}
                                """
                                ai_detection_result = get_gemini_response(ai_detection_prompt, is_history=False,
                                                                          preferred_language=lang).strip().replace('.',
                                                                                                                   '')
                                st.session_state.candidate_info["technical_answer_ai_detection"][
                                    question_text] = ai_detection_result

                                acknowledgment_prompt = f"""
                                Given the following technical question and a candidate's response, provide a very brief (1-2 sentences), neutral, and encouraging acknowledgment or transition phrase.
                                If the candidate's response seems brief, generic, or if it doesn't fully address the question, politely prompt them to "elaborate" or "provide more details" at the end of your acknowledgment.
                                Do NOT provide correct answers, evaluate the correctness of the response, or give away solutions. If the response is a clear non-answer (e.g., 'no', 'I don't know', 'skip', 'abc'), acknowledge that politely and suggest moving on.
                                Respond in {lang}.

                                Question: {question_text}
                                Candidate Response: {candidate_answer}

                                Your acknowledgment/transition:
                                """
                                acknowledgment = get_gemini_response(acknowledgment_prompt, is_history=False,
                                                                     preferred_language=lang).strip()

                                is_elaboration_request = "?" in acknowledgment or any(
                                    word in acknowledgment.lower() for word in
                                    ["elaborate", "further", "more details", "can you tell me"])

                                response_text_parts = [acknowledgment]
                                # Removed the AI Detection/Sentiment from here as requested
                                response_text_parts.append("\n---\n")

                                if is_elaboration_request:
                                    st.session_state.awaiting_elaboration = True
                                    st.session_state.last_question_for_elaboration = question_text
                                else:
                                    st.session_state.candidate_info["current_question_index"] += 1
                                    next_q_index = st.session_state.candidate_info["current_question_index"]
                                    total_questions = len(
                                        st.session_state.candidate_info["technical_questions_generated"])

                                    if next_q_index < total_questions:
                                        next_question = \
                                        st.session_state.candidate_info["technical_questions_generated"][next_q_index]
                                        response_text_parts.append(f"Question {next_q_index + 1}: {next_question}")
                                    else:
                                        response_text_parts.append(get_gemini_response(
                                            "Thank user for answering all technical questions and inform that all necessary information is collected.",
                                            is_history=False, preferred_language=lang
                                        ))
                                        st.session_state.conversation_stage = "conclude_interview"

                                response_text = "\n".join(response_text_parts)

                        elif st.session_state.conversation_stage == "conclude_interview":
                            # This block should typically be reached via the conversation_ending_keywords or after all questions are answered
                            # No user input is processed here, just transition to exit page
                            response_text = ""  # No response here, handled by page transition
                            st.session_state.page = "exit_page"
                            st.rerun()

                        else:
                            # Fallback for unexpected conversation stages
                            full_chat_history_for_llm = [{"role": m["role"], "parts": [m["content"]]} for m in
                                                         st.session_state.messages]
                            response_text = get_gemini_response(full_chat_history_for_llm, is_history=True,
                                                                preferred_language=lang)
                            if "sorry" in response_text.lower() or "understand" in response_text.lower():
                                response_text += "\n\n" + get_gemini_response(
                                    "Inform user chatbot is lost and ask to rephrase or tell what they want to do.",
                                    is_history=False, preferred_language=lang
                                )
                    st.markdown(response_text)
                    st.session_state.messages.append({"role": "assistant", "content": response_text})
                st.rerun()  # Rerun to update the UI based on prompt processing
        st.markdown("</div>", unsafe_allow_html=True)  # End chat-window-panel

    with summary_col:
        st.markdown("<div class='candidate-summary-panel'>", unsafe_allow_html=True)  # Start candidate-summary-panel
        st.subheader("Candidate Summary")
        st.markdown("---")

        info = st.session_state.candidate_info
        st.markdown(f"*Name:* {info['full_name'] if info['full_name'] else 'N/A'}")
        st.markdown(f"*Email:* {info['email'] if info['email'] else 'N/A'}")
        if info['linkedin_profile']:
            st.markdown(f"*LinkedIn:* [{info['linkedin_profile']}]({info['linkedin_profile']})")
        else:
            st.markdown(f"*LinkedIn:* N/A")
        st.markdown(f"*Company:* {info['current_company'] if info['current_company'] else 'N/A'}")
        st.markdown(
            f"*Experience:* {f'{info['years_experience']} years' if info['years_experience'] is not None else 'N/A'}")
        st.markdown(f"*Preferred Role:* {info['desired_positions'] if info['desired_positions'] else 'N/A'}")
        st.markdown(f"*Tech Stack:* {', '.join(info['tech_stack']) if info['tech_stack'] else 'N/A'}")

        if info['resume_uploaded']:
            st.markdown(":green[Resume Uploaded ✅]")
        else:
            st.markdown(":red[Resume Not Uploaded ❌]")

        st.markdown("</div>", unsafe_allow_html=True)  # End candidate-summary-panel

    st.markdown("</div>", unsafe_allow_html=True)  # End chat-main-container


def exit_page():
    st.markdown("<div class='welcome-container'>",
                unsafe_allow_html=True)  # Reusing welcome-container style for consistency
    st.markdown("<h1>🎉 Thank you!</h1>", unsafe_allow_html=True)
    full_name = st.session_state.candidate_info["full_name"] if st.session_state.candidate_info[
        "full_name"] else "Candidate"
    st.markdown(f"<h3>We've recorded your responses, {full_name}. Our recruiters will reach out shortly.</h3>",
                unsafe_allow_html=True)

    st.markdown("---")

    # Generate Hiring Recommendation Report
    with st.spinner("Generating hiring recommendation..."):
        info = st.session_state.candidate_info
        lang = info["preferred_language"]

        report_prompt = f"""
        You are an AI Hiring Manager. Based on the following candidate's profile and their performance in a technical screening, provide a concise hiring recommendation.
        Your recommendation should include:
        1. A clear "Hire", "Do Not Hire", or "Maybe" verdict.
        2. A brief justification for the verdict, considering:
           - Completeness and clarity of provided personal information.
           - Relevance of their experience and desired role to their tech stack.
           - Overall perceived quality and depth of their technical answers (DO NOT evaluate correctness, only perceived effort/engagement).
           - General sentiment from their technical answers.
           - Any red flags (e.g., consistently generic/AI-generated answers, lack of engagement).
           - Consideration of their years of experience and if the answers align with it.
        3. A summary of their strengths and areas for potential development based on the technical answers.

        Maintain a professional and objective tone.

        Candidate Information:
        Name: {info['full_name']}
        Email: {info['email']}
        Phone: {info['phone_number']}
        Current Company: {info['current_company']}
        Years of Experience: {info['years_experience']}
        Desired Positions: {info['desired_positions']}
        Location: {info['current_location']}
        Tech Stack: {', '.join(info['tech_stack'])}
        Resume Uploaded: {info['resume_uploaded']}
        LinkedIn Profile: {info['linkedin_profile'] if info['linkedin_profile'] else 'N/A'}

        Technical Questions and Answers:
        """
        for q, a in info["technical_Youtubes"].items():
            ai_detect = info["technical_answer_ai_detection"].get(q, 'N/A')
            sentiment = info["technical_answer_sentiment"].get(q, 'N/A')
            report_prompt += f"\n- Q: {q}\n  A: {a}\n  AI Detection: {ai_detect}, Sentiment: {sentiment}\n"

        all_sentiments = [s for s in info["technical_answer_sentiment"].values() if s != 'N/A']
        if all_sentiments:
            positive_count = all_sentiments.count("Positive 😊")
            negative_count = all_sentiments.count("Negative 😞")
            neutral_count = all_sentiments.count("Neutral 😐")
            report_prompt += f"\nOverall sentiment of technical answers: Positive ({positive_count}), Negative ({negative_count}), Neutral ({neutral_count})."
        else:
            report_prompt += "\nOverall sentiment of technical answers: Not enough data."

        report_prompt += "\n\nHiring Recommendation Report:"

        hiring_report = get_gemini_response(report_prompt, is_history=False, preferred_language=lang)
        st.markdown(f"### Hiring Recommendation:\n{hiring_report}")

    st.markdown("---")

    # Options buttons
    col1, col2 = st.columns(2)
    with col1:
        # Placeholder for download summary PDF
        st.button("📄 Download Summary PDF", key="download_summary_pdf")
    with col2:
        if st.button("🔄 Return to Home", key="return_to_home_exit"):
            st.session_state.page = "welcome"
            # Reset all session state for a fresh start
            st.session_state.messages = []
            st.session_state.candidate_info = {
                "full_name": None, "email": None, "phone_number": None, "country_code": None,
                "years_experience": None, "desired_positions": None, "current_location": None,
                "tech_stack": [], "technical_questions_generated": [], "current_question_index": 0,
                "technical_answer_ai_detection": {}, "technical_answer_sentiment": {},
                "tech_stack_to_question": {}, "preferred_language": "English",
                "resume_uploaded": False, "linkedin_profile": None, "current_company": None
            }
            st.session_state.conversation_stage = "greeting"
            st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)


# --- Main App Execution Flow ---
if st.session_state.page == "welcome":
    welcome_page()
elif st.session_state.page == "candidate_info_collection":
    candidate_info_collection_page()
elif st.session_state.page == "chatbot_interface":
    chatbot_interface()
elif st.session_state.page == "exit_page":
    exit_page()
