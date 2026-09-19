import streamlit as st
from pathlib import Path
from urllib.parse import quote
import shutil
import html

from src.rag import ask_question
from src.data_analysis import load_dataset, ask_dataset_question
from src.regression import predict_salary

# PAGE CONFIG
st.set_page_config(
    page_title="AI Data Analyst",
    page_icon="🤖",
    layout="wide"
)

# CHAT MANAGEMENT
def create_chat():
    chat_id = len(st.session_state.chats) + 1

    st.session_state.chats.append(
        {
            "id": chat_id,
            "title": "New Chat",
            "messages": []
        }
    )

    st.session_state.current_chat = chat_id

if "chats" not in st.session_state:
    st.session_state.chats = [
        {
            "id": 1,
            "title": "New Chat",
            "messages": []
        }
    ]

if "current_chat" not in st.session_state:
    st.session_state.current_chat = 1

if "dataset" not in st.session_state:
    st.session_state.dataset = None

if "dataset_name" not in st.session_state:
    st.session_state.dataset_name = None

# Track the selected source so the app can position the page at the
# latest chat message when switching into Salary Prediction.
if "previous_question_source" not in st.session_state:
    st.session_state.previous_question_source = None

if "salary_scroll_to_latest" not in st.session_state:
    st.session_state.salary_scroll_to_latest = False

# STATIC FILES
# Make the training Excel file available as a clickable browser link.
PROJECT_ROOT = Path(__file__).resolve().parent
STATIC_DIR = PROJECT_ROOT / "static"
DATA_DIR = PROJECT_ROOT / "data"

STATIC_DIR.mkdir(exist_ok=True)

TRAINING_FILE = DATA_DIR / "HR_Employee_Attrition.xlsx"
if TRAINING_FILE.exists():
    shutil.copy2(
        TRAINING_FILE,
        STATIC_DIR / TRAINING_FILE.name
    )

# SIDEBAR
with st.sidebar:

    st.header("AI Data Analyst")

    # Sync PDFs from documents/ to static/ so they can be opened directly in a new browser tab.
    documents_dir = Path("documents")
    static_dir = Path("static")
    static_dir.mkdir(exist_ok=True)

    for pdf_file in documents_dir.glob("*.pdf"):
        shutil.copy2(pdf_file, static_dir / pdf_file.name)

    # Compact document section
    document_html = """
    <style>
    .documents-title {
        margin: 0 0 5px 0;
        padding: 0;
        font-size: 14px;
    }

    .document-button {
        display: block;
        width: 100%;
        box-sizing: border-box;
        padding: 6px 10px;
        margin: 2px 0;
        border: 1px solid rgba(128, 128, 128, 0.35);
        border-radius: 8px;
        text-decoration: none !important;
        font-size: 14px;
        font-weight: 500;
        line-height: 1.2;
    }

    .document-button:hover {
        background-color: rgba(128, 128, 128, 0.18);
    }
    </style>

    <div class="documents-title">Available documents:</div>
    """

    for pdf_file in sorted(documents_dir.glob("*.pdf")):
        display_name = html.escape(
            pdf_file.stem.replace("_", " ").title()
        )
        pdf_url = f"/app/static/{quote(pdf_file.name)}"

        document_html += (
            f'<a class="document-button" href="{pdf_url}" target="_blank">'
            f'📄 {display_name}'
            f'</a>'
        )

    st.markdown(document_html, unsafe_allow_html=True)

    # DATASET UPLOAD
    st.markdown(
        """
        <div style="
            font-size: 14px;
            font-weight: 600;
            margin: 14px 0 6px 0;
        ">
            📊 Analyze a Dataset
        </div>
        """,
        unsafe_allow_html=True
    )

    uploaded_file = st.file_uploader(
        "Upload CSV or Excel",
        type=["csv", "xlsx", "xls"],
        label_visibility="collapsed"
    )

    if uploaded_file is not None:
        try:
            # Reload only when a different file is selected.
            if st.session_state.dataset_name != uploaded_file.name:
                st.session_state.dataset = load_dataset(uploaded_file)
                st.session_state.dataset_name = uploaded_file.name

        except Exception as exc:
            st.error(f"Could not load dataset: {exc}")

    if st.session_state.dataset is not None:
        df = st.session_state.dataset

        st.success(
            f"Loaded: {st.session_state.dataset_name}"
        )

        st.caption(
            f"{len(df):,} rows × {len(df.columns):,} columns"
        )

        with st.expander("Preview dataset", expanded=False):
            st.dataframe(
                df.head(10),
                use_container_width=True
            )

    # Choose what the chat should answer from.
    st.markdown(
        """
        <div style="
            font-size: 14px;
            font-weight: 600;
            margin: 10px 0 6px 0;
        ">
            🔎 Question Source
        </div>
        """,
        unsafe_allow_html=True
    )

    available_sources = (
        ["Dataset", "HR Documents", "Salary Prediction"]
        if st.session_state.dataset is not None
        else ["HR Documents", "Salary Prediction"]
    )

    if "question_source" not in st.session_state:
        st.session_state.question_source = available_sources[0]

    if st.session_state.question_source not in available_sources:
        st.session_state.question_source = available_sources[0]

    st.radio(
        "Question source",
        available_sources,
        key="question_source",
        label_visibility="collapsed"
    )

    question_source = st.session_state.question_source

    # If the user just switched into Salary Prediction, request a one-time scroll to the latest message after the page has rendered.
    if (
        question_source == "Salary Prediction"
        and st.session_state.previous_question_source != "Salary Prediction"
        and st.session_state.chats
    ):
        st.session_state.salary_scroll_to_latest = True

    st.session_state.previous_question_source = question_source

    if st.session_state.dataset is None:
        st.caption("Upload a dataset to enable data analysis.")

    # NEW CHAT + CHAT HISTORY
    st.markdown(
        '<div style="height: 12px;"></div>',
        unsafe_allow_html=True
    )

    if st.button(
        "＋  New Chat",
        type="primary",
        use_container_width=True
    ):
        create_chat()
        st.rerun()

    st.markdown(
        """
        <div style="
            font-size: 14px;
            font-weight: 600;
            margin: 14px 0 6px 0;
        ">
            💬 Chat History
        </div>
        """,
        unsafe_allow_html=True
    )

    # Show only chats that contain messages.
    history_chats = [
        chat for chat in st.session_state.chats
        if chat["messages"]
    ]

    if history_chats:
        # Show newest chats first.
        for chat in reversed(history_chats):
            is_current = chat["id"] == st.session_state.current_chat

            title = chat["title"]
            if len(title) > 32:
                title = title[:32] + "..."

            button_label = (
                f"●  {title}"
                if is_current
                else f"   {title}"
            )

            if st.button(
                button_label,
                key=f"chat_{chat['id']}",
                use_container_width=True
            ):
                st.session_state.current_chat = chat["id"]
                st.rerun()
    else:
        st.caption("No previous chats yet.")

    if len(history_chats) > 1:
        if st.button(
            "🗑️ Clear Chat History",
            use_container_width=True
        ):
            current = next(
                chat for chat in st.session_state.chats
                if chat["id"] == st.session_state.current_chat
            )

            st.session_state.chats = [current]
            st.rerun()

    st.divider()

    st.write("Powered by:")
    st.write("• LangChain")
    st.write("• Chroma")
    st.write("• Hugging Face Embeddings")
    st.write("• Ollama / Llama 3.2")

# TITLE
st.title("🤖 AI Data Analyst")

if question_source == "Salary Prediction":
    st.write(
        "Predict employee monthly income using the trained "
        "Linear Regression model."
    )
elif question_source == "Dataset":
    st.write(
        "Ask questions about your uploaded CSV or Excel dataset "
        "using natural-language data analysis."
    )
else:
    st.write(
        "Ask questions about the HR documents and employee data."
    )

# CURRENT CHAT
current_chat = next(
    chat for chat in st.session_state.chats
    if chat["id"] == st.session_state.current_chat
)

# CHAT WINDOW
chat_css = """
<style>
/* Give the page room for the fixed ChatGPT-style input. */
.block-container {
    padding-bottom: 110px;
}

/* Keep the chat area visually separated from the input. */
.chat-window {
    max-width: 900px;
    margin: 0 auto;
}

/* Green background for the AI/assistant avatar. */
div[data-testid="stChatMessageAvatarAssistant"] {
    background-color: #22c55e !important;
    border-radius: 8px !important;
}

div[data-testid="stChatMessageAvatarAssistant"] svg {
    color: white !important;
    fill: white !important;
}
</style>
"""

st.markdown(chat_css, unsafe_allow_html=True)

st.markdown('<div class="chat-window">', unsafe_allow_html=True)

if not current_chat["messages"] and question_source != "Salary Prediction":
    st.markdown(
        """
        <div style="
            text-align: center;
            padding: 110px 20px 80px 20px;
            color: #9aa0a6;
        ">
            <div style="font-size: 42px;">🤖</div>
            <h2 style="margin: 8px 0;">How can I help you?</h2>
            <p>Ask questions about HR documents, datasets, or use Salary Prediction.</p>
        </div>
        """,
        unsafe_allow_html=True
    )

elif current_chat["messages"]:
    for message in current_chat["messages"]:

        if message["role"] == "user":
            with st.chat_message("user"):
                st.write(message["content"])

        else:
            with st.chat_message("assistant"):
                st.write(message["content"])

                sql = message.get("sql")

                if sql:
                    with st.expander(
                        "🔍 Generated SQL",
                        expanded=False
                    ):
                        st.code(sql, language="sql")

                sources = message.get("sources", [])

                if sources:
                    with st.expander(
                        "📚 Sources",
                        expanded=True
                    ):
                        for source in sources:
                            file_name = source.get(
                                "file",
                                "Unknown document"
                            )
                            page_number = source.get(
                                "page",
                                None
                            )
                            content = source.get(
                                "content",
                                ""
                            )

                            if page_number is not None:
                                st.markdown(
                                    f"**📄 {file_name} — Page {page_number}**"
                                )
                            else:
                                st.markdown(
                                    f"**📄 {file_name}**"
                                )

                            if content:
                                st.caption(content)

                            st.divider()

st.markdown('</div>', unsafe_allow_html=True)

# MAIN INPUT AREA
if question_source == "Salary Prediction":
    # Three aligned information cards.
    info_col1, info_col2, info_col3 = st.columns(3)

    card_style = """
    <style>
    .prediction-info-card {
        box-sizing: border-box;
        height: 82px;
        padding: 15px 16px;
        border-radius: 8px;
        background: #19324b;
        border: 1px solid rgba(255,255,255,0.08);
        margin-bottom: 10px;
    }

    .prediction-info-title {
        font-weight: 600;
        margin-bottom: 9px;
    }

    .prediction-info-value {
        font-family: monospace;
        font-size: 12px;
    }

    .training-file-link {
        color: #42a5f5 !important;
        text-decoration: underline !important;
        font-family: monospace;
        font-size: 12px;
    }

    .training-file-link:hover {
        color: #90caf9 !important;
    }
    </style>
    """
    st.markdown(card_style, unsafe_allow_html=True)

    with info_col1:
        training_url = (
            "/app/static/"
            + quote(TRAINING_FILE.name)
        )

        if TRAINING_FILE.exists():
            st.markdown(
                f"""
                <div class="prediction-info-card">
                    <div class="prediction-info-title">
                        📄 Training Dataset
                    </div>
                    <div class="prediction-info-value">
                        <a class="training-file-link"
                           href="{training_url}"
                           target="_blank">
                            HR_Employee_Attrition.xlsx
                        </a>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                """
                <div class="prediction-info-card">
                    <div class="prediction-info-title">
                        📄 Training Dataset
                    </div>
                    <div class="prediction-info-value">
                        File not found
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

    with info_col2:
        st.markdown(
            """
            <div class="prediction-info-card">
                <div class="prediction-info-title">
                    🤖 Prediction Model
                </div>
                <div class="prediction-info-value">
                    Scikit-learn Linear Regression
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with info_col3:
        st.markdown(
            """
            <div class="prediction-info-card">
                <div class="prediction-info-title">
                    🎯 Prediction Target
                </div>
                <div class="prediction-info-value">
                    MonthlyIncome
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.caption(
        "Enter the employee details below. "
        "The model uses the same features used during training."
    )

    with st.form("salary_prediction_form"):

        row1 = st.columns(4)

        with row1[0]:
            prediction_age = st.number_input(
                "Age",
                min_value=18,
                max_value=70,
                value=30,
                step=1
            )

        with row1[1]:
            prediction_job_level = st.number_input(
                "Job Level",
                min_value=1,
                max_value=5,
                value=2,
                step=1
            )

        with row1[2]:
            prediction_total_working_years = st.number_input(
                "Total Working Years",
                min_value=0,
                max_value=50,
                value=6,
                step=1
            )

        with row1[3]:
            prediction_years_at_company = st.number_input(
                "Years At Company",
                min_value=0,
                max_value=50,
                value=4,
                step=1
            )

        row2 = st.columns(4)

        with row2[0]:
            prediction_years_in_role = st.number_input(
                "Years In Current Role",
                min_value=0,
                max_value=30,
                value=2,
                step=1
            )

        with row2[1]:
            prediction_education = st.number_input(
                "Education",
                min_value=1,
                max_value=5,
                value=4,
                step=1
            )

        with row2[2]:
            prediction_department = st.selectbox(
                "Department",
                [
                    "Research & Development",
                    "Sales",
                    "Human Resources"
                ]
            )

        with row2[3]:
            prediction_job_role = st.selectbox(
                "Job Role",
                [
                    "Research Scientist",
                    "Laboratory Technician",
                    "Manufacturing Director",
                    "Healthcare Representative",
                    "Manager",
                    "Sales Executive",
                    "Sales Representative",
                    "Human Resources",
                    "Research Director"
                ]
            )

        predict_clicked = st.form_submit_button(
            "🧠 Predict Monthly Income",
            type="primary",
            use_container_width=True
        )

    # When Salary Prediction is selected, scroll to the salary form itself so the input fields are immediately visible. The previous chat remains above it and naturally moves upward.
    if (
        question_source == "Salary Prediction"
        and st.session_state.salary_scroll_to_latest
    ):
        st.markdown(
            '<div id="salary-form-bottom-anchor"></div>',
            unsafe_allow_html=True
        )

        try:
            import inspect

            if (
                hasattr(st, "html")
                and "unsafe_allow_javascript" in inspect.signature(st.html).parameters
            ):
                st.html(
                    """
                    <script>
                    (() => {
                        const scrollToSalaryForm = () => {
                            const anchor = document.getElementById(
                                "salary-form-bottom-anchor"
                            );
                            if (anchor) {
                                anchor.scrollIntoView({
                                    behavior: "instant",
                                    block: "end"
                                });
                            }
                        };

                        // Allow Streamlit to finish rendering the form first.
                        setTimeout(scrollToSalaryForm, 50);
                        setTimeout(scrollToSalaryForm, 200);
                        setTimeout(scrollToSalaryForm, 500);
                    })();
                    </script>
                    """,
                    unsafe_allow_javascript=True
                )
        except Exception:
            pass

        st.session_state.salary_scroll_to_latest = False

    if predict_clicked:
        try:
            with st.spinner("Running Linear Regression model..."):
                predicted_salary = predict_salary(
                    age=prediction_age,
                    job_level=prediction_job_level,
                    total_working_years=prediction_total_working_years,
                    years_at_company=prediction_years_at_company,
                    years_in_current_role=prediction_years_in_role,
                    education=prediction_education,
                    department=prediction_department,
                    job_role=prediction_job_role
                )

            prediction_question = (
                f"Predict Monthly Income using the following inputs: "
                f"Age={prediction_age}, "
                f"Job Level={prediction_job_level}, "
                f"Total Working Years={prediction_total_working_years}, "
                f"Years At Company={prediction_years_at_company}, "
                f"Years In Current Role={prediction_years_in_role}, "
                f"Education={prediction_education}, "
                f"Department={prediction_department}, "
                f"Job Role={prediction_job_role}."
            )

            prediction_answer = (
                f"### 💰 Predicted Monthly Income\n\n"
                f"**{predicted_salary:,.2f}**\n\n"
                f"### 📋 Input Used\n\n"
                f"| Input | Value |\n"
                f"|---|---|\n"
                f"| Age | {prediction_age} |\n"
                f"| Job Level | {prediction_job_level} |\n"
                f"| Total Working Years | {prediction_total_working_years} |\n"
                f"| Years At Company | {prediction_years_at_company} |\n"
                f"| Years In Current Role | {prediction_years_in_role} |\n"
                f"| Education | {prediction_education} |\n"
                f"| Department | {prediction_department} |\n"
                f"| Job Role | {prediction_job_role} |\n\n"
                f"**Model:** Scikit-learn Linear Regression\n\n"
                f"**Training data:** HR_Employee_Attrition.xlsx\n\n"
                f"**Target:** MonthlyIncome"
            )

            current_chat["messages"].append(
                {
                    "role": "user",
                    "content": prediction_question
                }
            )

            current_chat["messages"].append(
                {
                    "role": "assistant",
                    "content": prediction_answer,
                    "sources": [],
                    "sql": None
                }
            )

            if current_chat["title"] == "New Chat":
                current_chat["title"] = "Salary Prediction"

            st.rerun()

        except Exception as exc:
            st.error(
                f"Could not make salary prediction: {exc}"
            )
else:
    question = st.chat_input(
        "Message AI Data Analyst..."
    )

    if question:

        question = question.strip()

        if question:

            current_chat["messages"].append(
                {
                    "role": "user",
                    "content": question
                }
            )

            if current_chat["title"] == "New Chat":
                current_chat["title"] = (
                    question[:35] + "..."
                    if len(question) > 35
                    else question
                )

            if (
                question_source == "Dataset"
                and st.session_state.dataset is not None
            ):
                with st.spinner(
                    "Analyzing dataset and generating answer..."
                ):
                    result = ask_dataset_question(
                        st.session_state.dataset,
                        question
                    )

                answer = result.get("answer", "")
                sources = []
                sql = result.get("sql")

            else:
                sql = None

                with st.spinner(
                    "Searching documents and generating answer..."
                ):
                    result = ask_question(question)

                if isinstance(result, dict):
                    answer = result.get("answer", "")
                    sources = result.get("sources", [])
                else:
                    answer = str(result)
                    sources = []

            current_chat["messages"].append(
                {
                    "role": "assistant",
                    "content": answer,
                    "sources": sources,
                    "sql": sql
                }
            )

            st.rerun()