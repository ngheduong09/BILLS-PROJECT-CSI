import hashlib
import json
import re
from io import BytesIO
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st
from PIL import Image
from google import genai


# Streamlit requires this to be the first Streamlit command.
st.set_page_config(
    page_title="Ember",
    page_icon="🔥",
    layout="wide",
)


# --- PATHS AND CONSTANTS ---

BASE_DIR = Path(__file__).parent
DATA_FILE = BASE_DIR / "extracted_data.csv"

ARCHIVE_COLUMNS = ["COMPANY", "DATE", "ADDRESS", "TOTAL", "CATEGORY"]
CATEGORY_OPTIONS = [
    "🍔 Food & Drinks",
    "⚡ Utilities",
    "🎉 Entertainment",
    "💼 Work",
    "🏠 Rent",
    "🤔 Other",
]
GEMINI_MODEL = "gemini-2.5-flash"


# --- GEMINI SETUP ---

def get_google_api_key():
    try:
        return st.secrets["GOOGLE_API_KEY"]
    except (FileNotFoundError, KeyError):
        return None


GOOGLE_API_KEY = get_google_api_key()
GEMINI_AVAILABLE = bool(GOOGLE_API_KEY)


@st.cache_resource
def get_gemini_client():
    if not GOOGLE_API_KEY:
        return None
    return genai.Client(api_key=GOOGLE_API_KEY)


# --- SESSION STATE ---

def empty_archive():
    return pd.DataFrame(columns=ARCHIVE_COLUMNS)


if "data_df" not in st.session_state:
    try:
        loaded_df = pd.read_csv(DATA_FILE)

        for column in ARCHIVE_COLUMNS:
            if column not in loaded_df.columns:
                loaded_df[column] = ""

        st.session_state.data_df = loaded_df[ARCHIVE_COLUMNS]

    except (FileNotFoundError, pd.errors.EmptyDataError):
        st.session_state.data_df = empty_archive()


if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


# Stores analyzed receipts so Streamlit reruns do not call Gemini repeatedly.
if "receipt_results" not in st.session_state:
    st.session_state.receipt_results = {}


# --- RECEIPT PROCESSING ---

def clean_json_text(text):
    text = (text or "").strip()

    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)

    return text.strip()


def normalize_receipt_data(raw_data):
    normalized = {}

    for field in ["COMPANY", "DATE", "ADDRESS", "TOTAL"]:
        value = raw_data.get(field, "")
        normalized[field] = "" if value is None else str(value).strip()

    return normalized


def process_image(image):
    """Extract receipt fields with Gemini Vision.

    This replaces the local EasyOCR + LayoutLM startup path that was causing
    repeated model downloads and Streamlit Community Cloud crashes.
    """
    client = get_gemini_client()

    if client is None:
        raise RuntimeError(
            "Google API Key is missing. Add GOOGLE_API_KEY to Streamlit Secrets."
        )

    prompt = """
Read this receipt image and extract the information below.

Return exactly one JSON object with these four keys:
{
  "COMPANY": "merchant or company name",
  "DATE": "purchase date as printed on the receipt",
  "ADDRESS": "merchant address",
  "TOTAL": "final amount paid"
}

Rules:
- Use an empty string when a field cannot be found.
- Do not invent information.
- Keep TOTAL as a short text value.
- Return JSON only, with no Markdown and no explanation.
"""

    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=[prompt, image],
        config={
            "temperature": 0,
            "response_mime_type": "application/json",
        },
    )

    response_text = clean_json_text(response.text)
    raw_data = json.loads(response_text)

    if not isinstance(raw_data, dict):
        raise ValueError("Gemini returned an unexpected response format.")

    return normalize_receipt_data(raw_data)


def file_fingerprint(file_bytes):
    return hashlib.sha256(file_bytes).hexdigest()


def save_archive():
    st.session_state.data_df.to_csv(DATA_FILE, index=False)


# --- MONEY PARSING FOR CHARTS ---

def parse_money(value):
    if pd.isna(value):
        return None

    text = str(value).strip()
    text = re.sub(r"[^\d,.\-]", "", text)

    if not text:
        return None

    if "," in text and "." in text:
        # Treat the last separator as decimal only when two digits follow it.
        last_comma = text.rfind(",")
        last_dot = text.rfind(".")
        decimal_position = max(last_comma, last_dot)
        digits_after = len(text) - decimal_position - 1

        if digits_after == 2:
            integer_part = re.sub(r"[,.]", "", text[:decimal_position])
            decimal_part = text[decimal_position + 1 :]
            text = f"{integer_part}.{decimal_part}"
        else:
            text = re.sub(r"[,.]", "", text)

    elif "," in text or "." in text:
        separator = "," if "," in text else "."
        parts = text.split(separator)

        if len(parts) > 2:
            text = "".join(parts)
        elif len(parts) == 2:
            before, after = parts
            # Three trailing digits usually represent a thousands separator.
            if len(after) == 3:
                text = before + after
            else:
                text = before + "." + after

    try:
        return float(text)
    except ValueError:
        return None


# --- PAGES ---

def page_home():
    st.header("Welcome back.")
    st.write(
        "Upload receipts and let Ember organize, understand, and remember them."
    )

    if not GEMINI_AVAILABLE:
        st.warning(
            "Receipt analysis is unavailable because GOOGLE_API_KEY is missing "
            "from Streamlit Secrets."
        )

    uploaded_files = st.file_uploader(
        "Drag receipts here or upload images to begin.",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
    )

    if not uploaded_files:
        return

    upload_items = []

    for uploaded_file in uploaded_files:
        file_bytes = uploaded_file.getvalue()
        fingerprint = file_fingerprint(file_bytes)

        upload_items.append(
            {
                "name": uploaded_file.name,
                "bytes": file_bytes,
                "fingerprint": fingerprint,
            }
        )

    st.caption(
        "Your images will not be analyzed until you press the button below."
    )

    analyze_clicked = st.button(
        f"🔍 Analyze {len(upload_items)} receipt(s)",
        type="primary",
        disabled=not GEMINI_AVAILABLE,
    )

    if analyze_clicked:
        for item in upload_items:
            fingerprint = item["fingerprint"]

            # Do not pay for or repeat an analysis already completed this session.
            if fingerprint in st.session_state.receipt_results:
                continue

            try:
                image = Image.open(BytesIO(item["bytes"])).convert("RGB")

                with st.spinner(f"Reading {item['name']}..."):
                    extracted_data = process_image(image)

                st.session_state.receipt_results[fingerprint] = {
                    "status": "success",
                    "data": extracted_data,
                }

            except Exception as error:
                st.session_state.receipt_results[fingerprint] = {
                    "status": "error",
                    "message": str(error),
                }

    analyzed_rows = []

    for item in upload_items:
        fingerprint = item["fingerprint"]
        result = st.session_state.receipt_results.get(fingerprint)

        st.markdown("---")
        st.subheader(f"Your story: `{item['name']}`")

        image = Image.open(BytesIO(item["bytes"])).convert("RGB")
        image_col, result_col = st.columns(2)
        image_col.image(image, caption="Original receipt", use_container_width=True)

        if result is None:
            result_col.info("Press **Analyze receipts** to read this image.")
            continue

        if result["status"] == "error":
            result_col.error(
                "Hmm... I couldn't read this receipt. "
                f"Technical detail: {result['message']}"
            )
            continue

        extracted_data = result["data"]
        result_col.success("🔥 Another memory kept")
        result_col.json(extracted_data)

        category = st.selectbox(
            f"Choose a category for {item['name']}:",
            CATEGORY_OPTIONS,
            key=f"category_{fingerprint}",
        )

        analyzed_rows.append(
            {
                **extracted_data,
                "CATEGORY": category,
            }
        )

    if analyzed_rows and st.button("🔥 Save analyzed receipts to Archive ⭐"):
        new_df = pd.DataFrame(analyzed_rows, columns=ARCHIVE_COLUMNS)
        st.session_state.data_df = pd.concat(
            [st.session_state.data_df, new_df],
            ignore_index=True,
        )
        save_archive()
        st.success(f"🧨 {len(new_df)} memory/memories saved to Archives!")
        st.balloons()


def page_data_storage():
    st.header("Data Archives 🗃️")
    st.write("Here are your traces, stored and kept safely.")

    if st.session_state.data_df.empty:
        st.info("No data yet! Upload a receipt to Ember.")
        return

    st.dataframe(
        st.session_state.data_df,
        hide_index=True,
        use_container_width=True,
    )

    st.markdown("---")

    csv = st.session_state.data_df.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="📥 Export Your Data (.csv)",
        data=csv,
        file_name="ember_receipts.csv",
        mime="text/csv",
        key="download-csv",
    )


def page_visualization():
    st.header("📈 Spending Insights")
    st.write("View informative charts based on your spending data 😁")

    df = st.session_state.data_df.copy()

    if df.empty:
        st.info("Your archive is empty. Upload a receipt to unlock insights.")
        return

    df["TOTAL_NUMERIC"] = df["TOTAL"].apply(parse_money)
    df["DATE_PARSED"] = pd.to_datetime(df["DATE"], errors="coerce")

    st.subheader("Choose the type of graph you want to see:")

    chart_type = st.radio(
        "Graph type:",
        (
            "🥧 Spending by Category",
            "📈 Spending Over Time",
            "💪 Top 5 Companies",
        ),
    )

    if chart_type == "🥧 Spending by Category":
        df_pie = df.dropna(subset=["TOTAL_NUMERIC", "CATEGORY"])

        if df_pie.empty:
            st.warning("Not enough category and total information.")
            return

        fig = px.pie(
            df_pie,
            names="CATEGORY",
            values="TOTAL_NUMERIC",
            title="Spending by category",
        )
        st.plotly_chart(fig, use_container_width=True)

    elif chart_type == "📈 Spending Over Time":
        df_line = df.dropna(subset=["TOTAL_NUMERIC", "DATE_PARSED"])

        if df_line.empty:
            st.warning("Not enough date and total information.")
            return

        daily_spending = (
            df_line.groupby(df_line["DATE_PARSED"].dt.date)["TOTAL_NUMERIC"]
            .sum()
            .reset_index()
        )

        fig = px.line(
            daily_spending,
            x="DATE_PARSED",
            y="TOTAL_NUMERIC",
            title="Spending over time",
            markers=True,
        )
        st.plotly_chart(fig, use_container_width=True)

    else:
        df_bar = df.dropna(subset=["TOTAL_NUMERIC", "COMPANY"])

        if df_bar.empty:
            st.warning("Not enough company and total information.")
            return

        top_companies = (
            df_bar.groupby("COMPANY")["TOTAL_NUMERIC"]
            .sum()
            .nlargest(5)
            .reset_index()
        )

        fig = px.bar(
            top_companies,
            x="COMPANY",
            y="TOTAL_NUMERIC",
            title="Top 5 companies by spending",
        )
        st.plotly_chart(fig, use_container_width=True)


def generate_finance_answer(question):
    client = get_gemini_client()

    if client is None:
        raise RuntimeError("Google API Key is missing.")

    data_context = st.session_state.data_df.to_string(index=False)

    prompt = f"""
You are Bill Fye, Ember's trustworthy personal-finance companion.

Use only the user's spending data below when making claims about their spending.

--- DATA ---
{data_context}
--- END DATA ---

Answer the user's question concisely and practically.
Use English unless the user writes in another language.
You may be lightly witty, but remain clear and useful.
If the question is unrelated to personal finance, guide the conversation back to finance.

Question: {question}
"""

    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt,
        config={"temperature": 0.4},
    )

    return response.text


def submit_chat_question(question):
    st.session_state.chat_history.append(("user", question))

    try:
        answer = generate_finance_answer(question)
    except Exception as error:
        answer = f"Bill Fye ran into an error: {error}"

    st.session_state.chat_history.append(("assistant", answer))
    st.rerun()


def page_chatbot():
    st.header("🤓 Bill Fye the Finance Guy")

    if not GEMINI_AVAILABLE:
        st.error(
            "This feature requires GOOGLE_API_KEY in Streamlit Secrets."
        )
        return

    if st.session_state.data_df.empty:
        st.warning("Bill Fye needs receipt data first. Upload a receipt!")
        return

    for role, text in st.session_state.chat_history:
        with st.chat_message(role):
            st.markdown(text)

    suggested_questions = [
        "What category did I spend the most on?",
        "How much did I spend this month?",
        "Analyze my buying trends 🤯",
    ]

    st.markdown("---")
    st.write("Ask Bill Fye 🤑:")

    columns = st.columns(len(suggested_questions))

    for column, question in zip(columns, suggested_questions):
        if column.button(question):
            submit_chat_question(question)

    user_prompt = st.chat_input("What do you want to ask Bill Fye today? 🤔")

    if user_prompt:
        submit_chat_question(user_prompt)


# --- SIDEBAR AND ROUTING ---

st.sidebar.title("🔥 Ember: Personal Finance powered by AI")

if not GEMINI_AVAILABLE:
    st.sidebar.warning(
        "GOOGLE_API_KEY is missing. Receipt analysis and Bill Fye are disabled."
    )

page = st.sidebar.radio(
    "Keep the transactions. Remember the warmth.",
    [
        "🏠 Upload your bill",
        "📊 Visualize",
        "🗃️ Archives",
        "🤓 Bill Fye the Finance Guy",
    ],
)

if page == "🏠 Upload your bill":
    page_home()
elif page == "📊 Visualize":
    page_visualization()
elif page == "🗃️ Archives":
    page_data_storage()
else:
    page_chatbot()
