import streamlit as st
import pandas as pd
import torch
from transformers import LayoutLMTokenizerFast, LayoutLMForTokenClassification
from PIL import Image, ImageDraw, ImageFont
import easyocr
import numpy as np
import google.generativeai as genai
import plotly.express as px

#supabase 
import streamlit as st
from supabase import create_client

# other imports...

supabase = create_client(
    st.secrets["SUPABASE_URL"],
    st.secrets["SUPABASE_KEY"]
)
try:
    response = supabase.table("receipts").select("*").limit(1).execute()
    print("✅ Supabase connected successfully!")
except Exception as e:
    print("❌ Supabase connection failed:", e)


# --- CẤU HÌNH VÀ TẢI TÀI NGUYÊN ---

# Cấu hình API Key cho Gemini (Lấy từ Google AI Studio)
# BẠN NÊN DÙNG st.secrets ĐỂ BẢO MẬT API KEY KHI DEPLOY
# Ví dụ: genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])
# Ở đây, để chạy local, bạn có thể dán key trực tiếp hoặc dùng secrets.toml
try:
    genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])
    GEMINI_AVAILABLE = True
except (FileNotFoundError, KeyError):
    GEMINI_AVAILABLE = False
    st.sidebar.warning("Không tìm thấy Google API Key. Tính năng Chatbot sẽ bị vô hiệu hóa.")


from pathlib import Path

BASE_DIR = Path(__file__).parent

MODEL_PATH = BASE_DIR / "layoutlm-sroie-finetuned-modern"
DATA_FILE = BASE_DIR / "extracted_data.csv" # File để lưu trữ dữ liệu

label2color = {
    'COMPANY': 'blue',
    'DATE': 'green',
    'ADDRESS': 'orange',
    'TOTAL': 'red'
}

import os



# Sử dụng cache của Streamlit để chỉ tải model và OCR reader một lần
@st.cache_resource
def load_resources():
    tokenizer = LayoutLMTokenizerFast.from_pretrained(MODEL_PATH)
    model = LayoutLMForTokenClassification.from_pretrained(MODEL_PATH)
    reader = easyocr.Reader(['en'], gpu=False)

    id2label = {0: "S-COMPANY", 1: "S-DATE", 2: "S-ADDRESS", 3: "S-TOTAL", 4: "O"}
    label2id = {label: id for id, label in id2label.items()}
    
    model.config.id2label = id2label
    model.config.label2id = label2id
    
    return tokenizer, model, reader

# Tải tài nguyên
tokenizer, model, reader = load_resources()
LABELS = [label.replace("S-", "") for label in model.config.id2label.values() if label != "O"]


# --- KHỞI TẠO SESSION STATE ---
# Session state để lưu dữ liệu giữa các lần tương tác
if 'data_df' not in st.session_state:
    try:
        # Thử tải dữ liệu đã lưu nếu có
        st.session_state.data_df = pd.read_csv(DATA_FILE)
    except FileNotFoundError:
        # Nếu không có, tạo DataFrame rỗng
        st.session_state.data_df = pd.DataFrame(columns=LABELS + ["CATEGORY"])

if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []


# --- CÁC HÀM XỬ LÝ (GIỮ NGUYÊN TỪ PHIÊN BẢN TRƯỚC) ---

def normalize_box(box, width, height):
    return [
        int(1000 * (box[0] / width)), int(1000 * (box[1] / height)),
        int(1000 * (box[2] / width)), int(1000 * (box[3] / height)),
    ]

def process_image(image):
    width, height = image.size
    ocr_results = reader.readtext(np.array(image))
    if not ocr_results: return None, None

    words = [res[1] for res in ocr_results]
    unnormalized_boxes = [[int(p) for p in box[0]] + [int(p) for p in box[2]] for box, _, _ in ocr_results]
    normalized_boxes = [normalize_box(box, width, height) for box in unnormalized_boxes]

    tokenized_inputs = tokenizer(words, padding="max_length", max_length=512, truncation=True, is_split_into_words=True, return_tensors="pt")
    word_ids = tokenized_inputs.word_ids()

    bbox_list = [normalized_boxes[word_idx] if word_idx is not None else [0,0,0,0] for word_idx in word_ids]
    bbox = torch.tensor(bbox_list).unsqueeze(0)

    model.eval()
    with torch.no_grad():
        outputs = model(input_ids=tokenized_inputs['input_ids'], bbox=bbox, attention_mask=tokenized_inputs['attention_mask'], token_type_ids=tokenized_inputs['token_type_ids'])
    
    predictions = outputs.logits.argmax(-1).squeeze().tolist()
    token_labels = [model.config.id2label[pred] for pred in predictions]

    word_level_predictions = []
    previous_word_idx = None
    for i, word_idx in enumerate(word_ids):
        if word_idx is None or word_idx == previous_word_idx: continue
        label = token_labels[i].replace("S-", "")
        word_level_predictions.append({"word": words[word_idx], "label": label, "box": unnormalized_boxes[word_idx]})
        previous_word_idx = word_idx

    extracted_data = {label: [] for label in LABELS}
    for item in word_level_predictions:
        if item["label"] in LABELS:
            extracted_data[item["label"]].append(item["word"])

    for label in LABELS:
        extracted_data[label] = ' '.join(extracted_data[label])

    return extracted_data, word_level_predictions

def draw_predictions(image, predictions):
    drawn_image = image.copy()
    draw = ImageDraw.Draw(drawn_image)
    try: font = ImageFont.truetype("arial.ttf", size=15)
    except IOError: font = ImageFont.load_default()

    for item in predictions:
        if item["label"] in label2color:
            draw.rectangle(item["box"], outline=label2color[item["label"]], width=2)
            draw.text((item["box"][0], item["box"][1] - 15), item["label"], fill=label2color[item["label"]], font=font)
    return drawn_image


# --- ĐỊNH NGHĨA CÁC TRANG (SECTIONS) ---
st.set_page_config(
    page_title="Ember",
    page_icon="🔥",
    layout="wide"
)

def page_home():
    st.header("Welcome back.")
    st.write("Upload receipts and let Ember organize, understand, and remember them.")
    
    uploaded_files = st.file_uploader(
        "Drag receipts here or upload images to begin.", 
        type=["jpg", "jpeg", "png"], 
        accept_multiple_files=True
    )

    if uploaded_files:
        new_data = []
        for uploaded_file in uploaded_files:
            st.markdown(f"---")
            st.subheader(f"Your story: `{uploaded_file.name}`")
            image = Image.open(uploaded_file).convert("RGB")
            
            col1, col2 = st.columns(2)
            col1.image(image, caption="OG image", use_container_width=True)

            with st.spinner("🔍 Reading your receipt..."):
                extracted_data, word_level_predictions = process_image(image)
            
            if extracted_data:
                annotated_image = draw_predictions(image, word_level_predictions)
                col2.image(annotated_image, caption="🔥 Another memory kept", use_container_width=True)
                
                st.write("Here's your story:")
                st.json(extracted_data)
                
                # Yêu cầu người dùng gán danh mục
                category_options = ["🍔 Food & Drinks", "⚡Utilities", "🎉 Entertainment", "💼 Work", "🏠 Rent", "🤔Other"]
                category = st.selectbox(f"Choose your category '{uploaded_file.name}':", category_options, key=uploaded_file.name)
                
                extracted_data["CATEGORY"] = category
                new_data.append(extracted_data)
            else:
                col2.error("Hmm... I couldn't read this receipt. Try a clearer photo.")
        
        if st.button("🔥 Save to Archive ⭐"):
            new_df = pd.DataFrame(new_data)
            st.session_state.data_df = pd.concat([st.session_state.data_df, new_df], ignore_index=True)
            st.session_state.data_df.to_csv(DATA_FILE, index=False)
            st.success(f"🧨Memory{len(new_data)} safely kept in `{DATA_FILE}`!")
            st.balloons()


# def page_data_storage():
#     st.header("Kho Dữ liệu Hóa đơn")
#     st.write("Dưới đây là toàn bộ dữ liệu đã được trích xuất và lưu trữ.")
#     if not st.session_state.data_df.empty:
#         st.dataframe(st.session_state.data_df)
        
#         # Chuyển đổi DataFrame thành CSV để người dùng có thể tải về
#         csv = st.session_state.data_df.to_csv(index=False).encode('utf-8')
#         st.download_button(
#            "Tải về file CSV",
#            csv,
#            "hoa_don_da_trich_xuat.csv",
#            "text/csv",
#            key='download-csv'
#         )
#     else:
#         st.info("Chưa có dữ liệu nào được lưu. Vui lòng tải lên hóa đơn ở trang chủ.")

def page_data_storage():
    st.header("Data Archives🗃️")
    st.write("Here are your traces, stored and kept safely")

    # Kiểm tra nếu DataFrame rỗng
    if st.session_state.data_df.empty:
        st.info("No data yet! Upload a receipt to Ember")
        return

    # --- TÍNH NĂNG XÓA TỪNG HÓA ĐƠN ---
    # Thêm một cột 'Xóa' vào DataFrame để hiển thị các nút bấm
    # Sử dụng st.data_editor để có thể tương tác
    st.write("Delete receiptss by ticking the box in the row and the button below..")
    
    # Chuyển DataFrame sang định dạng có thể chỉnh sửa
    # Thêm cột "delete" để người dùng chọn
    df_with_delete = st.session_state.data_df.copy()
    df_with_delete.insert(0, "Delete", False)
    
    # Hiển thị bảng dữ liệu có thể chỉnh sửa
    edited_df = st.data_editor(
    df_with_delete,
    hide_index=True,
    column_config={
        "Delete": st.column_config.CheckboxColumn(
            "🗑️ Delete",
            help="Select the receipts you want to remove."
        )
    },
    disabled=st.session_state.data_df.columns
)

    # Lấy danh sách các dòng được chọn để xóa
    st.write(edited_df.columns)
    rows_to_delete = edited_df[edited_df["Delete"]].index

    if st.button("🗑️ Delete Selected Receipts", type="primary", disabled=len(rows_to_delete) == 0):
        # Lấy lại DataFrame gốc từ session_state
        df_original = st.session_state.data_df
        # Xóa các hàng đã chọn
        df_updated = df_original.drop(index=rows_to_delete).reset_index(drop=True)
        
        # Cập nhật lại session_state và file CSV
        st.session_state.data_df = df_updated
        st.session_state.data_df.to_csv(DATA_FILE, index=False)
        
        st.success(f"Successfully remove {len(rows_to_delete)} receipt🧹")
        # Chạy lại script để cập nhật giao diện ngay lập tức
        st.rerun()

    st.markdown("---")

    # --- TÍNH NĂNG TẢI VỀ VÀ RESET ---
    col1, col2 = st.columns(2)

    # Cột 1: Nút tải về
    with col1:
        csv = st.session_state.data_df.to_csv(index=False).encode('utf-8')
        st.download_button(
           "📥 Export Your Data (.csv)",
           csv,
           "hoa_don_da_trich_xuat.csv",
           "text/csv",
           key='download-csv'
        )

    # Cột 2: Nút Reset
    with col2:
        if st.button("🔴 Clear My Archive", help="⚠️ This will permanently delete all saved receipts!"):
            # Tạo DataFrame rỗng
            empty_df = pd.DataFrame(columns=st.session_state.data_df.columns)
            
            # Cập nhật session_state và ghi đè file CSV
            st.session_state.data_df = empty_df
            st.session_state.data_df.to_csv(DATA_FILE, index=False)
            
            st.warning("✨ Your archive has been cleared.")
            st.rerun()

def page_visualization():
    st.header("📈 Spending Insights")
    st.write("View informative charts based on your spending data😁")
    
    df = st.session_state.data_df.copy()
    if df.empty:
        st.success("🔥 Your archive is now empty. Ready for a fresh start?")
        return

    # --- BƯỚC LÀM SẠCH VÀ CHUYỂN ĐỔI CHUNG ---
    df['TOTAL'] = df['TOTAL'].astype(str).str.replace(r'[^\d.]', '', regex=True)
    df['TOTAL'] = pd.to_numeric(df['TOTAL'], errors='coerce')
    df['DATE'] = pd.to_datetime(df['DATE'], errors='coerce')

    # ⭐️ SỬA LỖI: Xóa dòng dropna chung ở đây
    # df.dropna(subset=['TOTAL', 'DATE', 'CATEGORY'], inplace=True) 

    st.subheader("Choose the Type of Graph You Want to See:")
    
    chart_type = st.radio(
        "Graph Type:",
        ("🥧 Spending by Category (Pie chart)", 
         "📈 Spending Over Time (Line Graph)",
         "💪Top 5 Companies Most Spent on (Bar Chart)")
    )
    
    if chart_type == "🥧 Spending by Category (Pie chart)":
        # ⭐️ SỬA LỖI: Chỉ dropna cho các cột cần thiết cho biểu đồ này
        df_pie = df.dropna(subset=['TOTAL', 'CATEGORY'])
        if not df_pie.empty:
            fig = px.pie(df_pie, names='CATEGORY', values='TOTAL', title='Spending amount by category')
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("📭 Your archive(Danh mục, Tổng tiền) is empty. Upload a receipt to unlock your story")

    elif chart_type == "📈 Spending Over Time (Line Graph)":
        # ⭐️ SỬA LỖI: Chỉ dropna cho các cột cần thiết cho biểu đồ này
        df_line = df.dropna(subset=['TOTAL', 'DATE'])
        if not df_line.empty:
            daily_spending = df_line.groupby(df_line['DATE'].dt.date)['TOTAL'].sum().reset_index()
            fig = px.line(daily_spending, x='DATE', y='TOTAL', title='Tổng chi tiêu theo ngày', markers=True)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Not enough info (Ngày, Tổng tiền) to draw this chart!")
            
    elif chart_type == "💪Top 5 Companies Most Spent on (Bar Chart)":
        # ⭐️ SỬA LỖI: Chỉ dropna cho các cột cần thiết cho biểu đồ này
        df_bar = df.dropna(subset=['TOTAL', 'COMPANY'])
        if not df_bar.empty:
            top_companies = df_bar.groupby('COMPANY')['TOTAL'].sum().nlargest(5).reset_index()
            fig = px.bar(top_companies, x='COMPANY', y='TOTAL', title='💪Top 5 Companies Most Spent on (Bar Chart)')
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Not enough info (Công ty, Tổng tiền) to draw this graph!")

def page_chatbot():
    st.header("🤓Bill Fye the Finance Guy")
    
    if not GEMINI_AVAILABLE:
        st.error("This feature requires Google API Key. Arrange secrets to use.")
        return

    if st.session_state.data_df.empty:
        st.warning("Not enough info for Bill Nye. Let's upload you receipts!")
        return

    # Khởi tạo mô hình Gemini
    gemini_model = genai.GenerativeModel('gemini-2.5-flash')

    # Hiển thị lịch sử chat
    for role, text in st.session_state.chat_history:
        with st.chat_message(role):
            st.markdown(text)

    # Dữ liệu chi tiêu làm bối cảnh cho bot
    data_context = st.session_state.data_df.to_string()
    
    # Các câu hỏi gợi ý
    suggested_questions = [
        "What category did I spend the most on?",
        "How much did I spend this month?",
        "Analyze my buying trends🤯"
    ]
    
    st.write("---")
    st.write("Ask Bill Fye 🤑:")
    cols = st.columns(len(suggested_questions))
    for i, question in enumerate(suggested_questions):
        if cols[i].button(question):
            st.session_state.chat_history.append(("user", question))
            with st.chat_message("user"):
                st.markdown(question)
            
            with st.chat_message("assistant"):
                with st.spinner("☕ Bill Fye is reading your receipt..."):
                    prompt = f"""
                    You are Bill Fye, Ember's trustworthy companion and mascot.
                    Use the user's following spending data:
                    --- DATA ---
                    {data_context}
                    --- END OF DATA -- 
                    to provide practical, personalized financial insights.
                    Always reply in English unless the user explicitly writes in another language.
                    Answer questions  in concise, high-quality method, and occasionally witty manner : "{question}"
                    If a question is unrelated to personal finance, politely acknowledge it and guide the conversation back to finance.
                    """
                    response = gemini_model.generate_content(prompt)
                    st.markdown(response.text)
            st.session_state.chat_history.append(("assistant", response.text))
            st.rerun()


    # Nhận input từ người dùng
    if user_prompt := st.chat_input("What do you want to ask Bill Fye?"):
        st.session_state.chat_history.append(("user", user_prompt))
        with st.chat_message("user"):
            st.markdown(user_prompt)

        with st.chat_message("assistant"):
            with st.spinner("🤓Bill Fye is thinking..."):
                prompt = f"""
                You are Bill Fye, Ember's trustworthy companion and mascot.
                Use the user's following spending data:
                    --- DATA ---
                    {data_context}
                    --- END OF DATA -- 
                to provide practical, personalized financial insights.
                Always reply in English unless the user explicitly writes in another language.
                Answer questions  in concise, high-quality method, and occasionally witty manner : "{user_prompt}"
                If a question is unrelated to personal finance, politely acknowledge it and guide the conversation back to finance. 
                """
                response = gemini_model.generate_content(prompt)
                st.markdown(response.text)
        
        st.session_state.chat_history.append(("assistant", response.text))
        st.rerun()


# --- THANH SIDEBAR ĐIỀU HƯỚNG ---

st.sidebar.title("🔥 Ember: Personal Finance powered by AI.")
page = st.sidebar.radio(
    "Keep the transactions. Remember the warmth.",
    ["🏠Upload your bill", "📊Visualize", "🗃️Archives", "🤓Bill Fye the Finance Guy"]
)

if page == "🏠Upload your bill":
    page_home()
elif page == "📊Visualize":
    page_visualization()
elif page == "🗃️Archives":
    page_data_storage()
elif page == "🤓Bill Fye the Finance Guy":
    page_chatbot()


try:
    ...
except Exception:
    st.error("⚠️ Something went wrong while loading the AI model. Please try again later.")

