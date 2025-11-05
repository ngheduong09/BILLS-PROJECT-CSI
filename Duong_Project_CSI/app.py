import streamlit as st
import pandas as pd
import torch
from transformers import LayoutLMTokenizerFast, LayoutLMForTokenClassification
from PIL import Image, ImageDraw, ImageFont
import easyocr
import numpy as np
import google.generativeai as genai
import plotly.express as px

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


MODEL_PATH = "./layoutlm-sroie-finetuned-modern"
DATA_FILE = "extracted_data.csv" # File để lưu trữ dữ liệu

label2color = {
    'COMPANY': 'blue',
    'DATE': 'green',
    'ADDRESS': 'orange',
    'TOTAL': 'red'
}

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

def page_home():
    st.header("Trang chủ: Tải lên và Xử lý Hóa đơn")
    st.write("Chọn một hoặc nhiều file ảnh hóa đơn để trích xuất thông tin.")
    
    uploaded_files = st.file_uploader(
        "Chọn file ảnh", 
        type=["jpg", "jpeg", "png"], 
        accept_multiple_files=True
    )

    if uploaded_files:
        new_data = []
        for uploaded_file in uploaded_files:
            st.markdown(f"---")
            st.subheader(f"Kết quả cho: `{uploaded_file.name}`")
            image = Image.open(uploaded_file).convert("RGB")
            
            col1, col2 = st.columns(2)
            col1.image(image, caption="Ảnh Gốc", use_container_width=True)

            with st.spinner("Đang trích xuất..."):
                extracted_data, word_level_predictions = process_image(image)
            
            if extracted_data:
                annotated_image = draw_predictions(image, word_level_predictions)
                col2.image(annotated_image, caption="Kết quả Trích xuất", use_container_width=True)
                
                st.write("Thông tin trích xuất:")
                st.json(extracted_data)
                
                # Yêu cầu người dùng gán danh mục
                category_options = ["Tạp hóa", "Tiện ích", "Du lịch", "Vật tư Văn phòng", "Tiền thuê nhà", "Khác"]
                category = st.selectbox(f"Chọn danh mục cho hóa đơn '{uploaded_file.name}':", category_options, key=uploaded_file.name)
                
                extracted_data["CATEGORY"] = category
                new_data.append(extracted_data)
            else:
                col2.error("Không tìm thấy văn bản trong ảnh này.")
        
        if st.button("Lưu tất cả kết quả đã trích xuất"):
            new_df = pd.DataFrame(new_data)
            st.session_state.data_df = pd.concat([st.session_state.data_df, new_df], ignore_index=True)
            st.session_state.data_df.to_csv(DATA_FILE, index=False)
            st.success(f"Đã lưu thành công {len(new_data)} hóa đơn vào `{DATA_FILE}`!")
            st.balloons()


def page_data_storage():
    st.header("Kho Dữ liệu Hóa đơn")
    st.write("Dưới đây là toàn bộ dữ liệu đã được trích xuất và lưu trữ.")
    if not st.session_state.data_df.empty:
        st.dataframe(st.session_state.data_df)
        
        # Chuyển đổi DataFrame thành CSV để người dùng có thể tải về
        csv = st.session_state.data_df.to_csv(index=False).encode('utf-8')
        st.download_button(
           "Tải về file CSV",
           csv,
           "hoa_don_da_trich_xuat.csv",
           "text/csv",
           key='download-csv'
        )
    else:
        st.info("Chưa có dữ liệu nào được lưu. Vui lòng tải lên hóa đơn ở trang chủ.")

def page_visualization():
    st.header("Trực quan hóa Chi tiêu")
    st.write("Xem các biểu đồ thống kê dựa trên dữ liệu hóa đơn của bạn.")
    
    df = st.session_state.data_df.copy()
    if df.empty:
        st.warning("Không có dữ liệu để vẽ biểu đồ.")
        return

    # --- BƯỚC LÀM SẠCH VÀ CHUYỂN ĐỔI CHUNG ---
    df['TOTAL'] = df['TOTAL'].astype(str).str.replace(r'[^\d.]', '', regex=True)
    df['TOTAL'] = pd.to_numeric(df['TOTAL'], errors='coerce')
    df['DATE'] = pd.to_datetime(df['DATE'], errors='coerce')

    # ⭐️ SỬA LỖI: Xóa dòng dropna chung ở đây
    # df.dropna(subset=['TOTAL', 'DATE', 'CATEGORY'], inplace=True) 

    st.subheader("Chọn loại biểu đồ bạn muốn xem:")
    
    chart_type = st.radio(
        "Loại biểu đồ:",
        ("Chi tiêu theo danh mục (Biểu đồ tròn)", 
         "Chi tiêu theo thời gian (Biểu đồ đường)",
         "Top 5 nhà cung cấp chi tiêu nhiều nhất (Biểu đồ cột)")
    )
    
    if chart_type == "Chi tiêu theo danh mục (Biểu đồ tròn)":
        # ⭐️ SỬA LỖI: Chỉ dropna cho các cột cần thiết cho biểu đồ này
        df_pie = df.dropna(subset=['TOTAL', 'CATEGORY'])
        if not df_pie.empty:
            fig = px.pie(df_pie, names='CATEGORY', values='TOTAL', title='Tỷ trọng chi tiêu theo từng danh mục')
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Không có đủ dữ liệu (Danh mục, Tổng tiền) để vẽ biểu đồ này.")

    elif chart_type == "Chi tiêu theo thời gian (Biểu đồ đường)":
        # ⭐️ SỬA LỖI: Chỉ dropna cho các cột cần thiết cho biểu đồ này
        df_line = df.dropna(subset=['TOTAL', 'DATE'])
        if not df_line.empty:
            daily_spending = df_line.groupby(df_line['DATE'].dt.date)['TOTAL'].sum().reset_index()
            fig = px.line(daily_spending, x='DATE', y='TOTAL', title='Tổng chi tiêu theo ngày', markers=True)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Không có đủ dữ liệu (Ngày, Tổng tiền) để vẽ biểu đồ này.")
            
    elif chart_type == "Top 5 nhà cung cấp chi tiêu nhiều nhất (Biểu đồ cột)":
        # ⭐️ SỬA LỖI: Chỉ dropna cho các cột cần thiết cho biểu đồ này
        df_bar = df.dropna(subset=['TOTAL', 'COMPANY'])
        if not df_bar.empty:
            top_companies = df_bar.groupby('COMPANY')['TOTAL'].sum().nlargest(5).reset_index()
            fig = px.bar(top_companies, x='COMPANY', y='TOTAL', title='Top 5 nhà cung cấp có chi tiêu cao nhất')
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Không có đủ dữ liệu (Công ty, Tổng tiền) để vẽ biểu đồ này.")

def page_chatbot():
    st.header("Chatbot Tư vấn Tài chính")
    
    if not GEMINI_AVAILABLE:
        st.error("Tính năng này yêu cầu Google API Key. Vui lòng cấu hình secrets để sử dụng.")
        return

    if st.session_state.data_df.empty:
        st.warning("Chưa có dữ liệu chi tiêu để tư vấn. Vui lòng tải hóa đơn lên trước.")
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
        "Tôi đã chi tiêu nhiều nhất cho danh mục nào?",
        "Tổng chi tiêu của tôi trong tháng này là bao nhiêu?",
        "Phân tích xu hướng chi tiêu của tôi."
    ]
    
    st.write("---")
    st.write("Gợi ý cho bạn:")
    cols = st.columns(len(suggested_questions))
    for i, question in enumerate(suggested_questions):
        if cols[i].button(question):
            st.session_state.chat_history.append(("user", question))
            with st.chat_message("user"):
                st.markdown(question)
            
            with st.chat_message("assistant"):
                with st.spinner("Bot đang suy nghĩ..."):
                    prompt = f"""
                    Bạn là một trợ lý tài chính cá nhân. Dựa trên dữ liệu chi tiêu sau đây:
                    --- DỮ LIỆU ---
                    {data_context}
                    --- HẾT DỮ LIỆU ---
                    Hãy trả lời câu hỏi sau của người dùng một cách thân thiện và súc tích: "{question}"
                    """
                    response = gemini_model.generate_content(prompt)
                    st.markdown(response.text)
            st.session_state.chat_history.append(("assistant", response.text))
            st.rerun()


    # Nhận input từ người dùng
    if user_prompt := st.chat_input("Bạn muốn hỏi gì về chi tiêu của mình?"):
        st.session_state.chat_history.append(("user", user_prompt))
        with st.chat_message("user"):
            st.markdown(user_prompt)

        with st.chat_message("assistant"):
            with st.spinner("Bot đang suy nghĩ..."):
                prompt = f"""
                Bạn là một trợ lý tài chính cá nhân. Dựa trên dữ liệu chi tiêu sau đây:
                --- DỮ LIỆU ---
                {data_context}
                --- HẾT DỮ LIỆU ---
                Hãy trả lời câu hỏi sau của người dùng một cách thân thiện và súc tích: "{user_prompt}"
                """
                response = gemini_model.generate_content(prompt)
                st.markdown(response.text)
        
        st.session_state.chat_history.append(("assistant", response.text))
        st.rerun()


# --- THANH SIDEBAR ĐIỀU HƯỚNG ---

st.sidebar.title("Quản lý Tài chính Thông minh")
page = st.sidebar.radio(
    "Điều hướng",
    ["🏠 Tải Hóa Đơn", "📊 Trực Quan Hóa", "🗃️ Kho Dữ Liệu", "🤖 Chatbot Tư Vấn"]
)

if page == "🏠 Tải Hóa Đơn":
    page_home()
elif page == "📊 Trực Quan Hóa":
    page_visualization()
elif page == "🗃️ Kho Dữ Liệu":
    page_data_storage()
elif page == "🤖 Chatbot Tư Vấn":
    page_chatbot()