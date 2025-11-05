# Ứng dụng Trích xuất Thông tin Hóa đơn

Đây là dự án demo sử dụng Streamlit và mô hình LayoutLM để trích xuất thông tin từ hóa đơn.

## Hướng dẫn Cài đặt và Chạy

**Yêu cầu:**
- Python 3.9+
- pip

**Các bước thực hiện:**

1.  **Tải và giải nén project:**
    Giải nén file `.zip` của project này vào một thư mục trên máy tính của bạn.

2.  **Mở Terminal (hoặc Command Prompt):**
    Di chuyển vào thư mục gốc của project vừa giải nén.
    ```bash
    cd Duong_Project_CSI
    ```

3.  **Tạo môi trường ảo:**
    ```bash
    python3 -m venv venv
    ```

4.  **Kích hoạt môi trường ảo:**
    *   Trên macOS/Linux:
        ```bash
        source venv/bin/activate
        ```
    *   Trên Windows:
        ```bash
        venv\Scripts\activate
        ```

5.  **Cài đặt các thư viện cần thiết:**
    ```bash
    pip install -r requirements.txt
    ```

6.  **(Tùy chọn) Cấu hình Chatbot Gemini:**
    *   Tạo một thư mục có tên `.streamlit` bên trong `Duong_Project_CSI`.
    *   Trong thư mục `.streamlit`, tạo một file tên là `secrets.toml`.
    *   Thêm nội dung sau vào file `secrets.toml`, thay thế `YOUR_API_KEY_HERE` bằng API Key của bạn từ Google AI Studio:
        ```toml
        GOOGLE_API_KEY = "YOUR_API_KEY_HERE"
        ```

7.  **Chạy ứng dụng Streamlit:**
    ```bash
    streamlit run app.py
    ```

8.  Mở trình duyệt và truy cập vào địa chỉ `http://localhost:8501`.