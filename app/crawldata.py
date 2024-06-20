import os
import requests
from bs4 import BeautifulSoup
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph
# Hàm để crawl nội dung từ URL và lưu vào file
def crawl(url: str) -> str:
    try:
        # Gửi yêu cầu HTTP GET để lấy nội dung của trang
        response = requests.get(url)
        response.raise_for_status()  # Ném exception nếu yêu cầu không thành công

        # Tạo đối tượng BeautifulSoup để phân tích cú pháp HTML
        soup = BeautifulSoup(response.content, 'html.parser')

        # Khởi tạo biến để lưu trữ nội dung
        text = ""

        # Duyệt qua các phần tử trong thân trang (body)
        for element in soup.body.find_all(['p', 'ul'], recursive=True):
            if element.name == 'p':
                text += element.get_text() + "\n"
            elif element.name == 'ul':
                for li in element.find_all('li'):
                    text += "- " + li.get_text() + "\n"

        # Lấy tiêu đề của trang từ URL
        page_title = url.split('/')[-1]

        # Tạo đường dẫn đầy đủ đến file
        file_path = os.path.join("./data/", page_title + ".pdf")
        pdf_document = SimpleDocTemplate(file_path)
        pdf_elements = []
        styles = getSampleStyleSheet()

# Parse the HTML-like text into a Paragraph
        paragraph = Paragraph(text, styles["Normal"])

# Add the Paragraph to the PDF elements
        pdf_elements.append(paragraph)

# Build the PDF document
        pdf_document.build(pdf_elements)
        return file_path  # Trả về đường dẫn của file đã lưu
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"Failed to fetch URL: {str(e)}")