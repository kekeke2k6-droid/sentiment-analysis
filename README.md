# Dự án Phân tích cảm xúc IMDB (Sentiment Analysis)

Bộ dữ liệu IMDb chứa 50.000 bài đánh giá phim dùng cho xử lý ngôn ngữ tự nhiên (NLP) hoặc phân tích văn bản.

Đây là một bộ dữ liệu phục vụ cho bài toán phân loại cảm xúc nhị phân, có số lượng dữ liệu lớn hơn đáng kể so với các bộ dữ liệu chuẩn trước đây. Chúng tôi cung cấp 25.000 bài đánh giá phim có mức độ cảm xúc rõ ràng dùng để huấn luyện và 25.000 bài dùng để kiểm tra.

Đây là dự án Machine Learning / Deep Learning dùng để phân loại cảm xúc của các đánh giá phim IMDB thành:
- Positive (tích cực)
- Negative (tiêu cực)

Để biết thêm thông tin về bộ dữ liệu, vui lòng truy cập đường link sau.
http://ai.stanford.edu/~amaas/data/sentiment/

Dự án sử dụng các mô hình gồm Logistic Regression, Naive Bayes, SVM để so sánh hiệu quả giữa chúng.

## Cấu trúc thư mục

IMDB Dataset.csv: Tập dữ liệu (Dataset) chứa 50.000 review phim.

IMDB Processed.csv: Tập dữ liệu (Dataset) sau khi xử lý. 

svm_model.pkl, lr_model.pkl, nb_model.pkl, best_svm_mode.pkl: Mô hình học máy đã huấn luyện

eda.ipynb: Khai phá dữ liệu

experiment1.ipynb: So sánh hai phương pháp xử lý ngôn ngữ tự nhiên Bow và TF-IDF

experiment2.ipynb: So sánh các mô hình học máy với nhau

requirements.txt: Danh sách phiên bản các thư viện cần thiết.

## Mục tiêu của dự án

- Tiền xử lý dữ liệu văn bản (text preprocessing)
- Chuyển đổi text thành vector bằng TF-IDF / BoW
- Huấn luyện nhiều mô hình khác nhau
