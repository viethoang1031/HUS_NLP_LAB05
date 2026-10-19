# LAB 05 - Reflection

## So sánh RNN/LSTM/GRU với LAB 04

| Property | TF-IDF + LR | Avg Word2Vec + LR | RNN | LSTM | GRU |
|---|---|---|---|---|---|
| Sparse representation | v | | | | |
| Dense representation | | v | v | v | v |
| Word order | Limited (unigram) | x | v | v | v |
| Context-dependent state | x | x | v | v | v |
| Long dependency | x | x | Limited | Improved (lý thuyết) | Improved (lý thuyết) |
| Computational cost | Thấp | Thấp | Cao hơn | Cao hơn | Cao hơn |
| F1 trên test (dataset này) | 0.8158 | 0.6997 (LAB 04) | 0.5669 | 0.5375 | 0.5890 |

Số liệu RNN/LSTM/GRU lấy từ Experiment 3/4 (`results.csv`), chạy trên cùng dataset NLTK `movie_reviews` (2000 review, 1400/300/300) như LAB 04.

RNN/LSTM/GRU giải quyết được vấn đề gì mà Average Word2Vec không giải quyết tốt: cả ba đều mã hoá được thứ tự từ và duy trì context-dependent state qua công thức $h_t = f(x_t, h_{t-1})$, trong khi Average Word2Vec cộng trung bình vector từ nên mất hoàn toàn thông tin thứ tự (xem Bài 3 trong `calculations.md`: "dog bites man" và "man bites dog" cho cùng một vector trung bình).

RNN/LSTM/GRU vẫn còn vấn đề gì: trên dataset nhỏ (1400 mẫu train) với embedding khởi tạo ngẫu nhiên, cả ba đều cho F1 thấp hơn hẳn so với TF-IDF + Logistic Regression (0.5375-0.5890 so với 0.8158). Nguyên nhân không phải vì cơ chế sequence modeling sai, mà vì: (1) số tham số của các model này (hơn 1 triệu, xem `results.csv` mục `exp4`) quá lớn so với lượng dữ liệu train, dẫn tới overfit nhanh (xem Learning curves: GRU có train_loss giảm từ 0.69 xuống 0.11 nhưng val_loss tăng từ epoch 1); (2) không có pretrained embedding nên model phải học biểu diễn từ từ đầu chỉ với 1400 mẫu; (3) vanishing gradient vẫn ảnh hưởng tới RNN với sequence dài (Experiment 2: khi max_length tăng, F1 không tăng tương ứng). Computational cost cũng cao hơn nhiều: training time của RNN/LSTM/GRU tính bằng chục đến hàng trăm giây, so với dưới 1-2 giây của Logistic Regression trên TF-IDF.

## Vì sao LSTM không vượt RNN/GRU trên dataset này

Kết quả Experiment 4 (RNN F1=0.5669, LSTM F1=0.5375, GRU F1=0.5890) không khớp với kỳ vọng "LSTM luôn tốt hơn RNN" ở Prediction 2. Đây là minh hoạ trực tiếp cho tinh thần của Prediction 5 trong `prediction.md`: LSTM có nhiều tham số nhất (1,042,761, nhiều hơn RNN và GRU) nên dễ overfit hơn trên 1400 mẫu train, trong khi cơ chế cell state giúp ích chủ yếu khi sequence đủ dài và đủ dữ liệu để học đúng cách dùng các gate. GRU với ít tham số hơn LSTM (do không có cell state riêng) lại có F1 cao nhất trong ba model, phù hợp với mục 18 trong `W5.pdf`: GRU đơn giản hơn LSTM, đôi khi phù hợp hơn khi dữ liệu hạn chế.

## Prior lệch của model (positive bias)

Confusion matrix của GRU trên test set cho FP=80 cao hơn hẳn FN=54 (xem `error_analysis.md`), và trong phần Application, câu phủ định rõ ràng "I did not like this movie at all." vẫn bị dự đoán Positive (probability 0.669). Đây là dấu hiệu model học được một prior thiên lệch về lớp "positive" nhiều hơn là học được quan hệ ngữ nghĩa của phép phủ định, một giới hạn thực tế của việc train from scratch trên dataset nhỏ, không chỉ là giới hạn lý thuyết của kiến trúc recurrent.

## Giới hạn của recurrent model và động lực sang Attention

Dù RNN/LSTM/GRU có cơ chế $h_t = f(x_t, h_{t-1})$ cho phép xử lý sequence, thông tin vẫn phải đi qua tuần tự từng bước thời gian để tới được vị trí cuối cùng. Với sequence dài (ví dụ 100 token), thông tin ở token đầu tiên phải "sống sót" qua rất nhiều phép nhân Jacobian mới còn ảnh hưởng tới $h_T$ (xem mục "Vanishing gradient" trong `calculations.md`), và Experiment 6 cho thấy dấu hiệu thực nghiệm (không phải chứng minh lý thuyết đầy đủ) rằng vị trí của cụm từ quyết định sentiment có thể ảnh hưởng tới xác suất dự đoán.

Câu hỏi đặt ra ở cuối `W5.pdf`: có cách nào để một token "nhìn" trực tiếp các token quan trọng khác mà không phải truyền toàn bộ thông tin qua một chuỗi recurrent dài? Đây chính là động lực dẫn tới Attention và Query/Key/Value ở LAB 06, nơi context-aware representation được tính trực tiếp giữa các cặp token thay vì phải đi qua từng bước đệ quy.

## Tóm tắt

LAB 04 cho thấy: static representation không bằng sequence modeling về mặt khả năng mã hoá thứ tự. LAB 05 cho thấy: sequence modeling đúng là có cơ chế mạnh hơn (Bài 2, Bài 3 trong `calculations.md`), nhưng cơ chế mạnh hơn không tự động chuyển thành kết quả tốt hơn nếu thiếu dữ liệu hoặc pretrained embedding - đây là một bài học thực nghiệm quan trọng không kém gì bài học lý thuyết, và là lý do vì sao đánh giá một model NLP không nên chỉ dừng ở accuracy/F1 mà cần xem xét cả số tham số, training time, và error analysis chi tiết.
