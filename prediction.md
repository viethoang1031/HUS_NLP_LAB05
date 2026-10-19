# Prediction - LAB 05

Ghi trước khi chạy bất kỳ experiment nào. Đối chiếu với kết quả thực tế ở cuối file.

## Prediction 1

Prediction: RNN + Embedding có khả năng xử lý word order trực tiếp hơn TF-IDF + Logistic Regression.

Reason: TF-IDF biểu diễn văn bản bằng tần suất từ (bag-of-words), không mã hoá vị trí hay thứ tự xuất hiện của từ trong câu; hai câu có cùng tập từ nhưng khác thứ tự sẽ có vector TF-IDF giống nhau (trừ ảnh hưởng của n-gram nếu có dùng). RNN đọc token theo từng bước thời gian và cập nhật $h_t = f(x_t, h_{t-1})$, nên $h_t$ phụ thuộc vào cả nội dung lẫn vị trí của token trong chuỗi đã đọc.

Confidence: Cao.

## Prediction 2

Prediction: LSTM có khả năng xử lý dependency dài tốt hơn RNN thuần.

Reason: RNN thuần chỉ có một hidden state được cập nhật liên tục qua $\tanh$, gradient truyền qua nhiều bước dễ bị vanishing gradient (xem mục 9 trong tài liệu). LSTM thêm cell state $c_t$ được cập nhật chủ yếu qua phép cộng có trọng số bởi forget gate ($c_t = f_t c_{t-1} + i_t \tilde{c}_t$), con đường này cho phép gradient đi qua nhiều timestep mà không bị nhân lặp lại bởi đạo hàm $\tanh$/$W_{hh}$ ở mỗi bước, nên ít bị suy giảm hơn đường truyền hidden state thuần của RNN.

Confidence: Trung bình - cao (cơ chế rõ ràng nhưng mức độ cải thiện thực tế còn tuỳ dataset và độ dài sequence).

## Prediction 3

Prediction: Tăng sequence length từ 50 lên 200 sẽ làm thời gian training của RNN tăng xấp xỉ tuyến tính (hoặc hơn) theo số timestep, vì RNN phải thực hiện tuần tự nhiều bước forward/backward hơn cho mỗi sample, không thể song song hoá qua các timestep như CNN hay fully-connected layer.

Confidence: Cao.

## Prediction 4

Prediction: Nếu truncate toàn bộ review về 30 token, thông tin phân loại có thể bị ảnh hưởng, đặc biệt với các review dài (median/mean sequence length lớn hơn 30 nhiều - xem Experiment 1): phần kết luận hoặc từ khoá quyết định sentiment có thể nằm ở cuối review và bị cắt bỏ.

Confidence: Cao.

## Prediction 5

Prediction: LSTM không chắc chắn luôn cho accuracy cao hơn RNN trên mọi dataset. Với dataset nhỏ hoặc sequence ngắn, RNN đơn giản có thể đủ dùng và LSTM (nhiều tham số hơn) có thể không có lợi thế rõ ràng, thậm chí dễ overfit hơn hoặc cần nhiều dữ liệu/epoch hơn để phát huy hết khả năng của gate.

Confidence: Cao.

---

## Đối chiếu sau experiment

Prediction 1: Đúng về cơ chế nhưng sai về kết quả số cuối cùng trên dataset này. RNN có cơ chế khai thác thứ tự token ($h_t = f(x_t, h_{t-1})$) mà TF-IDF không có (xem Bài 3 trong `calculations.md` và Experiment 6), nhưng trên tập test, TF-IDF + Logistic Regression đạt accuracy 0.8133 / F1 0.8158, trong khi RNN + Embedding (max_length 150) chỉ đạt accuracy 0.5467 / F1 0.5669 (xem `results.csv`, `exp3`). Nguyên nhân chính không phải vì RNN "không dùng được" thông tin thứ tự, mà vì dataset chỉ có 1400 mẫu train và embedding được khởi tạo ngẫu nhiên (không dùng pretrained), nên RNN không đủ dữ liệu để học biểu diễn từ tốt, trong khi Logistic Regression trên TF-IDF là linear model đơn giản, ít tham số, học ổn định ngay cả khi ít dữ liệu.

Prediction 2: Sai trên số liệu cụ thể của lần chạy này - LSTM (F1 = 0.5375) thấp hơn cả RNN (F1 = 0.5669) và GRU (F1 = 0.5890) trên test set (xem `results.csv`, `exp4`). Cơ chế cell state của LSTM vẫn đúng về mặt lý thuyết (giúp kiểm soát information flow tốt hơn, xem `calculations.md`), nhưng LSTM có nhiều tham số nhất (1,042,761) trong khi dataset nhỏ, nên dễ overfit nhanh (xem Learning curves: val_loss của LSTM tăng liên tục từ epoch 2, từ 0.70 lên 1.47 ở epoch 10 trong khi train_loss giảm từ 0.70 xuống 0.097). Đây chính là minh chứng cho Prediction 5: kiến trúc phức tạp hơn không tự động tốt hơn.

Prediction 3: Đúng - training time tăng mạnh theo max_length: với RNN (Experiment 2), từ 9.15s (L=50) lên 21.64s (L=100) lên 116.35s (L=200); với GRU (Experiment 5), từ 8.91s (L=50) lên 17.59s (L=100) lên 153.11s (L=200) (xem `results.csv`).

Prediction 4: Đúng - với max_length = 50, chỉ 9.38% token trung bình của mỗi review được giữ lại; với max_length = 100 là 18.68%; với max_length = 200 là 36.98% (xem `results.csv`, `exp2`/`exp5`). Vì mean sequence length ~652 token, phần lớn nội dung review - bao gồm khả năng cả câu kết luận - bị cắt bỏ khi max_length nhỏ.

Prediction 5: Đúng, được xác nhận rõ - trên tập test, GRU (F1 = 0.5890) > RNN (F1 = 0.5669) > LSTM (F1 = 0.5375). LSTM không chỉ không vượt RNN mà còn thấp hơn, cho thấy LSTM "phức tạp hơn" không đảm bảo tốt hơn trên mọi dataset, đặc biệt với dataset nhỏ và embedding không pretrained như ở đây.

**Lưu ý thêm:** cả ba model RNN/LSTM/GRU đều thấp hơn đáng kể so với baseline TF-IDF + Logistic Regression trên dataset này (xem Experiment 3, Experiment 4). Đây là một kết quả thật, không phải lỗi: phản ánh đúng tinh thần Prediction 5 và nội dung mục 15/19 trong `W5.pdf` - sequence model có cơ chế mạnh hơn về mặt lý thuyết, nhưng cần đủ dữ liệu (hoặc pretrained embedding) để phát huy, nếu không thì một model tuyến tính đơn giản trên biểu diễn sparse vẫn có thể thắng. Xem thêm `error_analysis.md` và `reflection.md`.
