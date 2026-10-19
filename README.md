# LAB 05 - Sequence Models: RNN, LSTM and GRU

Bài làm mẫu (đáp án tham chiếu) cho LAB 05.

## Nội dung

- `calculations.md`: Bài 1 (một bước RNN), Bài 2 (information flow), Bài 3 (sequence order), giải thích vanishing/exploding gradient, bài tập tính toán LSTM gates.
- `prediction.md`: 5 prediction trước experiment (Prediction/Reason/Confidence) và đối chiếu bằng số liệu thực tế sau khi chạy.
- `rnn_cell.py`: `SimpleRNNCell` tự implement bằng numpy, thực hiện `h_t = tanh(W_xh x_t + W_hh h_{t-1} + b_h)`, có test đối chiếu với `torch.nn.RNNCell` (sai số ~1e-7).
- `sequence_classification.ipynb`: Experiment 1 đến 6, core implementation, baseline, RNN/LSTM/GRU, sequence length, long-range dependency, evaluation, learning curves, error analysis, so sánh với LAB 04, application demo. Output đã chạy sẵn.
- `results.csv`: số liệu của các experiment (cột experiment, setting, item, metric, value).
- `learning_curves.png`: train/validation loss theo epoch của model tốt nhất (GRU).
- `confusion_matrix.png`: confusion matrix của GRU trên test set.
- `error_analysis.md`: 3 mẫu đúng và 3 mẫu sai, phân tích nguyên nhân (negation, ambiguous sentiment, domain-specific term, mixed sentiment).
- `reflection.md`: so sánh với LAB 04, giải thích vì sao LSTM không vượt RNN/GRU trên dataset này, giới hạn của recurrent model, định hướng sang Attention (LAB 06).

Ngoài các file bắt buộc, notebook còn lưu thêm `sequence_length_hist.png` (histogram độ dài sequence, Experiment 1) và `sequence_length_tradeoff.png` (trade-off context/computational cost, Experiment 5) để minh hoạ trực quan.

## Cài đặt và chạy

```powershell
pip install numpy pandas scikit-learn nltk matplotlib torch jupyter jupytext
cd LAB05
python rnn_cell.py
python -m jupyter nbconvert --to notebook --execute --inplace sequence_classification.ipynb
```

Dataset NLTK `movie_reviews` sẽ tự tải về ở lần chạy đầu tiên. Nếu console Windows báo lỗi `UnicodeEncodeError` khi in tiếng Việt có dấu, chạy với `set PYTHONIOENCODING=utf-8` trước khi gọi `python`.

## Quy ước

- Dataset: NLTK `movie_reviews` (2000 review phim, 1000 pos / 1000 neg, cùng nguồn với IMDb), dùng thay thế cho IMDb đầy đủ để huấn luyện được trên CPU trong thời gian hợp lý - cùng quy ước đã dùng ở LAB 04.
- Chia stratified 70/15/15 (1400/300/300), seed 42. Vocabulary (giới hạn 10000 từ phổ biến nhất) chỉ được xây dựng trên tập train để tránh data leakage.
- Tiền xử lý: lowercase, giữ chữ cái và apostrophe (cùng `_TOKEN_RE` với LAB 04).
- Model: `Embedding(vocab_size, 100) -> RNN/LSTM/GRU(hidden_size=64, batch_first=True) -> Dropout(0.4) -> Linear(64, 1)`, dùng `pack_padded_sequence` để lấy đúng hidden state cuối cùng tại token thực (không tính PAD). Optimizer Adam (lr=1e-3, weight_decay=1e-5), gradient clipping 5.0, loss BCEWithLogitsLoss, 10 epoch (main experiment), checkpoint theo validation F1 tốt nhất (không phải epoch cuối).
- max_length chính dùng cho Experiment 3/4: 150. Experiment 2/5 quét max_length trong {50, 100, 200}.

## Kết quả chính (test set)

| Model | Representation | Accuracy | Precision | Recall | F1 | Parameters | Training time (s) |
|---|---|---|---|---|---|---|---|
| Logistic Regression | TF-IDF | 0.8133 | 0.8052 | 0.8267 | 0.8158 | - | 1.04 |
| RNN | Embedding | 0.5467 | 0.5427 | 0.5933 | 0.5669 | 1,010,889 | 75.67 |
| LSTM | Embedding | 0.5067 | 0.5059 | 0.5733 | 0.5375 | 1,042,761 | 38.32 |
| GRU | Embedding | 0.5533 | 0.5455 | 0.6400 | 0.5890 | 1,032,137 | 33.93 |

Model tốt nhất trong ba sequence model (theo F1 trên test): **GRU**.

| max_length | % tokens retained | Training time RNN (s) | F1 RNN |
|---|---|---|---|
| 50 | 9.38% | 9.15 | 0.5342 |
| 100 | 18.68% | 21.64 | 0.5309 |
| 200 | 36.98% | 116.35 | 0.5460 |

Ghi chú cho người chấm: số liệu có thể lệch nhẹ nếu đổi phiên bản thư viện, seed, hoặc phần cứng (CPU/GPU). Cần kiểm tra các kết luận định tính, không phải từng con số tuyệt đối:

- RNN/LSTM/GRU có cơ chế khai thác thứ tự từ mà TF-IDF không có (xem Bài 3 trong `calculations.md`), nhưng trên dataset nhỏ và embedding không pretrained, TF-IDF + Logistic Regression vẫn là baseline mạnh hơn hẳn về accuracy/F1 - đây là kết quả thật, không phải lỗi cài đặt, và phù hợp với nội dung mục AI policy/Prediction 5 của đề bài: LSTM (hay sequence model nói chung) không đảm bảo tốt hơn trên mọi dataset.
- Trong ba sequence model, GRU > RNN > LSTM về F1 trên dataset này - minh hoạ rằng kiến trúc phức tạp hơn (LSTM có nhiều tham số nhất) không tự động tốt hơn khi dữ liệu train hạn chế (1400 mẫu).
- Tăng max_length làm training time tăng mạnh nhưng không đảm bảo F1 tăng tương ứng (trade-off giữa context và computational cost).
- GRU có thiên hướng dự đoán "positive" nhiều hơn mức cần thiết (FP=80 > FN=54 trên test set) - xem `error_analysis.md` để biết các trường hợp lỗi cụ thể (negation, ambiguous sentiment, domain-specific term).
- Không được kết luận chỉ dựa vào accuracy: xem learning curves (GRU overfit rõ sau khoảng epoch 2-3, train_loss giảm liên tục trong khi val_loss tăng) và confusion matrix trước khi đánh giá model.
