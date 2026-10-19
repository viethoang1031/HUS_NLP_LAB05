# Error analysis - LAB 05

Model đánh giá: GRU, max_length = 150, vocab_size = 10002 (xem `results.csv`, `eval`).

Trên tập test (300 mẫu), model dự đoán sai 134/300 mẫu (44.7%), accuracy 0.5533, F1 0.5890. Confusion matrix: TN=70, FP=80, FN=54, TP=96 (xem `confusion_matrix.png`). FP (80) khá cao so với TN (70) cho thấy model có xu hướng thiên lệch, dự đoán "positive" nhiều hơn mức cần thiết trên các mẫu thực sự là "negative".

## Bảng mẫu

| # | Đoạn text (rút gọn) | True | Predicted | Phân tích |
|---|---|---|---|---|
| 1 | "dead-bang, an action thriller starring don johnson, lives up to its dumb title... dead-bang is definitely dead... should have called this one 'dead bore.'..." | negative | negative | Đúng. Văn bản chứa nhiều từ tiêu cực tường minh ("dumb", "dead", "dead bore"), không có phủ định/tương phản phức tạp, model dễ nắm được tín hiệu sentiment. |
| 2 | "synopsis: blond criminal psychologist sarah chooses to copulate with greasy tony... someone sends sarah rotting flowers, puts her name in the newspaper's obituary column, and kills her cat..." | negative | negative | Đúng. Nội dung mô tả cốt truyện theo hướng u ám ("rotting flowers", "obituary", "kills her cat"), từ vựng tiêu cực xuất hiện dày đặc, không phụ thuộc nhiều vào long-range dependency để nhận ra sentiment. |
| 3 | "you'd think it would be easy to spoof star trek, but try spoofing it without cruelly mocking it. that's a bit more difficult. the farfetched sci-fi series is, after all, enormously popular..." | positive | positive | Đúng, dù câu mở đầu có cấu trúc tương phản ("but try spoofing it without cruelly mocking it"). Văn bản đủ dài (150 token được giữ lại) nên model có nhiều tín hiệu tích lũy qua review để đi đến dự đoán đúng. |
| 4 | "tbwp is probably the single most profitable film ever. not surprising, considering its tiny us$100,000 budget, to date it has earned in excess of us$130 million. that's a shocking 100 times profit!..." | positive | negative | Sai. Đoạn đầu review bàn về thành công thương mại (doanh thu, lợi nhuận) chứ chưa đi thẳng vào đánh giá chất lượng phim; các từ như "shocking", nội dung xoay quanh tiền bạc có thể bị model hiểu nhầm là tín hiệu tiêu cực/trung tính. Đây là lỗi domain-specific/ambiguous sentiment hơn là lỗi do sequence modeling hay negation. |
| 5 | "capsule: the running gag pair of characters from all of kevin smith's films gets their own movie. the gags are sporadically funny. it is more than occasionally funny for teens who are fond of scatological humor and anti-gay jokes. the plot is weak and the leads are not a particularly funny co..." | negative | positive | Sai. Review có nhiều từ tích cực cục bộ ("funny" lặp lại nhiều lần, "sporadically funny", "more than occasionally funny") trước khi chuyển sang đánh giá tiêu cực ("the plot is weak", "not a particularly funny"). Đây là lỗi liên quan tới negation ("not... funny") và ambiguous/mixed sentiment: nhiều tín hiệu tích cực cục bộ ở đầu có thể lấn át tín hiệu tiêu cực xuất hiện muộn hơn trong câu bị cắt ở max_length 150. |
| 6 | "i'm not quite sure what to say about mars attacks!, which is obviously the work of a deranged genius. when tim burton's twisted alien invasion comedy really works, it's breathtaking and hilarious in equal measure. and when it doesn't work, it's just dull. i'm not even sure it works more of..." | positive | negative | Sai. Review có cấu trúc tương phản xen kẽ liên tục ("works... breathtaking and hilarious" / "doesn't work... just dull"), mang tính ambiguous sentiment thực sự (ngay cả người đọc cũng khó chắc chắn ngay từ đoạn đầu). Day là ví dụ về negation + contrast dồn dập khiến model khó tổng hợp đúng xu hướng chung. |

## Nhận xét chung

Không mặc định "LSTM/GRU không hiểu câu" cho mọi trường hợp sai. Với mẫu #1, #2, lỗi không xảy ra vì tín hiệu sentiment đủ tường minh và lặp lại. Với các mẫu sai (#4, #5, #6), nguyên nhân khả dĩ gồm:

- #4: ambiguous sentiment / domain-specific term - đoạn đầu review bàn chuyện doanh thu chứ chưa đánh giá phim, không phải lỗi do sequence modeling.
- #5: negation ("not a particularly funny") kết hợp với nhiều tín hiệu tích cực cục bộ xuất hiện trước đó trong câu - đây có thể là lỗi liên quan tới sequence modeling, vì model cần giữ được trạng thái đúng qua một chuỗi dài các tính từ tích cực trước khi gặp phần phủ định ở cuối.
- #6: negation + contrast lặp lại nhiều lần trong cùng một đoạn, cộng với ambiguous sentiment ngay cả về mặt ngôn ngữ tự nhiên - khó quy hoàn toàn cho một nguyên nhân duy nhất.

Ngoài ra, confusion matrix cho thấy model thiên về dự đoán "positive" (FP=80 > FN=54), phù hợp với quan sát ở phần Application trong notebook: câu phủ định đơn giản "I did not like this movie at all." vẫn bị dự đoán là Positive (probability 0.669). Đây là dấu hiệu cho thấy một phần lỗi không chỉ đến từ các trường hợp câu khó (negation, contrast) mà còn từ việc model học được một prior lệch (thiên vị lớp "positive") do dataset nhỏ và embedding khởi tạo ngẫu nhiên, không phải lỗi thuần tuý về sequence modeling.
