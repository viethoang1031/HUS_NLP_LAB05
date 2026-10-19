# Calculations - LAB 05

## Bài 1 - Một bước RNN

Cho:

$$
x_t = \begin{bmatrix} 1 \\ 0 \end{bmatrix}, \quad
h_{t-1} = \begin{bmatrix} 0.5 \\ 0.2 \end{bmatrix}
$$

$$
W_{xh} = \begin{bmatrix} 0.5 & 0.2 \\ 0.1 & 0.4 \end{bmatrix}, \quad
W_{hh} = \begin{bmatrix} 0.3 & 0.1 \\ 0.2 & 0.5 \end{bmatrix}, \quad
b_h = \begin{bmatrix} 0 \\ 0 \end{bmatrix}
$$

**Bước 1: Tính $W_{xh} x_t$**

$$
W_{xh} x_t =
\begin{bmatrix} 0.5 & 0.2 \\ 0.1 & 0.4 \end{bmatrix}
\begin{bmatrix} 1 \\ 0 \end{bmatrix}
=
\begin{bmatrix} 0.5 \times 1 + 0.2 \times 0 \\ 0.1 \times 1 + 0.4 \times 0 \end{bmatrix}
=
\begin{bmatrix} 0.5 \\ 0.1 \end{bmatrix}
$$

**Bước 2: Tính $W_{hh} h_{t-1}$**

$$
W_{hh} h_{t-1} =
\begin{bmatrix} 0.3 & 0.1 \\ 0.2 & 0.5 \end{bmatrix}
\begin{bmatrix} 0.5 \\ 0.2 \end{bmatrix}
=
\begin{bmatrix} 0.3 \times 0.5 + 0.1 \times 0.2 \\ 0.2 \times 0.5 + 0.5 \times 0.2 \end{bmatrix}
=
\begin{bmatrix} 0.15 + 0.02 \\ 0.10 + 0.10 \end{bmatrix}
=
\begin{bmatrix} 0.17 \\ 0.20 \end{bmatrix}
$$

**Bước 3: Tính $a_t = W_{xh} x_t + W_{hh} h_{t-1} + b_h$**

$$
a_t =
\begin{bmatrix} 0.5 \\ 0.1 \end{bmatrix}
+
\begin{bmatrix} 0.17 \\ 0.20 \end{bmatrix}
+
\begin{bmatrix} 0 \\ 0 \end{bmatrix}
=
\begin{bmatrix} 0.67 \\ 0.30 \end{bmatrix}
$$

**Bước 4: Tính $h_t = \tanh(a_t)$**

$$
h_t =
\begin{bmatrix} \tanh(0.67) \\ \tanh(0.30) \end{bmatrix}
=
\begin{bmatrix} 0.584980 \\ 0.291313 \end{bmatrix}
$$

**Kết quả:**

$$
h_t \approx \begin{bmatrix} 0.5850 \\ 0.2913 \end{bmatrix}
$$

---

## Bài 2 - Information flow

Sequence: "I did not like this movie", model đọc trái sang phải, cập nhật theo công thức $h_t = f(x_t, h_{t-1})$.

- $h_{\text{I}}$: chỉ chứa thông tin của token "I".
- $h_{\text{did}}$: chứa thông tin của "I did".
- $h_{\text{not}}$: chứa thông tin của "I did not". Đây là điểm "not" được đưa vào state, nhưng bản thân $h_{\text{not}}$ chưa chứa tín hiệu phủ định hoàn chỉnh vì chưa gặp từ cần phủ định.
- $h_{\text{like}}$: được tính từ $x_{\text{like}}$ và $h_{\text{not}}$ theo công thức

$$
h_{\text{like}} = \tanh\left(W_{xh}\, x_{\text{like}} + W_{hh}\, h_{\text{not}} + b_h\right)
$$

Vì $h_{\text{like}}$ phụ thuộc trực tiếp vào $h_{\text{not}}$ (qua số hạng $W_{hh} h_{\text{not}}$), thông tin "not" có thể ảnh hưởng trực tiếp đến cách model biểu diễn "like" tại bước này: cụm "not like" có thể được hidden state mã hoá khác với "like" đứng một mình.

- $h_{\text{this}}, h_{\text{movie}}$: tiếp tục mang theo (một phần) tín hiệu "not like" do được truyền đệ quy từ $h_{\text{like}}$.

**So sánh với Average Word2Vec:**

$$
\vec{x} = \frac{1}{n} \sum_{i=1}^{n} \vec{w}_i
$$

Phép cộng vector có tính giao hoán, không có bước nào phụ thuộc thứ tự xuất hiện trước/sau. Vector của "not" và "like" được cộng vào tổng theo cách giống hệt như nếu chúng xuất hiện ở vị trí khác hoặc cách xa nhau trong câu. Do đó Average Word2Vec không thể hiện được quan hệ kề cận "not like" như RNN.

---

## Bài 3 - Sequence order

$$
S_1 = \text{"dog bites man"}, \qquad S_2 = \text{"man bites dog"}
$$

Cùng vocabulary $\{dog, bites, man\}$.

**Average Word2Vec:**

$$
\vec{x} = \frac{1}{3}\left(\vec{w}_{dog} + \vec{w}_{bites} + \vec{w}_{man}\right)
$$

Phép cộng vector giao hoán nên biểu thức trên cho kết quả giống hệt nhau với cả $S_1$ và $S_2$, bất kể thứ tự các từ. Do đó Average Word2Vec cho ra cùng một vector cho $S_1$ và $S_2$, không thể phân biệt hai câu này dựa trên thứ tự.

**RNN:**

$$
h_t = \tanh\left(W_{xh} x_t + W_{hh} h_{t-1} + b_h\right)
$$

được tính tuần tự theo từng bước thời gian.

Với $S_1$ ("dog bites man"): $h_1 = f(\text{dog})$, $h_2 = f(\text{bites}, h_1)$, $h_3 = f(\text{man}, h_2)$.

Với $S_2$ ("man bites dog"): $h_1' = f(\text{man})$, $h_2' = f(\text{bites}, h_1')$, $h_3' = f(\text{dog}, h_2')$.

Vì số hạng $W_{hh} h_{t-1}$ xuất hiện trong công thức, $h_t$ phụ thuộc vào toàn bộ chuỗi đã đọc trước đó, không chỉ phụ thuộc vào tập hợp các token. Do phép nhân ma trận nói chung không giao hoán, thứ tự đưa token vào ảnh hưởng trực tiếp đến giá trị của $h_t$ tại mỗi bước. Vì vậy $h_3$ (biểu diễn cuối cùng của $S_1$) và $h_3'$ (của $S_2$) nói chung khác nhau dù cùng một tập hợp token.

Tức là RNN có cơ chế (thông qua phép nhân ma trận không giao hoán và sự phụ thuộc tuần tự vào $h_{t-1}$) để phân biệt $S_1$ và $S_2$ dựa trên thứ tự, còn Average Word2Vec thì không có cơ chế nào để làm điều đó.

---

## Vanishing gradient - giải thích

Trong RNN, hidden state được truyền qua nhiều timestep: $h_1 \to h_2 \to \cdots \to h_T$.

Đạo hàm của loss theo hidden state ở một bước xa được truyền qua tích các Jacobian:

$$
\frac{\partial h_T}{\partial h_t} = \prod_{k=t+1}^{T} \frac{\partial h_k}{\partial h_{k-1}}
$$

Mỗi Jacobian $\dfrac{\partial h_k}{\partial h_{k-1}}$ chứa thành phần đạo hàm của $\tanh$ (giá trị nằm trong $(0, 1]$, thường nhỏ hơn 1 tại vùng $\tanh$ đã bão hoà) nhân với $W_{hh}$.

Nếu sequence càng dài ($T - t$ càng lớn), số lượng thừa số trong tích càng nhiều. Nếu mỗi thừa số có độ lớn trung bình nhỏ hơn 1:

$$
\left|\prod_k a_k\right| \to 0
$$

tích giảm theo hàm mũ với số bước, nên gradient truyền về các timestep đầu tiên (xa output) bị suy giảm rất nhanh tới gần 0 - đây là **vanishing gradient**. Khi đó trọng số liên quan đến các token ở đầu sequence hầu như không được cập nhật, model khó học được dependency giữa các token ở xa nhau.

Ngược lại nếu các thừa số lớn hơn 1:

$$
\left|\prod_k a_k\right| \to \infty
$$

tích có thể tăng theo hàm mũ - đây là **exploding gradient**, làm quá trình training không ổn định.

Đây là lý do, động lực chính để LSTM/GRU đưa vào các gate và cell state nhằm kiểm soát luồng gradient tốt hơn.

---

## Bài tập tính toán - LSTM gates

Cho một timestep với:

$$
f_t = 0.9, \qquad i_t = 0.2, \qquad \tilde{c}_t = 0.5, \qquad c_{t-1} = 0.8
$$

**Tính:**

$$
c_t = f_t \, c_{t-1} + i_t \, \tilde{c}_t
$$

$$
c_t = 0.9 \times 0.8 + 0.2 \times 0.5 = 0.72 + 0.10 = 0.82
$$

**Trả lời:**

Nếu forget gate gần 1: $f_t\, c_{t-1}$ gần bằng $c_{t-1}$, tức model giữ lại gần như toàn bộ thông tin cũ trong cell state, chỉ cộng thêm một lượng nhỏ thông tin mới ($i_t \tilde{c}_t$).

Nếu forget gate gần 0: $f_t\, c_{t-1}$ gần bằng 0, tức model gần như loại bỏ thông tin cũ, cell state mới chủ yếu được quyết định bởi $i_t \tilde{c}_t$ (thông tin mới tại bước hiện tại).
