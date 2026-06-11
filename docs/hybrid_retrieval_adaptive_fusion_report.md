# Báo cáo phát triển: Hybrid Retrieval và Adaptive Fusion cho CLaRa

## 1. Tóm tắt

CLaRa là một mô hình Retrieval-Augmented Generation (RAG) sử dụng biểu diễn liên tục trong không gian ẩn để nối quá trình truy hồi tài liệu và sinh câu trả lời. Thay vì đưa toàn bộ văn bản truy hồi vào ngữ cảnh của mô hình ngôn ngữ, CLaRa nén mỗi tài liệu thành một chuỗi memory embeddings. Ở giai đoạn end-to-end, Query Reasoner ánh xạ câu hỏi thành một vector truy vấn trong cùng không gian với memory embeddings, sau đó chọn các tài liệu phù hợp bằng cosine similarity và một biến thể differentiable top-k dùng Straight-Through Estimator.

Thiết kế này có ưu điểm lớn về hiệu quả và khả năng tối ưu truy hồi - sinh trong cùng một không gian liên tục. Tuy nhiên, dense latent retrieval thường gặp hạn chế khi câu hỏi phụ thuộc mạnh vào exact lexical matching, chẳng hạn tên riêng, mã số, mốc thời gian, thuật ngữ y tế/pháp lý, tên sản phẩm, tên tổ chức, hoặc các từ hiếm. Những tín hiệu này đôi khi không được thể hiện đủ mạnh trong vector dense, trong khi các phương pháp sparse retrieval như BM25 lại đặc biệt hiệu quả vì dựa trực tiếp trên sự xuất hiện và độ hiếm của từ khóa.

Vì vậy, hướng phát triển được thực hiện trong project này là bổ sung Hybrid Retrieval cho CLaRa: kết hợp điểm truy hồi latent của CLaRa với điểm BM25 lexical. Đồng thời, thay vì dùng một trọng số cố định trong mọi trường hợp, project bổ sung cơ chế adaptive fusion để điều chỉnh trọng số giữa dense latent score và BM25 score theo đặc điểm của từng truy vấn. Cách làm này giữ nguyên kiến trúc lõi của CLaRa, không yêu cầu huấn luyện lại compressor, không cần thêm mô hình lớn, và có thể được bật/tắt bằng tham số dòng lệnh.

Kết quả triển khai hiện tại bao gồm:

- Bổ sung cấu hình Hybrid Retrieval vào `CLaRaConfig`.
- Bổ sung BM25 scoring nội bộ cho danh sách candidate documents đã có trong batch.
- Bổ sung hàm adaptive alpha để quyết định mức độ ưu tiên latent score hoặc BM25 score theo từng query.
- Tích hợp fusion vào luồng stage2 training và inference trước bước differentiable top-k.
- Bổ sung tham số CLI để bật/tắt và điều chỉnh hybrid retrieval.
- Bổ sung script ablation để chạy ba cấu hình: baseline latent retrieval, fixed hybrid retrieval, adaptive hybrid retrieval.
- Bổ sung suffix tên file output để các kết quả baseline/fixed/adaptive không ghi đè nhau.
- Bổ sung unit test nhẹ cho logic BM25 và fusion, không cần load mô hình 7B.

Do môi trường hiện tại không có `torch` và không có checkpoint/GPU cần thiết để chạy CLaRa đầy đủ, báo cáo này chưa đưa ra kết quả benchmark chính thức về EM, F1 hoặc recall trên NQ, HotpotQA, MuSiQue, 2Wiki. Tuy nhiên, phần triển khai đã được kiểm tra cú pháp bằng AST và đã có test logic sẵn sàng chạy trong môi trường có dependencies.

## 2. Bối cảnh và vấn đề nghiên cứu

### 2.1. CLaRa và truy hồi trong không gian latent

Trong RAG truyền thống, một pipeline thường gồm hai thành phần tách rời: retriever dùng để tìm tài liệu liên quan và generator dùng để sinh câu trả lời từ tài liệu đã truy hồi. Cách làm này có hai hạn chế phổ biến. Thứ nhất, nếu đưa nhiều tài liệu văn bản vào context, chi phí tính toán tăng nhanh theo độ dài input. Thứ hai, retriever và generator thường được tối ưu bằng các mục tiêu khác nhau, dẫn đến tình trạng retriever chọn tài liệu tốt theo tiêu chí retrieval nhưng chưa chắc tối ưu cho generation.

CLaRa giải quyết vấn đề này bằng cách nén tài liệu thành memory embeddings. Compressor tạo biểu diễn liên tục từ tài liệu, generator nhận các embeddings này như memory tokens thay vì nhận toàn bộ văn bản. Trong stage2/end-to-end, Query Reasoner tạo query representation, so sánh với document memory representations, rồi chọn top-k tài liệu để đưa vào generator. Nhờ vậy, quá trình truy hồi và sinh được nối lại trong cùng không gian latent.

Trong source code hiện tại, phần này nằm chủ yếu trong [modeling_clara.py](../openrlhf/models/modeling_clara.py). Ở stage2, luồng chính gồm:

1. Query được encode bởi `query_reasoner_adapter`.
2. Candidate documents được compressor encode thành memory embeddings.
3. Query representation và document representations được chuẩn hóa.
4. Cosine similarity được tính bằng `torch.bmm`.
5. `differentiable_topk()` chọn top-k tài liệu.
6. Embeddings của các tài liệu được chọn thay thế vào vị trí memory tokens trong prompt sinh.

Thiết kế gốc có dạng:

```text
score_latent(q, d_i) = cosine(q_rep, d_rep_i)
top_k = differentiable_topk(score_latent)
```

### 2.2. Hạn chế của dense latent retrieval

Dense retrieval thường mạnh ở matching ngữ nghĩa. Ví dụ, một query hỏi "Who founded the company?" có thể tìm được tài liệu chứa "the company was established by..." dù không trùng hoàn toàn từ khóa. Đây là lợi thế lớn so với sparse retrieval thuần túy.

Tuy nhiên, dense retrieval có một số điểm yếu thực tế:

- **Tên riêng và thực thể hiếm**: Query chứa tên người, địa danh, tổ chức hoặc sản phẩm ít gặp có thể bị mất tín hiệu sau khi nén vector.
- **Mốc thời gian và số liệu**: Các token như năm, ngày tháng, mã luật, số điều khoản, mã sản phẩm thường cần exact match.
- **Thuật ngữ chuyên ngành**: Trong y tế, pháp luật hoặc tài liệu doanh nghiệp, nhiều thuật ngữ hiếm nhưng rất quan trọng.
- **Query ngắn**: Query ngắn thường chứa ít ngữ cảnh ngữ nghĩa; nếu một hoặc hai token chính bị bỏ qua, retrieval dễ sai.
- **Candidate documents gần nghĩa nhưng khác thực thể**: Dense model có thể đưa tài liệu liên quan về chủ đề nhưng sai thực thể, ví dụ cùng nói về "The Collegian" nhưng ở các trường đại học khác nhau.

BM25 lại phù hợp với các trường hợp này vì dựa trên term frequency, document frequency và độ dài tài liệu. Nếu query chứa một tên riêng hoặc mã định danh xuất hiện rõ trong tài liệu, BM25 thường đẩy tài liệu đó lên cao.

Do đó, hướng Hybrid Retrieval là hợp lý: không thay dense latent retrieval, mà bổ sung một nguồn tín hiệu lexical để giảm lỗi ở các query phụ thuộc exact match.

## 3. Mục tiêu phát triển

Mục tiêu của phần phát triển này là xây dựng một cơ chế Hybrid Retrieval cho CLaRa với các yêu cầu sau:

1. **Giữ nguyên kiến trúc lõi của CLaRa**  
   Không thay đổi compressor, generator, LoRA adapters hoặc format memory token.

2. **Không yêu cầu huấn luyện bổ sung để dùng được ngay**  
   Hybrid Retrieval phải có thể bật trên checkpoint CLaRa đã huấn luyện.

3. **Tích hợp trực tiếp vào stage2 retrieval**  
   Fusion phải xảy ra trước `differentiable_topk`, để quyết định top-k tài liệu được đưa vào generator.

4. **Hỗ trợ cả fixed fusion và adaptive fusion**  
   Fixed fusion dùng một trọng số alpha cố định. Adaptive fusion thay đổi alpha theo query.

5. **Có thể cấu hình bằng CLI**  
   Người dùng có thể bật/tắt và điều chỉnh tham số mà không sửa code.

6. **Có workflow đánh giá rõ ràng**  
   Cần có script chạy baseline, fixed hybrid và adaptive hybrid để so sánh kết quả.

7. **Không ghi đè kết quả thí nghiệm**  
   File output cần có suffix riêng cho từng cấu hình.

8. **Có kiểm tra logic tối thiểu**  
   BM25 và fusion nên có unit test nhẹ, không cần load mô hình lớn.

## 4. Phương pháp đề xuất

### 4.1. Công thức fusion tổng quát

Với mỗi câu hỏi `q` và candidate document `d_i`, CLaRa gốc tính latent score:

```text
s_latent_i = cosine(q_rep, d_rep_i)
```

Trong phần phát triển này, ta bổ sung BM25 score:

```text
s_bm25_i = BM25(q, d_i)
```

Vì hai loại score có thang đo khác nhau, cần chuẩn hóa trước khi cộng. Latent cosine score thường nằm trong khoảng `[-1, 1]`, nên được đưa về `[0, 1]`:

```text
s_latent_norm_i = clip((s_latent_i + 1) / 2, 0, 1)
```

BM25 score được min-max normalize theo từng query:

```text
s_bm25_norm_i = (s_bm25_i - min_j s_bm25_j) / (max_j s_bm25_j - min_j s_bm25_j)
```

Nếu mọi BM25 score bằng nhau, vector BM25 normalized được đặt bằng 0 để tránh chia cho 0 và để latent retrieval tiếp tục chiếm ưu thế.

Điểm fusion cuối cùng:

```text
s_fused_i = alpha * s_latent_norm_i + (1 - alpha) * s_bm25_norm_i
```

Trong đó:

- `alpha` gần 1 nghĩa là ưu tiên CLaRa latent retrieval.
- `alpha` gần 0 nghĩa là ưu tiên BM25 lexical retrieval.

Sau đó, `s_fused` được đưa vào `differentiable_topk()` thay vì `s_latent`.

### 4.2. Fixed fusion

Ở chế độ fixed fusion, alpha là một hằng số:

```text
alpha = 0.75
```

Điều này nghĩa là latent retrieval vẫn là nguồn chính, BM25 đóng vai trò điều chỉnh. Đây là cấu hình an toàn vì không phá vỡ hành vi gốc của CLaRa quá mạnh. Nếu BM25 không có tín hiệu rõ, latent score vẫn quyết định. Nếu BM25 có tín hiệu mạnh, nó có thể thay đổi thứ hạng của candidate documents.

Fixed fusion phù hợp để làm baseline đầu tiên cho hướng hybrid.

### 4.3. Adaptive fusion

Fixed alpha có một hạn chế: không phải query nào cũng cần BM25 ở cùng mức độ. Có query cần suy luận ngữ nghĩa, có query lại phụ thuộc exact match. Vì vậy project bổ sung adaptive fusion.

Ý tưởng chính: tính một lexical signal cho mỗi query. Lexical signal càng cao thì query càng có khả năng cần BM25. Khi đó alpha được giảm xuống để BM25 có trọng số lớn hơn.

Adaptive alpha nằm trong khoảng:

```text
alpha_min <= alpha(q) <= alpha_max
```

Mặc định:

```text
alpha_min = 0.45
alpha_max = 0.90
```

Nếu query ít phụ thuộc lexical matching:

```text
alpha(q) gần 0.90
```

Nếu query phụ thuộc nhiều vào exact match:

```text
alpha(q) gần 0.45
```

Trong triển khai hiện tại, lexical signal được tính từ ba thành phần:

1. **Specific token ratio**  
   Tỷ lệ token có dấu hiệu là thực thể hoặc thuật ngữ cụ thể. Một token được xem là specific nếu:
   - chứa chữ số;
   - chứa dấu `-`, `/`, `.`;
   - có độ dài lớn;
   - có chữ cái đầu viết hoa.

2. **BM25 confidence**  
   Khoảng cách giữa BM25 normalized score của document top-1 và top-2. Nếu khoảng cách lớn, BM25 có một lựa chọn rõ ràng.

3. **Short specific query indicator**  
   Query ngắn nhưng có token cụ thể thường dễ phụ thuộc exact match.

Tín hiệu tổng hợp:

```text
lexical_signal =
    0.45 * specificity
  + 0.35 * bm25_confidence
  + 0.20 * short_specific_query
```

Sau đó:

```text
alpha(q) = alpha_max - lexical_signal * (alpha_max - alpha_min)
```

Thiết kế này là heuristic, không cần training. Ưu điểm là nhẹ, dễ giải thích, có thể chạy ngay trên checkpoint cũ. Nhược điểm là chưa tối ưu theo dữ liệu. Trong tương lai có thể thay bằng một learnable fusion head.

## 5. Thay đổi trong source code

### 5.1. Bổ sung cấu hình trong `CLaRaConfig`

File chính được chỉnh sửa là [openrlhf/models/modeling_clara.py](../openrlhf/models/modeling_clara.py). Trong `CLaRaConfig`, các tham số mới được thêm:

```python
hybrid_retrieval: bool = False
hybrid_alpha: float = 0.75
hybrid_adaptive_fusion: bool = False
hybrid_alpha_min: float = 0.45
hybrid_alpha_max: float = 0.90
bm25_k1: float = 1.2
bm25_b: float = 0.75
```

Các tham số này cũng được lưu vào config để checkpoint có thể ghi nhớ cấu hình hybrid retrieval.

Ý nghĩa:

- `hybrid_retrieval`: bật/tắt Hybrid Retrieval.
- `hybrid_alpha`: trọng số fixed alpha.
- `hybrid_adaptive_fusion`: bật/tắt adaptive alpha.
- `hybrid_alpha_min`: alpha nhỏ nhất trong adaptive mode.
- `hybrid_alpha_max`: alpha lớn nhất trong adaptive mode.
- `bm25_k1`, `bm25_b`: tham số BM25.

### 5.2. Bổ sung tokenizer lexical

Hàm `_lexical_tokens()` được thêm để chuyển query/document thành danh sách token lexical:

```python
tokens = re.findall(r"[A-Za-z0-9][A-Za-z0-9_./-]*", text or "")
return [tok.lower() for tok in tokens if tok.lower() not in LEXICAL_STOPWORDS]
```

Cách tokenization này đơn giản, không cần dependency ngoài, phù hợp với mục tiêu nhẹ và dễ tích hợp. Stopword list được định nghĩa trong `LEXICAL_STOPWORDS`.

### 5.3. Bổ sung BM25 scoring

Hàm `_compute_bm25_scores()` được thêm để tính BM25 score cho từng query trên danh sách candidate documents tương ứng.

Input:

- `questions`: danh sách query.
- `documents`: danh sách danh sách candidate documents.
- `device`, `dtype`: để output tensor khớp với latent score.

Output:

```text
Tensor shape [B, N]
```

Trong đó:

- `B`: batch size.
- `N`: số candidate documents mỗi query.

BM25 được tính nội bộ bằng `Counter`, không dùng thư viện ngoài. Điều này tránh thêm dependency và giúp code chạy được ngay trong môi trường hiện tại.

### 5.4. Bổ sung adaptive alpha

Hàm `_compute_hybrid_alpha()` trả về alpha cho từng query:

```text
shape [B, 1]
```

Nếu `hybrid_adaptive_fusion=False`, hàm trả về alpha cố định.

Nếu `hybrid_adaptive_fusion=True`, hàm tính alpha dựa trên:

- độ cụ thể của query;
- độ tự tin của BM25;
- query ngắn có token cụ thể hay không.

### 5.5. Bổ sung hàm fusion

Hàm `_fuse_retrieval_scores()` là điểm trung tâm của phần phát triển:

```python
fused_scores = alpha * latent_norm + (1.0 - alpha) * bm25_norm
```

Hàm này nhận:

- latent scores `[B, N]`;
- questions;
- documents.

Nếu `hybrid_retrieval=False`, hoặc không có raw documents, hàm trả lại latent scores gốc. Điều này giữ backward compatibility.

### 5.6. Tích hợp vào stage2 inference

Trong `generate_from_questions()`, ở nhánh `stage2_mips=False`, sau khi tính cosine score:

```python
scores = torch.bmm(...).squeeze(1)
```

project bổ sung:

```python
scores, _, _ = self._fuse_retrieval_scores(scores, questions, documents)
```

Như vậy, khi inference với candidate documents được truyền trực tiếp, CLaRa sẽ dùng fused score để chọn top-k.

Lưu ý: nhánh `stage2_mips=True` hiện vẫn dùng latent score gốc vì nhánh này truy hồi embeddings từ một service ngoài và không có raw document text để tính BM25. Nếu muốn hỗ trợ hybrid cho MIPS/ANN index, cần index thêm text hoặc precomputed lexical score.

### 5.7. Tích hợp vào stage2 training

Trong `_forward_stage2_batch()`, project bổ sung fusion ở cùng vị trí: sau cosine score và trước `differentiable_topk()`.

Luồng mới:

```text
query_reps
retrieved_doc_embeddings
latent_scores
BM25_scores
fused_scores
differentiable_topk(fused_scores)
selected embeddings
generator loss
```

Điểm quan trọng là BM25 chỉ ảnh hưởng đến lựa chọn top-k. Loss sinh vẫn là language modeling loss như CLaRa gốc. Vì BM25 là tín hiệu không khả vi theo text, nó không truyền gradient về compressor/query reasoner. Tuy nhiên, fused score vẫn đi qua `differentiable_topk()` và latent score vẫn nằm trong biểu thức fusion, nên gradient qua phần latent vẫn tồn tại.

### 5.8. Bổ sung tham số CLI

File [openrlhf/cli/train_sft.py](../openrlhf/cli/train_sft.py) được bổ sung các tham số:

```bash
--hybrid_retrieval
--hybrid_alpha
--hybrid_adaptive_fusion
--hybrid_alpha_min
--hybrid_alpha_max
--bm25_k1
--bm25_b
```

Ví dụ train stage2 với adaptive hybrid:

```bash
python -m openrlhf.cli.train_sft \
  --stage stage2 \
  --hybrid_retrieval \
  --hybrid_adaptive_fusion \
  --hybrid_alpha_min 0.45 \
  --hybrid_alpha_max 0.90 \
  ...
```

Ngoài ra, phần validation được thêm để đảm bảo:

- alpha nằm trong `[0, 1]`;
- `bm25_k1 > 0`;
- `bm25_b` nằm trong `[0, 1]`.

### 5.9. Sửa lỗi nhỏ khi load checkpoint

Trong `setup_model()`, trước đó code truyền:

```python
compress_rate=args.compress_rate
```

Trong khi `CLaRaConfig` dùng tên:

```python
compr_rate
```

Project đã sửa thành:

```python
compr_rate=args.compress_rate
```

Điều này quan trọng vì nếu load checkpoint với compression rate khác, override cũ có thể không có tác dụng.

### 5.10. Bổ sung tham số evaluation

File [evaluation/evaluate.py](../evaluation/evaluate.py) được bổ sung các tham số hybrid tương tự. Điều này cho phép chạy hybrid retrieval trên checkpoint cũ mà không cần train lại.

Ví dụ:

```bash
accelerate launch evaluate.py \
  --model_path clara_stage2_checkpoint \
  --stage stage2 \
  --dataset musique \
  --generation_top_k 5 \
  --hybrid_retrieval \
  --hybrid_adaptive_fusion
```

### 5.11. Tránh ghi đè kết quả evaluation

Trước khi chỉnh sửa, evaluator ghi kết quả theo format:

```text
{dataset}_{stage}_{gold_retrieval}_{generation_top_k}.jsonl
results_metrics_{stage}_{gold_retrieval}_{generation_top_k}.json
```

Nếu chạy baseline rồi chạy hybrid với cùng dataset và top-k, kết quả sẽ bị ghi đè. Project đã thêm suffix:

```text
_hybrid_fixed_a0p75
_hybrid_adaptive_a0p45-0p9
```

Ví dụ:

```text
musique_stage2_True_5.jsonl
musique_stage2_True_5_hybrid_fixed_a0p75.jsonl
musique_stage2_True_5_hybrid_adaptive_a0p45-0p9.jsonl
```

Điều này giúp quản lý ablation rõ ràng hơn.

### 5.12. Bổ sung script ablation

Script mới [scripts/evaluation_hybrid_ablation.sh](../scripts/evaluation_hybrid_ablation.sh) chạy ba cấu hình:

1. Baseline CLaRa latent retrieval.
2. Fixed hybrid retrieval.
3. Adaptive hybrid retrieval.

Các biến môi trường chính:

```bash
SAVE_PATH=/path/to/stage2_checkpoint
DATASETS=musique,hotpotqa,2wiki,nq
GENERATION_TOP_K=5
BATCH_SIZE=4
GOLD_RETRIEVAL_FLAG=--gold_retrieval
```

Lệnh chạy:

```bash
bash scripts/evaluation_hybrid_ablation.sh
```

### 5.13. Bổ sung unit test

File [tests/test_hybrid_retrieval.py](../tests/test_hybrid_retrieval.py) kiểm tra ba điểm:

1. BM25 ưu tiên document có exact query terms.
2. Fixed fusion có thể thay đổi top document nếu BM25 có tín hiệu mạnh.
3. Adaptive alpha giảm về phía BM25 khi query có dấu hiệu lexical-specific.

Test được thiết kế để không khởi tạo CLaRa 7B. Thay vào đó:

```python
model = CLaRa.__new__(CLaRa)
```

Sau đó gán các thuộc tính cần thiết và gọi trực tiếp helper functions.

## 6. Thiết kế thực nghiệm đề xuất

Để đánh giá đầy đủ Hybrid Retrieval + Adaptive Fusion, cần chạy ablation trên cùng checkpoint stage2.

### 6.1. Các cấu hình cần so sánh

| Cấu hình | Mô tả |
|---|---|
| Baseline | CLaRa gốc, chỉ dùng latent cosine score |
| Hybrid fixed | Fusion latent + BM25 với `alpha=0.75` |
| Hybrid adaptive | Fusion latent + BM25, alpha thay đổi theo query |

### 6.2. Dataset

Các benchmark phù hợp với repo hiện tại:

- Natural Questions (NQ)
- HotpotQA
- MuSiQue
- 2WikiMultiHopQA

Trong đó:

- NQ thường có nhiều query factoid, entity/date matching hữu ích.
- HotpotQA, MuSiQue, 2Wiki có nhiều multi-hop reasoning; cần kiểm tra BM25 có giúp recall hay làm nhiễu semantic retrieval.

### 6.3. Metric

Nên báo cáo hai nhóm metric.

Nhóm retrieval:

- Recall@1
- Recall@3
- Recall@5
- Precision@1
- Precision@3
- Precision@5

Nhóm generation:

- Exact Match
- Cover Exact Match hoặc Accuracy như code hiện tại
- F1
- Average output length

Lý do cần cả hai nhóm: Hybrid Retrieval có thể cải thiện recall nhưng chưa chắc cải thiện generation nếu generator không tận dụng tốt tài liệu được chọn. Ngược lại, generation metric tăng là bằng chứng mạnh hơn rằng retrieval cải thiện hữu ích cho downstream QA.

### 6.4. Kỳ vọng

Kỳ vọng chính:

- Hybrid Retrieval tăng recall ở query có entity/date/rare terms.
- Adaptive fusion ổn định hơn fixed fusion vì không ép BM25 quá mạnh với query cần semantic matching.
- Trên domain nhiều thuật ngữ chuyên biệt, hybrid adaptive có khả năng cải thiện rõ hơn baseline.

Rủi ro:

- BM25 có thể ưu tiên document có nhiều token trùng nhưng không trả lời đúng câu hỏi.
- Min-max normalization theo candidate set nhỏ có thể làm BM25 score bị phóng đại nếu chỉ có một document hơi trùng từ.
- Adaptive heuristic chưa được học từ dữ liệu nên có thể chưa tối ưu.

## 7. Kết quả triển khai

### 7.1. Kết quả về chức năng

Phần Hybrid Retrieval đã được tích hợp vào luồng stage2 chính của CLaRa. Khi không bật `--hybrid_retrieval`, hành vi mặc định giữ nguyên như baseline. Khi bật `--hybrid_retrieval`, điểm truy hồi latent được fuse với BM25 trước bước top-k.

Các thay đổi có tính backward compatible:

- Checkpoint cũ vẫn load được.
- Nếu không truyền raw documents, fusion tự fallback về latent scores.
- Nếu BM25 không có tín hiệu hoặc shape không khớp, fusion không can thiệp.
- Stage1 và stage1_2 không bị ảnh hưởng.

### 7.2. Kết quả kiểm tra kỹ thuật

Đã chạy kiểm tra cú pháp bằng Python AST:

```text
AST ok
```

Đã chạy unit test:

```text
Ran 3 tests
OK (skipped=3)
```

Các test bị skip vì môi trường hiện tại thiếu `torch`. Đây không phải lỗi logic của test; test đã được thiết kế để skip khi dependency deep learning chưa được cài. Trong môi trường đầy đủ của project, các test này sẽ chạy phần BM25/fusion mà không cần load model 7B.

### 7.3. Kết quả thực nghiệm benchmark

Hiện tại chưa có kết quả benchmark chính thức về Recall/EM/F1 vì chưa chạy inference/training trên checkpoint CLaRa stage2 trong môi trường GPU. Do đó chưa thể kết luận định lượng rằng Hybrid Retrieval cải thiện bao nhiêu phần trăm.

Báo cáo kết quả thực nghiệm nên được bổ sung sau khi chạy:

```bash
SAVE_PATH=/path/to/stage2_checkpoint \
DATASETS=musique,hotpotqa,2wiki,nq \
bash scripts/evaluation_hybrid_ablation.sh
```

Sau khi chạy, evaluator sẽ tạo các file metric riêng biệt cho từng cấu hình, ví dụ:

```text
results_metrics_stage2_True_5.json
results_metrics_stage2_True_5_hybrid_fixed_a0p75.json
results_metrics_stage2_True_5_hybrid_adaptive_a0p45-0p9.json
```

Các file này có thể dùng để điền bảng kết quả cuối cùng.

## 8. Phân tích tác động của thay đổi

### 8.1. Tác động đến chất lượng retrieval

Hybrid Retrieval bổ sung một tín hiệu mà CLaRa gốc chưa khai thác trực tiếp: exact lexical overlap. Tác động tích cực dự kiến rõ nhất ở các query có:

- tên người;
- địa danh;
- tên tổ chức;
- ngày tháng;
- số liệu;
- mã văn bản;
- thuật ngữ dài hoặc hiếm.

Ví dụ, với query:

```text
When was Timothy McVeigh executed?
```

BM25 có thể ưu tiên document chứa trực tiếp `Timothy`, `McVeigh`, `executed`, trong khi latent retrieval có thể chọn document nói chung về capital punishment nếu ngữ nghĩa gần.

### 8.2. Tác động đến generation

Generator của CLaRa không nhìn thấy raw text mà nhìn thấy compressed memory embeddings. Nếu hybrid retrieval chọn đúng tài liệu hơn, embeddings đưa vào generator sẽ chứa thông tin chính xác hơn. Do đó generation có thể cải thiện.

Tuy nhiên, cải thiện generation không tự động xảy ra. Nếu compressor nén mất chi tiết quan trọng, hoặc generator không khai thác tốt memory embeddings, recall tăng có thể không chuyển thành F1/EM tăng. Vì vậy cần đánh giá cả retrieval và generation.

### 8.3. Tác động đến chi phí tính toán

Chi phí BM25 nội bộ hiện tính trên candidate documents trong batch, không phải toàn bộ corpus. Vì vậy overhead tương đối thấp. Với stage2 hiện tại, mỗi query thường có số candidate hữu hạn, ví dụ 5, 20 hoặc vài chục documents. BM25 trên phạm vi này rẻ hơn nhiều so với forward pass của LLM.

Chi phí bổ sung gồm:

- tokenize lexical bằng regex;
- đếm term frequency bằng `Counter`;
- tính BM25 score;
- normalize và fuse tensor.

So với chi phí encode documents bằng LLM compressor và generation bằng decoder 7B, phần này nhỏ.

### 8.4. Tác động đến khả năng mở rộng

Triển khai hiện tại là reranking/fusion trên candidate set đã có. Nó chưa phải full hybrid search trên toàn corpus. Điều này phù hợp với code hiện tại vì stage2 nhận candidate documents từ dataset hoặc retrieval pipeline trước đó.

Nếu muốn mở rộng lên corpus lớn, có hai hướng:

1. Dùng BM25 để lấy một candidate set ban đầu, sau đó CLaRa latent retrieval rerank.
2. Dùng ANN/FAISS cho latent embeddings và BM25 index song song, sau đó fuse hai danh sách kết quả bằng Reciprocal Rank Fusion hoặc score fusion.

Phiên bản hiện tại là bước đầu an toàn để kiểm chứng giả thuyết hybrid trước khi xây dựng index lớn.

## 9. Hạn chế hiện tại

### 9.1. Chưa hỗ trợ hybrid cho `stage2_mips=True`

Nhánh `stage2_mips=True` gọi service ngoài để lấy embeddings truy hồi. Nhánh này không có raw document text, nên không thể tính BM25 trực tiếp. Hiện tại hybrid chỉ hoạt động khi documents được truyền vào model.

Để hỗ trợ MIPS/ANN, cần:

- lưu raw document text hoặc document id;
- truy xuất text tương ứng với candidate embeddings;
- tính BM25 trên candidate text;
- hoặc precompute sparse scores/index riêng.

### 9.2. Adaptive fusion đang là heuristic

Adaptive alpha hiện dựa trên quy tắc thủ công. Quy tắc này dễ giải thích và không cần training, nhưng chưa chắc tối ưu trên mọi dataset.

Một hướng cải tiến là học alpha bằng một module nhỏ:

```text
alpha = sigmoid(MLP([query_rep, lexical_features]))
```

Tuy nhiên hướng này cần thêm training data và kiểm soát overfitting.

### 9.3. Tokenization BM25 còn đơn giản

Regex tokenizer hiện phù hợp cho tiếng Anh và token dạng chữ/số. Nếu áp dụng cho tiếng Việt hoặc domain đặc biệt, cần cải thiện tokenization.

Ví dụ:

- tiếng Việt cần word segmentation tốt hơn;
- y tế có nhiều ký hiệu như `IL-6`, `TNF-alpha`, `HbA1c`;
- pháp luật có nhiều pattern như `Article 5`, `Section 12(b)`.

### 9.4. Normalization có thể ảnh hưởng thứ hạng

Latent score và BM25 score được chuẩn hóa khác nhau. Latent cosine được đưa từ `[-1, 1]` về `[0, 1]`, BM25 dùng min-max theo query. Nếu candidate set nhỏ, min-max normalization có thể làm chênh lệch BM25 trở nên lớn hơn thực tế.

Có thể thử các normalization khác:

- z-score;
- softmax temperature;
- rank-based normalization;
- reciprocal rank fusion.

### 9.5. Chưa có kết quả benchmark chính thức

Hiện chỉ có kết quả triển khai và kiểm tra cú pháp. Để kết luận khoa học đầy đủ, cần chạy ablation và báo cáo metric định lượng.

## 10. Hướng phát triển tiếp theo

### 10.1. Learnable adaptive fusion

Thay heuristic alpha bằng một head học được:

```text
features = [query_rep, bm25_top1, bm25_margin, specificity, query_length]
alpha = sigmoid(MLP(features))
```

Ưu điểm:

- alpha được tối ưu theo downstream loss;
- có thể học các pattern phức tạp hơn heuristic.

Nhược điểm:

- cần training;
- có nguy cơ overfit;
- cần thiết kế để không phá ổn định của differentiable top-k.

### 10.2. Reciprocal Rank Fusion

Thay score fusion bằng rank fusion:

```text
RRF(d) = 1 / (k + rank_latent(d)) + 1 / (k + rank_bm25(d))
```

RRF thường ổn định khi score scale giữa hai retriever rất khác nhau. Đây là hướng ablation tốt vì dễ triển khai và không cần training.

### 10.3. Hard negative mining

BM25 có thể tìm các hard negatives: tài liệu trùng nhiều từ khóa nhưng không chứa đáp án đúng. Dùng các negatives này để fine-tune Query Reasoner có thể tăng khả năng phân biệt entity gần giống.

### 10.4. Domain-specific lexical analyzer

Nếu project chuyển sang y tế/pháp luật/doanh nghiệp, có thể thêm tokenizer chuyên ngành:

- nhận diện mã thuốc, bệnh, gene, đơn vị đo;
- nhận diện điều luật, khoản, mục;
- nhận diện mã sản phẩm hoặc mã hợp đồng.

### 10.5. Hybrid indexing quy mô lớn

Về lâu dài, cần xây dựng song song:

- dense latent index cho CLaRa memory embeddings;
- sparse BM25 index cho raw text.

Sau đó fuse kết quả từ hai index trước khi đưa vào generator.

## 11. Kết luận

Phần phát triển Hybrid Retrieval + Adaptive Fusion là một hướng mở rộng phù hợp cho CLaRa vì giải quyết một hạn chế thực tế của dense latent retrieval: khả năng xử lý exact lexical matching, rare terms, tên riêng, số liệu và thuật ngữ chuyên ngành. Thay đổi này không thay thế thiết kế gốc của CLaRa mà bổ sung một tín hiệu sparse nhẹ, dễ kiểm soát và có thể bật/tắt bằng cấu hình.

Về mặt kỹ thuật, project đã thêm BM25 scoring, score normalization, fixed fusion, adaptive fusion, CLI flags, evaluation suffix, ablation script và unit test. Luồng stage2 giờ có thể dùng fused retrieval score trước bước differentiable top-k. Khi không bật hybrid, mô hình giữ nguyên hành vi baseline.

Kết quả hiện tại là kết quả triển khai và kiểm tra kỹ thuật: code đã pass AST parsing, unit test đã được thêm và sẵn sàng chạy trong môi trường có `torch`. Chưa có kết quả benchmark định lượng vì chưa chạy trên GPU/checkpoint thật. Do đó, kết luận khoa học ở thời điểm này là: phương pháp đã được tích hợp thành công và có cơ sở hợp lý để kỳ vọng cải thiện retrieval trong các query phụ thuộc lexical matching; tuy nhiên cần chạy ablation trên NQ, HotpotQA, MuSiQue và 2Wiki để xác nhận mức cải thiện thực tế.

Nếu kết quả ablation cho thấy adaptive hybrid cải thiện Recall@k và F1/EM mà không làm tăng nhiều latency, đây sẽ là một đóng góp thực dụng và có tính mở rộng tốt cho CLaRa, đặc biệt khi áp dụng vào các domain chuyên biệt như y tế, pháp luật và tài liệu doanh nghiệp.
