# Kết luận thực nghiệm Hybrid Retrieval trên MuSiQue Subset

## 1. Mục tiêu

Phần thực nghiệm này đánh giá nhanh hướng phát triển Hybrid Retrieval cho CLaRa. Mục tiêu là kiểm tra liệu việc kết hợp dense latent retrieval của CLaRa với sparse lexical retrieval BM25 có thể cải thiện hiệu quả truy xuất và chất lượng trả lời end-to-end hay không.

Đây là một pilot evaluation, không phải full benchmark.

## 2. Thiết lập thực nghiệm

Mô hình và cấu hình:

- Model: `CLaRa-7B-E2E`
- Compression setting: `compression-16`
- Dataset: `MuSiQue`
- Stage: `stage2`
- Gold retrieval context: có dùng `--gold_retrieval`
- Số mẫu đánh giá: `MAX_EVAL_SAMPLES=100`
- `GENERATION_TOP_K=2`
- `BATCH_SIZE=1`
- Quantization: `int4`
- Device map: `auto`

Ba cấu hình được so sánh:

| Phương pháp | Mô tả |
|---|---|
| Baseline | CLaRa latent retrieval gốc |
| Hybrid fixed | BM25 + latent fusion với `alpha=0.90`, `candidate_top_m=3` |
| Hybrid adaptive | Adaptive fusion với `alpha_min=0.75`, `alpha_max=0.95`, `candidate_top_m=3` |

## 3. Kết quả

Kết quả chính trên 100 mẫu đầu của MuSiQue:

| Phương pháp | EM | F1 | Acc | Recall@1 | Precision@1 | Avg output length |
|---|---:|---:|---:|---:|---:|---:|
| Baseline | 0.110 | 0.240 | 0.120 | 0.220 | 0.440 | 2.19 |
| Hybrid fixed | 0.050 | 0.193 | 0.080 | 0.235 | 0.470 | 2.09 |
| Hybrid adaptive | 0.050 | 0.193 | 0.080 | 0.240 | 0.480 | 2.09 |

So với baseline, Hybrid adaptive có thay đổi như sau:

| Chỉ số | Baseline | Hybrid adaptive | Chênh lệch |
|---|---:|---:|---:|
| EM | 0.110 | 0.050 | -0.060 |
| F1 | 0.240 | 0.193 | -0.047 |
| Acc | 0.120 | 0.080 | -0.040 |
| Recall@1 | 0.220 | 0.240 | +0.020 |
| Precision@1 | 0.440 | 0.480 | +0.040 |

Hybrid adaptive tốt hơn hybrid fixed một chút ở retrieval top-1:

- Recall@1: `0.235 -> 0.240`
- Precision@1: `0.470 -> 0.480`

Tuy nhiên, hai cấu hình fixed và adaptive có cùng EM, F1 và accuracy. Điều này cho thấy adaptive fusion có tác động nhỏ lên thứ hạng truy xuất, nhưng tác động đó chưa đủ để cải thiện chất lượng câu trả lời.

## 4. Phân tích

Kết quả cho thấy Hybrid Retrieval có một tín hiệu tích cực cục bộ: retrieval top-1 tăng nhẹ. Cụ thể, Hybrid adaptive tăng Recall@1 từ `0.220` lên `0.240`, và Precision@1 từ `0.440` lên `0.480`. Điều này cho thấy BM25 có thể bổ sung tín hiệu lexical có ích trong một số trường hợp.

Tuy nhiên, mức tăng retrieval này không chuyển hóa thành cải thiện end-to-end QA. Ngược lại, EM giảm từ `0.110` xuống `0.050`, F1 giảm từ `0.240` xuống `0.193`, và accuracy giảm từ `0.120` xuống `0.080`.

Có ba nguyên nhân chính:

1. MuSiQue là dataset multi-hop. Nhiều câu hỏi yêu cầu liên kết qua nhiều tài liệu, không chỉ cần exact lexical matching.
2. BM25 dễ bị hấp dẫn bởi lexical distractors, tức các tài liệu có trùng từ bề mặt với query nhưng không nằm trên reasoning path đúng.
3. Generation của CLaRa rất nhạy với thứ tự document được đưa vào context. Nếu hybrid làm thay đổi top-1 sang một document lexical đúng nhưng reasoning sai, answer có thể tệ hơn dù baseline có recall thấp hơn.

Vì vậy, kết quả này cho thấy retrieval recall và answer quality không phải lúc nào cũng tăng giảm cùng nhau. Hybrid Retrieval có thể cải thiện một phần ranking, nhưng nếu ranking mới làm thay đổi context theo hướng nhiều distractor hơn, EM/F1 vẫn giảm.

## 5. Kết luận về Hybrid Retrieval

Từ pilot evaluation n=100, chưa có bằng chứng đủ mạnh để tiếp tục đầu tư full benchmark cho Hybrid Retrieval theo dạng score fusion hiện tại.

Kết luận cụ thể:

- Implementation đã hoạt động end-to-end trên CLaRa-7B-E2E.
- Hybrid fixed và Hybrid adaptive đã sinh được output hợp lệ.
- Adaptive fusion có tăng nhẹ Recall@1 và Precision@1 so với baseline.
- Tuy nhiên, EM, F1 và accuracy đều giảm so với baseline.
- Lợi ích retrieval không chuyển hóa thành lợi ích generation.
- Trên MuSiQue, lexical sparse retrieval có nguy cơ cao kéo distractor vào top context.

> Hybrid Retrieval bổ sung tín hiệu lexical và có thể cải thiện top-1 retrieval nhẹ trong pilot subset. Tuy nhiên, với MuSiQue, các lexical distractor trong câu hỏi multi-hop làm cho BM25 fusion không cải thiện chất lượng trả lời end-to-end. Kết quả này cho thấy cần có cơ chế gating học được hoặc chuyển sang hướng thích nghi compressor/latent reasoning thay vì tiếp tục score fusion heuristic.
