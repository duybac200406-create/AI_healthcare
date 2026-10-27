AI Healthcare — Failure Analysis Framework

Mục tiêu: Phân tích các failure của AI Healthcare trước khi chọn problem/feature cho project.
Nguyên tắc: Không bắt đầu bằng “AI có thể làm gì?”, mà bắt đầu bằng “AI Healthcare đang thất bại ở đâu, tại sao, và failure đó có thể được chứng minh/cải thiện bằng thực nghiệm hay không?”.

1. AI Accuracy

Failure

AI có thể trả lời sai ở những tình huống nào?

Sai kiểu gì: hallucination, reasoning error, outdated knowledge, multimodal error, overconfidence...?

Failure nào có hậu quả nghiêm trọng nhất?

Root Cause

Vì sao model tạo ra lỗi?

Lỗi đến từ model, dữ liệu, prompt, retrieval, context hay inference?

Evidence

Benchmark / paper / clinical study nào chứng minh failure?

Dataset nào được dùng?

Metric nào được dùng?

Evidence có phù hợp với healthcare thực tế hay chỉ là benchmark phòng thí nghiệm?

Existing Solutions

Các phương pháp hiện tại giải quyết failure này như thế nào?

Fine-tuning?

RAG?

Medical LLM?

Tool use?

Verification / self-check?

Weakness

Existing solutions vẫn thất bại ở đâu?

Có trade-off accuracy / latency / cost / safety không?

Có phụ thuộc dataset hoặc domain không?

Research Gap

Khoảng trống nào vẫn chưa được giải quyết tốt?

Có failure cụ thể nào chưa được đánh giá đầy đủ?

Possible Intervention

Nhóm có thể can thiệp vào đâu?

Model / retrieval / verification / uncertainty / abstention / workflow?

Evaluation

Baseline là gì?

Proposed method là gì?

Metric nào?

Test set nào?

Có đánh giá hallucination, factuality, calibration, abstention không?

2. Memory / Health Records

Failure

AI quên thông tin sức khỏe?

Nhớ sai?

Dùng thông tin cũ?

Không xử lý thông tin mâu thuẫn?

Mất context qua nhiều phiên?

Root Cause

Context window?

Memory architecture?

Retrieval failure?

Poor data representation?

Không có timestamp / provenance / confidence?

Evidence

Có benchmark / paper / user study nào chứng minh?

Failure xuất hiện ở single-session hay longitudinal interaction?

Existing Solutions

Conversation memory

Long-term memory

Vector database

Structured health records

Knowledge graph

Retrieval-based memory

Weakness

Recall tốt nhưng precision kém?

Memory hallucination?

Stale memory?

Conflict resolution yếu?

Privacy / storage cost?

Research Gap

Làm sao duy trì health context chính xác khi dữ liệu thay đổi theo thời gian?

Làm sao phát hiện và xử lý conflicting health information?

Possible Intervention

Structured health memory

Provenance + timestamp

Confidence / source tracking

Conflict detection

Memory correction / deletion

Evaluation

Memory retrieval accuracy

Recall / precision của health facts

Conflict detection accuracy

Temporal consistency

Longitudinal QA performance

3. Conversation Safety

Failure

AI thay đổi câu trả lời vì user pushback?

AI chiều theo user thay vì evidence?

AI quá tự tin?

AI tạo reassurance sai?

AI làm tăng anxiety?

Root Cause

Sycophancy

Instruction hierarchy

Conversational pressure

Reward model / preference optimization

Poor uncertainty handling

Evidence

Medical sycophancy studies

Conversational safety evaluations

User studies về over-reliance / misinformation

Existing Solutions

System prompts

Safety policies

Refusal mechanisms

Constitutional / preference alignment

Verification

Weakness

Có thể fail trong multi-turn conversation?

Safety giảm khi user insist?

Refusal quá mức làm giảm usefulness?

Research Gap

Làm sao duy trì medical correctness dưới conversational pressure?

Làm sao phân biệt disagreement hợp lý và user-induced error?

Possible Intervention

Evidence-anchored dialogue

Contradiction detection

Stability checks across turns

Uncertainty-aware response policy

Safety escalation rules

Evaluation

Answer stability under user pressure

Sycophancy rate

Medical correctness

Appropriate refusal / escalation

Calibration

4. Triage / Emergency

Failure

AI bỏ sót dấu hiệu cấp cứu?

Phân loại mức độ nghiêm trọng sai?

Reassurance sai khiến user trì hoãn khám?

Đưa self-care khi cần professional care?

Root Cause

Poor symptom reasoning

Missing patient context

Threshold / decision boundary không phù hợp

Không có explicit escalation policy

Evidence

Clinical triage studies

Patient safety literature

Emergency symptom benchmarks

Existing Solutions

Rule-based red flags

Clinical decision support

Symptom checker

LLM triage

Human escalation

Weakness

False reassurance

False alarm

Poor calibration

Incomplete history

Không biết giới hạn của model

Research Gap

Làm sao tối ưu safe escalation thay vì chỉ tối ưu diagnosis accuracy?

Possible Intervention

Red-flag detector

Risk-aware triage

Abstention

Human handoff

Multi-stage safety gate

Evaluation

Sensitivity cho emergency cases

False-negative rate

False-positive rate

Calibration

Appropriate escalation rate

Time-to-escalation

5. Evidence / RAG

Failure

AI trả lời không có evidence?

Citation không support claim?

Retrieval sai tài liệu?

RAG nhưng answer vẫn hallucinate?

Evidence outdated hoặc không phù hợp patient context?

Root Cause

Poor retrieval

Chunking

Embedding mismatch

Query formulation

Source quality

Grounding failure

Evidence

Medical RAG benchmarks

Retrieval metrics

Grounded generation studies

Compare vanilla LLM vs RAG

Existing Solutions

Dense retrieval

Hybrid search

Re-ranking

Query expansion

Citation generation

Knowledge-grounded generation

Weakness

RAG không tự động làm answer tốt hơn

Retrieval errors propagate to generation

Citation có thể đúng tài liệu nhưng không support claim

Latency / cost tăng

Research Gap

Evidence quality nên được đánh giá ở mức claim, không chỉ document retrieval

Cần đánh giá cả retrieval + grounding + abstention

Possible Intervention

Claim-level evidence verification

Source quality ranking

Evidence confidence

Contradiction checking

Retrieval-aware abstention

Evaluation

Retrieval Recall@k / Precision@k

Citation correctness

Citation completeness

Groundedness

Factuality

Abstention quality

6. Privacy / Data

Failure

Health data bị lưu quá mức?

Dữ liệu có thể bị lộ?

Model nhớ thông tin nhạy cảm?

Data được dùng ngoài mục đích ban đầu?

User không kiểm soát được dữ liệu của mình?

Root Cause

Poor data governance

Excessive data collection

Insecure storage

Weak access control

Third-party data sharing

Lack of deletion / consent mechanisms

Evidence

Privacy/security research

Healthcare data regulations/guidelines

Medical AI governance literature

Existing Solutions

Encryption

Access control

Data minimization

Anonymization / pseudonymization

Consent management

Audit logs

Weakness

Anonymization không phải lúc nào cũng đảm bảo privacy

Utility giảm khi hạn chế dữ liệu quá mạnh

LLM-specific privacy risks còn phức tạp

Research Gap

Cân bằng giữa personalization và privacy

Kiểm soát health memory mà vẫn hữu ích

Possible Intervention

Minimal health memory

User-controlled memory

Sensitive-data filtering

Encryption / access control

Data lifecycle policy

Evaluation

Privacy leakage tests

Unauthorized retrieval rate

Data minimization score

Utility vs privacy trade-off

Deletion correctness

7. Trust / Commercialization

Failure

User tin AI quá mức?

AI tạo cảm giác như bác sĩ/người bạn có authority?

Engagement được tối ưu thay vì health outcome?

Commercial incentives ảnh hưởng recommendation?

Root Cause

Anthropomorphic design

Simulated authority

Poor uncertainty communication

Engagement-based optimization

Conflicts of interest

Evidence

Human-AI trust studies

Medical AI ethics

Over-reliance / automation bias studies

Analysis of commercial healthcare AI systems

Existing Solutions

Disclaimers

Explainability

Confidence indicators

Human oversight

Transparent recommendation source

Weakness

Disclaimer không đảm bảo user hiểu limitation

Explainability có thể tạo false confidence

“Friendly” interaction có thể tăng trust ngoài ý muốn

Research Gap

Làm sao đạt calibrated trust thay vì maximum trust?

Làm sao thiết kế AI hữu ích nhưng không giả lập medical authority?

Possible Intervention

Uncertainty-aware communication

Explicit capability boundaries

Evidence-linked answers

Risk-sensitive language

Conflict-of-interest disclosure

Evaluation

Trust calibration

Appropriate reliance

Over-reliance rate

Decision quality

User understanding of AI limitations

8. Long-term Health Relationship / Retention

Lưu ý: “Retention” không được mặc định là mục tiêu tối ưu. Trong healthcare, việc user quay lại nhiều hơn không đồng nghĩa health outcome tốt hơn.

Failure

User bỏ hệ thống sau một thời gian?

AI trở nên lặp lại / không hữu ích?

Historical context mất dần?

User phụ thuộc quá mức vào AI?

Không tạo ra measurable health improvement?

Root Cause

Low long-term utility

Weak personalization

Memory failure

Repetitive interaction

Wrong incentives

No measurable outcome

Evidence

Longitudinal healthcare AI studies

User retention / adherence literature

Digital health outcome studies

Existing Solutions

Personalization

Reminder systems

Long-term memory

Habit / adherence mechanisms

Human-in-the-loop care

Weakness

Retention có thể trở thành vanity metric

Engagement không đồng nghĩa clinical benefit

Personalization có thể làm tăng privacy risk / dependency

Research Gap

Đo long-term usefulness thay vì chỉ DAU/session length

Liên kết AI interaction với measurable health outcome hoặc appropriate action

Possible Intervention

Outcome-oriented personalization

Longitudinal health summaries

Periodic evidence-based review

Human escalation

User-controlled health history

Evaluation

Appropriate-action rate

Longitudinal task completion

Health-plan adherence (khi phù hợp)

User-reported usefulness

Unnecessary interaction rate

Trust / dependency indicators

Retention chỉ là secondary metric

Cross-Category Comparison

Sau khi hoàn thành 8 mục, chấm mỗi failure theo:

Criterion

Question

Severity

Failure có hậu quả nghiêm trọng đến đâu?

Evidence

Có bằng chứng đủ mạnh không?

Frequency

Failure xảy ra thường xuyên không?

Measurability

Có metric rõ ràng không?

Benchmarkability

Có baseline/dataset để so sánh không?

Feasibility

Nhóm sinh viên có thể giải quyết trong 1 semester không?

Novelty

Còn khoảng trống nghiên cứu không?

Product Value

Nếu giải quyết tốt, người dùng có thực sự hưởng lợi không?

Final Selection Rule

Không chọn project chỉ vì:

Ý tưởng nghe hay

Có thể dùng LLM

Có thể làm chatbot

Có thể thêm RAG

Có nhiều feature

Chỉ ưu tiên failure khi có chuỗi lập luận:

Failure → Root Cause → Evidence → Existing Solutions → Weakness → Research Gap → Intervention → Evaluation

Và cuối cùng phải trả lời được:

“Nếu giải quyết failure này tốt hơn baseline, chúng ta chứng minh điều đó bằng số liệu nào?”