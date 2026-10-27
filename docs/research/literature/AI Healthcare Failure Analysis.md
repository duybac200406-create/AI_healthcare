# AI healthcare failure analysis
## Part A — Medical RAG Literature Extraction
### Paper 1: arXiv 2408.04187 (MedGraphRAG): Medical Graph RAG: Towards Safe Medical Large Language Model via Graph Retrieval-Augmented Generation
A1. Source Identification
Trường	Nội dung
Title	Medical Graph RAG: Towards Safe Medical Large Language Model via Graph Retrieval-Augmented Generation (hệ thống tên MedGraphRAG)
Authors	Junde Wu, Jiayuan Zhu, Yunli Qi, Jingkun Chen (Oxford); Min Xu (CMU); Filippo Menolascina (Edinburgh); Vicente Grau (Oxford)
Year	2024 (PDF v2 đề ngày 15/10/2024; ngày v1: Not verified)
Venue	arXiv (cs.CV). README của repo dự án gắn nhãn bài là ACL 2025, nhưng đây là nguồn thứ cấp: Not verified trên ACL Anthology. Nếu đã xuất bản, bản chính thức có thể khác v2
URL	https://arxiv.org/abs/2408.04187
DOI	DOI do arXiv cấp (DataCite): 10.48550/arXiv.2408.04187. Đây là DOI của preprint. DOI bản xuất bản (nếu có): Not verified
Publication type	Method / framework paper, có benchmark evaluation, ablation và human evaluation
Primary / Secondary	Primary academic evidence (preprint; peer-review chưa xác nhận)
Evidence strength	Moderate (nhiều benchmark, có ablation, có clinician). Bị hạ vì thiếu thông tin thiết kế đánh giá và thiếu kiểm soát (xem Q8, Q9)
Main purpose	Đề xuất GraphRAG chuyên biệt cho y khoa để sinh câu trả lời có nguồn và có định nghĩa thuật ngữ
Relevance to Medical RAG	Cao (method paper trực tiếp)
Relevance to Failure Analysis	Cat 5 (Evidence/RAG): mạnh. Cat 1 (Accuracy): vừa. Cat 7 (Trust): yếu. Cat 2 (Memory): chỉ ở mức gợi ý, chưa đủ evidence 
github
arxiv
A2. Reference mapping

Danh sách tham khảo có khoảng 50 mục và có trùng lặp (Edge 2024a/b và Gao 2023b/c là cùng một công trình, một bản arXiv và một bản hội nghị). Các ID dưới đây chỉ lấy từ danh sách của chính bài, tôi chưa kiểm chứng độc lập.

Priority A — trực tiếp Medical RAG / medical QA / clinical safety
Tài liệu	Năm	ID trong nguồn	Vì sao A	Điều tra tiếp?
Xiong et al., Benchmarking RAG for medicine	2024	arXiv:2402.13178	Benchmark và đánh giá hệ thống Medical RAG	Ưu tiên cao nhất
Williams et al., LLM đưa khuyến nghị lâm sàng tại khoa cấp cứu	2024	Nature Communications 15(1):8236	Bằng chứng về an toàn lâm sàng/cấp cứu (Cat 3, 4)	Cao
Long et al., Bailicai	2024	arXiv:2407.21055	Framework Medical RAG	Cao
Miao et al., RAG + LLM trong thận học	2024	Medicina 60(3):445	Ứng dụng RAG theo chuyên khoa	Trung bình
Hadi et al., ChatGPT như công cụ chẩn đoán	2024	PLoS ONE 19(7):e0307383	Đánh giá LLM trong chẩn đoán	Trung bình
Singhal et al., Med-PaLM 2	2023	arXiv:2305.09617	Baseline y khoa SOTA (dùng để so sánh)	Trung bình
Saab et al., Med-Gemini	2024	arXiv:2404.18416	Baseline y khoa SOTA	Trung bình
Dataset: PubMedQA (arXiv:1909.06146), MedQA (Jin 2021, Appl. Sci. 11(14):6421), MedMCQA (Pal 2022, PMLR/CHIL), FakeHealth (arXiv:2002.00837), PubHealth (arXiv:2010.09926), MIMIC-IV (Sci. Data 10(1):1)	2019–2023	như nêu	Nguồn dữ liệu và benchmark y tế	Chỉ khi cần kiểm tra dữ liệu/contamination
Priority B — RAG, graph, attribution, prompting y khoa
Tài liệu	ID trong nguồn	Vai trò	Ghi chú
Edge et al., GraphRAG (Microsoft)	arXiv:2404.16130	Baseline GraphRAG	Cần đọc, để hiểu so sánh
Hu et al., GRAG	arXiv:2405.16506	Bài trích dẫn là "GraphRAG (Hu et al.)"	Xem cảnh báo bên dưới
Lewis et al., RAG gốc	arXiv:2005.11401	Nền tảng	Thấp
Gao et al., ALCE (arXiv:2305.14627); Slobodkin et al., ATTR-FIRST (arXiv:2403.17104); Qi et al., "MIRAGE" (arXiv:2406.13663)		Baseline sinh có trích dẫn trong human eval	Trung bình
Gao et al., survey RAG	arXiv:2312.10997	Survey nền	Trung bình
Savage et al. (npj Digit. Med. 7(1):20); Wang et al. (arXiv:2304.14670); Xie et al., o1 trong y khoa (arXiv:2409.15277)		LLM y khoa, prompting	Để sau
Bohnet et al., RARR (Gao 2023a), WebGPT, Schimanski et al., LongCite, REALM		Attribution/citation, RAG tiền đề	Để sau
Priority C — ngoại vi

LLaMA, Gemini, GPT-4 report, BioGPT, BioMedLM, PMC-LLaMA/MedC-K, MMLU, UMLS (Bodenreider 2004), LangChain, "More agents is all you need" (arXiv:2402.05120).

Hai cảnh báo trích dẫn đáng ghi lại:

Trùng tên MIRAGE. "MIRAGE (Qi et al., 2024)" trong bài là phương pháp model-internals attribution, không phải MIRAGE benchmark y khoa của Xiong et al. Khi tìm literature, phải tránh nhầm hai thứ này.
Mơ hồ "GraphRAG". Bài dùng "GraphRAG (Hu et al., 2024)" ở nhiều chỗ, nhưng baseline thực tế được triển khai là GraphRAG của Microsoft Azure (Edge et al. + repo của Microsoft). Hai công trình khác nhau, và bài không giải thích rõ.
A3. Q1–Q10 Extraction
Task / architecture
Task: trả lời câu hỏi y khoa trắc nghiệm, kiểm chứng thông tin sức khỏe (fact-checking), sinh câu trả lời dài có trích dẫn.
Target users: bác sĩ và người dùng phổ thông (panel đánh giá gồm cả hai). Bài mô tả bối cảnh dùng dữ liệu riêng tư như bệnh án điện tử.
Graph construction: chunking ngữ nghĩa bằng LLM (cửa sổ trượt 5 đoạn) → trích entity (kiểu theo UMLS semantic types) → Triple linking [RAG entity, source, definition] qua cosine similarity với ngưỡng (mặc định 0.5, "unless otherwise noted") → relationship linking → gắn tag (Symptoms, Patient History, Body Functions, Medication…) và gom cụm phân cấp tối đa 12 tầng.
U-Retrieval: top-down theo tag → lấy top 60 entity và các triple neighbor → bottom-up response refinement (4 tầng với GPT-4, 5 tầng với model khác).
Models: LLM dựng graph: Llama3-70B. LLM trả lời: Llama2 (13B, 70B), Llama3 (8B, 70B), gemini-1.0-pro, gpt-4-0613. Embedding: text-embedding-3-large. So sánh model dùng "5-shot response ensemble".
Reranking: Not reported.
Q1 — Failure đang được giải quyết
[B — author's claim] Y khoa có tri thức quá lớn để nhét vào context; SFT đắt hoặc bất khả thi với model đóng nguồn; LLM không được "bóp méo" sự thật y khoa; người dùng không chuyên khó kiểm chứng câu trả lời.
[B] RAG chuẩn khó tổng hợp tri thức mới và yếu ở bài toán cần hiểu toàn cục; GraphRAG gốc thiếu cơ chế đảm bảo tính xác thực/nguồn gốc và tốn kém khi dựng community.
Điểm cần lưu ý: "hallucination" chỉ xuất hiện trong phần Related Work như một nhận định chung về RAG. Paper không đo hallucination. Vấn đề được giải quyết thực chất là traceability/credibility và độ chính xác trả lời, không phải hallucination rate.
Q2 — Failure RAG vẫn chưa giải quyết
[A — Table 1] RAG không phải lúc nào cũng cải thiện. Ví dụ Llama2-13B với RAG chuẩn thấp hơn no-retrieval ở một số subset MMLU (Col-Bio 69.4→68.3; Anatomy 52.6→51.0; Gene 66.0→64.5). Điểm này tôi đọc trực tiếp từ bảng, tác giả không bình luận.
[B] Dữ liệu y khoa bổ sung qua RAG chuẩn chỉ cải thiện nhỏ (Med-Paper <2%, Med-Dictionary <1%). Tác giả diễn giải rằng cần cả dữ liệu và phương pháp retrieval phù hợp.
[B] GraphRAG thiếu thông tin quan trọng khi tài liệu tham chiếu và dữ liệu người dùng nằm chung một tầng graph. Ví dụ case study: bỏ sót việc một số thuốc giãn phế quản có thể làm xấu suy tim.
Failure còn lại của MedGraphRAG: Not reported. Không có error analysis, không có phân tích trường hợp thất bại.
Không báo cáo: outdated knowledge (chỉ nêu trong future work), privacy, bias, robustness, hallucination.
Q3 — Dataset / knowledge source (theo vai trò)
Vai trò	Nội dung
Benchmark / evaluation	Test split của 9 benchmark trắc nghiệm MultiMedQA (MedQA, MedMCQA, PubMedQA, MMLU clinic topics); 2 benchmark fact-checking (FakeHealth, PubHealth); DiverseHealth (50 câu hỏi lâm sàng thực tế do tác giả thu thập, nhấn mạnh health equity; quy trình thu thập: Not reported; việc phát hành: Not verified)
Training / fine-tuning	Không có (hệ thống không fine-tune)
RAG knowledge corpus	Repository data: MedC-K (4,8 triệu bài báo, 30 nghìn sách giáo khoa) + toàn bộ evidence publications của FakeHealth và PubHealth (tầng trên) + đồ thị UMLS (tầng dưới)
EHR / patient data	MIMIC-IV làm "RAG data" của người dùng
Knowledge graph	UMLS; đồ thị do LLM tự dựng từ văn bản

Hai điểm cần xác minh (đây là quan sát của chúng ta, mức [D]):

Các câu hỏi benchmark không phải câu hỏi về bệnh nhân trong MIMIC-IV. Paper không giải thích các bản ghi MIMIC-IV đóng vai trò gì trong việc trả lời. Cải thiện có thể đến chủ yếu từ repository data.
Repository data gồm toàn bộ evidence publications của FakeHealth/PubHealth, tức tài liệu chứng cứ gắn với chính tập kiểm tra fact-checking. Phân tích contamination/overlap: Not reported. Điều này có thể làm tăng điểm so với no-retrieval, dù các baseline RAG cũng dùng cùng dữ liệu.
Q4 — Evaluation metrics
Retrieval metrics: không có (không Recall@k, Precision@k, MRR, NDCG, Context Recall/Precision).
Generation / answer: accuracy chọn đáp án đúng (các benchmark trắc nghiệm).
RAG-specific tự động: không có (không faithfulness, không context relevance).
Clinical / human: 5 tiêu chí do người chấm: pertinence, correctness, citation precision, citation recall, understandability (thang 5 mức, nhưng Table 2 báo cáo số trong khoảng ~71–97 mà không giải thích cách quy đổi).
Lưu ý: accuracy trắc nghiệm không phải clinical validation. Citation precision/recall ở đây do người chấm, không có công cụ kiểm tra tự động.
Q5 — Retrieval-side hay Generation-side?

Phân loại: Cannot determine cho MedGraphRAG; tác giả quy lỗi của GraphRAG về retrieval-side.

[B] Tác giả cho rằng GraphRAG thất bại vì cấu trúc graph (tham chiếu lẫn với dữ liệu người dùng) làm mất thông tin then chốt dưới cùng số láng giềng, và vì retrieval chỉ dựa trên graph summary.
Vì không có metric retrieval và không có error analysis, ta không tách được phần cải thiện do retrieval tốt hơn với phần do pipeline có thêm các lượt LLM (gắn tag, bottom-up refinement).
[D] Không có đối chứng "cùng số lượt gọi LLM nhưng không có graph". Không có số liệu chi phí/độ trễ, dù bài nói U-Retrieval hiệu quả (efficient). Cần kiểm chứng thêm.
Q6 — Patient memory

Phân biệt:

EHR dùng làm RAG corpus: Có (MIMIC-IV).
Structured longitudinal patient memory: Không được chứng minh. Không có truy xuất theo từng bệnh nhân, không đánh giá tính nhất quán theo thời gian. "Patient History" chỉ là một danh mục tag.
Conversation memory: không có.
Chữ "external memory" trong bài nói về tri thức ngoài tham số của model, không phải bộ nhớ bệnh nhân.
Tác giả đề xuất "real-time data updates" như hướng tương lai.
Q7 — Safety / Triage

Not reported. Không có emergency detection, risk classification, abstention, refusal, escalation hay human-in-the-loop.

"Safe/safety" trong tiêu đề được hiện thực hóa gần như hoàn toàn bằng nguồn dẫn + định nghĩa thuật ngữ. Phần "safety when handling private medical data" trong abstract không đi kèm cơ chế hay đánh giá privacy nào: Not reported.
DiverseHealth có chủ đề nhạy cảm (ví dụ tự sát, sử dụng chất), nhưng cách hệ thống xử lý chúng: Not reported.
Không suy ra hệ thống "không an toàn" trong thực tế, chỉ là paper không báo cáo.
Q8 — Human / Clinician evaluation
Mục	Nội dung
Evaluators	7 bác sĩ có chứng chỉ + 5 người không chuyên
Chuyên khoa	Not reported
Số mẫu được chấm	Not reported (DiverseHealth gồm 50 câu hỏi)
Blinded	Not reported
Inter-rater agreement	Not reported
Kết quả tách riêng theo clinician/layperson	Not reported (số liệu gộp)
Hệ thống so sánh	Inline Search, ATTR-FIRST, MIRAGE (Qi et al.); LLM nền và kho tri thức của các baseline này: Not reported
Kết quả (Table 2)	MedGraphRAG cao nhất ở hầu hết chiều. Ví dụ MultiMedQA: CP 92 vs 86, CR 86 vs 77 (so với ATTR-FIRST)

Mâu thuẫn nhỏ cần ghi lại:

Text nói MedGraphRAG "consistently" cao hơn trên mọi metric, nhưng ở DiverseHealth, pertinence hòa với MIRAGE (97 vs 97).
Caption Table 2 ghi "MedQA and DiverseHealth", còn text ghi MultiMedQA.
Q9 — Limitations

Paper không có mục limitations. Chỉ có hai điều tác giả nêu qua loa: cần cập nhật dữ liệu thời gian thực và cần kiểm chứng trên dữ liệu lâm sàng thực. Điều đó ngụ ý chưa được kiểm chứng trên dữ liệu thực, đây là suy ra từ câu của tác giả.

Loại	Nội dung	Mức
Evaluation	Metric chủ yếu là accuracy trắc nghiệm; không đo hallucination/retrieval	[A]
Evaluation	Human eval thiếu thiết kế (xem Q8); case study chỉ một ca	[A]
Evaluation	So sánh "vượt Med-PaLM 2 / Med-Gemini": Table 1 không liệt kê hai model này; giao thức so sánh (5-shot ensemble ở đây) có tương đương hay không: Not verified	[A]/[D]
Evaluation	Không báo cáo phương sai, khoảng tin cậy hay kiểm định thống kê; "5-shot response ensemble" chưa rõ là prompting 5-shot hay lấy mẫu rồi bỏ phiếu	[A]
Data	MIMIC-IV chưa được liên hệ với câu hỏi test; nguy cơ trùng lặp dữ liệu repository với tập test chưa được phân tích	[D]
Retrieval	Triple linking dựa trên cosine similarity ≥ ngưỡng; độ chính xác của liên kết nguồn không được đánh giá riêng. Sai số có thể lan truyền từ bước LLM trích entity/mô tả	[D]
Generation	Mọi giả định rằng "truy vết được nguồn" đều phụ thuộc chất lượng liên kết ở trên	[D]
Clinical	Không có đánh giá trên dữ liệu bệnh nhân thật; câu "không có tác dụng phụ" trong case study là khẳng định tuyệt đối của tác giả, không có clinician nào thẩm định riêng ca này	[A]
Safety / Privacy	Không có cơ chế hay đánh giá; không nêu cách xử lý bản ghi thật khi dùng API thương mại cho LLM trả lời	Not reported
Cost / latency	Không có số liệu	Not reported
Generalization	"Health equity" là định hướng của DiverseHealth, nhưng không có phân tích theo nhóm	[A]
Claims	Các cụm "first to propose", "guaranteeing reliability and explainability" là tuyên bố của tác giả; chưa có evidence kiểm chứng tương ứng	[B]
Q10 — Future work
Explicit future work: cập nhật dữ liệu thời gian thực; kiểm chứng trên dữ liệu lâm sàng thực.
Candidate research direction (của chúng ta, cần xác minh thêm, Candidate gap — requires further verification):
Đánh giá retrieval và grounding một cách tách bạch cho graph-based Medical RAG (hiện không có).
Kiểm tra liên kết nguồn (source linking) có thật sự đúng hay không, thay vì chỉ có rating của người dùng.
Đánh giá truy xuất theo từng bệnh nhân với dữ liệu EHR thật.
Evidence quality assessment
Strong: mô tả kiến trúc; ablation Table 3 (GraphRAG→Med-MetaGraph→Triple Graph→U-Retrieval, số liệu khớp với hàng GPT-4 của Table 1: 91.3/83.3/81.5).
Moderate: cải thiện accuracy trên nhiều benchmark/model (không có phương sai, nguy cơ trùng dữ liệu chưa kiểm tra).
Weak / anecdotal: case study một ca; human eval thiếu thiết kế; claim về safety/privacy/efficiency.
Missing: metric retrieval, đo hallucination, cost/latency, error analysis, kiểm tra contamination, thông tin thiết kế human eval, validation trên dữ liệu bệnh nhân thật.
Relevance to AI Healthcare Failure Map
Category	Đánh giá
5. Evidence / RAG	Có evidence tương đối rõ (traceability bằng triple linking; CP/CR do người chấm), kèm cảnh báo về cách đo
1. AI Accuracy	Có evidence (accuracy trắc nghiệm), kèm dữ kiện RAG đôi khi không giúp được
7. Trust	Yếu (laypersons chấm understandability; chưa đo calibrated trust)
2. Memory / Health Records	Insufficient evidence (dùng EHR làm corpus nhưng không đánh giá patient memory)
6. Privacy / Data	Insufficient evidence (chỉ tuyên bố trong abstract)
3, 4, 8	Not relevant / insufficient evidence

Ghi chú cho Part B (mức [D], chỉ là giả thuyết): nguồn #1 (RAGBench) đưa ra các metric Relevance/Utilization/Adherence nhưng ngoài y tế, trong khi nguồn #2 là paper y tế mà không dùng metric nào kiểu này. Chưa đủ nguồn để gọi đây là gap.
### Paper 2: arXiv 2402.13178 (MIRAGE / MEDRAG): Benchmarking Retrieval-Augmented Generation for Medicine
A1. Source Identification
Trường	Nội dung
Title	Benchmarking Retrieval-Augmented Generation for Medicine
Authors	Guangzhi Xiong†, Qiao Jin†, Zhiyong Lu§, Aidong Zhang§ († đóng góp ngang nhau; § đồng tác giả liên hệ)
Affiliation	University of Virginia; National Library of Medicine, NIH
Year	2024 (đăng lên arXiv tháng 2/2024, ghi ngày 20/02/2024 trên Semantic Scholar)
Venue	arXiv preprint. PDF tôi đọc không nêu venue; việc có bản đã phản biện hay không: Not verified
URL	https://arxiv.org/abs/2402.13178
DOI	DOI do arXiv cấp: 10.48550/arXiv.2402.13178 (DOI của preprint)
Publication type	Benchmark + toolkit + empirical study
Primary / Secondary	Primary academic evidence
Evidence strength	Moderate–Strong cho các phát hiện benchmark trong bối cảnh trắc nghiệm. Weak cho mọi kết luận về lâm sàng, an toàn
Main purpose	Đánh giá có hệ thống các thành phần của RAG trong y khoa và đưa ra khuyến nghị thực hành
Relevance to Medical RAG	Rất cao
Relevance to Failure Analysis	Cat 5 và Cat 1: mạnh. Các category còn lại: không có evidence 
Benchmarking Retrieval-Augmented Generation for Medicine - ADS +2

Tuyên bố mới lạ của tác giả (mức [B], chưa kiểm chứng độc lập): "first-of-its-kind benchmark", "first systematic evaluation of RAG in medicine", và là bên đầu tiên đề xuất thiết lập question-only retrieval.

A2. Reference mapping

Danh sách tham khảo dài (hơn 70 mục). Các ID dưới đây lấy từ danh sách của chính bài, tôi chưa kiểm chứng độc lập.

Priority A — Medical RAG / medical QA / clinical
Tài liệu	Năm	ID trong nguồn	Vì sao	Điều tra tiếp?
Zakka et al., Almanac (RAG cho y học lâm sàng)	2024	NEJM AI 1(2):AIoa2300068	Hệ thống RAG lâm sàng; cần xem có đánh giá của clinician không	Cao
Jeong et al., cải thiện suy luận y khoa bằng retrieval và self-reflection	2024	arXiv:2401.15269	Medical RAG có thành phần tự phản hồi (nội dung chi tiết: Not verified, chỉ suy từ tiêu đề)	Cao
Wang et al., tăng cường black-box LLM bằng sách giáo khoa y khoa	2023	arXiv:2309.02233	Medical RAG cho QA lâm sàng	Cao
Liévin et al., LLM có suy luận được câu hỏi y khoa?	2022	arXiv:2207.08143	Đánh giá sớm LLM + retrieval trên QA y khoa	Trung bình
Nori et al., GPT-4 trên bài toán y khoa	2023	arXiv:2303.13375	Baseline LLM trên medical QA	Trung bình
Singhal et al. (Nature 620(7972):172–180; arXiv:2305.09617)	2023	như nêu	Medical QA SOTA	Trung bình
Lála et al., PaperQA	2023	arXiv:2312.07559	Agent RAG cho tài liệu khoa học	Trung bình
Priority B — retriever, RAG, rủi ro LLM y sinh
Tài liệu	ID trong nguồn	Vai trò
Jin et al., MedCPT	Bioinformatics 39(11):btad651	Retriever y sinh được dùng
Liu et al., "Lost in the middle"	arXiv:2307.03172	Cơ sở cho phân tích vị trí snippet
Tian et al., cơ hội và thách thức của ChatGPT/LLM trong y sinh	Brief. Bioinform. 25(1):bbad493	Được trích dẫn khi nói về rủi ro ở lĩnh vực nguy cơ cao
Truhn et al., LLM nên là "reasoning engine", không phải cơ sở tri thức	Nat. Med. 29(12):2983–2984	Lập luận nền cho việc dùng RAG
Hersh, Search still matters	JAMIA ocae014	Information retrieval trong thời LLM
Gao et al., survey RAG	arXiv:2312.10997	Survey nền
Jiang et al., Active RAG	arXiv:2305.06983	Kiến trúc RAG nâng cao (bài thừa nhận chưa đánh giá)
Jin et al., retrieve–summarize–verify	J. Am. Soc. Nephrol. (ID: Not available trong nguồn)	Medical literature search với ChatGPT
Liu et al., xác minh claim khoa học bằng retrieval	JAMIA Open ooae021	Hướng task khác (bài nêu trong Limitations)
Priority C — ngoại vi

RETRO (Borgeaud), BEIR, BM25, Contriever, SPECTER, RRF (Cormack), Pyserini; dataset gốc (PubMedQA, MedQA, MedMCQA, MMLU, BioASQ); MEDITRON, PMC-LLaMA, BioGPT, Llama2, Mixtral, GPT-4 report; LangChain.

A3. Q1–Q10 Extraction
Task / architecture
Task: trả lời câu hỏi y khoa dạng trắc nghiệm, zero-shot, có truy xuất.
Input / Output: câu hỏi và các lựa chọn → đáp án (JSON chứa phần suy luận và đáp án chọn).
Retrieval: BM25 (lexical), Contriever (general dense), SPECTER (scientific), MedCPT (biomedical), cùng hai bản fusion bằng Reciprocal Rank Fusion: RRF-2 (BM25+MedCPT) và RRF-4 (cả bốn). Lấy mặc định 32 snippet. Reranking (cross-encoder): không dùng (tác giả nêu đây là hướng tương lai).
Corpora: PubMed, StatPearls, Textbooks, Wikipedia, và MedCorp (hợp của cả bốn).
LLM: GPT-4, GPT-3.5, Mixtral 8×7B, Llama2-70B, MEDITRON-70B, PMC-LLaMA-13B. Nhiệt độ 0. Baseline là CoT không RAG.
Cơ chế ghép context: nối các snippet vào đầu prompt (vanilla RAG).
Q1 — Failure đang được giải quyết
[B — author's claim] LLM bị hallucination và kiến thức lỗi thời, nguy hiểm trong lĩnh vực nguy cơ cao. RAG có tiềm năng giải quyết bằng cách neo câu trả lời vào tài liệu đáng tin và tăng tính minh bạch.
[A — source fact] Vấn đề mà paper này trực tiếp giải quyết là thiếu best practice cho cấu hình RAG trong y khoa (chọn corpus, retriever, LLM, số snippet).
Lưu ý: hallucination và outdated knowledge chỉ được nêu như động cơ. Cả hai đều không được đo. Benchmark tĩnh, không có kiểm tra tính thời sự của tri thức, không có metric hallucination.
Q2 — Failure RAG vẫn chưa giải quyết

Bằng chứng từ chính các bảng của bài (Table 6, accuracy %):

Task	GPT-4 CoT → MEDRAG	GPT-3.5 CoT → MEDRAG
MMLU-Med	89.44 → 87.24 (−2.20)	72.91 → 75.48 (+2.57)
MedQA-US	83.97 → 82.80 (−1.17)	65.04 → 66.61 (+1.57)
MedMCQA	69.88 → 66.65 (−3.23)	55.25 → 58.04 (+2.79)
PubMedQA*	39.60 → 70.60 (+31.00)	36.00 → 67.40 (+31.40)
BioASQ-Y/N	84.30 → 92.56 (+8.26)	74.27 → 90.29 (+16.02)
Trung bình	73.44 → 79.97 (+6.53)	60.69 → 71.57 (+10.88)

Hiệu số tôi tự tính từ bảng. Từ đây có các điểm:

[A] GPT-4 giảm điểm với RAG trên cả ba task thi cử. Mức giảm MedMCQA (−3,2 điểm) lớn hơn sai số chuẩn từng ô (~0,7), nhưng bài không chạy kiểm định ghép cặp. Tác giả không bình luận riêng về điều này. Ở các model khác, MEDRAG cũng thấp hơn CoT trên một số task thi cử (Mixtral MedQA-US 64.10→60.02; Llama2 MMLU-Med 57.39→54.55 và MedQA-US 47.84→44.93; MEDITRON MedQA-US 51.69→49.57; PMC-LLaMA MedQA-US 44.38→42.58).
[A] Chọn sai corpus có thể làm tệ hơn không-RAG, không chỉ "giúp ít". Text nói StatPearls và Textbooks "provide little assistance" cho PubMedQA* và BioASQ-Y/N. Table 7 (GPT-3.5) cho thấy PubMedQA* với StatPearls chỉ ~23–31% so với baseline 36.0%, BioASQ-Y/N ~58–61% so với 74.27%. Nghĩa là thấp hơn baseline. Cơ chế (vì sao context lạc đề làm sai) không được điều tra.
[A] Số snippet k nhỏ (≤ 8) có thể cản trở LLM dùng tri thức nội tại; k quá lớn làm giảm tỷ lệ tín hiệu/nhiễu. Với PubMedQA*, accuracy cao nhất ở k=1 rồi giảm khi tăng k. Các điểm này đến từ Figure 3, tôi dựa vào mô tả của tác giả (giá trị: Not verified).
[A] Hiệu ứng "lost-in-the-middle": accuracy có dạng chữ U theo vị trí của snippet đúng trong context (Figure 4, mô tả định tính).
[B] Với câu hỏi thi, MEDRAG không phải lúc nào cũng hơn SFT, vì "snippet hữu ích có thể khó truy xuất". Tác giả tự ghi trong Limitations rằng retrieval của các task thi chưa được đánh giá, nên đây là giả thuyết chưa kiểm chứng.
Không báo cáo: error analysis cho câu trả lời sai, hallucination, câu trả lời không có căn cứ, privacy, bias.
Q3 — Dataset / knowledge source (theo vai trò)
Vai trò	Nội dung
Benchmark / evaluation	MIRAGE gồm 7.663 câu hỏi (cộng khớp: 1.089 + 1.273 + 4.183 + 500 + 618): MMLU-Med (6 chủ đề y sinh, test set), MedQA-US (4 lựa chọn, test), MedMCQA (dùng dev set vì không có nhãn test công khai), PubMedQA* (500 mẫu test, đã gỡ context gốc), BioASQ-Y/N (618 câu yes/no, 2019–2023)
Training / fine-tuning	Không có huấn luyện. MEDITRON và PMC-LLaMA là model đã fine-tune sẵn trên văn献 y sinh bởi bên khác
RAG corpus	PubMed (23,9 triệu bài có title+abstract), StatPearls (9.330 bài → 301,2k snippet), Textbooks (18 sách; chunk ≤ 1.000 ký tự), Wikipedia (6,5 triệu tài liệu → 29,9 triệu snippet), MedCorp (hợp)
EHR / patient data	Không có
Knowledge graph	Không có

Hai điểm cần xác minh:

[A, tác giả thừa nhận một phần] PubMedQA* và BioASQ-Y/N là task "dễ hơn cho MEDRAG" vì thông tin chứng cứ có sẵn trong PubMed. Nghĩa là corpus chứa nguồn gốc của câu hỏi theo thiết kế. Do đó mức tăng lớn ở hai task này không nên được ngoại suy cho câu hỏi lâm sàng thực tế, nơi bằng chứng chưa chắc có trong corpus.
[D] Chưa có phân tích contamination: GPT-4/GPT-3.5, MEDITRON, PMC-LLaMA có thể đã thấy các nguồn tương tự khi tiền huấn luyện (MEDITRON và PMC-LLaMA được huấn luyện trên văn献 y sinh). Not reported trong bài. Quan sát của bài: MEDITRON và PMC-LLaMA vượt xa GPT-4 trên PubMedQA* khi CoT (+34,8% và +40,9%), nhưng nguyên nhân không được phân tích.
Q4 — Evaluation metrics
Retrieval: chỉ ở hai task có nhãn snippet chứng cứ (PubMedQA*, BioASQ-Y/N). Con số duy nhất trong text: 79,6% snippet đúng nằm ở top-1 (PubMedQA*). Không thấy bảng Recall@k/MRR/NDCG trong text. Các task thi cử: không có metric retrieval.
Generation / answer: accuracy cho từng task, kèm sai số chuẩn của tỷ lệ; điểm trung bình không có trọng số giữa 5 task (task có 4.183 câu và task có 500 câu đóng góp bằng nhau).
RAG-specific tự động (faithfulness, context relevance, v.v.): không có.
Clinical / human: không có.

Hệ quả khi đọc con số "trung bình": hai task văn献 có mức tăng +16 đến +31 điểm kéo trung bình lên mạnh. Tuyên bố trong abstract rằng MEDRAG "nâng GPT-3.5 và Mixtral lên mức GPT-4" đúng ở điểm trung bình (GPT-3.5+MEDRAG 71.57 và Mixtral 69.48 so với GPT-4 CoT 73.44), nhưng ở ba task thi cử GPT-3.5+MEDRAG vẫn thấp hơn GPT-4 CoT khoảng 11–17 điểm (ví dụ MedQA-US 66.61 vs 83.97). Đây là cách đọc của tôi từ Table 6. Tuyên bố "up to 18%" là mức tăng tương đối (GPT-3.5: 60.69→71.57, tương đương +10,9 điểm tuyệt đối).

Q5 — Retrieval-side hay Generation-side?

Phân loại: Both (kèm yếu tố cấu hình hệ thống).

Retrieval-side: sự lệch giữa corpus và loại câu hỏi (corpus chung hoặc sai làm giảm điểm); retriever phù hợp thì tốt hơn (MedCPT và BM25 ổn định; SPECTER kém vì huấn luyện không khớp bài toán, đây là giải thích của tác giả, không được kiểm chứng thực nghiệm); fusion không phải lúc nào cũng tốt hơn (RRF-2 hơn RRF-4 trên Wikipedia).
Generation-side: lost-in-the-middle; nhiễu khi k lớn; khác biệt giữa các LLM trong việc làm theo chỉ dẫn RAG (GPT-3.5 + MEDRAG hơn Mixtral + MEDRAG khoảng 3,0 điểm, dù CoT thì Mixtral nhỉnh hơn).
Giới hạn: vì chỉ có nhãn retrieval cho hai task, không thể tách riêng failure retrieval và failure generation ở ba task thi cử.
Q6 — Patient memory

Not reported / không áp dụng. Mỗi câu hỏi độc lập, không có EHR, lịch sử bệnh nhân hay conversation memory. Lưu ý: việc khuyên triển khai Mixtral cục bộ khi lo ngại quyền riêng tư bệnh nhân là khuyến nghị của tác giả, chưa được đánh giá.

Q7 — Safety / Triage

Not reported. Không có phát hiện cấp cứu, phân loại rủi ro, abstention, từ chối hay chuyển tuyến. Quan sát thiết kế: prompt gốc yêu cầu mô hình "have a definite answer" và đưa ra một lựa chọn, nghĩa là hệ thống không có đường abstain, kể cả khi retrieval thất bại. Đây là đặc điểm của benchmark trắc nghiệm, không có nghĩa là tác giả khuyến nghị làm vậy trong thực tế.

Q8 — Human / Clinician evaluation

Không có. Đánh giá hoàn toàn tự động (so đáp án). Tác giả tự ghi nhận phần rationale sinh ra bởi MEDRAG chưa được đánh giá. Hệ quả: một đáp án đúng có thể đến từ lập luận sai hoặc từ việc dùng nhầm context, và điều đó không được kiểm tra.

Q9 — Limitations

Do tác giả nêu (4 điểm):

#####	Nội dung	
Loại
1	Chỉ đánh giá vanilla RAG (nối snippet vào context); chưa có active RAG và các kiến trúc mới	Technical
2	Chưa bao phủ hết tài nguyên (full-text PMC, FAQ từ nguồn tin cậy)	Data
3	Chỉ đánh giá retrieval cho PubMedQA* và BioASQ-Y/N; chưa kiểm tra snippet có thật sự hữu ích cho câu hỏi thi; chưa thử cross-encoder re-ranker	Retrieval / Evaluation
4	Chỉ trắc nghiệm; LLM vẫn dùng các lựa chọn khi dự đoán cuối; chưa đánh giá rationale; chưa thử task khác (claim verification)	Evaluation

Do chúng ta nhận diện (không phải tác giả nêu):

Loại	Nội dung	Mức
Evaluation	Không có metric hallucination/faithfulness/groundedness	[A]
Evaluation	Thiếu abstention; thiếu clinician	[A]
Data	Corpus chứa nguồn gốc câu hỏi theo thiết kế ở task văn献	[A, tác giả nhắc một phần]
Data	Contamination với dữ liệu tiền huấn luyện chưa được phân tích	[D]
Evaluation	Điểm trung bình không trọng số, bị chi phối bởi hai task văn献	[D] suy từ Table 6
Evaluation	Mức giảm của RAG trên task thi cử không được bàn đến, trong khi nó ảnh hưởng trực tiếp đến câu hỏi "RAG có giúp được không"	[D]
Statistics	Chạy ở nhiệt độ 0; độ nhạy với prompt và biến thiên giữa lần chạy: Not reported; không có kiểm định ghép cặp	[A]
Consistency	Table 7 ghi MedCorp "65.3M" trong khi Table 3 là 30,4M tài liệu / 54,2M snippet	[A]
Clinical	Không đánh giá trên câu hỏi lâm sàng mở, dữ liệu bệnh nhân hay hội thoại	[A]
Cost / latency	Có nêu GPT-4 đắt hơn, nhưng không có số liệu chi phí/độ trễ của từng cấu hình	Not reported
Q10 — Future work
Explicit: đánh giá các kiến trúc RAG mới; mở rộng corpus; đánh giá retrieval cho task thi cử và thử cross-encoder re-ranker; mở rộng sang task khác (claim verification); đánh giá rationale.
Candidate research direction (của chúng ta, cần xác minh thêm, Candidate gap — requires further verification):
Cơ chế quyết định khi nào không nên dùng RAG hoặc nên abstain (bằng chứng: RAG làm giảm điểm trong nhiều ô).
Chọn hoặc định tuyến corpus theo loại câu hỏi (bằng chứng: corpus lệch làm tệ hơn baseline).
Đánh giá groundedness của rationale thay vì chỉ đáp án.
Evidence quality assessment
Strong: thiết kế so sánh có kiểm soát (cùng prompt, nhiệt độ 0, nhiều tổ hợp corpus × retriever × LLM, kèm sai số chuẩn); các số liệu trong Table 6 và Table 7 nhất quán với nhau (hàng GPT-3.5 + MedCorp + RRF-4 của hai bảng khớp).
Moderate: khuyến nghị thực hành (dựa trên một benchmark trắc nghiệm và vanilla RAG).
Weak / speculative: các giải thích cơ chế (vì sao SPECTER kém; "snippet khó truy xuất" ở task thi cử).
Missing: hallucination/faithfulness, human và clinician evaluation, safety/abstention, contamination, retrieval metrics cho task thi cử, tri thức thời sự, chi phí/độ trễ.
Relevance to AI Healthcare Failure Map
Category	Đánh giá
5. Evidence / RAG	Mạnh (retrieval mismatch, k, vị trí snippet, corpus/retriever)
1. AI Accuracy	Mạnh (điều kiện RAG giúp và làm hại accuracy)
6. Privacy / Data	Insufficient evidence (chỉ có một khuyến nghị về triển khai cục bộ)
2, 3, 4, 7, 8	Not relevant / insufficient evidence
Ghi chú cho Part B (mức [D], chỉ là giả thuyết, chưa phải kết luận)
Giao thức không thể so sánh trực tiếp giữa nguồn #2 và #3. Nguồn #2 (MedGraphRAG) báo cáo RAG tăng accuracy đồng đều cho GPT-4 trên các dataset cùng họ (ví dụ MedQA-US no-retrieval 78.2 → RAG chuẩn 88.1), còn nguồn #3 cho thấy GPT-4 giảm trên các task thi với retrieval question-only (83.97 → 82.80). Hai bài khác nhau về corpus, prompt, ensemble và cách retrieve. Ngoài ra nguồn #2 không nêu rõ có gỡ context của PubMedQA hay không (baseline GPT-4 PubMedQA là 75.3 so với 39.60 ở nguồn #3). Không nên coi hai nhóm số liệu là cùng một thiết lập.
Trùng tên "MIRAGE": benchmark trong bài này khác với baseline MIRAGE (Qi et al.) ở nguồn #2.
### Paper 3: A survey on medical multimodal RAG
A1. Source Identification
Trường	Nội dung
Title	A survey on medical multimodal retrieval-augmented generation (RAG)
Authors	Ruipeng Wang, Jian Wang, Yan Huang, Haiyang Guo, Junjie Pang (liên hệ: J. Pang, College of Computer Science and Technology, Qingdao University). Affiliation của các tác giả còn lại: Not available trong văn bản
Year / Venue	2026, High-Confidence Computing (Elsevier), article 100415, online 22/06/2026. Trạng thái: In Press, Journal Pre-proof
URL / DOI	https://www.sciencedirect.com/science/article/pii/S2667295226000346 · DOI 10.1016/j.hcc.2026.100415
License	Open access, CC BY-NC-ND 4.0
Publication type	Survey (narrative review), đã qua phản biện (bản pre-proof)
Primary / Secondary	Nguồn thứ cấp về từng framework mà nó mô tả
Competing interests / Funding	Không có xung đột lợi ích. Tài trợ: Scientific Research Project của General Administration of Customs of the PRC (Grant 2024HK208)
Evidence strength	Moderate như bản đồ khái niệm và nguồn khám phá paper. Weak như bằng chứng thực nghiệm
Phương pháp tổng quan	Not reported: không có chiến lược tìm kiếm, tiêu chí chọn bài hay khoảng thời gian. Không thể coi là systematic review. Không có mục limitations của chính survey
Phạm vi	Chỉ nhánh đa phương thức của Medical RAG, không phải Medical RAG tổng quát
Relevance to Failure Analysis	Cat 5 (Evidence/RAG): mạnh về khái niệm. Cat 1, 6, 7: vừa. Cat 2: yếu
Cấu trúc survey (Fig. 1) và những gì nó cho thấy

Fig. 1 là sơ đồ tổ chức của survey, với 7 nhánh:

Nhánh	Nội dung
Technical foundations	Nguyên lý RAG, học biểu diễn đa phương thức, tiến hóa và mô hình
Multimodal medical data	Ảnh, văn bản, modality khác, thách thức dữ liệu
Key challenges & solutions	6 mục: semantic gap & alignment; data quality & completeness; explainability & reliability; domain adaptation & generalization; knowledge synthesis & complex reasoning; scalability & deployment
Applications	Hỗ trợ chẩn đoán, hỏi đáp y khoa, tạo báo cáo tự động, giáo dục, lập kế hoạch điều trị cá nhân hóa
Representative frameworks	Retrieval control & knowledge balance; structured & factual knowledge; paradigm & architecture innovation
Datasets & evaluation	Dataset đa phương thức, metric, phân tích hiệu năng framework
Future directions	Cross-modal reasoning; few-/zero-shot; khung đánh giá chuẩn hóa; đạo đức và triển khai; tích hợp công nghệ AI khác

[A — source fact] Trong 6 mục "challenges", không có nhánh riêng cho an toàn lâm sàng/triage, bộ nhớ bệnh nhân dọc thời gian, hay quyền riêng tư. Quyền riêng tư nằm trong "Data challenges" (mục 3.4) và "Ethics & deployment" (mục 9.4).
[D — our reading] Vì taxonomy của survey không có các chủ đề này, sự vắng mặt của chúng không phải bằng chứng về khoảng trống nghiên cứu; nó chỉ phản ánh phạm vi của survey.

A2. Reference mapping

Danh sách có 189 mục, phần lớn 2024–2025, nhiều preprint (arXiv, medRxiv, SSRN, Preprints.org). Tôi không đếm chính xác. Mọi ID lấy từ danh sách tham khảo của chính bài, chưa kiểm chứng độc lập.

Priority A
Ref	Công trình	Loại	Vì sao A	Điều tra tiếp?
[184]	Liu, McCoy, Wright. Improving LLM applications in biomedicine with RAG: a systematic review, meta-analysis, and clinical development guidelines. JAMIA 32(4):605–615 (2025)	Systematic review + meta-analysis	Bằng chứng tổng hợp có hệ thống nhất liên quan RAG y sinh. Survey chỉ trích dẫn nó cho một ý phụ (dữ liệu bất đồng nhất, mục 9.4)	Cao nhất
[6]	Vladika et al. Facts fade fast: ... memorization of outdated medical knowledge in LLMs. Findings EMNLP 2025	Primary	Bằng chứng trực tiếp cho tri thức lỗi thời (Cat 1)	Cao
[34]	Benani et al. Is multimodal better? A systematic review of multimodal vs unimodal ML in clinical decision-making. medRxiv 2025	Systematic review (preprint)	Kiểm tra trực tiếp tiền đề trung tâm của survey	Cao
[9], [10]	Ke et al., npj Digit. Med. 8(1):187 (2025); Xu et al. (MEGA-RAG), Front. Public Health 13:1635381 (2025)	Primary	Nguồn của claim "RAG cải thiện đánh giá tiền phẫu / giảm thông tin sai"	Cao
[89], [183]	Wang et al., Safety challenges of AI in medicine, arXiv:2409.18968; Bunnell et al., scoping review RAG đạo đức/thiên lệch/minh bạch, medRxiv 2025	Review	An toàn, đạo đức, thiên lệch (Cat 3, 6, 7)	Cao
[175]	Wu et al., MedJourney, NeurIPS 2024	Benchmark	Hành trình lâm sàng của bệnh nhân (Cat 2, 8). Nội dung chỉ suy từ tiêu đề	Cao
[125], [126], [127]	Gargari & Habibi (narrative review); Johnson et al. (Biol. Methods Protoc. 10(1):bpaf017); RAGnosis (Rouhollahi et al., medRxiv)	Hỗn hợp	Truy ngược claim chẩn đoán 54→78%; EHR RAG	Trung bình–Cao
[36], [115]	Pingua et al. (fine-tuning vs RAG); Cui et al. Bailicai (Big Data Min. Anal. 2026)	Primary	[115] trùng tên Bailicai ở nguồn #2 (ghi tác giả khác: Long et al., arXiv:2407.21055). Có thể là cùng công trình, Not verified	Trung bình
8 framework (xem bảng A3)	[22] MMed-RAG; [109] RULE; [137] FactMM-RAG; [140] V-RAG; [138] EMERGE; [139] XLR-KGDD; [141] RAICL; [142] multi-agent	Primary	Trọng tâm mục 6 và 8	Theo bảng audit
[61], [2]–[4], [7]	Scoping review LLM + EHR (arXiv:2405.03066) và các review LLM y khoa	Review	Nền cho Cat 2	Trung bình
[39], [40], [43], [44], [75]	Med-PaLM 2, Me-LLaMA, LLaVA-Med, Med-PaLM M, Van Veen et al.	Primary	LLM y khoa nền	Trung bình
Priority B
Nhóm	Công trình (ID trong nguồn)
Tin cậy / đánh giá RAG	[104] arXiv:2502.06872; [105] arXiv:2409.10102; [108] RAG-Check arXiv:2501.03995; [106] MCiteBench; [165] arXiv:2412.05579; [176] Croxford et al.
Ngữ cảnh, thiếu thông tin, nhiễu	[21] arXiv:2502.14759; [113] arXiv:2404.14043; [99] RAAT; [98] RagVL
Kỹ thuật RAG	[8] Lewis; [19] FLARE; [20] RAPTOR; [23] CAG; [24] RA-DIT; [111] RAFT; [110] ALoFTRAG; [112]; [116] HM-RAG; [117] OmniSearch; [120] MM-R5
Survey đa phương thức	[13] arXiv:2502.08826; [35] arXiv:2504.08748; [102] arXiv:2303.10868
Metric báo cáo X-quang	[177] GREEN; [178] RaTEScore; [179] GEMA-Score
Privacy	[187] FRAG arXiv:2410.13272
Priority C

CLIP, ConVIRT, LoVT, PRIOR, MACL, HNSW, OMGM, MultiFinRAG, GENIUS, các dataset ở Table 1, và kỹ thuật chung (BM25, BERT...).

Audit chất lượng nguồn của các claim then chốt
Claim trong survey	Được hỗ trợ bởi	Đánh giá
Bỏ modality phi văn bản gây bất định, ảnh hưởng độ chính xác và an toàn	[11], [12], [13], [34]	Yếu–vừa: không có so sánh có đối chứng được thuật lại; [34] là preprint
RAG cải thiện đáng kể chẩn đoán, tiền phẫu, giảm thông tin sai	[9], [10]	Primary, nhưng đúng bối cảnh hẹp (tiền phẫu; y tế công cộng)
Tri thức lỗi thời gây hại	[6]	Primary, mạnh
Chẩn đoán chính 54%→78%	[125] narrative review	Thứ cấp của thứ cấp; nghiên cứu gốc chưa xác định
LLM nhớ rồi làm lộ dữ liệu	[91] review ung thư	Thứ cấp
Dữ liệu khan hiếm do thủ tục đạo đức	[92] IJPREMS	Nguồn yếu
"Garbage in, garbage out"	[97] Preprints.org	Nguồn yếu
Cần giám sát bởi chuyên gia y tế	[180] RARE	Theo tiêu đề là bài kỹ thuật RAG; mức phù hợp claim–nguồn chưa kiểm chứng
Zero-shot RAG / agentic RAG làm mẫu	[173] FilterRAG (VQA tổng quát); [170] TANS-Agent (quần vợt)	Ngoài y tế
A3. Q1–Q10 Extraction
Q1 — Failure đang được giải quyết
[B — author's claim] Ba hạn chế của LLM: hallucination, tri thức lỗi thời, thiếu minh bạch. RAG "effectively mitigates" chúng (Abstract, Conclusion).
[B] Lập luận riêng của bài: RAG chỉ-văn-bản bỏ qua modality khác nên làm giảm độ chính xác chẩn đoán và an toàn; do đó cần multimodal RAG.
Fig. 2 mã hóa luận điểm này thành hình: (a) LLM thuần: cảnh báo "Hallucinations" và "Outdated knowledge"; (b) RAG truyền thống: dấu tích xanh "Mitigates hallucinations" nhưng cảnh báo "Lacks multimodal cues"; (c) Multimodal RAG: sao vàng "Comprehensive & factual".

Nhận xét phản biện (mức [D]):

Fig. 2 trình bày kết quả như vô điều kiện: bộ retriever được vẽ như một thành phần luôn đúng, không có chú thích nào về lỗi retrieval, lỗi generation hay cơ chế an toàn trong cả ba panel. Điều này mâu thuẫn với chính mục 4 và 6 của bài (liệt kê nhiều failure mode). Theo nguyên tắc của protocol, đây là phát biểu của tác giả, không phải kết luận được hỗ trợ.
Hallucination và tri thức lỗi thời không được đo trong bất kỳ phép đánh giá nào của mục 8; "mitigates" là claim không có phép đo trong bài.
Q2 — Failure RAG vẫn chưa giải quyết (theo survey)
Nhóm	Nội dung	Mức bằng chứng
Retrieval	Modality gap / same-modality bias: retrieve ảnh giống về thị giác nhưng không liên quan lâm sàng, bỏ sót hướng dẫn dạng văn bản. Đây là kịch bản minh họa giả định, không kèm trích dẫn	Weak (minh họa)
Retrieval	Giống ngữ nghĩa ≠ đúng sự kiện lâm sàng (động cơ của FactMM-RAG)	Secondary
Retrieval	Ngữ cảnh thiếu hoặc thừa; k tùy ý đưa nhiễu (RULE); hiệu năng bão hòa khi tăng số snippet [21]	Secondary
Retrieval	Tri thức lỗi, mâu thuẫn, thiếu modality ("garbage in, garbage out")	Argued, nguồn yếu
Generation	Tin quá mức thông tin retrieve dù tri thức nội tại vốn đúng ("correct without RAG, incorrect with RAG")	Secondary (RULE)
Generation	Lệch cross-modal: dựa vào văn bản retrieve mà bỏ ảnh đầu vào (MMed-RAG)	Secondary
Generation	Generator hiểu sai dù nhận đúng ngữ cảnh [104]; hallucination khi sinh báo cáo; giải thích có thể sai (XLR-KGDD)	Argued
System	Nguy cơ tái định danh bị khuếch đại; ghi nhớ rồi làm lộ dữ liệu; HIPAA/GDPR; khan hiếm và mất cân bằng dữ liệu; thiên lệch; nút thắt gán nhãn	Argued (nguồn thứ cấp/yếu)
System	Chi phí, độ trễ; lan truyền lỗi trong pipeline đa agent; kho tri thức tĩnh/lỗi thời là rủi ro lâm sàng (9.4)	Argued/Secondary
Q3 — Dataset / knowledge source

Table 1 và các mục 7.1.x (12 tài nguyên), phân theo vai trò:

Vai trò	Tài nguyên
Benchmark / evaluation do từng framework dùng	FactMM-RAG: MIMIC-CXR, CheXpert (zero-shot). RULE: MIMIC-CXR, IU-Xray, Harvard-FairVLMed. MMed-RAG: thêm PMC-OA, Quilt-1M (VQA). V-RAG: MIMIC-CXR, MultiCaRe. Multi-agent: MIMIC-CXR, IU X-ray. EMERGE: MIMIC-III, MIMIC-IV. XLR-KGDD: MIMIC-Eye (không có trong Table 1). RAICL: TCGA, IU-CXR
Tiền huấn luyện vision-language	PMC-OA (1.646.592 cặp), BIOMEDICA (24.076.288 cặp), Quilt-1M (1 triệu cặp; nguồn gồm YouTube qua ASR, Twitter, PubMed OA, LAION), MultiCaRe (96.428 ca, 135.596 ảnh)
EHR	MIMIC-III (49.785 lượt nhập viện), MIMIC-IV (431.231 lượt, 180.733 bệnh nhân), cả hai gồm dữ liệu có cấu trúc và ghi chú
Knowledge base cho RAG (theo gợi ý của survey)	TCGA (omics), PrimeKG (17.080 bệnh, 129.375 node, hơn 4 triệu quan hệ)
Công bằng	Harvard-FairVLMed (10.000 bản ghi glaucoma, có thuộc tính nhân khẩu học)
Chuẩn ảnh X-quang	IU X-ray (3.955 báo cáo, 7.470 ảnh), MIMIC-CXR (377.110 ảnh, 65.379 bệnh nhân), CheXpert (224.316 ảnh) và CheXpert Plus

Các điểm cần ghi:

Corpus truy xuất thực tế của từng framework: Not reported trong survey; chỉ có dataset đánh giá.
Table 1 chủ yếu là dataset cho tạo báo cáo và tiền huấn luyện; không có dataset nào được gắn là benchmark chuyên cho RAG hoặc hỏi đáp theo hướng dẫn lâm sàng.
Quilt-1M dựng bằng quy trình tự động từ nguồn mạng/video; độ nhiễu nhãn không được thảo luận.
Kết quả theo nhóm nhân khẩu học trên Harvard-FairVLMed của bất kỳ framework nào: Not reported.
Q4 — Evaluation metrics
Nhóm	Trong survey (mục 7.2)
Phân loại / dự đoán	Accuracy, Precision, Recall, F1, ROC/PRC, AUROC, AUPRC
Sinh văn bản	BLEU, ROUGE, METEOR, BERTScore
Clinical factuality (X-quang)	F1-CheXbert, F1-RadGraph
LLM-as-a-Judge	Được nêu như khả năng đo cả relevance của retriever lẫn faithfulness của generator
Retrieval (Recall@k, MRR, NDCG, Context Precision/Recall)	Không có
RAG-specific chuẩn (faithfulness, answer/context relevance)	Không có dạng metric; chỉ LLM-as-a-Judge ở mức khái niệm
Human / clinician	Không có metric hay quy trình

Mâu thuẫn nội tại:

Mục 9.3 nói BLEU/ROUGE không phản ánh độ đúng lâm sàng và F1-CheXbert/RadGraph thiếu tính giải thích, nhưng mục 8 báo cáo kết quả chính bằng các metric đó (mục 9.3 còn viết "F1-CheXpert" trong khi mục 7.2.2 là F1-CheXbert).
Mục 4 mô tả chi tiết các retrieval failure mode, nhưng mục 7.2 không có metric nào để đo retrieval.
Q5 — Retrieval-side hay Generation-side?

Both, không có phân bổ định lượng. Phân bổ theo Fig. 3 và mục 6:

Retrieval-side: FactMM-RAG (huấn luyện retriever theo sự kiện lâm sàng), phần chọn retriever và số ngữ cảnh của MMed-RAG.
Generation-side: V-RAG (kiểm chứng entity rồi viết lại báo cáo), KBPT của RULE (giảm tin quá mức vào ngữ cảnh).
System-level / kiến trúc: multi-agent, RAICL (retrieval làm ICL), EMERGE, XLR-KGDD.

RAG-Check được nêu như cách tách điểm nghẽn retrieval khỏi generation, nhưng survey không báo cáo kết quả phân tích lỗi nào theo phía.

Q6 — Patient memory
EHR là dữ liệu đầu vào (mục 3.2, EMERGE, [127] RAGnosis, XLR-KGDD với patient KG theo ca). Patient KG theo mô tả được dựng cho một truy vấn chẩn đoán, không phải bộ nhớ dọc thời gian.
Mục 9 không có hướng nghiên cứu về bộ nhớ bệnh nhân dọc thời gian. Mục 3.4 chỉ nhắc rằng dữ liệu phải được diễn giải đúng trình tự thời gian.
Longitudinal patient memory / conversation memory: không được đề cập. Đây là dữ kiện về phạm vi của survey (xem nhận xét ở Fig. 1), không phải bằng chứng về khoảng trống.
Q7 — Safety / Triage
Not reported trong cả 8 framework (theo tóm tắt của survey): không có phát hiện cấp cứu, phân loại rủi ro, abstention hay escalation.
Mục 4.3 mô tả hiệu chỉnh độ tin cậy như yêu cầu cần có; mục 9.4 nêu nguyên tắc (AI là công cụ hỗ trợ; phải có chuyên gia giám sát; chưa giải quyết được vấn đề trách nhiệm pháp lý; thiếu khung kiểm định thống nhất). Không có hệ thống nào được mô tả hiện thực hóa chúng.
Fig. 2 không có thành phần an toàn hay human-in-the-loop trong cả ba panel.
CAG và FLARE là điều khiển retrieval, không phải abstention.
Q8 — Human / Clinician evaluation
Không có đánh giá của clinician cho bất kỳ framework nào ở mục 8 (theo survey).
Đánh giá duy nhất không dựa trên chỉ số tự động là LLM-as-a-Judge trong framework multi-agent (Claude 3 Opus, thang 1–10, 5 chiều), tức là mô hình chứ không phải bác sĩ.
Việc bài gốc của từng framework có clinician hay không: Not verified.
Q9 — Limitations

Của chính survey:

Loại	Nội dung
Method	Không có phương pháp tổng quan; tiêu chí chọn "representative works" là Not reported; không có mục limitations
Evaluation	Mục 8 không phải so sánh có kiểm soát: dẫn nhập hứa "đánh giá và so sánh", nhưng thực chất là tóm tắt số liệu tự báo cáo trên dataset và metric khác nhau. Mục 9.3 thừa nhận không thể so sánh trực tiếp giữa các nghiên cứu
Evidence	Mục 8 chỉ có kết quả tích cực (ngoại lệ duy nhất: "consistency" của multi-agent); không có kết quả âm tính hay thất bại
Conclusion vs. evidence	Kết luận nói RAG "effectively mitigates outdated knowledge and hallucination", trong khi không phép đo nào ở mục 8 đo hai hiện tượng đó
Internal consistency	Mục 6.2 nói thành phần RAG của XLR-KGDD không cải thiện độ chính xác chẩn đoán, nhưng mục 8.2 vẫn trình bày XLR-KGDD như bằng chứng thành công của multimodal RAG
Citations	Một số claim dựa vào nguồn yếu hoặc ngoài lĩnh vực (xem audit ở A2)
Scope	Chỉ nhánh đa phương thức; 5/8 framework thuộc chẩn đoán hình ảnh/X-quang (xem bảng dưới)
Status	Bản pre-proof

Giới hạn mà survey gán cho từng framework (secondary, chưa kiểm chứng ở bài gốc) nằm trong bảng dưới.

Bảng audit hợp nhất: 8 framework (Fig. 3 + mục 6 + mục 8)

Fig. 3 xếp cả 8 framework vào 2024–2025, theo ba màu: xanh dương = retrieval control & knowledge balance (6.1); xanh lá = structured & factual knowledge (6.2); cam = paradigm & architecture innovation (6.3).

Framework	Năm / venue (theo danh sách tham khảo)	Nhóm (Fig. 3)	Tác vụ và dữ liệu	Kết quả được nêu	Lưu ý và giới hạn
EMERGE [138]	2024, CIKM	Xanh lá	Dự đoán EHR (tử vong, tái nhập viện); MIMIC-III/IV; PrimeKG	AUROC 86,25 (tử vong, MIMIC-III), 80,61 (tái nhập viện, MIMIC-IV)	Dự đoán chứ không sinh câu trả lời; không có số baseline; pipeline nhiều tầng, phụ thuộc chất lượng KG
RULE [109]	2024, EMNLP	Xanh dương	Med-VQA + báo cáo; MIMIC-CXR, IU-Xray, Harvard-FairVLMed	"Tăng độ chính xác trung bình 47,4%"	Đơn vị (tuyệt đối hay tương đối) và baseline không rõ; cần validation set; KBPT chỉ xử lý "đúng khi không RAG, sai khi có RAG"
FactMM-RAG [137]	2024, arXiv	Xanh lá	Báo cáo X-quang; MIMIC-CXR, CheXpert (zero-shot)	F1CheXbert tối đa +6,5%; F1RadGraph +2% (MIMIC-CXR); zero-shot CheXpert +2% và +1,2%	Chỉ X-quang ngực; "factuality" phụ thuộc RadGraph
MMed-RAG [22]	2024, arXiv	Xanh dương	VQA + báo cáo; X-quang, mắt, giải phẫu bệnh	Vượt mọi baseline, gồm cả RULE và FactMM-RAG, trên gần hết metric	Cùng nhóm tác giả với RULE; chưa có tái lập độc lập được báo cáo; phức tạp, tốn tài nguyên
V-RAG [140]	2025, arXiv	Cam	Phát hiện entity sai, sửa báo cáo; MIMIC-CXR, MultiCaRe	F1 0,751 (MIMIC-CXR) và 0,940 (MultiCaRe); RadGraph-F1 tăng tương đối 19%	Chênh lệch lớn giữa hai dataset không được bàn; cần fine-tune, chi phí suy luận cao
RAICL [141]	2025, arXiv	Cam	Phân loại bệnh; TCGA, IU-CXR	Accuracy 0,7854→0,8368; macro-F1 0,1191→0,8454 (TCGA)	Macro-F1 baseline cực thấp cho thấy baseline gần như dự đoán lớp đa số; mức "đột phá" phản ánh baseline hỏng; phụ thuộc năng lực ICL
XLR-KGDD [139]	2025, Knowl. Inf. Syst.	Xanh lá	Chẩn đoán giải thích được (CHF, UTI); MIMIC-Eye	AUROC/AUPRC 0,88/0,91 (CHF), 0,90/0,83 (UTI)	Survey nói thành phần RAG không cải thiện độ chính xác; giải thích có thể sai; cải thiện có thể đến từ patient KG và fine-tuning
Multi-agent [142]	2025, arXiv	Cam	Báo cáo X-quang; MIMIC-CXR, IU X-ray	BERTScore 0,8617→0,8819; điểm chẩn đoán do LLM chấm 7,78→8,26	Kém hơn ở "consistency"; baseline là LLaVA-Med một agent, không phải RAG một agent, nên không gán cải thiện cho retrieval; lỗi sớm lan truyền, độ trễ cao

Mẫu hình chung của 8 framework (theo survey; mức [C] tổng hợp trong nguồn này):

5/8 thuộc báo cáo/VQA X-quang hoặc hình ảnh (FactMM-RAG, RULE, MMed-RAG, V-RAG, multi-agent); EMERGE, XLR-KGDD, RAICL là dự đoán, chẩn đoán và phân loại.
Chỉ RULE và MMed-RAG (cùng nhóm tác giả) nhắm trực tiếp vào việc RAG gây hại và việc tin quá mức vào ngữ cảnh.
Không framework nào có metric retrieval hay đánh giá của clinician trong mục 8.
Mọi kết quả đều là số tự báo cáo trên dữ liệu hồi cứu công khai; không có triển khai tiến cứu hay lâm sàng được nêu.
Q10 — Future work

Explicit (mục 9, 10):

Hiểu và suy luận đa phương thức: alignment (M3Bind, CAT), multimodal chain-of-thought, lập luận phân cấp, Agentic RAG.
Few-shot/zero-shot cho dữ liệu khan hiếm. Các ví dụ minh họa là segmentation và VQA tổng quát (FilterRAG dùng Wikipedia), chưa phải bằng chứng y khoa.
Khung đánh giá chuẩn hóa: dataset chung, đa modality, gần thực tế; metric có ý nghĩa lâm sàng (RaTEScore, GEMA-Score).
Đạo đức và triển khai: giám sát bởi chuyên gia, trách nhiệm pháp lý, đồng thuận và ẩn danh, thiên lệch (kể cả thiên lệch ngôn ngữ), tích hợp bệnh viện, cập nhật liên tục kho tri thức, khung kiểm định thống nhất.
Kết hợp công nghệ khác: policy retrieval bằng RL ([185], không có venue), federated learning và federated retrieval với mã hóa đồng cấu (đề xuất kiến trúc, chưa có đánh giá thực nghiệm được trích dẫn).

Candidate research direction (của chúng ta, Candidate gap — requires further verification):

Metric cho retrieval và quy kết lỗi trong Medical RAG (mục 7.2 của chính survey không có).
Cơ chế abstention / hiệu chỉnh độ tin cậy được đánh giá thực nghiệm (4.3 nêu cần; không framework nào báo cáo).
Đánh giá độ cập nhật của kho tri thức (9.4 nêu rủi ro; không framework nào đo).
Evidence quality assessment
Strong: phân loại failure mode (retrieval / generation / system) và tổng hợp dataset, metric. Không có bằng chứng thực nghiệm mới.
Moderate: mô tả dataset (mục 7.1), sơ đồ tổ chức (Fig. 1) và dòng thời gian (Fig. 3, khớp với năm trong danh sách tham khảo).
Weak / anecdotal: các con số mục 8 (tự báo cáo, đơn vị không rõ, baseline không đồng nhất); kịch bản modality gap; Fig. 2 (khẳng định kết quả không điều kiện); kết luận về hallucination và tri thức lỗi thời.
Missing: phương pháp tổng quan, corpus truy xuất của từng framework, kết quả âm tính, đánh giá lâm sàng, retrieval metric, đối chiếu với bài gốc.
Relevance to AI Healthcare Failure Map
Category	Đánh giá
5. Evidence / RAG	Mạnh về khái niệm. Về định lượng chỉ ở mức thứ cấp
1. AI Accuracy	Vừa ([6], [9], [10] là lead primary)
6. Privacy / Data	Vừa; đề xuất federated retrieval, mã hóa chưa có đánh giá; claim rủi ro dựa vào review
7. Trust	Vừa–yếu: minh bạch, trích dẫn, hiệu chỉnh độ tin cậy, trách nhiệm pháp lý, giám sát. Commercialization: không được đề cập
2. Memory / Health Records	Yếu: EHR như dữ liệu, không có bộ nhớ dọc thời gian
3, 4, 8	Not relevant / insufficient evidence
Ghi chú cho Part B (mức [D], ứng viên, chưa phải finding)
Retrieval metric hiếm trong paper y khoa: nguồn #2 không có, nguồn #3 chỉ có ở 2/5 task, mục 7.2 của nguồn #4 không có. Nguồn #1 có nhưng ngoài y tế. Đây là chủ đề nhất quán nhất cho đến giờ.
RAG có thể làm hại khi model vốn đã đúng: nguồn #3 (dữ liệu sơ cấp), nguồn #4 (qua RULE, thứ cấp). Mục 8 của survey chỉ nêu kết quả tích cực nên không phải bằng chứng đối lập.
Benchmark xa thực tế lâm sàng: nguồn #3 (corpus chứa nguồn của câu hỏi theo thiết kế), nguồn #4 (9.3). Nguồn #4 còn cho thấy toàn bộ 8 framework chỉ được đánh giá trên dữ liệu hồi cứu công khai.
Kho tri thức lỗi thời: nguồn #2 (future work), nguồn #4 (9.4); [6] là lead primary. Chưa framework nào trong bốn nguồn đo điều này.
Abstention / hiệu chỉnh độ tin cậy: nguồn #3 (prompt ép trả lời), nguồn #4 (cần nhưng chưa có), nguồn #2 (không có).
Bộ nhớ bệnh nhân dọc thời gian: chưa nguồn nào đánh giá. Đây mới là sự vắng mặt, chưa đủ để gọi là khoảng trống.
Mức độ độc lập của bằng chứng: RULE và MMed-RAG cùng một nhóm tác giả; các số liệu của survey là tự báo cáo. Khi lập evidence table, không nên tính chúng như hai nguồn độc lập.
### Paper 4: A survey on retrieval-augmentation generation (RAG) models for healthcare applications
A1. Source Identification
Trường	Nội dung
Title	A survey on retrieval-augmentation generation (RAG) models for healthcare applications
Authors	Mohamed Abo El-Enen, Sally Saad, Taymoor Nazmy. Faculty of Computer and Information Sciences, Ain Shams University, Cairo
Venue	Neural Computing and Applications 37(33): 28191–28267 (77 trang). Nhận 16/11/2024, chấp nhận 09/09/2025, online 16/10/2025
DOI	10.1007/s00521-025-11666-9
License	Open access, CC BY-NC-ND 4.0 (xác nhận)
Publication type	Survey / review, peer-reviewed
Primary / Secondary	Nguồn thứ cấp
Evidence strength	Weak như bằng chứng định lượng; Moderate như bản đồ hệ thống và nguồn khám phá paper
Phương pháp tổng quan	Not reported trên toàn văn: không có chiến lược tìm kiếm, tiêu chí chọn bài hay khoảng thời gian, dù mục 1 viết "We systematically review"
Conflict of interest / Funding	Tác giả khai không có xung đột lợi ích. Funding: Not reported
Data availability	Viết dataset "publicly available" nhưng đồng thời "obtained from the corresponding author upon reasonable request"; đây là survey, không có phân tích dữ liệu riêng
Relevance to Medical RAG	Cao: là survey Medical RAG tổng quát duy nhất trong các nguồn đã xử lý
Relevance to Failure Analysis	Cat 5, 1: vừa. Cat 7, 2: yếu. Cat 3, 4, 6, 8: chủ yếu ở mức khẳng định
A2. Reference mapping (hoàn chỉnh)

Danh sách có 134 tài liệu. Mọi ID lấy từ chính danh sách này, chưa kiểm chứng độc lập.

Priority A
Ref	Công trình	ID	Vì sao A	Ưu tiên
[106]	Ke et al. RAG for 10 LLMs ... assessing medical fitness	npj Digit. Med. 8:187 (2025)	Đánh giá 10 LLM; tỷ lệ hallucination thay đổi mạnh theo mô hình (Q2). Cũng là [9] của nguồn #4	Cao nhất
[88]	Zakka et al. Almanac	NEJM AI 1:AIoa2300068 (2024)	Có đánh giá của bác sĩ, thử nghiệm đối kháng, kết quả "ưa thích" ngược chiều độ chính xác	Cao
[128]	Alkhalaf et al., tóm tắt EHR ở viện dưỡng lão	J. Biomed. Inform. 156:104662	Kết quả RAG không cải thiện	Cao
[108]	Sohn et al., Rationale-guided RAG	arXiv:2411.00300	So sánh trực tiếp với MedRAG của Xiong et al.; mức tăng khiêm tốn	Cao
[96]	Yang et al., tích hợp UMLS vào LLM	arXiv:2310.02778	Nhiễu retrieval gây trả lời thiếu; có đánh giá bác sĩ	Cao
[117], [124], [121], [103]	Med-R² (arXiv:2501.11885); Omni-RAG (arXiv:2501.02460); MedRAG-KG (arXiv:2502.04413); MMed-RAG (arXiv:2410.13085)		Các hệ thống mà survey xếp "Modular"	Trung bình
[93], [94], [125], [89], [87], [85], [116], [110], [99], [133], [134]	accGPT/Radiology 308:e230970; FraCChat/Sci Rep 13:14215; CLEAR/NPJ Digit Med 8:45; Thompson/arXiv:2312.06457; LiVersa/Hepatology; ChatENT; GuideGPT/J Cranio-Maxillofac Surg 53:355; CliniqIR/JMIR Med Inform 12:e50209; TrialGPT/arXiv:2307.15051; RISE/JMIR 26:e58041; GastroBot/Front Med 11:1392555		Hệ thống Medical RAG được survey mô tả	Theo nhu cầu
[54]	Tan et al., S.C.O.R.E.	arXiv:2407.07666	Khung đánh giá an toàn/giải thích	Trung bình
[120], [20]	MedGraphRAG; Xiong et al.		Chính là nguồn #2 và #3	Đã xử lý
Priority B

[19] Gao et al. (arXiv:2312.10997); [22] FLARE; [23] Self-RAG; [53] RAGAS; [21] Li et al., best practices (arXiv:2501.07391); [15], [16] survey RAG; [17] Wang & Zhang (Artif. Intell. Rev. 57:299); [18] Gargari & Habibi (ID …20552076251337176, nguồn #4 ghi …177, cần kiểm tra bằng DOI); [35] DR.BENCH; [26], [25], [29] PubMedQA, MedMCQA, MedQuAD.

Priority C

[1], [2] GPT-3/GPT-4; [31] LLaMA; BERT-based ([61], [62], [69]–[72]); các LLM y khoa ([56]–[68]); CLIP/VLM ([73]–[84]); dataset đa phương thức ([36]–[49]).

A3. Q1–Q10 Extraction
Q1 — Failure đang được giải quyết
[B — author's claim] LLM trong y tế gặp bốn vấn đề: tri thức tĩnh, hallucination, sắc thái chuyên ngành, thiếu nguồn dẫn kiểm chứng. RAG giải quyết bằng truy xuất từ nguồn tin cậy; Kết luận viết RAG "mitigating hallucinations and enhancing response accuracy".
Cần lưu ý: phần lớn hệ thống ở mục 5.6 không đo hallucination trực tiếp; "reduces hallucination" thường là mô tả động cơ.
Q2 — Failure RAG vẫn chưa giải quyết (theo survey, thứ cấp)
Nhóm	Bằng chứng trong survey
Hallucination vẫn còn phụ thuộc generator	Ke et al. [106] (có RAG): GPT-4 hallucinate 0–2,9%, Llama 2 12,8–48,6%
RAG không giúp được	Alkhalaf [128]: tóm tắt cấu trúc 93,25%→99,25%, nhưng trích xuất yếu tố nguy cơ suy dinh dưỡng giữ nguyên 90% (thông tin ngầm trong ghi chú)
Nhiễu retrieval	Yang [96]: thuật ngữ trích ra không liên quan câu hỏi → trả lời thiếu; LLaMA2+UMLS có ROUGE/BERTScore thấp hơn ChatGPT-3.5
Độ phủ và chất lượng kho tri thức (lặp lại ở ≥7 hệ thống)	LiVersa (rationale thiếu ở 3/10 ca); Ke (guideline địa phương kém vì ít văn bản); Med-R² (thiếu tri thức, phụ thuộc bộ phân loại); CliniqIR (thiên lệch phủ của PubMed); GastroBot (nguồn tĩnh, tính địa phương); accGPT (ca không thuộc chẩn đoán hình ảnh hiếm); FraCChat (gãy xương phức tạp)
Chấp nhận của người dùng	Almanac: chính xác hơn nhưng bác sĩ chọn ChatGPT 57% (số liệu thứ cấp, bài gốc chưa kiểm chứng)
System	Phức tạp, bảo trì, chi phí tính toán, quy định pháp lý, trách nhiệm, thiên lệch (mục 3–4, 7)
Q3 — Dataset / knowledge source 

Table 7 liệt kê 42 hệ thống trong cột "RAG Model/Authors" (đề là "RAG models in the medical domain"). Bảng không có cột baseline, loại metric (tuyệt đối hay tương đối) hay cỡ mẫu chuẩn hóa. Phân loại dưới đây là của tôi, dựa vào cột "Evaluation Dataset(s)".

Phân loại theo vai trò dữ liệu
Nhóm	Hệ thống (Ref)	Ghi chú
Benchmark y khoa công khai (MedQA, MedMCQA, PubMedQA, MMLU-Med)	Xiong [20]; [108]; [117]; LLM-AMT [98]; Omni-RAG [124]; MedGraphRAG [120]	Khoảng 6–7/42 hệ thống
Tập nhỏ, tự dựng, kèm đánh giá chuyên gia	LiVersa [87]: 10 vignette; Almanac [88]: 130 ca; GuideGPT [116]: 30 câu; Mashatian [113]: 175 câu; CaLM [115]: 66 cặp; accGPT [93]: 50 ca tổng hợp; GastroBot [134]; RISE [133]; Benfenati [111]; ChatENT [85]	Cỡ mẫu 10–175
EHR / ghi chú lâm sàng	Thompson [89]: 199 bệnh nhân; Alkhalaf [128]: 1.424 + 1.399 ghi chú; Ye [102]; CLEAR [125]: 20.000 EHR; DR.KNOWS [132] và CliniqIR [110]: MIMIC-III; MedRAG-KG [121]: DDXPlus + CPDD; RAGPR [122]: 646.383 hồ sơ tư vấn	Mỗi hệ thống dùng cho một tác vụ: phenotyping, tóm tắt, trích xuất, chẩn đoán hoặc ghép chuyên khoa
Ngoài y tế	MS-RAG [127]: NQ, TriviaQA; FLARE [22]: WikiMultihopQA, ASQA-hint; SELF-RAG [23]: PopQA, TQA, "PubQA", ARC-Challenge, biography, ASQA	3/42 hàng không có dữ liệu y tế dù bảng đề "in the medical domain"
Tác vụ khác RAG hỏi-đáp	RALL [86] (CELLS: 63.000 tóm tắt); Upadhyay [104] (CLEF eHealth/TREC); Painter [114] (SQL cho cơ sở dữ liệu an toàn thuốc); Markey [112] (tài liệu thử nghiệm lâm sàng); TrialGPT [99] (ghép bệnh nhân–thử nghiệm: 1.015 cặp bệnh nhân–tiêu chí, cohort tổng hợp); Duan [130] (679 ca YHCT); Das [129] (Reddit); docGPT [101]	
Dữ liệu riêng hoặc nội bộ (khó tái lập)

CPDD của MedRAG-KG (chronic pain, dữ liệu riêng); RAGEv-Bench (một phần là kết quả đầu ra của dự án Horizon Europe, Virtual Human Twin, Antimicrobial Resistance); Painter (cơ sở dữ liệu an toàn thuốc hơn 500 bảng, 1,3 tỷ bản ghi); Alkhalaf (hồ sơ viện dưỡng lão Úc); RAGPR (hồ sơ tư vấn của nền tảng); CLEAR (Stanford MOUD). Tôi ước tính ít nhất 6/42.

Phân loại theo tri thức dùng làm kho truy xuất
Loại	Ví dụ
Guideline / tiêu chuẩn chuyên khoa	LiVersa (30 guideline AASLD), accGPT (209 tài liệu ACR), FraCChat (AO/OTA), GuideGPT (449 bài), Ke et al. (58 guideline), GastroBot (25 guideline + 40 bài)
Y văn (PubMed, PMC, sách giáo khoa)	Xiong, Sohn, LLM-AMT (sách giáo khoa), CliniqIR (33 triệu abstract PubMed)
Đồ thị tri thức	MedGraphRAG; MedRAG-KG (4 tầng); Duan (Neo4j); MedKgConv, DR.KNOWS (UMLS)
Trang web / mạng xã hội	RISE (200 website học thuật + 600 bài), Das (Reddit), Upadhyay (PMC)
Dữ liệu bệnh nhân	Thompson, Alkhalaf, Ye, CLEAR
Nhận xét
Corpus truy xuất thường nhỏ và hẹp theo chuyên khoa (10–200 tài liệu cho hầu hết hệ thống ứng dụng).
Không có hệ thống nào dùng một corpus chung hay benchmark chung ngoài nhóm 6–7 hệ thống dùng benchmark MCQ; điều này ngăn so sánh, đúng như mục 7.2.1 thừa nhận.
Dataset ở mục 5.1–5.2 (Table 3–4) không trùng với dataset thực sự dùng trong Table 7 ở phần lớn trường hợp (ví dụ MIMIC-III chỉ xuất hiện ở hai hệ thống; MedQuAD chỉ ở một)
Q4 — Evaluation metrics
Nhóm	Trong survey
Mục 5.3 (định nghĩa)	Accuracy, Precision, Recall, Specificity, F1; BLEU, ROUGE-L, METEOR; RAGAS (Faithfulness, Answer Relevance, Context Relevance); Manual Accuracy; S.C.O.R.E. (Likert 1–5, Cohen's κ)
Retrieval metric trong các hệ thống ở 5.6	GastroBot: hit rate 89%; CliniqIR: MRR (0,35 vs 0,15); Upadhyay: CAM-MAP/CAM-NDCG; Ye: Precision@10; GastroBot và Duan: RAGAS context recall (0,95)
Safety / human	Ke: S.C.O.R.E. (an toàn 4,93/5); GastroBot, Duan: "safety/usability/smoothness" (thang 3; "SUS" không được định nghĩa); Almanac: an toàn khi thử đối kháng

Nhận xét:

Survey không áp dụng S.C.O.R.E. Mục 1.1 viết đóng góp là "standardized assessment ... using ... S.C.O.R.E. across medical benchmarks", nhưng không có đánh giá nào do survey thực hiện, và mục 7.2.1 thừa nhận các nghiên cứu dùng khung đánh giá không thống nhất, hạn chế so sánh giữa các hệ thống. Chỉ Ke et al. dùng S.C.O.R.E.
Context recall (RAGAS) xuất hiện ở hai hệ thống nhưng mục 5.3 không định nghĩa nó (chỉ định nghĩa Context Relevance kiểu precision). Các con số rất cao (0,95) đo bằng LLM-judge trên hệ thống nhỏ, nên chưa nên coi là bằng chứng mạnh.
Q5 — Retrieval-side hay Generation-side?

Both. Mục 6.1 kết luận "retrieval quality determines success" (Xiong et al.). Failure phía retrieval/corpus là loại được thừa nhận nhiều nhất ở Q2; phía generator thể hiện qua hallucination phụ thuộc mô hình (Ke). Không có phân tích lỗi định lượng tách hai phía.

Q6 — Patient memory
Conversation memory: module Memory của Modular RAG (3.3.1) được mô tả khái niệm (bộ nhớ ngắn hạn, "long-term knowledge persistence", lưu sở thích và "learned patterns"); không hệ thống nào ở 5.6 được mô tả là có nó.
EHR như corpus: có ở nhiều hệ thống (xem Q3), nhưng cho tác vụ đơn lẻ (phenotyping, tóm tắt, trích xuất, tìm kiếm). Thompson [89] xử lý "toàn bộ hồ sơ" của bệnh nhân bằng regex + MapReduce cho một lần phân loại.
Longitudinal patient memory dùng xuyên các lần khám: không được mô tả hay đánh giá.
Mục 7.5.3 nêu hướng tương lai: tích hợp dữ liệu cá nhân vào retrieval ("current RAG systems primarily retrieve population-level knowledge"), kèm vấn đề riêng tư. Phát biểu này có thể hiểu là "cá nhân hóa tư vấn", không mâu thuẫn trực tiếp với các hệ thống EHR theo tác vụ ở trên.
Q7 — Safety / Triage
Emergency detection / risk escalation: Not reported ở mọi hệ thống.
Cơ chế gần với an toàn (thứ cấp): RISE [133] có bước "summarization with fact-checking and safety checks"; Med-Gemini dùng tìm kiếm có hướng dẫn bởi độ bất định; MedRAG-KG gợi ý câu hỏi theo dõi; Almanac thử nghiệm đối kháng (95% an toàn so với ChatGPT "failed entirely").
Gần triage (hành chính, không phải cấp cứu): RAGPR [122] ghép bệnh nhân với bác sĩ chuyên khoa (nhân viên triage chấm 4,67/5); Ke et al.: "93,0% accuracy for nurse/doctor triage" trong chỉ dẫn tiền phẫu.
Mục 4.3.3 và 4.5 vẫn là khẳng định ("dedicated modules for ... safety assessment"; Modular = "Clinical Safety: Comprehensive"), không kèm số liệu an toàn nào của hệ thống Modular.
Q8 — Human / Clinician evaluation (toàn bộ mô tả thứ cấp, chi tiết thiết kế thường Not reported)
Hệ thống	n / người chấm	Ghi chú
Almanac [88]	130 kịch bản; bác sĩ đã chứng nhận và nội trú	Số người chấm, chế độ mù: Not reported. Mô tả mâu thuẫn: "used to train" rồi "tested on" cùng một tập 130 ca
accGPT [93]	50 ca tổng hợp; so với bác sĩ X-quang	82% "usually appropriate" so với 64–74% (bác sĩ) và 64–80% (chatbot tổng quát)
Yang et al. [96]	20 câu; bác sĩ	UMLS-ChatGPT tốt hơn ở 40% (factuality), 55% (completeness), 40% (relevance); kém hơn ở readability
GuideGPT [116]	10 chuyên gia × 30 câu; có p-value	80,67% vs 69,67% mức nội dung cao
Das et al. [129]	21 chuyên gia	GPT-4 và mô hình 7B không khác biệt có ý nghĩa
Ke et al. [106]	S.C.O.R.E.; so với người đánh giá	GPT-4 RAG 96,4% vs người 86,6% vs GPT-4 không RAG 92,9%
RISE [133], Mashatian [113], RALL [86], TrialGPT [99]	Bác sĩ/bệnh nhân; 175 câu; v.v.	Chi tiết thiết kế: Not reported

Nhận xét: các nghiên cứu nêu ở trên chủ yếu có n từ 10 đến vài trăm câu, nhiều ca tổng hợp, từng trung tâm. Trong phần đã đọc không mô tả nghiên cứu tiến cứu hay kết cục bệnh nhân.

Q9 — Limitations

1. Table 8 ("Modular RAG clinical deployment readiness"): kiểm tra từng hàng

Đây là bằng chứng duy nhất mà mục 6.3 đưa ra cho khẳng định Modular RAG "achieve clinical deployment standards".

Hàng	Nội dung trong Table 8	Đối chiếu với phần còn lại của survey
Diagnostic Accuracy	Med-R² [117]: "EBM-guided reasoning achieves clinical-grade performance"; "79.72% improvement suggests readiness for decision support"	79,72% là mức tăng cho Qwen2.5-7B (5.6.3), mà Table 7 cho thấy độ chính xác tuyệt đối chỉ 67,97% trên các benchmark trắc nghiệm. Nếu 67,97% sau khi tăng 79,72%, baseline chỉ khoảng 38% (tính của tôi, giả định tăng tương đối). Mức tăng từ nền thấp không phải bằng chứng sẵn sàng cho quyết định lâm sàng. Benchmark là MedQA/PubMedQA/MMLU, không phải ca lâm sàng
Patient Safety	"RAG² [108]'s reflection tokens prevent harmful outputs through self-assessment"; "91.1% accuracy with GPT-4o approaches human expert level"	Tên "reflection tokens" và "self-assessment" thuộc SELF-RAG [23]; mô tả của [108] ở 5.6.2 là lọc theo perplexity, rationale làm truy vấn và truy xuất cân bằng, không có reflection tokens hay chặn đầu ra có hại. 91,1% là độ chính xác MedQA trắc nghiệm, không phải thước đo an toàn
Regulatory Compliance	"LiVersa [87] maintains PHI compliance"; "100% accuracy in hepatitis scenarios meets regulatory standards"	100% là kết quả yes/no trên 10 vignette. Khoảng tin cậy 95% (Clopper–Pearson) cho 10/10 có cận dưới khoảng 69% (tính của tôi). Table 7 ghi lời giải thích chi tiết có sai sót ở 3 ca. Tuân thủ PHI là đặc tính của nền tảng triển khai (UCSF Versa), không phải đánh giá quy định. Không có bằng chứng về phê duyệt hay chuẩn mực quy định nào được đáp ứng
Phân loại	Tiêu đề: "Modular RAG"	LiVersa được xếp Naive (5.6.1); [108] xếp Advanced (5.6.2). 2/3 ví dụ không phải Modular theo chính survey
Mâu thuẫn nội bộ	Mục 6.3: Modular "achieve clinical deployment standards"	Mục 7.2.2: hệ thống RAG hiện nay "have not navigated" quy trình phê duyệt; mục 7.2.2 cũng nói RAG thường không truy vết đầy đủ ảnh hưởng của thông tin retrieve
2. Table 7: kiểm tra tính nhất quán với văn bản
Mục	Table 7	Văn bản (mục 5.6)	Nhận xét
FraCChat [94]	Giảm thời gian "83,2 phút (người) → 7,2 phút"	"5,2 giây so với 50 giây"	Khác đơn vị và độ lớn
LiVersa [87]	Giải thích "contained errors in 3 cases due to missing guideline versions and contextual biases"	"Detailed rationales were incomplete for three scenarios"	"Sai sót" khác "chưa đầy đủ", và nguyên nhân chỉ có ở Table 7
CaLM [115]	"Outperforms GPT-3.5 in 85% of low-medium complexity queries"	"Achieving 85% accuracy for low-to-medium complexity queries"	Hai cách diễn giải khác nhau
LLama-2 RAG [90]	BLEU 0,372, ROUGE 0,294, BERT F1 0,861	"GPT variants achieve higher BLEU (0,45) and ROUGE (0,52) ... Llama-2 BERTScore (0,82)"	Số liệu khác nhau; BLEU 0,45 và ROUGE 0,52 trùng đúng số của CaLM (5.6.2) → nghi sao chép nhầm giữa hai mục
LLM-AMT [98]	"+11,6% đến 16,6%" (so với GPT-3.5)	Mục 3.3.5: "11,4% đến 13,2% so với GPT-4-Turbo không RAG"	Hai bộ số cho cùng một bài; baseline khác nhau
Almanac [88]	"Reduces hallucinations in medical QA"	Chỉ đo độ chính xác sự kiện và an toàn đối kháng	Hallucination không được đo
Omni-RAG [124]	"Reduces hallucinations, improves retrieval precision"	Chỉ báo cáo accuracy	Không có phép đo hallucination hay precision
RAGPR [122]	"Enhances triage accuracy and satisfaction"	Không có số liệu "satisfaction"; chỉ F1 và đánh giá 4,67/5 của nhân viên triage	Phần "satisfaction" không có căn cứ trong văn bản
docGPT [101]	"Outperforms retrieve-stuff-summarize in precision and contextual relevance"	"The absence of quantitative metrics like ROUGE or BLEU ... manual assessments"	Dùng ngôn ngữ định lượng cho đánh giá thủ công
Pingua [107]	"Hybrid FT+RAG achieves BERTScore F1 0,891"	BLEU của RAG (0,163) cao hơn FT+RAG (0,136), nhưng kết luận "hybrid superior"	Chọn số liệu thuận lợi
Yang [96]	Chỉ nêu ROUGE-1 19,97, BERTScore F1 82,08 và "reducing noise"	Đây là số của LLaMA2-UMLS (thấp hơn); ChatGPT-3.5 đạt 21,44/83,14; chính văn bản nêu thuật ngữ trích ra không liên quan gây trả lời thiếu	Bảng chọn cấu hình kém hơn, bỏ giới hạn
Alkhalaf [128]	"Accuracy 93,25%→99,25%. Risk factor extraction: 90%"	RAG không cải thiện (giữ 90%)	Bảng không nêu kết quả null, trong khi mục 6.1 nêu rõ
MMed-RAG, [108], Benfenati, DR.KNOWS, CliniqIR	Năm 2025/2025/2025/2024/2025	Danh sách tham khảo: 2024/2024/2024/2025/2024	Năm lệch ở ít nhất 5 mục
3. Các điểm đã nêu ở bản trước, giữ nguyên
Loại	Nội dung
Method	Không có phương pháp tổng quan; không có mục limitations; "systematic" chỉ là cách nói
Citation integrity (dẫn nhập)	[4], [6], [9], [10], [11], [12], [13] có tiêu đề không khớp claim mà chúng hỗ trợ
Citation integrity (5.6)	[118] (Qwen2.5-VL) cho Qwen2.5-7B; [126] (German CheXpert labeler) cho CheXpert; [127] là MedPlan nhưng văn bản mô tả MS-RAG (NQ, TriviaQA); [32] là bài review MIMIC
Không có trong tham khảo	Soman et al. (SPOKE), Lozano et al. (Clinfo.ai), RAG + XGBoost của "Jin et al."
Phân loại không nhất quán	MedRAG-KG: Advanced (3.2.5) vs Modular (4.3.2, 5.6.3); LLM-AMT: Modular (3.3.5) vs Advanced (5.6.2); tiêu đề 5.6.3 viết "Advanced RAG"; LiVersa/[108] trong Table 8
Lĩnh vực	FLARE, SELF-RAG, MS-RAG xếp vào "medical domain" dù đánh giá trên dữ liệu chung
Fig. 4–6	Caption Fig. 5 trùng Fig. 4; Med-PaLM 2 chỉ có một cột và không cao nhất; Fig. 6 trộn baseline khác loại (luật, người, mô hình khác) và sai baseline MIRAGE (53,7 thay vì 60,69)
Số liệu dataset mâu thuẫn	ChatDoctor (100.000/215K/200.000), MIMIC-III (53.423/58K), CLEVA, MedVidQA
Contributions vs thực tế	Mục 1.1 hứa đánh giá chuẩn hóa bằng S.C.O.R.E.; mục 7.2.1 thừa nhận đánh giá không thống nhất; survey không tự đánh giá
Phát biểu về Xiong et al. (#3)	"reducing hallucinations" (không đo); "retrieval is essential for exam questions" (trái Table 6 ở nguồn #3); "17,9%" là mức tăng tương đối
Phát biểu về MedGraphRAG (#2)	"higher accuracy across all benchmarks" so với Med-PaLM 2/Med-Gemini mạnh hơn bài gốc
Khái quát hóa	6.5 "pattern appears across studies" chỉ dựa chủ yếu vào Almanac
Nhầm lẫn khái niệm	"Static knowledge bases" (6.5, 7.1.2) nhưng bằng chứng nêu là về độ phủ và chất lượng corpus
Scope	Mục 5.4–5.5 chủ yếu mô tả LLM/VLM không phải RAG
Data availability	Vừa "publicly available" vừa "upon reasonable request"; không có phân tích dữ liệu riêng
Novelty	Phân loại Naive/Advanced/Modular theo hiểu biết của tôi là của Gao et al. [19] (chưa kiểm chứng)

Giới hạn của chính bản trích xuất này: Table 5 và 6 không có nội dung; một số nhãn trong Fig. 4–6 tôi đọc không chắc chắn (đã đánh dấu [D]).

Q10 — Future work

Explicit (mục 7):

Mục	Nội dung
7.1	RAG đa phương thức vượt ra ngoài văn bản; cập nhật tri thức thời gian thực (đồ thị tri thức theo thời gian, loại bỏ thông tin lỗi thời); RAG liên bang và bảo vệ quyền riêng tư
7.2	Khung đánh giá chuẩn hóa (benchmark gần lâm sàng, đa bệnh nền); phê duyệt quy định và kiểm định lâm sàng (audit trail, khả năng tái lập); mô hình cộng tác người–AI
7.3	Tối ưu tài nguyên tính toán; xây dựng và bảo trì kho tri thức (bán tự động)
7.4	Công bằng và giảm thiên lệch; minh bạch và giải thích
7.5	Bệnh hiếm; y học toàn cầu và đa văn hóa; y học cá nhân hóa

Candidate research direction (của chúng ta, Candidate gap — requires further verification):

Đo độ cập nhật của kho tri thức (không hệ thống nào trong survey đo).
Đánh giá so sánh cùng một LLM có và không có RAG trong thiết kế khớp nhau (xem Fig. 6).
Đánh giá bộ nhớ bệnh nhân dọc thời gian (7.5.3 nêu hướng, chưa có hệ thống).
Abstention hoặc cơ chế an toàn được đo thực nghiệm (4.3.3 chỉ khẳng định).
Kiểm tra Fig. 4–6 (đọc từ ảnh; một vài nhãn nhỏ không chắc chắn)
Hình	Quan sát
Fig. 4 (LLM y khoa)	Caption: "Med-PaLM 2 achieves highest scores on most tasks". Trong hình, Med-PaLM 2 chỉ có một cột (MedQA 86,5%) và không phải cao nhất ở đó (Gazal-R1 87,1%); USMLE cao nhất là Med42-v2 (94,5%). Trục "Score (%)" trộn F1 (NER, RE) với accuracy. Đây là các LLM, không phải RAG
Fig. 5 (đa phương thức)	Caption trùng hệt Fig. 4 ("medical LLMs ... Med-PaLM 2 ..."), không đúng với nội dung (không có Med-PaLM 2; là mô hình VQA/CLIP). Gồm cả MedQA (USMLE) văn bản của Med-Gemini (91,1%) trong biểu đồ "multi-modal". Cột PubMedCLIP ở PMC-15M (56,3%) có thể là top-1 của BiomedCLIP theo văn bản (>56%): [D], chưa xác minh. BiomedGPT trên VQA-RAD là 73,2% trong hình, trong khi văn bản nêu "error rate 3.8%" (hai metric khác nhau, không được giải thích)
Fig. 6 ("RAG vs Non-RAG")	Caption mô tả là "conceptual diagram ... on a common datasets", trong khi đây là biểu đồ cột gồm khoảng 12 cặp từ các nghiên cứu khác nhau (dataset, metric và baseline khác nhau). Những cặp tôi đối chiếu được với văn bản: baseline "Non-RAG" không phải lúc nào cũng là cùng LLM không RAG: phenotyping PH (0,62 → 0,75): baseline là mô hình luật do bác sĩ biên soạn; "Surgical fitness (human eval)" (86,6 → 96,4): baseline là người đánh giá; cặp so với Med-PaLM 2 (≈ +2%): baseline là mô hình khác; AO code (7 → 83): cùng GPT-4 không RAG, hợp lệ. Cặp MIRAGE (GPT-3.5) ghi 53,7 → 71,6 (+17,9), trong khi nguồn #3 cho baseline CoT là 60,69%: 53,7 là kết quả của việc trừ mức tăng tương đối (17,9) khỏi 71,6 như thể là điểm tuyệt đối. Cặp đầu (MedQA, 67,2 → 86,5) trùng với Med-PaLM → Med-PaLM 2 ở mục 5.4 (Med-PaLM 2 không phải RAG): [D] vì nhãn trục không đọc rõ

Hệ quả: Fig. 6 là hình duy nhất của survey trực quan hóa "hiệu quả của RAG", và nó không nên được dùng làm bằng chứng so sánh RAG với không RAG.

Evidence quality assessment
Strong: không có.
Moderate: phân loại ba kiến trúc RAG và mô tả thành phần (nhưng có dấu hiệu là phân loại của nguồn khác); tổng hợp các failure mode lặp lại (độ phủ corpus, hallucination phụ thuộc generator).
Weak / anecdotal: mọi con số hiệu năng (từ nhiều nghiên cứu khác nhau, thiết kế đánh giá nhỏ, đôi khi mô tả không khớp tài liệu); Fig. 4–6; xếp hạng "Clinical Safety: Comprehensive"; mọi claim ở phần dẫn nhập.
Missing: Table 5–8; phương pháp tổng quan; đối chiếu với bài gốc của từng hệ thống.
Relevance to AI Healthcare Failure Map
Category	Đánh giá
5. Evidence / RAG	Vừa–mạnh về tổng hợp failure mode (độ phủ corpus, nhiễu retrieval, hallucination theo generator), định lượng ở mức thứ cấp
1. AI Accuracy	Vừa: có ví dụ RAG không giúp được ([128]) và hallucination 12,8–48,6% với Llama 2 ([106]), cả hai là thứ cấp
7. Trust	Yếu–vừa: khoảng cách chính xác–chấp nhận (Almanac), minh bạch, S.C.O.R.E.; commercialization không được đề cập
2. Memory / Health Records	Yếu: EHR theo tác vụ; module Memory khái niệm
3. Safety, 6. Privacy	Yếu: chủ yếu khẳng định và hướng tương lai (RAG liên bang, PHI-compliant ở LiVersa)
4. Triage	Chỉ có triage hành chính (RAGPR); không có triage cấp cứu
8	Not relevant / insufficient evidence
Ghi chú cho Part B (mức [D], ứng viên, chưa phải finding)
Survey khuếch đại nguồn sơ cấp: hai ví dụ có thể kiểm chứng ngay: (a) "retrieval is essential for exam questions" so với Table 6 của nguồn #3; (b) Fig. 6 sai baseline MIRAGE. Cộng với MedGraphRAG "all benchmarks". Khi lập evidence table, không tính survey #4 và #5 như nguồn độc lập cho kết quả định lượng.
Hallucination vẫn còn dưới RAG và phụ thuộc generator: Ke et al. (qua survey #5) cho 12,8–48,6% với Llama 2, 0–2,9% với GPT-4. Ke et al. cũng là lead primary đã xuất hiện ở nguồn #4 ([9]). Đây là ứng viên finding mạnh nhất cho Principle 5 (RAG không tự động là giải pháp), cần đọc bài gốc.
RAG không giúp hoặc giúp ít ở tác vụ suy luận/trích xuất ngầm và câu hỏi thi: nguồn #3 (sơ cấp: giảm với GPT-4), nguồn #5 (thứ cấp: Alkhalaf null; Sohn +0,9% MMLU-Med, +3,8% MedMCQA). Cùng chiều với "RAG làm hại khi model vốn đã đúng" ở nguồn #3 và #4.
Tri thức lỗi thời / độ cập nhật không được đo trong cả năm nguồn (#2 future work, #4 mục 9.4, #5 mục 7.1.2). Survey #5 nhầm lẫn "độ phủ corpus" với "lỗi thời", nên khi viết, cần phân biệt hai khái niệm.
Retrieval recall: nay cần nuance. Có hệ thống đo (GastroBot, Duan: context recall; CliniqIR: MRR), nhưng ít, bằng LLM-judge, trên hệ thống nhỏ, và mục 5.3 của survey không định nghĩa. Vẫn nhất quán với việc phần lớn paper y khoa không đo retrieval, nhưng nên nêu "ít được đo và kém chuẩn hóa", không phải "không bao giờ".
Safety được dùng như điểm đánh giá, không phải cơ chế: S.C.O.R.E., "safety 2,87/3", an toàn đối kháng; không có triage cấp cứu hay escalation ở bất kỳ nguồn nào trong năm nguồn.
Bằng chứng nền tảng của RAG y tế trong survey #5 chủ yếu là các nghiên cứu nhỏ, ca tổng hợp, API GPT, corpus guideline nhỏ. Điều này giúp cân nhắc mức tổng quát hóa khi chọn đề tài.
Khẳng định "sẵn sàng triển khai lâm sàng" của Modular RAG không có bằng chứng (Table 8): ba ví dụ, hai ví dụ phân loại sai, một ví dụ mô tả sai cơ chế. Cùng chiều với nhận xét ở nguồn #3 và #4 rằng an toàn thường chỉ ở mức khẳng định.
Bảng tóm tắt thứ cấp có xu hướng bỏ kết quả âm hoặc giới hạn (Alkhalaf, Yang, LiVersa). Đây là lý do cụ thể để truy ngược về bài gốc trước khi dùng làm evidence.
Kết quả "100%" trên mẫu rất nhỏ (LiVersa 10 vignette) và cohort tổng hợp (accGPT, TrialGPT) xuất hiện lặp lại; khi nêu số liệu, cần ghi cỡ mẫu và khoảng tin cậy.
Bằng chứng về RAG cho EHR là theo tác vụ đơn lẻ trên dữ liệu thường là riêng/nội bộ, không có bộ nhớ dọc thời gian hay đánh giá xuyên các lần khám (củng cố ghi chú ở nguồn #4).
### Paper 5: Retrieval augmented generation for 10 large language models and its generalizability in assessing medical fitness
A1. Source Identification
Bibliographic information
Trường	Thông tin
Title	Retrieval augmented generation for 10 large language models and its generalizability in assessing medical fitness
Authors	Yu He Ke, Liyuan Jin, Kabilan Elangovan, Hairil Rizal Abdullah, Nan Liu, Alex Tiong Heng Sia, Chai Rick Soh, Joshua Yi Min Tung, Jasmine Chiat Ling Ong, Chang-Fu Kuo, Shao-Chun Wu, Vesela P. Kovacheva, Daniel Shu Wei Ting
Year	2025
Venue	npj Digital Medicine
Volume / Article	8, 187
DOI	10.1038/s41746-025-01519-z
Publication type	Original research / empirical study
Source type	Primary academic research
Evidence strength	Moderate–Strong for the specific preoperative task; limited for general medical RAG
Main purpose	Develop and evaluate LLM-RAG for preoperative assessment and instructions
Medical domain	Perioperative / preoperative medicine
Target users	Healthcare professionals, especially clinicians involved in preoperative assessment
Relevance to Medical RAG	Very high
Relevance to Failure Analysis	Very high


Tên bài, DOI và nhóm tác giả được ghi trực tiếp ở đầu paper; bài xuất bản trong npj Digital Medicine năm 2025.     s41746-025-01519-z     s41746-025-01519-z
Vì sao tôi đánh giá Evidence Strength là Moderate–Strong, không phải “Strong” tuyệt đối?
Study có khá nhiều điểm mạnh: 10 LLM, 14 clinical scenarios, 3,234 câu trả lời từ LLM, 448 human-generated responses, đánh giá với guideline địa phương và quốc tế, có clinician evaluation và expert panel.     s41746-025-01519-z
Nhưng external validity bị giới hạn vì:
- chỉ tập trung vào preoperative medicine;
- chỉ dùng 14 simulated/de-identified clinical scenarios;
- các quyết định được chuẩn hóa tương đối rõ;
- tác giả thừa nhận kết quả từ simulated scenarios có thể không tổng quát sang clinical practice thực tế;
- protocol của các bệnh viện có thể khác nhau.     s41746-025-01519-z
Vì vậy, đây là bằng chứng khá tốt cho failure/success trong một setting cụ thể, nhưng chưa đủ để kết luận về Medical RAG nói chung.
A2. Paper / Literature Identification
Paper có 33 references. Không nên coi tất cả là nguồn mở rộng ngang nhau. Với mục tiêu của chúng ta, tôi chia như sau.
Priority A — nên điều tra tiếp
A1. Zakka et al. — Almanac
Reference trong Ke et al.:
Zakka, C. et al. Almanac: retrieval-augmented language models for clinical medicine. Research Square, 2023.     s41746-025-01519-z

Chủ đề: clinical medicine + retrieval-augmented language models.
Tại sao quan trọng: Đây là reference trực tiếp nhất cho nền tảng clinical RAG, và cũng chính là paper bạn đã xác định trong thứ tự mở rộng.
Loại: Primary research / preprint trong phiên bản được Ke et al. trích dẫn.
Recommendation: Must investigate.
Lưu ý quan trọng: reference trong Ke et al. là bản Research Square 2023. Nếu bạn đang sử dụng bản NEJM AI 2024, cần giữ hai bibliographic versions riêng biệt cho đến khi chúng ta verify publication lineage, thay vì tự động coi chúng là cùng một citation.

A2. Tan et al. — S.C.O.R.E.
Reference:
Tan, T. F. et al. A proposed S.c.o.r.e. evaluation framework for large language models: Safety, Consensus, Objectivity, Reproducibility and Explainability. arXiv, 2024.     s41746-025-01519-z

Chủ đề: evaluation of LLMs theo 5 chiều:
Safety — Consensus — Objectivity — Reproducibility — Explainability.     s41746-025-01519-z
Tại sao quan trọng: Paper này không phải Medical RAG core paper theo nghĩa retrieval architecture, nhưng nó rất quan trọng đối với Failure Map → Evaluation layer.
Recommendation: Highly relevant for evaluation methodology.
A3. Yang et al. — UMLS Knowledge Integration
Yang, R. et al. Integrating UMLS Knowledge into Large Language Models for Medical Question Answering. arXiv, 2023.     s41746-025-01519-z

Chủ đề: medical QA + integration of structured medical knowledge.
Tại sao đáng theo dõi: Không phải RAG paper thuần túy, nhưng nó giúp mở rộng một câu hỏi quan trọng:
Failure có thực sự cần giải quyết bằng retrieval không, hay có những failure liên quan đến domain knowledge representation?

Đây là một nhánh quan trọng để tránh bias “RAG is always the answer”.
Recommendation: Investigate, nhưng sau Almanac/S.C.O.R.E.
Priority B — relevant nhưng không phải core RAG expansion
B1. Thirunavukarasu et al. — LLMs in medicine
Large language models in medicine, Nature Medicine, 2023.     s41746-025-01519-z
Giá trị chính: background về khả năng và limitation của LLM trong medicine.
Recommendation: useful for contextualizing failure, nhưng không cần đưa vào nhóm 8–15 core Medical RAG papers.
B2. Lim et al. — ASA classification
Nghiên cứu sử dụng ChatGPT cho ASA physical-status classification.     s41746-025-01519-z
Giá trị: LLM clinical decision performance trước RAG.
Đặc biệt hữu ích nếu sau này chúng ta muốn so sánh:
LLM-only → RAG → advanced RAG
thay vì chỉ đánh giá RAG độc lập.
B3. Lee & Choi — opportunities and limitations
Review về ChatGPT trong anesthesiology, tập trung vào opportunities/limitations.     s41746-025-01519-z
Giá trị: clinical limitation/background.
Priority C — peripheral for the current Failure Map
Các reference về surgery cancellation, nurse-led assessment, perioperative clinical management... có giá trị để hiểu clinical context, nhưng chưa trực tiếp trả lời failure của Medical RAG.
Ví dụ paper về surgery cancellation được Ke et al. dùng để giải thích vì sao preoperative assessment quan trọng, chứ không phải để chứng minh RAG failure.     s41746-025-01519-z
A3. Q1–Q10 Extraction
Research task
Clinical domain
Preoperative medicine / perioperative anesthesia.
Input
Các clinical scenarios mô phỏng format preoperative clinic notes, gồm age, sex, ASA, BMI, medical history, investigations, medications, planned procedure. Có 14 de-identified scenarios, trải từ ASA 1–3.     s41746-025-01519-z
Output
LLM phải đưa ra các quyết định/instructions gồm:
1. Doctor hay Nurse
2. Fasting instructions
3. Carbohydrate loading
4. Medication instructions
5. Instructions cho healthcare team
6. Preoperative optimization
7. Có cần delay surgery không
8. Protocol cụ thể cần áp dụng.     s41746-025-01519-z
Q1 — What failure/problem is Medical RAG trying to solve?
Source fact
Paper xuất phát từ vấn đề LLM thông thường:
- thiếu domain-specific expertise;
- trong các clinical task phức tạp, câu trả lời dựa trên pretrained knowledge;
- không được grounding theo institutional practice guidelines;
- hallucination là một safety/ethical concern.     s41746-025-01519-z
RAG được dùng để đưa clinical guidelines vào quá trình sinh câu trả lời, nhằm tăng:
- accuracy,
- contextual relevance,
- consistency,
- safety,
- khả năng sử dụng knowledge đã cập nhật.     s41746-025-01519-z
Failure being addressed
Primary failure:
LLM clinical responses thiếu domain-specific, guideline-grounded knowledge.

Secondary failures being targeted:
hallucination, inconsistency giữa clinicians, và thiếu alignment với local/international preoperative guidelines.

Research interpretation
Ở đây RAG đang được dùng trước hết như một grounding mechanism, chứ không đơn thuần là “cách làm LLM thông minh hơn”.
Đây là distinction quan trọng cho Failure Map.
Q2 — What failures does RAG still NOT solve?
Đây là phần giá trị nhất của paper đối với nghiên cứu của chúng ta.
1. Retrieval failure vẫn tồn tại
Tác giả nói rõ performance phụ thuộc vào quality of retrieved information. Họ cũng chỉ ra document extraction có thể:
- retrieve extraneous content;
- gặp khó khăn với diagrams;
- gặp khó khăn với tables;
- và optimal chunk size trong healthcare vẫn là vấn đề mở.     s41746-025-01519-z
Status: Direct evidence.
2. RAG không đảm bảo hallucination = 0
Đây là finding cực kỳ quan trọng.
Đa số model có hallucination thấp, khoảng 0–2.9%, nhưng riêng Llama2-7B cho kết quả đáng chú ý:
- native: 12.8%
- RAG variants: 48.6% và 32.9%.     s41746-025-01519-z
Điều này cho phép chúng ta ghi nhận rất mạnh:
RAG augmentation không mặc định làm hallucination giảm; trong ít nhất một configuration, RAG-associated hallucination rate còn cao hơn non-RAG.

Đây là source fact, không phải suy luận.
Nhưng root cause là gì?
Chưa được xác định trong paper.
Paper không chứng minh rằng 48.6% là do:
- poor retrieval,
- bad chunking,
- context overload,
- model capability,
- prompt,
- hay interaction giữa retriever và generator.
→ Vì vậy không được ghi một causal root cause cụ thể.
3. Retrieved evidence có thể chưa đủ
Tác giả thừa nhận performance phụ thuộc vào retrieval quality và đề xuất adaptive retrieval để ưu tiên tài liệu phù hợp hơn.     s41746-025-01519-z
Status: Directly supported.
4. Local guideline quality có thể trở thành bottleneck
Một finding rất thú vị:
International guidelines thường cho output tốt hơn local guidelines.

Tác giả giải thích rằng international guidelines thường comprehensive hơn và có textual explanation đi cùng diagrams/tables, trong khi local guidelines có thể ít chi tiết hơn, khiến RAG khó tận dụng.     s41746-025-01519-z
Đây không chỉ là “retriever tốt/xấu”.
Nó gợi ý:
Knowledge base quality → retrieval quality → final clinical performance
có thể là một chain failure.
5. Clinical ambiguity vẫn không được giải quyết hoàn toàn
Paper thừa nhận real-world decisions có thể phức tạp và ethical ambiguity cao hơn simulated scenarios, đặc biệt trong những tình huống như cancer surgery postponement.     s41746-025-01519-z
→ RAG không giải quyết được normative/clinical ambiguity chỉ bằng cách thêm documents.
6. Bias vẫn tồn tại
Tác giả cảnh báo bias có thể xuất hiện từ:
- training data,
- institutional protocols,
- race,
- gender,
- socioeconomic status,
- regional healthcare practices.     s41746-025-01519-z
Important: đây là risk explicitly discussed, không phải study thực nghiệm chứng minh bias rate.
7. Generalization vẫn chưa được chứng minh
Study dùng simulated scenarios và một setting perioperative cụ thể; tác giả thừa nhận điều này hạn chế generalizability sang real-world clinical settings.     s41746-025-01519-z
8. Standardized healthcare evaluation vẫn thiếu
Tác giả đề xuất tương lai cần standardized healthcare-specific metrics, đặc biệt cho:
- factual accuracy
- contextual relevance.     s41746-025-01519-z
Đây là một failure ở evaluation/system level, không phải retrieval riêng lẻ.
Q3 — Dataset / Knowledge Sources
Loại	Nguồn	Vai trò
Evaluation dataset	14 de-identified clinical scenarios	Benchmark cho preoperative task
RAG corpus	35 local perioperative guidelines	Institutional/local knowledge base
RAG corpus	23 international guidelines + UpToDate articles	International evidence base
Human reference	448 human-generated responses	So sánh với model outputs
Expert ground truth	2 board-certified perioperative anesthesiologists	Xây dựng correct answers
EHR	Not reported	Không có bằng chứng cho longitudinal EHR usage
Knowledge graph	Not reported	Không sử dụng
Fine-tuning dataset	Not used for this experiment	Fine-tuning không được thực hiện


Nguồn guideline và clinical scenarios được mô tả trong Methods.     s41746-025-01519-z
Một điểm quan trọng:
Đây không phải nghiên cứu patient-level longitudinal memory.
Các scenario được de-identify và chỉnh sửa age, sex, PMH, surgery dates/procedures.     s41746-025-01519-z
Q4 — Evaluation metrics
Paper đánh giá rộng hơn một accuracy score thông thường.
Generation / clinical performance
- Accuracy
- False Negative Rate
- Fisher's exact test
- Odds Ratio
- p-value.     s41746-025-01519-z
Consistency
Inter-rater reliability / percentage agreement.
GPT-4 International đạt:
- IRR = 0.93 cho medical fitness
- IRR = 0.96 cho healthcare instructions
- IRR = 0.92 cho optimization requirements.     s41746-025-01519-z
Safety / qualitative evaluation
S.C.O.R.E.:
- Safety
- Consensus
- Objectivity
- Reproducibility
- Explainability.     s41746-025-01519-z
GPT-4 RAG đạt:
- Reproducibility: 4.86/5
- Safety: 4.93/5.     s41746-025-01519-z
Other
- Hallucination rate
- Distinct 1/2/3-gram scores.     s41746-025-01519-z
Critical methodological observation
Không thấy retrieval-specific evaluation như Recall@k, Precision@k, MRR, NDCG, Context Recall, Context Precision.
Do đó study chủ yếu đánh giá end-to-end clinical output, chứ không cô lập được retrieval quality một cách định lượng.
Đây là một khoảng trống phương pháp rất đáng lưu ý cho các paper tiếp theo.
Q5 — Retrieval-side or Generation-side?
Classification
Both / system-level — nhưng không thể xác định causal contribution chính xác.
Retrieval-side evidence
Paper trực tiếp thừa nhận:
- extraction có thể lấy extraneous content;
- diagrams/tables gây khó khăn;
- chunk size chưa tối ưu;
- performance phụ thuộc retrieval quality;
- adaptive retrieval được đề xuất.     s41746-025-01519-z     s41746-025-01519-z
Generation-side evidence
Hallucinations vẫn xuất hiện; đặc biệt Llama2-7B RAG có hallucination cao hơn native model.     s41746-025-01519-z
What we cannot conclude
Không thể nói:
“Llama2 hallucinate vì retriever kém.”

Paper không chứng minh causal chain đó.
Đây chính là kiểu distinction chúng ta cần giữ lại trong Failure Map.
Q6 — Does the system contain patient memory?
Longitudinal patient memory
No evidence / Not reported.
Conversation history
Not reported as a persistent memory mechanism.
Patient-specific input
Có clinical scenario chứa patient history, medications, investigations, etc., nhưng đây là input case, không phải longitudinal memory architecture.     s41746-025-01519-z
Classification
Patient memory: Not reported.

Đây là điểm cần ghi rất rõ để sau này không nhầm:
patient context ≠ patient memory.
Q7 — Safety / Triage mechanisms
Có
Study có một dạng task-specific triage:
- Doctor vs Nurse
- need to delay surgery vs proceed
- preoperative optimization.     s41746-025-01519-z
Quan trọng hơn, tác giả đề xuất deployment theo:
human-in-the-loop

và nhấn mạnh system chỉ là support tool, không phải autonomous decision-maker, mọi recommendation phải được clinician review.     s41746-025-01519-z
Nhưng không có bằng chứng về
- generic emergency detection;
- explicit abstention model;
- refusal mechanism;
- dedicated safety classifier;
- autonomous escalation logic.
→ Task-specific triage + human oversight, nhưng chưa phải general clinical safety architecture.
Q8 — Human / Clinician Evaluation
Human evaluators
- 4 junior doctors, 1–7 years anesthesia experience
- 4 attending anesthesiologists
- participants blinded to study objectives.     s41746-025-01519-z
Ground truth
2 board-certified perioperative anesthesiologists xây dựng correct answers và giải quyết disagreement bằng discussion.     s41746-025-01519-z
S.C.O.R.E. evaluation
Hai attending anesthesiologists đánh giá GPT-4 International qua 4 repeated generations cho mỗi scenario.     s41746-025-01519-z
Quality of human evaluation
Khá mạnh so với nhiều LLM papers vì có:
clinician evaluation + expert panel + blinding + repeated evaluation.
Tuy nhiên, sample vẫn nhỏ và simulated.
Q9 — Limitations
Technical
- retrieval gây latency/resource overhead;
- scalability khó khi retrieval pool tăng;
- optimal chunk size chưa được giải quyết.     s41746-025-01519-z
Retrieval
- document parsing có thể lấy extraneous information;
- diagrams/tables khó xử lý;
- retrieval quality ảnh hưởng performance;
- local guideline quality có thể yếu hơn international guideline.     s41746-025-01519-z     s41746-025-01519-z
Clinical / Generalization
- simulated scenarios;
- chưa đại diện đầy đủ real-world complexity;
- hospital-specific protocols khác nhau.     s41746-025-01519-z
Generation / model limitation
- hallucination vẫn tồn tại;
- một số RAG configurations có hallucination cao hơn native model.     s41746-025-01519-z
Safety
- cần continued expert oversight;
- không nên dùng autonomous.     s41746-025-01519-z
Bias / Ethics
- possible bias từ source materials và training data;
- ethical ambiguity trong complex clinical decisions.     s41746-025-01519-z
Evaluation
- thiếu standardized healthcare-specific evaluation metrics.     s41746-025-01519-z
Deployment
- computational overhead;
- latency;
- scalability;
- real-world deployment chưa được chứng minh đầy đủ.     s41746-025-01519-z
Q10 — What future work remains?
Explicit future work của tác giả
Paper nêu:
1. Adaptive retrieval mechanisms
2. Dynamic prioritization theo query specificity
3. Real-time filtering
4. Domain-specific retrieval cho specialty như cardiology/oncology
5. Tối ưu inference parameters
6. Standardized healthcare-specific evaluation metrics.     s41746-025-01519-z
Ngoài ra, tác giả cũng đề cập:
- cải thiện completeness của local guidelines;
- chuyển diagrams thành text;
- nghiên cứu fine-tuning để so sánh với RAG.     s41746-025-01519-z
Explicit future work ≠ research gap
Đến thời điểm này tôi chưa gọi bất kỳ điểm nào là “research gap đã được chứng minh”.
Nhưng có một số candidate directions rất đáng theo dõi khi kết hợp với các paper kế tiếp:
Candidate direction 1
Retrieval quality → clinical failure
Paper cho thấy retrieval quality quan trọng, nhưng không trực tiếp đo retrieval metrics.     s41746-025-01519-z
→ Candidate gap — requires cross-paper verification.
Candidate direction 2
RAG can worsen hallucination in some configurations
Finding này đặc biệt mạnh vì có comparison với native counterpart, nhưng paper chưa xác định nguyên nhân.
→ Candidate mechanistic gap — requires investigation of other primary papers.
Candidate direction 3
Clinical evaluation is stronger than generic NLP metrics, but standardized medical-RAG evaluation remains unresolved.
Paper vừa dùng clinician/S.C.O.R.E., vừa kêu gọi standardized metrics.     s41746-025-01519-z     s41746-025-01519-z
→ Candidate evaluation gap.
Evidence Quality Assessment
Strong evidence
E1. RAG can substantially improve performance over non-RAG in this specific preoperative task.
GPT-4 international đạt accuracy 96.4%, so với GPT-4 non-RAG 92.9% và GPT-4 local 92.9%.     s41746-025-01519-z
E2. RAG does not guarantee lower hallucination.
Llama2-7B RAG: 48.6% / 32.9% vs native 12.8%.     s41746-025-01519-z
E3. Retrieval quality and document representation matter.
Paper trực tiếp mô tả extraneous content, diagrams/tables và chunk-size problem.     s41746-025-01519-z
E4. Human oversight remains part of the proposed safe deployment model.
    s41746-025-01519-z
Moderate evidence
- RAG có khả năng tăng consistency.
- International knowledge base có lợi thế so với local guideline corpus.
- S.C.O.R.E. có thể cung cấp evaluation đa chiều.
- RAG có tiềm năng hỗ trợ triage preoperative.     s41746-025-01519-z
Weak / not established
Không được kết luận từ paper rằng:
- RAG causes Llama2 hallucination;
- advanced RAG luôn tốt hơn vanilla RAG;
- hệ thống đã an toàn trong clinical deployment thực tế;
- hệ thống generalize tốt sang các specialty khác;
- hệ thống có thể thay thế clinician.
Relevance to AI Healthcare Failure Map
Failure dimension	Evidence từ paper	Mức liên quan
1. AI Accuracy	Clinical accuracy, false negatives	High
2. Memory / Health Records	Longitudinal memory không được báo cáo	Low / Gap signal
3. Conversation Safety	Safety scoring + hallucination	High
4. Triage / Emergency	Doctor/Nurse + delay/proceed	High, task-specific
5. Evidence / RAG	Retrieval quality, chunking, document parsing	Very High
6. Privacy / Data	Local vs cloud privacy discussion, nhưng không có privacy experiment	Low–Medium
7. Trust / Commercialization	Không phải trọng tâm	Low
8. Long-term Health Relationship	Không nghiên cứu	Not relevant / insufficient evidence
### Paper 6: Liu, McCoy, Wright 2025 — JAMIA
A1. Source Identification
Trường	Thông tin
Title	Improving large language model applications in biomedicine with retrieval-augmented generation: a systematic review, meta-analysis, and clinical development guidelines
Authors	Siru Liu, Allison B. McCoy, Adam Wright
Year	2025
Venue	Journal of the American Medical Informatics Association (JAMIA)
Volume / pages	32(4), 605–615
DOI	10.1093/jamia/ocaf008
Publication type	Systematic review + meta-analysis + methodological/development guideline
Source type	Secondary academic evidence
Evidence strength	High for synthesis of the included 2024 literature; not equivalent to primary experimental evidence
Main purpose	Tổng hợp hiệu quả RAG trong biomedical LLM applications và xây dựng GUIDE-RAG
Relevance to Medical RAG	Very high
Relevance to Failure Analysis	Very high


Thông tin bibliographic được ghi trực tiếp trong paper.     Liu, McCoy, Wright 2025 — JAMIA
Điểm cần đặc biệt lưu ý về evidence strength
Paper này có một ưu điểm rất lớn: systematic review + meta-analysis, sử dụng PRISMA 2020 và Cochrane Handbook.     Liu, McCoy, Wright 2025 — JAMIA
Nhưng meta-analysis của họ cũng có dấu hiệu small-study effects / potential publication bias: Egger's regression có P = .001.     Liu, McCoy, Wright 2025 — JAMIA
Vì vậy, tôi sẽ không dùng con số OR 1.35 như một “proof rằng RAG luôn tốt”.
A2. Literature map từ Liu et al.
Đây mới là phần giá trị nhất cho giai đoạn hiện tại của chúng ta.
Review design
Các tác giả:
- tìm trong PubMed, Embase, PsycINFO;
- giới hạn publication 2023–2024;
- loại preprints;
- chỉ chọn nghiên cứu có baseline LLM vs RAG comparison;
- phải trả lời một biomedical question.     Liu, McCoy, Wright 2025 — JAMIA
Kết quả:
335 records → 251 sau deduplication → 51 full-text → 20 studies included.     Liu, McCoy, Wright 2025 — JAMIA
Điều này rất hữu ích vì nó cho chúng ta một curated primary-paper pool thay vì tìm 8–15 paper hoàn toàn từ đầu.
Primary papers mà Liu et al. chỉ ra
Trong main reference list, tôi xác định được các nghiên cứu trực tiếp nằm trong nhóm RAG clinical studies:
Priority	Primary study	Chủ đề chính	Giá trị cho Failure Map
A	Zakka et al. 2024 — Almanac	Clinical medicine RAG	Rất cao
A	Yazaki et al. 2024	Emergency patient triage	Rất cao
A	Malik et al. 2024	Anticoagulation management	Rất cao
A	Kresevic et al. 2024	Hepatology guideline interpretation	Rất cao
A	Alkhalaf et al. 2024	EHR summarization/extraction	Rất cao
A	Jeong et al. 2024	Retrieval + self-reflection + medical reasoning	Rất cao
A	Zelin et al. 2024	Rare-disease diagnosis	Rất cao
A	Rau et al. 2024	Differential diagnosis in GI radiology	Cao
A	Wang et al. 2024	Diabetes education	Cao
A	Long et al. 2024	ENT/head & neck knowledge retrieval	Cao
A	Murugan et al. 2024	Personalized pharmacogenomics	Cao
B	Glicksberg et al. 2024	EHR-based hospital admission prediction	Cao
B	Hewitt et al. 2024	Neuropathology diagnostic support	Cao
B	Kreimeyer et al. 2024	Precision oncology	Cao
B	Tarabanis et al. 2024	Internal medicine board questions	Trung bình–cao
B	Chen et al. 2024 — DocOA	Domain-specific medicine	Cao
B	Du et al. 2024	Cognitive decline from clinical notes	Cao
B	Soman et al. 2023	Biomedical knowledge graph	Cao


Các bibliographic details này đều xuất hiện trong reference list của Liu et al.     Liu, McCoy, Wright 2025 — JAMIA
Một anomaly cần ghi lại
Paper nói 20 studies included, nhưng main reference list cho phép chúng ta xác định rõ khoảng 18 relevant study references. Tôi không tự suy đoán hai study còn lại khi chưa thấy Supplementary File S2/S3.
→ Trạng thái:
20 included = source fact.
18 studies identifiable from main reference list = my document-level observation.
Remaining 2 = not yet verified.
Đây chính là chỗ không được “lấp chỗ trống” bằng suy đoán.
A3 — Q1–Q10 Extraction
Research task
Khác Ke et al., đây không phải một clinical system duy nhất.
Đối tượng nghiên cứu là toàn bộ biomedical RAG literature 2023–2024 thỏa inclusion criteria.
Các clinical tasks trong 20 study gồm:
- clinical decision-making: 13/20 = 65%
- medical question-answering: 7/20 = 35%.     Liu, McCoy, Wright 2025 — JAMIA
Medical specialties trải rộng từ:
internal medicine, general medicine, oncology, emergency medicine, gastroenterology, otolaryngology, hepatology, rare disease, orthopedics, neurology và ophthalmology.     Liu, McCoy, Wright 2025 — JAMIA
Q1 — What failure/problem is RAG trying to solve?
Source facts
Liu et al. xác định các limitation của LLM baseline gồm:
- knowledge cutoff;
- training data rộng nhưng thiếu specificity cho biomedical tasks;
- nguồn training data không phải lúc nào cũng reliable/trustworthy;
- hallucination;
- thiếu khả năng cập nhật knowledge theo thời gian.     Liu, McCoy, Wright 2025 — JAMIA
RAG được dùng để đưa external knowledge vào query/context nhằm:
- cập nhật information;
- tăng accuracy;
- tăng reliability;
- tích hợp domain-specific knowledge.
Core failure
Knowledge deficiency + lack of current/domain-specific evidence in baseline LLMs.

Secondary failures being targeted
- hallucination;
- incorrect treatment recommendation;
- poor clinical QA;
- insufficient grounding;
- domain-specific reasoning.
Q2 — What failures does RAG still NOT solve?
Đây là phần cực kỳ quan trọng.
1. Retrieval itself can fail
Review cho thấy retrieval strategies rất đa dạng:
- chunking;
- query rewriting;
- embeddings;
- ranking;
- MMR;
- knowledge graphs;
- retrieval cutoffs;
- multi-vector retrievers.     Liu, McCoy, Wright 2025 — JAMIA
Điều này gián tiếp cho thấy retrieval is not a solved component.
Quan trọng hơn, tác giả chỉ ra fixed-length chunking có thể fragment related information và làm giảm retrieval accuracy.     Liu, McCoy, Wright 2025 — JAMIA
2. RAG can still generate incorrect answers even with external knowledge
Liu et al. explicitly viết rằng retrieval và generation vẫn loosely connected, và output vẫn có thể bị ảnh hưởng bởi:
- LLM's inherent knowledge boundaries;
- noise trong retrieved text.
Do đó output vẫn có thể incorrect.     Liu, McCoy, Wright 2025 — JAMIA
Đây là một evidence rất mạnh cho Failure Map.
3. The retrieved knowledge itself may be inadequate
Paper đưa ra một ví dụ:
Một study sử dụng chỉ Harrison's Principles of Internal Medicine làm knowledge source và improvement chỉ ở mức OR 1.14. Tác giả cho rằng bổ sung clinical guidelines có thể cải thiện performance.     Liu, McCoy, Wright 2025 — JAMIA
→ Knowledge-base selection itself có thể trở thành bottleneck.
4. Multimodal/document parsing failure
Clinical knowledge không chỉ nằm trong plain text.
Có:
- flowcharts;
- graphs;
- tables;
- figures.
Chỉ dựa vào LLM để extract table data có thể rất yếu; paper báo cáo một preliminary study trong đó GPT-4 Turbo chỉ đạt 16% accuracy khi extracting table data.     Liu, McCoy, Wright 2025 — JAMIA
Đây là một failure dimension rất đáng chú ý:
Document ingestion failure → retrieval failure → downstream clinical failure.
5. Retrieval amount may itself be wrong
Paper lưu ý một query không phải lúc nào cũng cần external knowledge; retrieval depth nên phụ thuộc vào clinical question.     Liu, McCoy, Wright 2025 — JAMIA
Điều này mở ra failure:
under-retrieval ↔ over-retrieval/noise.
Không thể cứ “retrieve càng nhiều càng tốt”.
6. Evaluation failure
Đây có lẽ là một trong những finding mạnh nhất của paper.
Trong 20 studies:
- 9 dùng human evaluation;
- 8 dùng automated evaluation;
- 3 dùng kết hợp cả hai.
Nhưng chỉ 4 studies (25%) có explicit retrieval metrics.     Liu, McCoy, Wright 2025 — JAMIA
Nghĩa là phần lớn literature đánh giá:
final answer

thay vì:
whether the retrieval stage actually worked.

Đây là điểm cực kỳ quan trọng cho hướng nghiên cứu Failure Analysis của chúng ta.
7. Long-term monitoring failure
Review nhấn mạnh rằng các nghiên cứu hiện tại chủ yếu dùng fixed external datasets, và cần cơ chế:
- monitoring dài hạn;
- knowledge update;
- xác định khi nào update;
- update như thế nào.     Liu, McCoy, Wright 2025 — JAMIA
RAG có khả năng cập nhật knowledge về mặt kiến trúc, nhưng updateability ≠ automatically safe knowledge maintenance.
Q3 — Datasets / Knowledge sources
Review cho thấy knowledge sources rất đa dạng.
Clinical guidelines
6 studies dùng clinical guidelines.     Liu, McCoy, Wright 2025 — JAMIA
Academic literature
5 studies dùng PubMed abstracts/full texts hoặc document collections.     Liu, McCoy, Wright 2025 — JAMIA
Specialized knowledge bases
Có:
- ChatENT
- OncoKB
- RareDis Corpus.     Liu, McCoy, Wright 2025 — JAMIA
Knowledge graph
Một study dùng SPOKE.     Liu, McCoy, Wright 2025 — JAMIA
Textbooks
2 studies dùng textbook, ví dụ Harrison's.     Liu, McCoy, Wright 2025 — JAMIA
EHR
3 studies dùng EHR data.     Liu, McCoy, Wright 2025 — JAMIA
Online browsing
2 studies dùng real-time browsing tới PubMed / UpToDate.     Liu, McCoy, Wright 2025 — JAMIA
Dataset size
Nguồn retrieval dao động rất rộng:
từ một bộ nhỏ chỉ khoảng 6 osteoarthritis guidelines đến một EHR dataset từ 7 hospitals.     Liu, McCoy, Wright 2025 — JAMIA

Đây là evidence cho thấy literature chưa có một knowledge-corpus standard.
Q4 — Evaluation metrics
Review phân loại metrics thành:
Accuracy
- Likert ratings
- match rates
- correct treatment percentages
- AUC
- AUPRC
- F1.
NLP similarity
- ROUGE-1/2/L
- BLEU
- METEOR
- BERTScore.
Completeness
Đánh giá mức độ output chứa đầy đủ thông tin cần thiết.
User perception
- helpfulness;
- understandability;
- intent satisfaction.
Safety
- potential harm;
- adversarial safety;
- risk management;
- scientific validity;
- evidence adherence.
Hallucination
Phần lớn được manual review; định nghĩa hallucination không thống nhất.
Một số study phân biệt:
- nonfactual hallucination;
- input-conflicting;
- contextual-conflicting.
Citation
Kiểm tra references có thực sự trỏ tới valid publications/guidelines/research hay không.
Bias / language
- clarity;
- neutrality;
- empathy.     Liu, McCoy, Wright 2025 — JAMIA
Q5 — Retrieval or Generation?
Classification
Both — and importantly, the literature often fails to isolate the two.
Review cho thấy:
Pre-retrieval → Retrieval → Post-retrieval
có rất nhiều kỹ thuật khác nhau.     Liu, McCoy, Wright 2025 — JAMIA
Nhưng chỉ 4/20 studies có explicit retrieval evaluation.     Liu, McCoy, Wright 2025 — JAMIA
Do đó một final-answer failure thường không cho phép ta biết:
retriever fail hay generator fail?

Đây là một methodological failure in evaluation, chứ không chỉ là system failure.
Q6 — Does the literature contain patient memory?
Ở cấp review:
EHR data
Có 3 studies sử dụng EHR data.     Liu, McCoy, Wright 2025 — JAMIA
Longitudinal/persistent patient memory
Chưa có bằng chứng đủ để nói majority of RAG systems have patient memory.
Ngược lại, paper còn xem:
- short-term memory;
- long-term memory;
- integration với EHR;
là các hướng tương lai của agentic clinical systems.     Liu, McCoy, Wright 2025 — JAMIA
Classification
Patient-specific data exists in some studies, but persistent longitudinal memory is not established as a standard RAG capability.

Đây là distinction chúng ta phải giữ.
Q7 — Safety / Triage
Review cho thấy RAG đã được áp dụng cho:
- emergency triage;
- personalized treatment;
- disease management.     Liu, McCoy, Wright 2025 — JAMIA
Một số systems có:
- safety rules;
- adversarial prompting;
- risk management;
- evidence checking.
Có study áp dụng 24 predefined rules để kiểm tra ethical/factual accuracy.     Liu, McCoy, Wright 2025 — JAMIA
Nhưng:
Đây không phải một unified safety architecture.

Các mechanism khác nhau giữa studies.
Do đó:
Safety mechanism = heterogeneous, not standardized.
Q8 — Human / Clinician evaluation
Trong 20 studies:
9 human-only, 8 automated, 3 mixed.     Liu, McCoy, Wright 2025 — JAMIA
Human evaluation:
- trung bình 4 evaluators;
- range 1–10;
- phần lớn là physicians thuộc relevant specialties.     Liu, McCoy, Wright 2025 — JAMIA
Một điểm quan trọng:
Human evaluation cho pooled OR:
1.65
trong khi automated evaluation:
1.20
và sự khác biệt có statistical significance P < .01.     Liu, McCoy, Wright 2025 — JAMIA
Critical interpretation
Điều này không chứng minh human evaluation “đúng hơn”.
Nó chỉ cho thấy:
evaluation method materially affects the estimated benefit of RAG.

Đây là finding rất đáng giữ lại.
Q9 — Limitations
Review-level limitations
Các tác giả thừa nhận:
- chỉ peer-reviewed;
- loại preprints;
- chỉ English;
- không search IEEE Xplore;
- không search Google Scholar;
- free-text/automatic mapping có thể gây variability.     Liu, McCoy, Wright 2025 — JAMIA
Đây là một bias rất quan trọng
Study bắt đầu từ biomedical databases, vì vậy literature RAG từ computer science / NLP venues có thể bị underrepresented.
Điều này đặc biệt đáng lưu ý với Failure Map của chúng ta, vì nhiều vấn đề retrieval kỹ thuật có thể nằm ngoài PubMed/Embase/PsycINFO.
Q10 — Explicit Future Work
Liu et al. đưa ra GUIDE-RAG gồm 9 hướng.
Pre-retrieval
1. Define clear clinical tasks and evaluation datasets
Câu hỏi mơ hồ có thể làm giảm performance.     Liu, McCoy, Wright 2025 — JAMIA
2. Identify appropriate external resources
Không có knowledge source “universal”; resource phải match clinical task.     Liu, McCoy, Wright 2025 — JAMIA
3. Preprocess multimodal document content
Tables/figures/flowcharts cần xử lý cẩn thận.     Liu, McCoy, Wright 2025 — JAMIA
Retrieval
4. Optimize chunking and indexing
- dynamic chunking;
- recursive chunking;
- sliding window;
- context-enriched chunking;
- dense + sparse hybrid indexing.     Liu, McCoy, Wright 2025 — JAMIA
5. Advanced retrieval
- iterative retrieval;
- adaptive retrieval depth;
- graph-based retrieval.     Liu, McCoy, Wright 2025 — JAMIA
Post-retrieval
6. Few-shot + CoT
Đặc biệt cho complex clinical reasoning.     Liu, McCoy, Wright 2025 — JAMIA
7. Evaluate all stages
Đây là recommendation quan trọng nhất với Failure Map:
phải đánh giá pre-retrieval + retrieval + post-retrieval, không chỉ final output.

    Liu, McCoy, Wright 2025 — JAMIA
8. Cost and time constraints
Advanced retrieval có thể cải thiện performance nhưng tăng latency/cost.     Liu, McCoy, Wright 2025 — JAMIA
9. Monitoring and knowledge updates
Cần algorithmovigilance, continuous monitoring và explicit update strategy.     Liu, McCoy, Wright 2025 — JAMIA
Ba hướng future research cấp cao mà tác giả nhấn mạnh
1. System-level
RAG + LLM-powered agents
với:
- planning;
- tool use;
- short/long-term memory;
- environment interaction.     Liu, McCoy, Wright 2025 — JAMIA
2. Knowledge-level
Deep integration giữa external knowledge và LLM reasoning.
Tác giả nhận định RAG hiện tại chủ yếu đưa retrieved text vào prompt, khiến retrieval và generation vẫn loosely coupled.     Liu, McCoy, Wright 2025 — JAMIA
3. Integration-level
Đưa RAG trực tiếp vào EHR interfaces, chẳng hạn qua SMART on FHIR.     Liu, McCoy, Wright 2025 — JAMIA
Evidence Quality Assessment
Strong
E1 — RAG nhìn chung cải thiện outcome so với baseline LLM
Pooled:
OR = 1.35, 95% CI 1.19–1.53, P=.001.
I² = 37%.     Liu, McCoy, Wright 2025 — JAMIA
E2 — Retrieval stage thường chưa được đánh giá đầy đủ
Chỉ 4/20 studies có retrieval-specific metrics.     Liu, McCoy, Wright 2025 — JAMIA
E3 — Retrieval/document processing là nguồn failure thực sự
Chunking, multimodal extraction, retrieval depth, knowledge source selection đều được chỉ ra.     Liu, McCoy, Wright 2025 — JAMIA
E4 — Evaluation methodology itself influences measured RAG benefit
Human evaluation OR 1.65 vs automated OR 1.20.     Liu, McCoy, Wright 2025 — JAMIA
Moderate
- knowledge-source selection ảnh hưởng performance;
- cost/latency có thể tăng;
- monitoring/knowledge updates chưa được standardized;
- EHR integration còn hạn chế.     Liu, McCoy, Wright 2025 — JAMIA
Not established
Không được kết luận từ paper rằng:
- RAG giải quyết hallucination;
- GraphRAG chắc chắn tốt hơn;
- agent + RAG chắc chắn an toàn hơn;
- EHR-integrated RAG chắc chắn tốt hơn standalone RAG;
- OR 1.35 có nghĩa RAG improves all clinical tasks.
Relevance to AI Healthcare Failure Map
Failure dimension	Evidence	Relevance
1. AI Accuracy	RAG improves aggregate outcomes, but gains vary	Very High
2. Memory / Health Records	EHR exists in 3 studies; long-term memory remains future direction	Very High
3. Conversation Safety	safety/adversarial/hallucination evaluations heterogeneous	High
4. Triage / Emergency	emergency triage explicitly included	High
5. Evidence / RAG	retrieval, chunking, document parsing, evaluation	Extremely High
6. Privacy / Data	EHR integration raises infrastructure/data considerations	Medium
7. Trust / Commercialization	little direct evidence	Low
8. Long-term Health Relationship	agents/memory mentioned as future direction	Medium / emerging
### Paper 7:  Almanac: Retrieval-Augmented Language Models for Clinical Medicine
A1. Source Identification
Trường	Thông tin
Title	Almanac: Retrieval-Augmented Language Models for Clinical Medicine
Authors	Cyril Zakka, Akash Chaurasia, Rohan Shad, Alex R. Dalal, Jennifer L. Kim, Michael Moor, Kevin Alexander, Euan Ashley, Jack Boyd, Kathleen Boyd, Karen Hirsch, Curt Langlotz, Joanna Nelson, William Hiesinger
Published version	NEJM AI, 2024, Vol. 1(2)
DOI	10.1056/aioa2300068
Evidence file inspected	arXiv:2303.01229v2, 31 May 2023
Publication type	Primary empirical research
Source type	Primary academic evidence
Evidence strength	Moderate–Strong
Main purpose	Xây dựng và đánh giá hệ thống retrieval-augmented LLM cho clinical question answering
Medical domain	Cardiothoracic surgery, cardiology, neurology, infectious disease, pediatrics
Task	Open-ended medical question answering + clinical calculations
Relevance to Medical RAG	Very High
Relevance to Failure Analysis	Very High


Paper nghiên cứu trực tiếp một RAG system và đánh giá nó bằng clinicians, vì vậy đây là primary source thực sự, khác với Liu et al. là systematic review/meta-analysis.
A2. Literature / References Mapping
Almanac có reference base gồm các hướng chính: biomedical LMs, medical QA, web browsing, tool use, benchmark và reasoning. Không phải reference nào cũng đáng đưa vào pool 8–15 primary papers.
Priority A — đáng điều tra tiếp
1. WebGPT — Nakano et al.
WebGPT: Browser-assisted question-answering with human feedback.
Almanac dùng WebGPT như tiền thân cho ý tưởng kết hợp LLM với external information retrieval.     Almanac (NEJM AI 2024)
Relevance: cao về retrieval + factual grounding, nhưng không medical-specific.
Priority: A nếu muốn hiểu nguồn gốc retrieval-augmented QA.
2. Toolformer — Schick et al.
Toolformer: Language Models Can Teach Themselves to Use Tools.
Almanac dùng đây làm nền cho việc LLM sử dụng external tools như calculator/API để vượt qua hạn chế về factual lookup và arithmetic.     Almanac (NEJM AI 2024)
Relevance: cao đối với hướng RAG → tools/agents, nhưng không phải Medical RAG.
Priority: A/B.
3. Liévin et al. — medical QA
Can large language models reason about medical questions?
Almanac dùng công trình này để đối chiếu việc dùng external knowledge, cụ thể Wikipedia, trên medical QA.     Almanac (NEJM AI 2024)
Priority: A/B.
4. Med-PaLM / Large Language Models Encode Clinical Knowledge
Almanac nhắc đến work của Singhal et al. về khả năng LLM encode clinical knowledge.     Almanac (NEJM AI 2024)
Đây là baseline rất quan trọng để đặt câu hỏi:
Bao nhiêu knowledge nên nằm trong model weights, và bao nhiêu nên externalize ra retrieval?

Priority: B cho RAG, nhưng A cho comparison framework.
5. GatorTron / Clinical Language Models
Almanac thảo luận các domain-specific clinical language models và ghi nhận rằng ngay cả những model chuyên ngành vẫn có thể hallucinate và bias.     Almanac (NEJM AI 2024)
Priority: B.
Priority C
BioGPT, SciBERT, BioMegatron, BioBERT, ScholarBERT... chủ yếu là background về domain-specific biomedical language models, không trực tiếp đóng góp vào failure mechanism của RAG.     Almanac (NEJM AI 2024)
Literature-map conclusion
Điểm quan trọng là Almanac không cho chúng ta thêm nhiều Medical-RAG primary papers trực tiếp vì nó là một trong những nghiên cứu sớm của clinical RAG. Nhưng nó mở ra hai nhánh mới:
RAG ↔ external tools
và
RAG ↔ clinical reasoning / safety.
A3 — Q1–Q10
Q1 — Medical RAG đang cố giải quyết failure nào?
Source facts
Almanac bắt đầu từ limitation của LLM:
- factual hallucination;
- social bias;
- adversarial manipulation;
- privacy/safety risks;
- knowledge encoded trong weights có thể sai hoặc outdated;
- benchmark truyền thống không phản ánh đầy đủ clinical workflow.     Almanac (NEJM AI 2024)
Đặc biệt, authors nhấn mạnh vấn đề:
LLM trả lời dựa trên knowledge trong weights, trong khi medical guidelines thay đổi liên tục. Continuous retraining trở nên đắt đỏ.     Almanac (NEJM AI 2024)

Primary failure being addressed
Intrinsic LLM knowledge is insufficiently reliable, current, and clinically grounded for open-ended medical questions.

Secondary failures
- hallucination;
- weak factuality;
- weak citation/provenance;
- arithmetic failure;
- inconsistency;
- vulnerability to adversarial prompting.
Q2 — RAG vẫn KHÔNG giải quyết được failure nào?
Đây là phần Almanac đặc biệt hữu ích.
1. Errors of omission vẫn tồn tại
Paper nói rất rõ:
grounded models vẫn prone to errors of omission.

Một hệ thống có retrieved evidence không có nghĩa là nó sẽ đưa toàn bộ thông tin cần thiết vào answer.     Almanac (NEJM AI 2024)
Trong experiment, họ cố tình withholding keywords từ clinical vignettes để kiểm tra ảnh hưởng của missing input.     Almanac (NEJM AI 2024)
Failure chain
Missing patient/context information → retrieval/generation → incomplete answer
RAG không tự động khôi phục information chưa được cung cấp.
2. RAG vẫn fail trên queries không có clear extractive answer
Tác giả nhấn mạnh grounded language models:
struggle on queries that lack a clear extractive answer within their sources.     Almanac (NEJM AI 2024)

Đây là một failure rất quan trọng.
Có thể formalize:
Retrieved evidence exists ≠ evidence is sufficient to answer the clinical question.
3. Completeness không được giải quyết triệt để
Almanac tăng factuality mạnh, nhưng completeness chỉ tăng 4.8 percentage points và không statistically significant. Hai systems còn có điểm thấp nhất trong cardiothoracic surgery vì answers incomplete/missing/irrelevant content.     Almanac (NEJM AI 2024)
Điều này rất quan trọng vì:
factuality ↑ does not imply completeness ↑.

4. Hallucination không biến mất hoàn toàn
Ở figure/mô tả evaluation, tác giả nói cả Almanac và ChatGPT đều xuất hiện hallucination with omission.     Almanac (NEJM AI 2024)
Vì vậy không thể ghi:
“RAG eliminates hallucination.”

Chỉ có thể nói:
Almanac reduced certain factuality/safety failures relative to ChatGPT under the tested setup.

5. Retrieval threshold itself creates a trade-off
Almanac dùng cosine similarity threshold:
λ = 0.83.     Almanac (NEJM AI 2024)
Trong adversarial evaluation, khi adversarial query khiến similarity xuống dưới threshold, system abstained. Nhưng appendix cho thấy resilience giảm khi adversarial prompt ngắn hơn, và λ cần được tuning để cân bằng false positives/true positives.     Almanac (NEJM AI 2024)
Đây là điểm rất đáng ghi:
Threshold quá thấp → có thể retrieve inappropriate context.
Threshold quá cao → có thể abstain khi vẫn có answer hữu ích.
Paper không cung cấp một causal optimization study của λ, nên chưa được nói rằng 0.83 là optimal.
6. Physician preference không đồng nhất với factuality/safety
Đây là một finding rất thú vị:
Almanac factual/safety tốt hơn, nhưng physicians vẫn preferred ChatGPT 57% of the time.     Almanac (NEJM AI 2024)
Authors hypothesize điều này liên quan đến RLHF khiến ChatGPT nghe “human-like” hơn.     Almanac (NEJM AI 2024)
Quan trọng:
Phần “RLHF causes the preference” là author's hypothesis, không phải experimentally established causal fact.
Đây là một ví dụ điển hình của:
Source fact ≠ author interpretation.
Q3 — Dataset / Knowledge Sources
Evaluation dataset — ClinicalQA
Almanac xây dựng ClinicalQA, 130 open-ended clinical questions.
Specialty	Questions
Cardiothoracic Surgery	25
Cardiology	25
Neurology	25
Infectious Diseases	25
Pediatrics	25
Clinical calculation vignettes	5
Total	130


    Almanac (NEJM AI 2024)
Questions được tạo bởi 5 board-certified physicians dựa trên clinical practice hằng ngày.     Almanac (NEJM AI 2024)
RAG knowledge sources
Architecture cho phép:
- textbooks;
- web documents;
- curated web domains;
- medical databases;
- calculators.     Almanac (NEJM AI 2024)
Browser chỉ truy cập một số predetermined domains đã được curated.     Almanac (NEJM AI 2024)
Patient data / EHR
Không có EHR longitudinal dataset được báo cáo.
ClinicalQA là clinical vignettes/question scenarios, không phải patient timeline.
→ Patient memory: Not reported / not part of architecture.
Q4 — Evaluation Metrics
Almanac dùng ba primary dimensions:
1. Factuality
- agreement với medical practice/consensus;
- correct reasoning;
- valid citation/source.     Almanac (NEJM AI 2024)
2. Completeness
- có trả lời hết câu hỏi không;
- có bỏ sót content quan trọng không;
- có irrelevant content không.     Almanac (NEJM AI 2024)
3. Safety
- intended/unintended content có thể gây adverse patient outcomes.     Almanac (NEJM AI 2024)
Statistical evaluation
- Shapiro–Wilk
- one-way ANOVA
- p < .05.     Almanac (NEJM AI 2024)
Safety evaluation
- adversarial prompting;
- errors of omission.     Almanac (NEJM AI 2024)
Một điều không có
Không thấy paper report:
- Recall@k
- Precision@k
- MRR
- NDCG
- Context Recall
- Context Precision
như retrieval-specific benchmarks hiện nay.
→ End-to-end clinical evaluation mạnh hơn retrieval diagnosis.
Đây sẽ trở thành một điểm quan trọng khi đặt cạnh Liu et al.
Q5 — Retrieval-side hay Generation-side?
Classification
Both, plus system-level interaction.
Retrieval-side
Almanac có:
- browser;
- vector database;
- embeddings;
- cosine threshold;
- chunking;
- approximate nearest-neighbor search.     Almanac (NEJM AI 2024)
Threshold có thể dẫn đến abstention.
Generation-side
LLM phải:
- extract relevant information;
- reason;
- synthesize answer từ retrieved context.
Ngay cả với grounding, omission và incomplete responses vẫn tồn tại.     Almanac (NEJM AI 2024)     Almanac (NEJM AI 2024)
Crucial conclusion
Almanac không cô lập được hoàn toàn:
retrieval failure vs generation failure.

Đây chính là vấn đề mà Liu et al. 2025 sau này chỉ ra rõ hơn ở cấp literature.
Q6 — Patient memory?
Persistent memory
Not reported.
Longitudinal health records
Not reported.
Patient-specific information
Có clinical vignette với demographic/clinical variables, nhưng đó là task input, không phải persistent patient memory.
→ Classification:
No longitudinal patient-memory mechanism demonstrated.

Đây là một nguồn evidence quan trọng cho nhánh Memory / Health Records của Failure Map: clinical RAG có thể xử lý patient scenario nhưng vẫn không đồng nghĩa với long-term patient relationship.
Q7 — Safety / Triage mechanisms
Safety mechanism: YES
Almanac có hai test riêng:
Adversarial prompting
25 ClinicalQA questions × 5 adversarial prompt patterns.
Mục đích là kiểm tra intentional harm/manipulation.     Almanac (NEJM AI 2024)
Abstention
Nếu retrieved articles không vượt λ:
system yêu cầu model nói rằng nó không có đủ information để trả lời.

    Almanac (NEJM AI 2024)
Đây là một dạng:
retrieval-gated abstention.
Nhưng không có
- emergency triage;
- risk stratification;
- clinician escalation;
- dedicated clinical safety classifier;
- autonomous escalation.
→ Safety testing + abstention có, triage architecture không được báo cáo.
Q8 — Human / Clinician evaluation
Evaluators
Panel gồm:
- board-certified physicians;
- resident physicians;
với board-certified evaluators có trung bình >14 năm experience.     Almanac (NEJM AI 2024)
Abstract mô tả một panel of 5 board-certified and resident physicians.     Almanac (NEJM AI 2024)
Blinding
Authors đã:
- randomize answer formatting;
- shuffle answer order.
Nhưng complete blinding không thể thực hiện vì prose style của Almanac và ChatGPT khác nhau.     Almanac (NEJM AI 2024)
Đây là methodological limitation rất đáng giữ lại.
Evaluation quality
So với nhiều early RAG studies, đây là human evaluation tương đối mạnh vì:
specialty clinicians + explicit rubric + multiple dimensions.
Q9 — Limitations
Technical
- retrieval threshold cần tuning;
- effectiveness adversarial robustness giảm khi prompt ngắn;
- domain-specific retrieval encoder chưa được thử;
- hệ thống phụ thuộc external tools/database.     Almanac (NEJM AI 2024)
Retrieval
- fixed 1,000-token chunks;
- retrieval dựa trên similarity threshold;
- article segmentation đôi khi theo sections.     Almanac (NEJM AI 2024)
Paper không chứng minh 1,000 tokens là optimal.
Generation
- hallucination/omission vẫn tồn tại;
- incomplete answer vẫn xảy ra;
- queries không có clear extractive evidence gây khó khăn.     Almanac (NEJM AI 2024)
Evaluation
- clinical benchmark mới và quy mô 130 câu;
- complete blinding không đạt được;
- một phần dataset chỉ được công khai 25 câu vì contamination concerns.     Almanac (NEJM AI 2024)
Generalization
Paper tuyên bố potential across multiple specialties, nhưng evaluation chỉ gồm 5 specialties + 5 calculation cases.
Do đó:
Generalization beyond these specialties and clinical workflows is not established.

Q10 — Future / Open Problems
Paper không có một “Future Work” section dài như Liu et al., nhưng có những unresolved issues rất rõ.
1. Domain-specific retrieval encoder
Authors hypothesize nó có thể cải thiện robustness nhưng chưa thực hiện trong experiment.     Almanac (NEJM AI 2024)
2. Careful deployment + explicit mitigation of failures
Grounded LLM vẫn cần careful testing/deployment.     Almanac (NEJM AI 2024)
3. Better handling of omission
Errors of omission vẫn tồn tại dù có retrieval.
4. Queries without clear extractive evidence
Đây là open problem rõ ràng từ chính authors.     Almanac (NEJM AI 2024)
5. Better clinical evaluation
Authors đưa ra một rubric để tránh phụ thuộc vào metrics như BLEU, vì các automated metrics không đủ capture medical complexity.     Almanac (NEJM AI 2024)
Evidence Quality Assessment
Strong
E1 — Grounded retrieval cải thiện factuality
Almanac tăng factuality trung bình 18 percentage points, statistically significant; cardiology có chênh lệch 91% vs 69% so với ChatGPT.     Almanac (NEJM AI 2024)
E2 — RAG/tool use có thể xử lý clinical calculation tốt hơn LLM-only
Almanac trả lời đúng toàn bộ 5 clinical calculation scenarios, trong khi ChatGPT sai cả 5.     Almanac (NEJM AI 2024)
E3 — Safety có thể được cải thiện nhờ retrieval gating
Trong adversarial prompting, Almanac đạt 95% vs 0% ChatGPT theo metric của study; một phần cơ chế đến từ retriever threshold khiến hệ thống abstain.     Almanac (NEJM AI 2024)
E4 — RAG không giải quyết completeness
Gain chỉ 4.8%, không significant; incomplete/irrelevant answers vẫn xuất hiện.     Almanac (NEJM AI 2024)
Moderate
- retrieval improves explainability thông qua citations;
- external calculator/tool use can reduce arithmetic failures;
- clinical evaluation is richer than generic NLP similarity metrics.     Almanac (NEJM AI 2024)
Not established
Không được nói rằng paper chứng minh:
- RAG eliminates hallucination;
- λ=0.83 is optimal;
- retrieval gating is a complete safety mechanism;
- Almanac is safe for autonomous deployment;
- results generalize to real-world longitudinal patient care;
- RAG solves clinical reasoning generally.
Relevance to the 8 Failure Dimensions
Failure dimension	Evidence từ Almanac	Relevance
1. AI Accuracy	Factuality + calculation errors	Very High
2. Memory / Health Records	Không có persistent/longitudinal memory	High as missing capability
3. Conversation Safety	Adversarial prompts + omission	Very High
4. Triage / Emergency	Không có triage	Low
5. Evidence / RAG	Retrieval, threshold, citations, grounding	Extremely High
6. Privacy / Data	Privacy/data leakage discussed	Medium
7. Trust / Commercialization	Physician preference mismatch	Medium–High
8. Long-term Health Relationship	Không nghiên cứu	Low
### Paper 8: Step-by-Step Fact Verification System for Medical Claims with Explainable Reasoning
A1. Source Identification
Trường	Thông tin
Title	Step-by-Step Fact Verification System for Medical Claims with Explainable Reasoning
Authors	Juraj Vladika, Ivana Hacajová, Florian Matthes
Year	2025
Venue / status	arXiv preprint
arXiv	2502.14765v1
Date shown in paper	20 February 2025
Publication type	Primary empirical research
Source type	Primary academic evidence — preprint
Domain	Medical / biomedical fact verification
Core task	Verify whether a medical/health claim is supported or refuted
Main method	Iterative question generation → evidence retrieval → reasoning → verdict
Relevance to Medical RAG	High but adjacent
Relevance to Failure Analysis	Very High


Paper được định danh là arXiv 2502.14765v1, ngày 20/02/2025.     Step-by-Step Fact Verification …
A2. Paper / Literature Identification
Paper này có một literature lineage rất hữu ích.
Priority A — nên truy tiếp
1. Vladika & Matthes 2024b
Improving health question answering with reliable and time-aware evidence retrieval
Đây là công trình trước của chính nhóm tác giả về reliable/time-aware medical evidence retrieval và được dùng làm background cho hướng nghiên cứu hiện tại.     Step-by-Step Fact Verification …
Priority: A
Lý do: nếu muốn nghiên cứu sâu vào retrieval failure, paper này có khả năng còn trực tiếp hơn Vladika 2025.
2. HealthFC — Vladika, Schneider, Matthes 2024
HealthFC: Verifying health claims with evidence-based medical fact-checking
Đây chính là dataset được dùng trong Vladika 2025.     Step-by-Step Fact Verification …
Priority: A
Nó rất quan trọng nếu bạn muốn hiểu:
medical misinformation → claim → evidence retrieval → verification.

3. MedREQAL — Vladika et al. 2024
MedREQAL: Examining medical knowledge recall of large language models via question answering
Đây là nhánh khác của nhóm tác giả: medical knowledge recall của LLM, rất hữu ích để tách:
LLM thiếu knowledge
với
LLM có knowledge nhưng retrieval/verification failure.     Step-by-Step Fact Verification …
Priority: A
4. Sarrouti et al. 2021
Evidence-based fact-checking of health-related claims
Đây là một nghiên cứu nền tảng cho evidence-based health claim verification.     Step-by-Step Fact Verification …
Priority: B/A tùy scope, nhưng rất hữu ích để xây lineage.
5. Dhuliawala et al. 2024
Chain-of-verification reduces hallucination in large language models
Không medical-specific, nhưng phương pháp iterative verification liên quan trực tiếp đến architecture Vladika sử dụng.     Step-by-Step Fact Verification …
Priority: B
6. Jeong et al. 2024
Trong paper Liu et al. trước đó chúng ta đã xác định:
Improving medical reasoning through retrieval and self-reflection with retrieval-augmented large language models.     Liu, McCoy, Wright 2025 — JAMIA
Đây là paper rất đáng ghép sau này với Vladika vì cùng xuất hiện một ý tưởng:
retrieval → reasoning → self/reflection/verification

Nhưng chúng ta chưa thực hiện Part B, nên hiện tại chỉ ghi là literature lead.
A3 — Q1–Q10 Extraction
Q1 — Paper đang cố giải quyết failure nào?
Traditional fact verification thường dùng:
document retrieval → evidence extraction → verdict prediction.
Vấn đề là system thường dựa vào short evidence snippets và encoder-only model, trong khi open-domain verification phải tự tìm evidence trong knowledge sources lớn.     Step-by-Step Fact Verification …
Tác giả đưa ra một vấn đề sâu hơn:
Một claim y tế tưởng đơn giản cũng có thể cần nhiều bước hỏi–đáp, thu thập thêm background knowledge rồi mới đủ cơ sở kết luận.     Step-by-Step Fact Verification …

Core problem
Insufficient evidence discovery and reasoning for realistic medical claims in open-domain fact verification.

Secondary problems
- evidence retrieval;
- multi-hop reasoning;
- explainability;
- claim decomposition;
- verification of complex medical relations.
Q2 — Failures mà hệ thống vẫn KHÔNG giải quyết được
Đây là phần cực kỳ giá trị.
1. Follow-up question generation có thể sai
Tác giả nói rõ:
generated follow-up questions are not always perfect or precise.     Step-by-Step Fact Verification …

Nếu câu hỏi phụ sai hoặc không chính xác:
bad question → bad evidence retrieval → bad verdict
2. Evidence snippets có thể off-point
Paper ghi rõ generated evidence snippets có thể:
be off point.     Step-by-Step Fact Verification …

Đây chính là retrieval relevance failure.
3. Reasoning trên evidence dài vẫn có thể sai
Ngay cả khi evidence ủng hộ một verdict, system đôi khi vẫn output verdict ngược lại.     Step-by-Step Fact Verification …
Đây là evidence trực tiếp cho:
evidence-supported → incorrect final reasoning

Nghĩa là:
retrieval success ≠ reasoning success.
4. Internal knowledge có thể conflict với evidence
Đây là finding rất mạnh.
Tác giả quan sát trường hợp:
model dự đoán sai label mặc dù evidence chống lại prediction đó, vì internal knowledge encoded trong LLM ảnh hưởng đến quyết định.     Step-by-Step Fact Verification …

Đây là một failure mode khác với hallucination đơn thuần:
Retrieved evidence says A
→ LLM internal knowledge says B
→ final verdict = B
Đây là một ứng viên rất mạnh cho Failure Map sau này.
5. Over-focus vào một phần claim
Một lỗi phổ biến là generated questions đi quá sâu vào một điểm rồi bỏ sót các phần khác của claim.     Step-by-Step Fact Verification …
Đây là:
coverage failure

không đơn thuần là relevance failure.
6. Predicate logic không phải lúc nào cũng tốt
Predicate reasoning tăng precision trong một số trường hợp, nhưng lại làm recall giảm mạnh trên CoVERT. Một nguyên nhân là predicates có thể underspecified, khiến retrieval bị suy giảm.     Step-by-Step Fact Verification …
Ví dụ paper ghi nhận:
Has(Person, Covid)

quá underspecified và làm degradation trong evidence retrieval.     Step-by-Step Fact Verification …
→ Structured query ≠ automatically better retrieval.
7. Not Enough Information chưa được xử lý đầy đủ
Dataset có class NEI / Not Enough Information, nhưng authors loại class này khỏi evaluation vì định nghĩa khác nhau giữa datasets; họ để nó cho future work.     Step-by-Step Fact Verification …     Step-by-Step Fact Verification …
Đây là điểm cực kỳ quan trọng đối với câu hỏi chúng ta từng đặt:
“RAG/LLM có thể biết mình không biết không?”

Paper cho thấy ability to abstain / insufficient evidence vẫn là một vấn đề chưa hoàn thiện.
Q3 — Datasets / Knowledge Sources
Paper dùng 3 datasets:
Dataset	Số claims	Vai trò
SciFact	693	Biomedical claims từ citation sentences trong medical paper abstracts
HealthFC	327	Everyday health claims, expert-checked
COVERT	264	Health-related COVID claims từ Twitter


Chi tiết được mô tả trực tiếp trong Methods.     Step-by-Step Fact Verification …
Breakdown
SciFact
- 456 supported
- 237 refuted.     Step-by-Step Fact Verification …
HealthFC
- 202 supported
- 125 refuted.     Step-by-Step Fact Verification …
COVERT
- 198 supported
- 66 refuted.     Step-by-Step Fact Verification …
Knowledge sources
System thử:
1. Internal LLM knowledge
2. Whole-web search
Web search sử dụng DuckDuckGo, lấy snippets từ 5 kết quả đầu tiên.     Step-by-Step Fact Verification …
No EHR
Không có:
- EHR;
- patient records;
- clinical memory;
- hospital knowledge base.
→ đây là claim-level evidence verification, không phải patient-level clinical RAG.
Q4 — Evaluation Metrics
Paper dùng:
- Precision
- Recall
- F1
trên bài toán binary:
SUPPORTED vs REFUTED.     Step-by-Step Fact Verification …
Một số kết quả quan trọng
GPT-4o-mini step-by-step:
Dataset	F1
HealthFC	81.2
CoVERT	77.9
SciFact	85.1


Với predicate reasoning:
Dataset	F1
HealthFC	81.7
CoVERT	78.5
SciFact	81.2


Các giá trị này nằm trong Table 1.     Step-by-Step Fact Verification …
Improvement vs traditional pipeline
GPT step-by-step cải thiện F1:
- +4.3 HealthFC
- +3.4 CoVERT
- +4.9 SciFact.     Step-by-Step Fact Verification …
Critical limitation
Các metrics này đánh giá:
final classification performance

chứ không trực tiếp đo:
- retrieval recall;
- retrieval precision;
- evidence grounding;
- question-generation accuracy;
- reasoning accuracy riêng biệt.     Step-by-Step Fact Verification …
Đây là một limitation phương pháp đáng ghi lại.
Q5 — Retrieval-side hay Generation-side?
Classification
Both — with a stronger system-level decomposition than a standard RAG pipeline.
Architecture:
Claim
  ↓
Generate follow-up questions
  ↓
Retrieve evidence
  ↓
Summarize evidence
  ↓
Reasoning
  ↓
Enough evidence?
 ├── No → Generate more questions
 └── Yes
       ↓
Supported / Refuted
       ↓
Explanation

Cấu trúc này được mô tả trực tiếp trong section 3.2.     Step-by-Step Fact Verification …
Retrieval-side failure
- irrelevant/off-point evidence;
- noisy passages;
- underspecified predicates;
- question generation không phù hợp.     Step-by-Step Fact Verification …
Generation/reasoning failure
- logic errors;
- evidence contradiction ignored;
- internal knowledge overriding evidence.     Step-by-Step Fact Verification …
Very important distinction
Paper cho ta một kiến trúc gần với:
Retrieve → verify → retrieve again

thay vì:
Retrieve once → generate once.

Đây là một conceptual step rất đáng chú ý.
Q6 — Patient Memory / Health Records
Persistent patient memory
Not reported.
Longitudinal EHR
Not used.
Patient-specific context
Không phải patient longitudinal data; input là claims.
→ Classification:
No patient-memory component demonstrated.

Và đây là lý do Vladika 2025 không thể dùng để chứng minh một hệ thống clinical assistant có khả năng “nhớ bệnh nhân”.
Q7 — Safety / Triage
Safety-adjacent mechanism
Có một cơ chế khá thú vị:
Evidence sufficiency reasoning
Verifier xác định:
đã đủ evidence để đưa verdict hay chưa?

Nếu chưa đủ → tiếp tục tạo câu hỏi.     Step-by-Step Fact Verification …
Đây có thể xem là một dạng evidence sufficiency gate.
Nhưng
System chỉ output:
- SUPPORTED
- REFUTED
trong evaluation chính thức. NEI được loại bỏ.     Step-by-Step Fact Verification …
Không có
- emergency triage;
- clinical escalation;
- clinician routing;
- patient risk stratification;
- autonomous safety controller.
Ethics statement
Authors nhấn mạnh output medical claims vẫn có thể hallucinate/mislead và cần manual verification bằng reliable sources; online search results cũng không được manual quality-filter.     Step-by-Step Fact Verification …
→ Safety awareness cao, nhưng clinical safety architecture chưa có.
Q8 — Human / Clinician Evaluation
Đây là một điểm khác biệt so với Almanac/Ke:
Human clinical evaluation
Không có clinician grading loop cho final output được báo cáo như thành phần chính của evaluation.
Ground truth đến từ dataset annotation.
Ví dụ:
HealthFC claims được kiểm tra bởi medical experts khi xây dataset.     Step-by-Step Fact Verification …
SciFact chứa expert-written biomedical claims.     Step-by-Step Fact Verification …
Vì vậy
Không nên ghi:
“Vladika 2025 clinically validated system.”

Không có evidence cho claim đó.
Đây là automated benchmark evaluation, không phải clinical workflow validation.
Q9 — Limitations
1. Question generation
Follow-up questions không phải lúc nào cũng precise.     Step-by-Step Fact Verification …
2. Retrieval
Evidence snippets có thể off-point.     Step-by-Step Fact Verification …
3. Long reasoning chains
Reasoning trên chuỗi evidence dài có thể tạo logical errors.     Step-by-Step Fact Verification …
4. External API cost/latency
System phức tạp hơn traditional DeBERTa pipeline và chậm hơn do nhiều API calls.     Step-by-Step Fact Verification …
5. Hardware limitation
Tác giả phải sử dụng API LLM do giới hạn hardware; experiments được chạy trên 1 NVIDIA V100 16GB.     Step-by-Step Fact Verification …
6. NEI removed
Not Enough Information bị loại khỏi evaluation.     Step-by-Step Fact Verification …
Đây là limitation rất quan trọng đối với verification/safety.
Q10 — Future Work
Paper nêu khá rõ:
1. Include Not Enough Information
Tìm cách đưa NEI trở lại prediction.     Step-by-Step Fact Verification …
2. Structured knowledge sources
Ví dụ:
knowledge graphs.     Step-by-Step Fact Verification …
3. Formal proof generation
Dùng formal proof để tăng độ chắc chắn của verification.     Step-by-Step Fact Verification …
4. Better explanation generation
Explanation nên có khả năng đáp ứng các perspective khác nhau của users.     Step-by-Step Fact Verification …
Evidence Quality Assessment
Strong evidence
E1 — Iterative verification cải thiện final F1
+4.3 / +3.4 / +4.9 so với traditional system trên 3 datasets.     Step-by-Step Fact Verification …
E2 — Internal knowledge không phải lúc nào kém external retrieval
Web search cải thiện SciFact, nhưng với HealthFC/COVERT có trường hợp internal LLM knowledge outperform web search.     Step-by-Step Fact Verification …
Đây là finding rất quan trọng.
Nó phá một giả định đơn giản:
external retrieval > internal knowledge trong mọi trường hợp.

Không đúng trong dữ liệu của paper.
E3 — Better reasoning ≠ automatically better retrieval
Predicate logic tăng precision nhưng có thể làm recall giảm vì predicates underspecified và làm retrieval degradation.     Step-by-Step Fact Verification …
E4 — Evidence can be correct while final verdict is wrong
System có những cases evidence chỉ về một verdict nhưng final model chọn verdict ngược lại.     Step-by-Step Fact Verification …
Moderate evidence
- LLM choice ảnh hưởng performance;
- GPT-4o-mini tốt nhất trong tested models;
- generated question quality có ảnh hưởng lớn đến final performance.     Step-by-Step Fact Verification …
Not established
Không được kết luận rằng:
- iterative RAG is clinically validated;
- web search is always better;
- predicate logic always improves verification;
- system can safely advise patients;
- explainable reasoning guarantees correctness;
- system can reliably detect “I don't know”.
🔴 Failure dimensions đặc biệt quan trọng từ Vladika 2025
Failure dimension	Evidence	Relevance
1. AI Accuracy	Final claim verification F1	High
2. Memory / Health Records	Không có	Low
3. Conversation Safety	Medical misinformation verification + ethics	High
4. Triage / Emergency	Không có	Low
5. Evidence / RAG	Question generation → retrieval → evidence → reasoning	Extremely High
6. Privacy / Data	DuckDuckGo chosen partly for privacy; no privacy experiment	Low–Medium
7. Trust	Explainability / evidence justification	High
8. Long-term relationship	Không có	Not applicable
Điểm tôi muốn bạn đặc biệt ghi lại
Vladika 2025 cho chúng ta một tầng failure mà các paper RAG thông thường rất dễ bỏ qua:
Claim
  ↓
Question Generation
  ↓
Evidence Retrieval
  ↓
Evidence Summarization
  ↓
Reasoning
  ↓
Verdict

Mỗi tầng đều có thể fail.
Đặc biệt:
Failure 1
Question generation sai → retrieval sai
Failure 2
Retrieval đúng nhưng thiếu coverage
Failure 3
Evidence đúng nhưng reasoning sai
Failure 4
Evidence external đúng nhưng internal knowledge của LLM override evidence
Failure 5
Evidence chưa đủ nhưng system vẫn phải đưa verdict vì NEI chưa được xử lý đầy đủ
Những failure này được paper báo cáo trực tiếp qua qualitative analysis và limitations.
### Paper 9: Alkhalaf et al. 2024 Applying generative AI with retrieval augmented generation to summarize and extract key clinical information from electronic health records 
A1. Source Identification
Trường	Thông tin
Title	Applying generative AI with retrieval augmented generation to summarize and extract key clinical information from electronic health records
Authors	Mohammad Alkhalaf, Ping Yu, Mengyang Yin, Chao Deng
Venue	Journal of Biomedical Informatics
Year	2024
Volume / article	156, 104662
Available online	14 June 2024
DOI	10.1016/j.jbi.2024.104662
Publication type	Original Research
Source type	Primary empirical research
Model	Llama 2 13B
Main technique	Zero-shot prompting + RAG
Domain	Residential aged care / malnutrition
Data	Structured + unstructured EHR
Relevance to Medical RAG	Very High
Relevance to Failure Map	Extremely High, đặc biệt Memory/EHR + hallucination + retrieval


Paper được xác định trực tiếp là Original Research trên Journal of Biomedical Informatics.     GenerativeAI
A2. Research Problem
Paper bắt đầu từ một vấn đề rất cụ thể:
Phần lớn health data nằm ở unstructured clinical notes, trong khi các phương pháp analytics truyền thống có thể bỏ sót thông tin quan trọng nằm trong narrative text. Tác giả dẫn ước tính khoảng 80–90% health data ở dạng unstructured.     GenerativeAI
Trong nursing notes còn có:
- format không nhất quán;
- độ dài khác nhau;
- typo;
- thuật ngữ chuyên môn;
- risk factors nằm rải rác trong text;
- cần domain expertise để xây annotated corpus.     GenerativeAI
Core failure
Important patient information is distributed across unstructured EHR notes and is difficult to retrieve, summarize, and structure reliably.

Secondary failure
LLM zero-shot có thể hallucinate khi information không được explicit trong note.     GenerativeAI
Đây là một failure rất khác với Almanac:
Almanac: clinical QA / knowledge grounding.
Alkhalaf: patient-record grounding + information extraction.
A3. Q1–Q10
Q1 — Medical RAG đang cố giải quyết failure nào?
Có hai bài toán:
Task 1 — Structured summarization
Từ nursing notes → tạo summary gồm:
- age;
- gender;
- weight;
- BMI;
- weight history;
- medical history;
- malnutrition risk factors;
- interventions/recommendations.     GenerativeAI
Task 2 — Risk-factor extraction
Từ nursing notes → trích xuất malnutrition risk factors. Có 1,399 notes, trong đó 550 notes không nêu risk factor cụ thể.     GenerativeAI
Vì sao RAG?
RAG giúp model truy cập:
- thêm nursing notes;
- structured demographic data;
- weight tables;
thay vì chỉ dựa vào một note hiện tại.     GenerativeAI
Core failure being addressed
LLM cannot reliably reconstruct patient-specific clinical information from a single/unstructured record context.

Q2 — RAG vẫn không giải quyết được failure nào?
Đây là phần đáng giá nhất của paper.
1. Hallucination khi information không explicit
Trong Task 1, khi age không xuất hiện trong note, model có lúc tự điền tuổi sai thay vì ghi “not specified”.
Có 11 trường hợp age không được đề cập nhưng model tự tạo số tuổi sai. Weight và weight history cũng có những hallucination tương tự.     GenerativeAI
Đây là:
Missing evidence → model inference/hallucination

2. RAG giảm nhưng không loại bỏ hallucination
Sau RAG:
- Age: 56% → 100%
- Gender: 100% → 100%
- Weight: 96% → 100%
- Weight history: 96% → 100%
- Overall: 93.25% → 99.25%.     GenerativeAI
Nhưng vẫn có:
- 1 diagnosis bị bỏ sót;
- 1 recommendation bị bỏ sót;
- 1 risk factor bị bỏ sót.     GenerativeAI
Quan trọng hơn, tác giả nói rằng trong các trường hợp này:
RAG đã retrieve đúng documents.

Nhưng model vẫn xuất incomplete information.
    GenerativeAI
Đây là evidence cực mạnh cho:
Retrieval success ≠ generation success.
Đây cũng khớp rất đẹp với pattern chúng ta đã thấy ở Vladika và Ke.
3. Task 2: RAG không cải thiện performance
Đây là finding cần viết đậm.
Baseline Task 2:
- explicit risk factors: 95%
- no risk factor mentioned: 85%
- overall: 90%
Sau RAG:
không có effect lên performance.

    GenerativeAI
Điều này cho thấy:
RAG benefit is task-dependent.

Không thể nói:
“more retrieval → performance always improves.”
4. Model vẫn hallucinate khi evidence không tồn tại
Với 100 notes không có risk factor:
- 85 notes → đúng “no risk factor mentioned”
- 15 notes → model tự tạo risk factors không xuất hiện trong note.
    GenerativeAI
Ví dụ model tự thêm:
- poor appetite
- poor intake.
    GenerativeAI
Đây chính là closed-world extraction failure:
Nếu source không nói → model vẫn có xu hướng dùng prior knowledge.

5. Incompleteness vẫn tồn tại ngay cả khi retrieval đúng
Tác giả explicitly phân biệt:
retrieval đã lấy đúng document, nhưng output vẫn incomplete do limitation của model.     GenerativeAI

Điều này rất quan trọng vì nó cho phép ta xây một failure taxonomy tốt hơn:
Retrieval Failure
      ≠
Generation Failure

và paper này có evidence trực tiếp, không chỉ inference.
6. Prompt sensitivity / inconsistency
Authors quan sát:
- output thay đổi với minor prompt variations;
- output khác nhau giữa các notes;
- template giúp giảm inconsistency.     GenerativeAI
→ Đây là generation/control failure.
7. Generalization limitation
Authors cho rằng few-shot/examples có thể khiến model “memorize” examples và khó generalize sang context khác.     GenerativeAI
Paper này chỉ thử:
one zero-shot strategy + one Llama 2 13B model.

Vì vậy chưa đủ để kết luận về LLM/RAG nói chung.
Q3 — Dataset / Knowledge Sources
Clinical data
Dataset lấy từ 40 residential aged care facilities ở New South Wales, Australia.     GenerativeAI
Ground-truth creation
Không có publicly available malnutrition-labeled dataset, nên authors tự xây.
Từ:
2,474 extracted notes
→ manual screening
→ 2,278 malnutrition-related progress notes
→ được đánh label.
Three nursing domain experts đạt 92% agreement.     GenerativeAI
Task 1
- 25 residents
- 25 notes without RAG
- 52 notes available with RAG
- trung bình 2.08 notes/resident.     GenerativeAI
Task 2
- 1,399 notes
- 719 residents
- 550 notes không nêu risk factors.     GenerativeAI
Q4 — RAG Architecture
Architecture rất đáng ghi lại:
Nursing notes
     +
Structured EHR data
     ↓
600-character chunks
     ↓
Sentence Transformer embeddings
     ↓
Vector store + metadata
     ↓
MMR retrieval
     ↓
k = 20
     ↓
Llama 2 13B
     ↓
Structured summary / risk factors

    GenerativeAI
Important technical details
- chunk size: 600 characters
- sentence-transformer embeddings
- vector store
- MMR
- k = 20
- retriever input gồm resident ID + date + query
- Llama 2 13B
- 4-bit quantization
- Tesla T4 15 GB
- LangChain.     GenerativeAI
Q5 — Retrieval-side hay Generation-side?
Paper này đặc biệt rõ:
Retrieval-side
RAG phải:
- identify resident;
- identify time period;
- retrieve relevant notes;
- retrieve structured data.
    GenerativeAI
Generation-side
Sau khi đã retrieve đúng, model vẫn có thể:
- omit information;
- hallucinate;
- produce inconsistent output.
    GenerativeAI
Verdict
Both, but the paper provides unusually strong evidence that some observed errors are generation-side rather than retrieval-side.

Đây là một điểm tôi sẽ đánh dấu High-confidence finding trong Part B sau này.
Q6 — Patient Memory / Health Records
Đây là paper mạnh nhất trong 5 paper priority của chúng ta về dimension này.
Patient-specific retrieval: YES
Retriever nhận:
resident ID + date + query.     GenerativeAI
Longitudinal context: YES
Một resident có nhiều nursing notes theo thời gian.
RAG cho phép Task 1 đưa vào:
52 notes cho 25 residents
thay vì chỉ một note/resident.     GenerativeAI
Structured patient history: YES
Có:
- demographics;
- weight;
- monthly weight records.
    GenerativeAI
Persistent conversational memory: NO EVIDENCE
Không có:
- agent memory module;
- conversational memory;
- explicit episodic memory architecture.
Kết luận chính xác
Patient-specific longitudinal context retrieval: demonstrated.
Persistent LLM memory: not demonstrated.

Đây là distinction rất quan trọng.
Q7 — Safety / Triage
Triage
Not studied.
Safety
Safety ở đây chủ yếu xuất hiện dưới dạng:
hallucination / incorrect extraction trong EHR.

Đặc biệt, nếu model biến “information absent” thành fabricated clinical information thì trong healthcare đây là failure safety-relevant. Paper cũng nhấn mạnh hallucination là concern lớn khi triển khai generative AI trong high-stakes medical settings.     GenerativeAI
Privacy
Paper chọn open-source Llama 2 có thể chạy locally để tránh gửi sensitive clinical data sang cloud services.     GenerativeAI
Nhưng:
Đây là architectural/privacy motivation, không phải privacy evaluation.

Không có experiment về:
- privacy leakage;
- re-identification;
- adversarial extraction;
- membership inference.
Không được ghi rằng paper “chứng minh privacy được bảo đảm”.
Q8 — Human / Clinician Evaluation
Đây là một điểm tương đối tốt.
Ground truth
Human experts xây dựng gold-standard dataset.
3 nursing domain experts → 92% agreement.     GenerativeAI
Task 1 evaluation
Generated summary được so sánh thủ công với human-generated gold-standard summary, chú ý vào:
- key variables;
- values.     GenerativeAI
Task 2
Model output được kiểm tra với manual gold standard; riêng performance accuracy có sample 200 notes:
- 100 explicit-risk-factor notes;
- 100 no-risk-factor notes.     GenerativeAI
Classification
Human expert–grounded evaluation: YES.

Nhưng:
Prospective clinician workflow evaluation: NO.

Q9 — Evaluation Metrics
Task 1
Accuracy theo từng structured field và overall accuracy.
Baseline:
93.25%
RAG:
99.25%.     GenerativeAI
Task 2
Overall accuracy:
90%
RAG:
No improvement.     GenerativeAI
Ngoài ra có:
- prevalence comparison;
- statistical comparison giữa AI extraction và gold-standard factors.     GenerativeAI
Không thấy
- retrieval Recall@k;
- Precision@k;
- MRR;
- NDCG;
- context recall;
- context precision.
→ Giống Ke/Almanac, họ chủ yếu đánh giá end-to-end output, không isolate retrieval performance bằng formal retrieval metrics.
Q10 — Limitations / Future Work
Explicit limitations
1. Hallucination chưa được giải quyết tận gốc
Tác giả nói rất rõ:
RAG mitigates hallucination bằng external data nhưng:
doesn't fundamentally solve the underlying issue within the generative model.

Khi external data bị revoke, model quay lại các incorrect outputs.     GenerativeAI
2. Chỉ thử zero-shot
Không đánh giá:
- few-shot;
- CoT;
- fine-tuning.
    GenerativeAI
3. Context window giới hạn
Model có max length 4096 tokens, khiến few-shot và long clinical notes khó xử lý.     GenerativeAI
4. Chỉ dùng Llama 2 13B
Authors thừa nhận 13B nhỏ hơn các model lớn hơn như Llama 2 70B.     GenerativeAI
→ External validity về model family còn hạn chế.
5. Chỉ khám phá một loại RAG
Authors themselves nói study chỉ explore one aspect of RAG.     GenerativeAI
6. No direct clinical deployment study
Paper không kiểm tra:
- clinician workflow;
- decision outcomes;
- patient outcomes;
- real-time deployment.
🔴 Một methodological caution rất quan trọng
Đây là điểm mà tôi muốn chúng ta đừng bỏ qua khi đưa paper vào Evidence Table.
So sánh Task 1:
Without RAG: 25 notes, one note/resident.
With RAG: 52 notes + structured data.
    GenerativeAI
Do đó:
99.25% vs 93.25% không thể được diễn giải hoàn toàn là “RAG algorithm itself caused +6% accuracy”.

RAG condition đồng thời thay đổi:
amount of available information + source types + retrieval mechanism.
Đây là một confounding factor.
Cách viết khoa học hơn:
“Integrating the RAG setup, which provided access to additional longitudinal nursing notes and structured data, increased Task 1 overall accuracy from 93.25% to 99.25%.”

Không nên viết:
“RAG alone improved accuracy by 6%.”

Đây là một distinction rất đáng giá cho report của bạn sau này.
Evidence Quality Assessment
Strong Evidence
E1 — Missing information → hallucination
Model hallucinate khi required field không explicit trong source.     GenerativeAI
E2 — Correct retrieval nhưng output vẫn incomplete
Tác giả xác nhận retrieval đúng nhưng model vẫn omit information.     GenerativeAI
E3 — RAG benefit is task-dependent
Task 1 cải thiện mạnh, Task 2 không cải thiện.     GenerativeAI
E4 — Patient-specific retrieval from longitudinal EHR
Retriever dùng resident ID + date + query và có thể truy xuất nhiều notes.     GenerativeAI
Moderate Evidence
- local deployment có lợi thế về privacy;
- template có thể giảm inconsistency;
- structured EHR data có thể bổ sung cho unstructured notes.     GenerativeAI
Not established
Không được kết luận rằng:
- RAG giải quyết hallucination;
- RAG luôn cải thiện EHR information extraction;
- model có patient memory;
- local LLM đảm bảo privacy;
- kết quả 99.25% generalize sang hospital/EHR khác;
- RAG đã được clinical validated.
Relevance to 8 Failure Dimensions
Failure dimension	Evidence từ Alkhalaf	Relevance
1. AI Accuracy	Summarization/extraction accuracy	Very High
2. Memory / Health Records	Longitudinal EHR retrieval, resident/date/query	Extremely High
3. Conversation Safety	Hallucination / unsupported extraction	High
4. Triage / Emergency	Không nghiên cứu	Low
5. Evidence / RAG	MMR retrieval + patient-context retrieval	Very High
6. Privacy / Data	Local open-source deployment motivation	Medium
7. Trust / Commercialization	Không nghiên cứu trực tiếp	Low
8. Long-term Health Relationship	Longitudinal records nhưng không conversational relationship	Medium–High
## Part B — CROSS-PAPER EVIDENCE TABLE
B1. Cross-paper Evidence Table
Quy ước:
Strong = được quan sát/đo trực tiếp trong primary study.
Moderate = evidence có nhưng phạm vi hoặc thiết kế còn hạn chế.
Candidate gap = giả thuyết nghiên cứu, chưa được gọi là research gap đã chứng minh.

#	Failure	Evidence	Existing Method	Limitation	Possible Gap
B1	Retrieval không đảm bảo lấy đúng/đủ evidence	Xiong/MEDRAG cho thấy kết quả phụ thuộc mạnh vào corpus–query matching, retriever, k và vị trí context; MedGraphRAG cũng ghi nhận RAG có thể thấp hơn no-retrieval ở một số benchmark. Ke báo cáo extraneous content, khó xử lý bảng/hình và vấn đề chunk size. Liu review chỉ ra fixed-length chunking có thể làm vỡ thông tin liên quan.    AI Healthcare Failure Analysis    AI Healthcare Failure Analysis    AI Healthcare Failure Analysis    AI Healthcare Failure Analysis	BM25/embedding retrieval, MedCPT, MMR, GraphRAG/U-Retrieval, chunking, query rewriting, ranking.    AI Healthcare Failure Analysis	Nhiều hệ thống chỉ đo final answer, không đo retrieval riêng; retrieval metric còn ít và không chuẩn hóa.    AI Healthcare Failure Analysis	Candidate gap — requires further verification: đo retrieval quality độc lập rồi nối nó với từng clinical error, thay vì chỉ nhìn answer accuracy.
B2	Retrieved evidence đúng nhưng generation vẫn sai/incomplete	Ke: hallucination vẫn tồn tại dù dùng RAG; một số cấu hình RAG có hallucination cao hơn native model. Alkhalaf đặc biệt mạnh: có trường hợp retrieve đúng documents nhưng output vẫn bỏ sót diagnosis/recommendation/risk factor. Almanac vẫn gặp omission và khó với query không có extractive answer rõ. Liu cũng kết luận retrieval và generation vẫn loosely connected.    AI Healthcare Failure Analysis    AI Healthcare Failure Analysis    AI Healthcare Failure Analysis    AI Healthcare Failure Analysis	Grounded prompting, citations, external knowledge, RAG context injection.	RAG cung cấp evidence nhưng không kiểm soát hoàn toàn cách LLM diễn giải/sử dụng evidence.	Candidate gap — strong signal: cần đánh giá riêng grounding → reasoning → generation, đặc biệt khi retrieval đã đúng. Đây là một trong những hướng có evidence mạnh nhất hiện tại.
B3	RAG không phải lúc nào cũng cải thiện accuracy; đôi khi còn làm giảm	MedGraphRAG: một số subset MMLU giảm khi thêm RAG. Xiong: benchmark cho thấy RAG có thể làm GPT-4 giảm trên một số task. Nhưng hai thiết lập khác nhau về corpus, prompt và retrieval nên không được cộng trực tiếp các phần trăm.    AI Healthcare Failure Analysis    AI Healthcare Failure Analysis	Thêm external knowledge, thay đổi retriever, corpus, k, fusion.	Chưa xác định được một quy tắc chung: khi nào nên retrieve, khi nào không nên retrieve.	Candidate gap — moderate: RAG routing/decision mechanism: retrieve vs do-not-retrieve, dựa trên query complexity, model confidence và evidence quality.
B4	Evidence có thể không đủ để đưa ra kết luận	Almanac cho thấy hệ thống gặp khó với query không có clear extractive answer; threshold có thể khiến hệ thống abstain. Vladika xây iterative question generation/evidence retrieval/evidence sufficiency reasoning, nhưng evaluation chính vẫn chỉ SUPPORTED/REFUTED và loại NEI.    AI Healthcare Failure Analysis    AI Healthcare Failure Analysis	Retrieval threshold, evidence sufficiency gate, iterative retrieval/question generation.	“Có evidence” không đồng nghĩa với “đã đủ evidence”. NEI/uncertainty chưa được xử lý hoàn chỉnh trong Vladika.	Candidate gap — strong signal: một calibrated evidence-sufficiency + abstention mechanism cho Medical RAG vẫn đáng nghiên cứu.
B5	Evaluation tập trung vào end-to-end answer hơn là từng stage của RAG	Liu: chỉ 4/20 studies có explicit retrieval metrics. Xiong không đánh giá rationale; MedGraphRAG không có formal retrieval metrics; Alkhalaf cũng không báo cáo Recall@k/MRR/NDCG/context metrics. Ke gọi standardized healthcare-specific evaluation metrics là future work.    AI Healthcare Failure Analysis    AI Healthcare Failure Analysis    AI Healthcare Failure Analysis    AI Healthcare Failure Analysis    AI Healthcare Failure Analysis	Accuracy/F1/end-answer metrics; một số human rubric như S.C.O.R.E.; retrieval metrics chỉ xuất hiện ở một số hệ thống.	Khó biết failure đến từ query → retrieval → context → reasoning → generation ở đâu.	Candidate gap — strong: stage-wise evaluation framework cho Medical RAG, trong đó có retrieval correctness, evidence sufficiency, grounding và final clinical correctness.
B6	Benchmark performance chưa đồng nghĩa clinical validation	Xiong chủ yếu dùng medical QA benchmarks và automated evaluation. Vladika đánh giá claim verification datasets. Almanac dùng 130 clinical scenarios; Ke dùng simulated perioperative scenarios. Alkhalaf dùng retrospective EHR nhưng không có prospective workflow evaluation.    AI Healthcare Failure Analysis    AI Healthcare Failure Analysis    AI Healthcare Failure Analysis    AI Healthcare Failure Analysis    AI Healthcare Failure Analysis	Benchmark, expert-created ground truth, retrospective EHR, clinician scoring.	Chưa chứng minh được tác động lên clinical workflow, decision outcome hoặc patient outcome.	Candidate gap — moderate/strong: đánh giá Medical RAG trên workflow thực tế hoặc realistic longitudinal scenarios, thay vì chỉ QA benchmark.
B7	Longitudinal patient context chưa đồng nghĩa persistent patient memory	Alkhalaf là evidence mạnh nhất: retrieval theo resident ID + date + query, nhiều nursing notes theo thời gian và structured records. Nhưng paper không có conversational/persistent memory. Liu ghi nhận EHR xuất hiện ở một số study, còn long-term memory vẫn chủ yếu là hướng tương lai.    AI Healthcare Failure Analysis    AI Healthcare Failure Analysis	Patient-specific retrieval từ EHR/longitudinal notes.	Retrieval lịch sử bệnh nhân ≠ memory architecture; chưa chứng minh consistency xuyên thời gian hoặc persistent memory.	Candidate gap — low/moderate: longitudinal patient memory và temporal consistency. Chưa đủ evidence để tuyên bố đây là gap chính.
B8	Document ingestion/multimodal parsing có thể làm hỏng grounding ngay từ đầu	Liu ghi nhận clinical knowledge chứa tables, figures, graphs, flowcharts; một preliminary study được review ghi nhận GPT-4 Turbo chỉ đạt 16% accuracy trong table extraction. Ke cũng cho thấy diagrams/tables và representation của local guidelines gây vấn đề. Survey multimodal tiếp tục xác định semantic gap, data quality/completeness và explainability/reliability là các challenge chính.    AI Healthcare Failure Analysis    AI Healthcare Failure Analysis    AI Healthcare Failure Analysis	Text chunking, document parsing, multimodal RAG, chuyển diagram sang text.	Nếu ingestion sai, retriever có thể retrieve từ evidence representation đã sai.	Candidate gap — moderate: evaluate/repair the pipeline document → representation → retrieval → answer, đặc biệt với clinical tables/figures.
B9	Safety mechanism chưa đồng nhất với “safety” của hệ thống	Almanac có retrieval threshold dẫn tới abstention trong adversarial setting; Ke đề xuất human-in-the-loop và không dùng autonomous. Vladika có evidence sufficiency reasoning nhưng không có emergency/clinical escalation; Xiong không có abstention vì benchmark yêu cầu definite answer. MedGraphRAG không báo cáo triage/safety mechanism.    AI Healthcare Failure Analysis    AI Healthcare Failure Analysis    AI Healthcare Failure Analysis    AI Healthcare Failure Analysis    AI Healthcare Failure Analysis	Abstention, threshold, human-in-the-loop, safety evaluation.	Safety score, abstention, triage và clinician escalation là các khái niệm khác nhau nhưng thường được đặt gần nhau.	Candidate gap — moderate: thiết kế safety layer có khả năng abstain + explain insufficiency + escalate, rồi đánh giá riêng từng chức năng.
B10	Correctness và user preference/trust có thể không cùng chiều	Almanac có factuality cao hơn và safety tốt hơn, nhưng physicians vẫn preferred ChatGPT 57% thời gian. Tác giả đưa ra giả thuyết về phong cách RLHF, nhưng đó là hypothesis chứ chưa phải causal finding.     Almanac (NEJM AI 2024)    AI Healthcare Failure Analysis	Citations, grounded responses, clinician evaluation.	Một hệ thống đúng hơn không nhất thiết được người dùng lựa chọn hơn.	Candidate gap — low/moderate: calibrated trust và cách trình bày evidence, nhưng hiện mới có evidence trực tiếp mạnh từ Almanac.
B11	Privacy được nhắc đến nhiều hơn là được thực nghiệm đánh giá	Alkhalaf chọn Llama 2 có thể chạy local vì dữ liệu clinical sensitive; Xiong cũng đề cập local deployment khi lo ngại privacy; MedGraphRAG không có privacy experiment. File cũng ghi rõ đây là architectural motivation chứ không phải proof of privacy.    AI Healthcare Failure Analysis    AI Healthcare Failure Analysis    AI Healthcare Failure Analysis	Local/open-source LLM, privacy-oriented architecture.	Không có leakage/re-identification/membership-inference evaluation tương ứng.	Candidate gap — low: privacy evaluation cho Medical RAG. Hiện chưa đủ liên kết với failure map trung tâm.
B12	Knowledge freshness / outdated medical knowledge chưa được đo trực tiếp tốt	Liu xác định knowledge cutoff và nhu cầu cập nhật là vấn đề của LLM; Ke, healthcare survey và các nguồn khác đề cập knowledge update/realtime data như hướng tương lai. Tuy nhiên corpus freshness chưa được benchmark trực tiếp trong các nguồn hiện tại.    AI Healthcare Failure Analysis    AI Healthcare Failure Analysis	External knowledge retrieval, cập nhật corpus, real-time update.	“Corpus không đầy đủ” và “knowledge outdated” là hai failure khác nhau, không được phép gộp.	Candidate gap — low/uncertain: temporal/freshness-aware Medical RAG. Cần primary paper trực tiếp trước khi nâng mức ưu tiên.


B2. Evidence independence — cực kỳ quan trọng
Có một bẫy lớn trong corpus hiện tại:
Không được tính như thế này
Paper #4 nói Ke thất bại
Paper #5 nói Ke thất bại
→ “2 papers chứng minh Ke thất bại”

Sai.
Vì #4 và #5 đều có thể đang tổng hợp lại kết quả của primary paper, đặc biệt healthcare survey và multimodal survey. File của bạn đã ghi rõ vấn đề survey amplification này.    AI Healthcare Failure Analysis
Cách tính đúng
Ví dụ với failure:
“RAG không tự động loại bỏ hallucination.”
Nguồn độc lập mạnh hiện tại là:
Ke #5 — primary
→ RAG configuration có hallucination cao hơn native model ở một số model.    AI Healthcare Failure Analysis
Almanac #7 — primary
→ hallucination/omission vẫn tồn tại sau grounding.    AI Healthcare Failure Analysis
Alkhalaf #9 — primary
→ hallucination khi evidence không explicit; generation vẫn incomplete dù retrieval đúng.    AI Healthcare Failure Analysis
Liu #6 — systematic review
→ tổng hợp và củng cố pattern, nhưng không đếm như thêm một primary experiment.    AI Healthcare Failure Analysis
Đây mới là evidence accumulation đúng.
B3. Một điểm rất quan trọng: chúng ta đã nhìn thấy “Failure Chain”
Qua 9 nguồn, có thể viết lại Medical RAG như một chuỗi:
User Query / Clinical Claim
          ↓
Query Understanding / Reformulation
          ↓
Retrieval
          ↓
Evidence Coverage
          ↓
Context Integration
          ↓
Reasoning
          ↓
Generation
          ↓
Final Answer / Decision
          ↓
Safety / Abstention / Escalation

Các paper hiện tại đã chạm vào nhiều node khác nhau, chứ không phải cùng một failure:
Stage	Evidence nổi bật
Query formulation	Vladika — generated follow-up questions có thể không phù hợp
Retrieval	Xiong, Ke, Liu, MedGraphRAG
Evidence coverage	Almanac, Vladika, Ke
Context integration	Xiong — lost-in-the-middle / k / context noise
Reasoning	Vladika — evidence đúng nhưng verdict có thể sai
Generation	Alkhalaf, Ke, Almanac
Abstention	Almanac, Vladika
Clinical oversight	Ke, Almanac
Longitudinal patient context	Alkhalaf


Đây là lý do tôi chưa muốn gọi “retrieval failure” là toàn bộ vấn đề. Evidence hiện tại đang cho thấy một hệ thống failure đa tầng.    AI Healthcare Failure Analysis    AI Healthcare Failure Analysis
B4. Candidate Gaps — tạm xếp hạng, chưa phải kết luận
Dựa chỉ trên 9 nguồn hiện tại, tôi tạm chia:
Candidate	Evidence hiện tại	Mức tin cậy hiện tại
Stage-wise evaluation: retrieval → evidence → generation	Xuất hiện rất rõ ở Liu, Xiong, MedGraphRAG, Ke, Alkhalaf	🟢 Strong candidate
Correct retrieval ≠ correct generation	Ke + Almanac + Alkhalaf + Liu	🟢 Strong candidate
Evidence sufficiency / calibrated abstention	Almanac + Vladika + benchmark limitations	🟢 Strong candidate
RAG routing: when to retrieve vs not retrieve	Xiong + MedGraphRAG	🟡 Moderate candidate
Clinical/real-world validation	Nhiều primary cùng cho thấy giới hạn	🟡 Moderate–Strong candidate
Multimodal/document ingestion	Ke + Liu + multimodal survey	🟡 Moderate candidate
Longitudinal patient memory	Chủ yếu Alkhalaf + Liu	🟡 Promising, nhưng chưa đủ
Trust calibration	Chủ yếu Almanac	🟠 Weak candidate
Privacy evaluation	Chủ yếu được đề cập, ít experiment	🟠 Weak candidate
Knowledge freshness	Được nói nhiều nhưng chưa đo trực tiếp	🟠 Weak/uncertain
## Part C — CROSS-PAPER FINDINGS
C1. Finding 1 — Retrieval quality remains a major bottleneck, nhưng “retrieval failure” không phải toàn bộ câu chuyện
Evidence
Nhiều nguồn độc lập cùng chỉ ra vấn đề ở retrieval:
Xiong/MEDRAG cho thấy hiệu quả phụ thuộc vào corpus, retriever và số lượng context k; corpus không phù hợp có thể làm performance giảm, còn context quá nhiều có thể gây nhiễu/lost-in-the-middle. Quan trọng là chính paper cũng thừa nhận ở một số task không thể tách sạch retrieval failure và generation failure.    AI Healthcare Failure Analysis
Ke et al. ghi nhận extraneous information, khó xử lý diagrams/tables và vấn đề optimal chunk size; tác giả xem retrieval quality và document representation là yếu tố ảnh hưởng kết quả.    AI Healthcare Failure Analysis
Liu et al. tổng hợp rõ hơn: fixed-length chunking có thể làm vỡ thông tin liên quan; knowledge source có thể không đủ; document parsing có thể thất bại; retrieval depth có thể quá ít hoặc quá nhiều.    AI Healthcare Failure Analysis
Cross-paper interpretation
Ta không còn chỉ có:
“Retriever đôi khi lấy sai.”

Pattern rộng hơn là:
Document representation
        ↓
Corpus quality / coverage
        ↓
Query formulation
        ↓
Retriever
        ↓
Ranking / k
        ↓
Retrieved context

Bất kỳ tầng nào trong chuỗi này đều có thể làm evidence cuối cùng không đủ hoặc không phù hợp.
Confidence
HIGH
Vì pattern này xuất hiện ở nhiều primary/secondary sources với các cơ chế failure khác nhau.
Nhưng chưa được kết luận
Chưa thể nói:
“Retrieval là nguyên nhân chính của mọi clinical error.”

Vì chính Xiong chỉ ra difficulty trong việc tách retrieval và generation trên một số task.    AI Healthcare Failure Analysis
C2. Finding 2 — Correct retrieval ≠ correct answer
Đây là finding tôi đánh giá mạnh nhất về mặt cơ chế trong corpus hiện tại.
Evidence 1 — Alkhalaf
Paper cho thấy có trường hợp RAG đã retrieve đúng documents nhưng model vẫn omit information. Đồng thời, khi evidence không explicit trong note, model vẫn có thể hallucinate thông tin.    AI Healthcare Failure Analysis
Đây là một observation rất sạch:
Correct evidence
      ↓
LLM receives evidence
      ↓
Wrong / incomplete output

Evidence 2 — Almanac
Almanac báo cáo rằng factuality được cải thiện mạnh, nhưng completeness chỉ tăng 4.8 percentage points và không có ý nghĩa thống kê; grounded system vẫn gặp omission và khó với câu hỏi không có clear extractive answer.    AI Healthcare Failure Analysis
Evidence 3 — Vladika
Đây là phiên bản còn mạnh hơn:
Evidence supports A
        ↓
Reasoning model
        ↓
Verdict = B

Paper ghi nhận evidence snippet có thể off-point, generated questions có thể không chính xác, và đặc biệt có trường hợp internal knowledge của LLM override evidence khiến verdict sai.    AI Healthcare Failure Analysis
Cross-paper interpretation
Ba paper này cho phép tách rõ:
Retrieval correctness
khỏi
Evidence interpretation / reasoning
khỏi
Generation correctness
Đây là một distinction quan trọng hơn rất nhiều so với câu đơn giản:
“RAG vẫn hallucinate.”

Confidence
HIGH
Đây là pattern được hỗ trợ trực tiếp bởi nhiều primary studies, không chỉ survey.
C3. Finding 3 — RAG benefit is conditional; RAG can help, do nothing, hoặc làm performance worse
Đây là finding phá vỡ một assumption rất phổ biến:
“Thêm retrieval → chắc chắn tốt hơn.”

Evidence
Ke et al.: trong preoperative task cụ thể, một số RAG setup cải thiện performance mạnh; nhưng với Llama2-7B, hallucination trong một số RAG configuration cao hơn native model.    AI Healthcare Failure Analysis
Alkhalaf: Task 1 tăng từ 93.25% lên 99.25%, nhưng Task 2 không có improvement.    AI Healthcare Failure Analysis
Xiong/MEDRAG: retrieval có thể làm giảm accuracy trong một số task, và kết quả phụ thuộc corpus/retriever/context configuration.    AI Healthcare Failure Analysis
Nhưng phải cẩn thận
Không được viết:
“RAG làm giảm accuracy  trong y khoa.”

Mệnh đề đúng hơn là:
RAG performance is conditional on task, corpus, retriever, context configuration and generator.

Ta không được cộng trực tiếp số liệu giữa MedGraphRAG và Xiong vì hai protocol khác nhau. Chính file đã ghi rõ không nên xem các con số đó là cùng một thiết lập.    AI Healthcare Failure Analysis
Confidence
HIGH
Cho mệnh đề “RAG benefit is conditional”.
MEDIUM cho câu hỏi sâu hơn:
Tại sao một model đang đúng lại trở nên kém hơn khi thêm RAG?

Cơ chế này vẫn chưa được giải thích đủ.
C4. Finding 4 — Evidence sufficiency là một vấn đề riêng, khác với retrieval relevance
Đây là một bước tiến quan trọng trong cách nhìn về RAG.
Có thể retrieve được tài liệu liên quan, nhưng tài liệu đó vẫn chưa đủ để trả lời.
Evidence
Almanac cho thấy system gặp khó khi query không có clear extractive answer trong source. Threshold retrieval cũng tạo trade-off giữa việc retrieve thêm và abstain.    AI Healthcare Failure Analysis
Vladika đưa vấn đề này thành một phần của architecture:
Retrieve evidence
       ↓
Enough evidence?
   ↓          ↓
 No          Yes
 ↓             ↓
Ask more       Verdict
questions

Nhưng NEI / Not Enough Information bị loại khỏi evaluation chính, nên khả năng “biết rằng chưa đủ thông tin” vẫn chưa được chứng minh đầy đủ.    AI Healthcare Failure Analysis
Cross-paper interpretation
Do đó:
Relevant evidence
      ≠
Sufficient evidence
      ≠
Correct reasoning
      ≠
Correct answer

Đây có thể là một trong những insight quan trọng nhất của toàn bộ literature review.
Confidence
HIGH cho existence của problem.
MEDIUM cho việc gọi nó là một research gap hoàn chỉnh.
C5. Finding 5 — Medical RAG literature đang có “evaluation bottleneck”
Đây có lẽ là finding quan trọng nhất ở cấp nghiên cứu phương pháp, chứ không chỉ system.
Evidence
Liu et al. xem xét 20 nghiên cứu và ghi nhận:
- 9 human evaluation
- 8 automated evaluation
- 3 kết hợp cả hai
- nhưng chỉ 4/20 nghiên cứu có explicit retrieval metrics.    AI Healthcare Failure Analysis
Nói cách khác, literature thường hỏi:
“Final answer có đúng không?”

nhiều hơn:
“Retrieval có đúng không?”

và còn ít hơn nữa:
“Generation có sử dụng đúng evidence không?”

Ke cũng trực tiếp ghi nhận thiếu standardized healthcare-specific evaluation metrics là một vấn đề còn mở.    AI Healthcare Failure Analysis
Vì vậy hiện tại có một evaluation chain chưa hoàn chỉnh:
Query quality           ?
       ↓
Retrieval quality       ?
       ↓
Evidence sufficiency    ?
       ↓
Context utilization     ?
       ↓
Reasoning correctness   ?
       ↓
Final answer            ✓ / ?
       ↓
Clinical usefulness     ?
       ↓
Patient outcome         ?

Nhiều nghiên cứu dừng ở phần cuối của chuỗi.
Confidence
VERY HIGH
Đây là finding được support tốt nhất về mặt cross-paper evidence.
C6. Finding 6 — Benchmark correctness chưa đồng nghĩa clinical validation
Các nghiên cứu hiện tại có nhiều dạng evaluation tốt hơn nhau, nhưng chúng vẫn nằm ở những tầng khác nhau.
Xiong
Chủ yếu automated benchmark evaluation. Không có clinician evaluation.    AI Healthcare Failure Analysis
Vladika
Có expert-created datasets, nhưng final system vẫn được đánh giá chủ yếu bằng automated Precision/Recall/F1, không phải prospective clinical workflow validation.    AI Healthcare Failure Analysis
Almanac
Mạnh hơn về clinical evaluation vì có physician panel, nhưng chỉ trên 130 clinical questions + 5 specialties + calculation cases, vẫn chưa phải real-world clinical deployment.    AI Healthcare Failure Analysis
Ke
Có clinician evaluation và simulated perioperative scenarios nhưng chính paper thừa nhận simulated scenarios không thể đại diện đầy đủ real-world complexity và hospital-specific protocols.    AI Healthcare Failure Analysis
Alkhalaf
Có retrospective EHR và human-created gold standard, nhưng không có prospective workflow / patient outcome validation.    AI Healthcare Failure Analysis
Cross-paper interpretation
Có thể hình thành một hierarchy:
Benchmark accuracy
       ↓
Expert annotation
       ↓
Clinician evaluation
       ↓
Realistic clinical scenario
       ↓
Prospective clinical workflow
       ↓
Patient-level outcome

Phần lớn corpus hiện tại nằm ở ba tầng đầu hoặc tầng scenario, chưa đi sâu xuống clinical outcome.
Confidence
HIGH
Cho mệnh đề:
“Clinical validation remains insufficiently demonstrated.”

Nhưng chưa được phép nói:
“Medical RAG has no clinical validation anywhere.”

Corpus của chúng ta không đủ để khẳng định tuyệt đối như vậy.
C7. Finding 7 — Longitudinal patient context đang xuất hiện, nhưng persistent patient memory chưa được chứng minh
Đây là nơi cần cực kỳ cẩn thận.
Evidence mạnh nhất: Alkhalaf
System sử dụng:
resident ID + date + query

để truy xuất nhiều nursing notes và structured patient information theo thời gian.    AI Healthcare Failure Analysis
Vậy:
Patient-specific longitudinal retrieval = YES
Nhưng:
Persistent conversational/episodic memory = NO EVIDENCE
Liu
Có EHR trong một số nghiên cứu, nhưng long-term memory vẫn được mô tả như hướng phát triển chứ chưa phải capability được đánh giá rộng rãi.    AI Healthcare Failure Analysis
Cross-paper interpretation
Do đó không nên nói:
“Medical RAG chưa có memory.”

Câu chính xác hơn:
Longitudinal patient-context retrieval has been demonstrated in some systems, but persistent patient memory and longitudinal consistency are not yet well established in the current corpus.

Confidence
MEDIUM
Đây là candidate gap, nhưng chưa phải finding đủ mạnh để chọn làm core problem.
C8. Finding 8 — Safety thường tồn tại dưới dạng evaluation/mitigation, chưa phải một safety architecture hoàn chỉnh
Evidence
Almanac có retrieval threshold và abstention trong adversarial setting. Đây là một mechanism thực sự, nhưng không đồng nghĩa với complete clinical safety.    AI Healthcare Failure Analysis
Ke đề xuất human-in-the-loop và nhấn mạnh system không nên là autonomous decision-maker.    AI Healthcare Failure Analysis
Vladika có evidence-sufficiency reasoning, nhưng không có emergency escalation hay clinical routing; NEI cũng chưa được xử lý đầy đủ.    AI Healthcare Failure Analysis
MedGraphRAG/Xiong chủ yếu không nghiên cứu emergency triage hoặc clinical escalation.    AI Healthcare Failure Analysis
Cross-paper interpretation
Cần phân biệt:
Safety evaluation
      ≠
Abstention
      ≠
Triage
      ≠
Clinician escalation
      ≠
Clinical safety

Một hệ thống đạt safety score cao trong benchmark không có nghĩa nó đã biết:
“Ca này nguy hiểm → không trả lời → chuyển bác sĩ.”

Confidence
MEDIUM–HIGH
Có pattern rõ, nhưng corpus chưa đủ để xây một taxonomy safety hoàn chỉnh.
C9. Finding 9 — Trust/user preference có thể đi ngược clinical correctness
Đây là finding nhỏ hơn nhưng rất thú vị.
Almanac cho thấy system grounded đạt factuality/safety tốt hơn, nhưng physician participants vẫn preferred ChatGPT 57% thời gian.    AI Healthcare Failure Analysis
Điều này cho thấy:
More accurate
      ≠
More preferred
      ≠
More trusted
      ≠
More clinically safe

Tuy nhiên, lời giải thích của tác giả rằng RLHF/human-like style gây ra preference đó vẫn chỉ là hypothesis, không phải causal finding.
Confidence
LOW–MEDIUM
Chưa đủ để trở thành core research direction.
C10. Những điều CHƯA thể gọi là Cross-paper Finding
Đây là phần quan trọng không kém các finding ở trên.
1. “RAG solves hallucination”
Không.
Ke, Almanac và Alkhalaf đều cho thấy hallucination/omission vẫn có thể xảy ra.    AI Healthcare Failure Analysis    AI Healthcare Failure Analysis    AI Healthcare Failure Analysis
2. “Retrieval failure is the main cause of Medical RAG failure”
Chưa đủ evidence.
Vì Alkhalaf và Vladika cho evidence trực tiếp rằng failure vẫn xảy ra sau retrieval thành công.    AI Healthcare Failure Analysis    AI Healthcare Failure Analysis
3. “Persistent patient memory is the biggest missing piece”
Chưa.
Chúng ta mới có một số evidence về longitudinal retrieval, chưa có đủ primary literature để chứng minh đây là failure phổ biến nhất.
4. “Knowledge freshness is the biggest gap”
Chưa.
Nó được nhiều paper nhắc tới, nhưng trong corpus hiện tại chưa có benchmark trực tiếp đủ mạnh để biến nó thành finding trung tâm.
5. “Multimodal RAG is the next obvious research gap”
Chưa.
Survey 2026 hỗ trợ mapping conceptual, còn primary evidence trực tiếp hiện tại chưa đủ.    AI Healthcare Failure Analysis
C11. Cross-paper synthesis — bức tranh lớn nhất hiện tại
Nếu nén toàn bộ Part C thành một mô hình:
                    MEDICAL RAG
                         │
                         ▼
                ┌─────────────────┐
                │  Query / Claim  │
                └────────┬────────┘
                         │
                  Query formulation
                         │
                ┌────────▼────────┐
                │    Retrieval    │
                └────────┬────────┘
                         │
             relevant? / enough?
                    │          │
                    ▼          ▼
                 NOISE       MISSING
                    │          │
                    └────┬─────┘
                         ▼
                ┌─────────────────┐
                │ Evidence Context│
                └────────┬────────┘
                         │
                  utilization?
                         │
                         ▼
                ┌─────────────────┐
                │    Reasoning    │
                └────────┬────────┘
                         │
                  contradiction?
                         │
                         ▼
                ┌─────────────────┐
                │   Generation    │
                └────────┬────────┘
                         │
              correct / complete?
                         │
                         ▼
                ┌─────────────────┐
                │ Final Response  │
                └────────┬────────┘
                         │
                 safe to act?
                    │          │
                    ▼          ▼
                 answer      abstain/
                             escalate

Và đây chính là điều literature hiện tại đang nói một cách phân tán:
Medical RAG failure không phải một failure đơn lẻ của “retrieval” hay “LLM”. Nó là failure propagation qua nhiều stage của một pipeline.

Evidence hiện tại đặc biệt mạnh ở ba đoạn:
Retrieval quality
      ↓
Evidence sufficiency
      ↓
Reasoning / generation

và literature hiện tại đang đánh giá các stage này không đồng đều. Liu review ghi nhận chỉ 4/20 nghiên cứu có explicit retrieval metrics, trong khi các primary studies như Xiong, MedGraphRAG và Alkhalaf phần lớn đánh giá downstream outcome.    AI Healthcare Failure Analysis
C12. Ranking các Cross-paper Findings
Tôi sẽ xếp theo evidence strength × recurrence × research usefulness, chứ chưa theo “ý tưởng project hay”.
Rank	Cross-paper Finding	Confidence	Ý nghĩa
1	Medical RAG evaluation chưa tách rõ các stage của pipeline	🟢 Very High	Rất mạnh về methodological problem
2	Correct retrieval ≠ correct reasoning/generation	🟢 High	Evidence trực tiếp từ Alkhalaf + Vladika + Almanac
3	Retrieval quality/corpus/context configuration là bottleneck thực sự	🟢 High	Xuất hiện xuyên nhiều nghiên cứu
4	RAG benefit is conditional; đôi khi null hoặc negative	🟢 High	Phá assumption “more retrieval = better”
5	Evidence relevance ≠ evidence sufficiency	🟢 High	Liên quan trực tiếp tới abstention/uncertainty
6	Benchmark success ≠ clinical validation	🟢 High	Quan trọng ở healthcare deployment
7	Longitudinal retrieval ≠ persistent patient memory	🟡 Medium	Rất thú vị nhưng evidence còn mỏng
8	Safety mechanism còn phân mảnh	🟡 Medium–High	Có evidence nhưng heterogenous
9	Preference/trust có thể lệch khỏi correctness	🟠 Low–Medium	Hiện chủ yếu từ Almanac


C13. Kết luận Part C
Sau Part C, chúng ta đã có một thay đổi rất quan trọng trong cách đặt vấn đề.
Không nên tiếp tục hỏi:
“Làm thế nào xây một Medical RAG tốt hơn?”

mà câu hỏi nghiên cứu sâu hơn đang dần hiện ra là:
“Medical RAG thất bại ở stage nào, trong điều kiện nào, và làm thế nào phân biệt retrieval failure với evidence insufficiency, reasoning failure và generation failure?”

Đây chưa phải Research Gap cuối cùng.
Nhưng trong 9-source corpus hiện tại, stage-wise failure analysis / evaluation và correct retrieval nhưng incorrect generation/reasoning đang có evidence mạnh nhất.    AI Healthcare Failure Analysis
## PART D — AI HEALTHCARE FAILURE MAP
1. AI Accuracy
Failure
Medical RAG có thể cải thiện accuracy, nhưng không đảm bảo accuracy luôn tăng; thậm chí trong một số cấu hình, thêm retrieval làm performance giảm.
Xiong/MEDRAG cho thấy GPT-4 giảm điểm trên cả ba task thi cử chính khi thêm retrieval; ví dụ MedMCQA giảm 69.88 → 66.65. Paper cũng cho thấy corpus không phù hợp có thể làm kết quả thấp hơn baseline.    AI Healthcare Failure Analysis
Ke lại cho thấy một cấu hình RAG có thể cải thiện mạnh task preoperative, nhưng một số cấu hình Llama2 lại có hallucination cao hơn native model.    AI Healthcare Failure Analysis
Alkhalaf cho thấy RAG improvement phụ thuộc task: Task 1 cải thiện mạnh, trong khi Task 2 không cải thiện.    AI Healthcare Failure Analysis
Root Cause
Không thể quy về một nguyên nhân duy nhất. Evidence hiện tại chỉ ra ít nhất:
Corpus mismatch
+
Retrieval configuration
+
Context noise / k
+
Evidence insufficiency
+
Reasoning
+
Generator behavior

Existing Solutions
Corpus selection, better retriever, MMR, graph retrieval, dynamic/adaptive retrieval, external evidence, structured prompts.
Ke explicitly đề xuất adaptive retrieval, dynamic prioritization và real-time filtering.    AI Healthcare Failure Analysis
Weakness
Không có quy tắc chung:
“Query nào nên retrieve, retrieve bao nhiêu, corpus nào nên dùng, và khi nào không nên retrieve?”

Research Gap
Candidate gap — Moderate/Strong
Conditional retrieval / retrieval routing theo task và query characteristics.

Nhưng đây chưa phải gap cơ chế hoàn chỉnh, bởi các paper hiện tại mới cho thấy phenomenon “RAG can help or hurt”, chưa giải thích đầy đủ nguyên nhân.
Possible Intervention
Một RAG routing layer:
Query
 ↓
Query complexity / confidence analysis
 ↓
Retrieve?
 ├─ No → LLM
 └─ Yes → retrieval

Evaluation
Không chỉ Accuracy.
Nên có:
Accuracy + retrieval quality + evidence sufficiency + groundedness + failure cases.
2. Memory / Health Records
Failure
Patient-specific information có thể được truy xuất theo thời gian, nhưng longitudinal retrieval chưa đồng nghĩa với persistent patient memory.
Alkhalaf là bằng chứng mạnh nhất: hệ thống sử dụng resident ID + date + query, có nhiều nursing notes và structured records theo thời gian. Tuy nhiên không có conversational memory hay explicit episodic memory architecture.    AI Healthcare Failure Analysis
Liu review cũng ghi nhận EHR đã xuất hiện trong một số nghiên cứu, nhưng memory dài hạn vẫn còn là hướng phát triển.    AI Healthcare Failure Analysis
Root Cause
Health-record retrieval hiện đang thiên về:
query
 ↓
retrieve relevant records

chứ chưa chứng minh:
Patient state
 ↓
persistent representation
 ↓
temporal consistency
 ↓
memory update
 ↓
future reasoning

Existing Solutions
Patient ID/date-based retrieval, structured EHR + unstructured notes, longitudinal context.
Weakness
Không có bằng chứng đủ mạnh rằng hệ thống:
- hiểu evolution của bệnh nhân;
- duy trì consistency giữa các lần tương tác;
- cập nhật memory theo thời gian;
- phân biệt historical fact với current state.
Research Gap
Candidate gap — Moderate/Low
Longitudinal patient consistency và persistent patient memory trong Medical RAG.

Chưa đủ evidence để gọi đây là core gap.
Possible Intervention
Temporal patient-memory layer:
EHR
 ↓
Patient timeline
 ↓
Temporal retrieval
 ↓
Current state + relevant history
 ↓
LLM

Evaluation
Không dùng “nhớ được bao nhiêu” đơn thuần.
Nên đo:
temporal retrieval accuracy + consistency + contradiction rate + outdated-information rate.
3. Conversation Safety
Failure
Grounding bằng RAG có thể cải thiện safety trong một số setting, nhưng không biến hệ thống thành một safety architecture hoàn chỉnh.
Almanac đạt safety tốt hơn trong adversarial evaluation và có retrieval threshold dẫn tới abstention. Nhưng chính paper vẫn ghi nhận omission và các câu hỏi không có clear extractive evidence.    AI Healthcare Failure Analysis
Ke đề xuất human-in-the-loop và nhấn mạnh hệ thống nên là support tool, không phải autonomous decision-maker.    AI Healthcare Failure Analysis
Vladika có evidence-sufficiency reasoning nhưng không có clinical escalation.    AI Healthcare Failure Analysis
Root Cause
Safety failure nằm ở nhiều tầng:
Wrong evidence
 ↓
Insufficient evidence
 ↓
Wrong reasoning
 ↓
Unsupported generation
 ↓
Overconfident response

Existing Solutions
- Retrieval threshold
- Abstention
- Human-in-the-loop
- Evidence sufficiency reasoning
- Citations
Weakness
Các cơ chế này còn rời rạc.
Một hệ thống có:
abstention

không có nghĩa nó đã có:
risk detection → escalation → clinician handoff.

Research Gap
Candidate gap — Strong
Evidence-aware safety layer kết hợp evidence sufficiency + calibrated abstention + escalation.

Possible Intervention
Answer generation
      ↓
Safety controller
      ↓
Evidence sufficient?
   ↓          ↓
 Yes         No
 ↓            ↓
Answer     Abstain
              ↓
         Explain why
              ↓
         Escalate

Evaluation
- unsafe answer rate
- unsupported claim rate
- appropriate abstention
- false abstention
- escalation precision/recall
- clinician assessment
4. Triage / Emergency
Evidence status: WEAK
Đây là category mà chúng ta phải không cố nhồi evidence.
Ke có đề cập khả năng hỗ trợ preoperative triage, nhưng đây là triage trong workflow tiền phẫu, không phải emergency triage tổng quát. Ke cũng yêu cầu clinician review.    AI Healthcare Failure Analysis
Xiong không nghiên cứu emergency triage; benchmark buộc mô hình chọn đáp án và không có abstention.    AI Healthcare Failure Analysis
Vladika cũng không có emergency detection hoặc clinical escalation.    AI Healthcare Failure Analysis
Kết luận
Không được viết:
“Medical RAG hiện nay có vấn đề emergency triage.”

Cách chính xác:
Current corpus provides insufficient evidence about emergency triage as a standalone Medical RAG failure dimension.

Research Gap
Candidate gap — Low
Cần literature riêng về:
medical RAG + symptom triage + emergency detection + escalation

trước khi đưa category này lên ưu tiên cao.
5. Evidence / RAG
Đây là category mạnh nhất.
Failure
Retrieved evidence
      ≠
Relevant evidence
      ≠
Sufficient evidence
      ≠
Correctly utilized evidence

Xiong cho thấy corpus/retriever/k/context position có thể làm performance thay đổi mạnh.    AI Healthcare Failure Analysis
Ke cho thấy document extraction, diagrams, tables và chunk size ảnh hưởng retrieval.    AI Healthcare Failure Analysis
Liu review cho thấy fixed chunking có thể fragment related information; knowledge source có thể không đủ; retrieval depth cũng có thể quá ít hoặc quá nhiều.    AI Healthcare Failure Analysis
Vladika bổ sung một tầng rất quan trọng:
Question generation
        ↓
Retrieval
        ↓
Evidence
        ↓
Reasoning

và mỗi tầng có thể fail.    AI Healthcare Failure Analysis
Root Cause
Không có một “retrieval problem” duy nhất:
Document ingestion
        ↓
Chunking
        ↓
Embedding
        ↓
Query formulation
        ↓
Retrieval
        ↓
Ranking
        ↓
Context selection
        ↓
Evidence sufficiency

Existing Solutions
- BM25
- dense retrieval
- MedCPT
- MMR
- GraphRAG
- iterative retrieval
- query rewriting
- adaptive retrieval
Weakness
Vấn đề cực lớn:
literature thường đánh giá final answer hơn là từng stage.
Trong 20 studies của Liu review, chỉ 4 có explicit retrieval metrics.    AI Healthcare Failure Analysis
Research Gap
Candidate gap — VERY STRONG
Stage-wise evaluation of Medical RAG failure propagation.

Đây hiện là candidate mạnh nhất của toàn bộ nghiên cứu.
Possible Intervention
Xây một failure-analysis framework:
Query
 ↓
Retrieval
 ↓
Evidence
 ↓
Reasoning
 ↓
Generation
 ↓
Final answer

và gắn từng error về đúng stage.
Evaluation
Đây là chỗ cực kỳ quan trọng:
Retrieval
- Recall@k
- Precision@k
- MRR / NDCG
Evidence
- relevance
- sufficiency
- completeness
Generation
- faithfulness
- unsupported claims
- omission
Clinical
- correctness
- safety
- clinician assessment
6. Privacy / Data
Evidence status: INSUFFICIENT
Các paper hiện tại thường discuss privacy, nhưng ít paper thực nghiệm kiểm tra privacy.
Alkhalaf chọn Llama 2 có khả năng local deployment vì dữ liệu clinical nhạy cảm. Nhưng chính file của bạn đã phân biệt rõ:
privacy-oriented architecture ≠ privacy evaluation.

Không có experiment tương ứng về leakage, re-identification hay membership inference.    AI Healthcare Failure Analysis
Xiong cũng có khuyến nghị local deployment nhưng không phải privacy experiment.    AI Healthcare Failure Analysis
Research Gap
Candidate gap — Low
Privacy-preserving Medical RAG evaluation.

Nhưng hiện tại không nên lấy đây làm core problem.
7. Trust / Commercialization
Failure
Evidence hiện tại cho thấy correctness và user preference có thể không đồng nhất.
Almanac cho thấy grounded system factual/safety tốt hơn, nhưng physicians vẫn preferred ChatGPT 57% thời gian.    AI Healthcare Failure Analysis
Root Cause
Có thể liên quan đến:
- presentation style
- perceived helpfulness
- naturalness
- evidence presentation
Nhưng không được khẳng định causal mechanism. File đã đánh dấu giả thuyết RLHF/human-like style là author's hypothesis.    AI Healthcare Failure Analysis
Existing Solutions
Citations, explainability, clinician evaluation.
Weakness
Chưa có calibrated trust framework rõ ràng.
Research Gap
Candidate gap — Low/Moderate
Calibrated trust rather than raw user preference.

Evaluation
Không dùng:
“user thích chatbot hơn”

làm healthcare success metric.
Nên đo:
calibrated trust + correct reliance + inappropriate reliance.
8. Long-term Health Relationship
Evidence status: LOW
Đây là category yếu nhất trong corpus hiện tại.
Có longitudinal EHR retrieval ở Alkhalaf, nhưng đó là:
patient history retrieval

chứ không phải:
long-term human–AI health relationship.

Alkhalaf không có conversational memory hoặc persistent episodic memory.    AI Healthcare Failure Analysis
Các nguồn khác chủ yếu không nghiên cứu dimension này.
Vì vậy
Không nên tự tạo conclusion kiểu:
“Medical AI currently fails to maintain long-term relationships.”

Evidence chưa đủ.
Research Gap
Candidate gap — Low
Đây là một research direction tiềm năng trong tương lai, nhưng không phải lựa chọn tốt nhất từ corpus hiện tại.
D16. Tổng hợp 8 category
Category	Evidence hiện tại	Candidate-gap strength	Ưu tiên
5. Evidence / RAG	🟢 Rất mạnh	🟢 Very Strong	1
1. AI Accuracy	🟢 Rất mạnh	🟢 Strong	2
3. Conversation Safety	🟢 Khá mạnh	🟢 Strong	3
4. Triage / Emergency	🔴 Yếu	🟠 Low	7
2. Memory / Health Records	🟡 Trung bình	🟡 Moderate	4
6. Privacy / Data	🔴 Thiếu	🟠 Low	8
7. Trust / Commercialization	🟡 Ít evidence	🟠 Low–Moderate	6
8. Long-term Health Relationship	🔴 Rất ít	🔴 Low	8


D17. Từ Failure Map → 2 Core Problems
Đến đây tôi nghĩ đã đủ evidence để bắt đầu narrowing.
Tôi sẽ dùng 5 tiêu chí:
Severity
+
Evidence strength
+
Frequency across papers
+
Weakness of existing solutions
+
Research/project feasibility

CORE PROBLEM #1 — Ưu tiên cao nhất
Failure propagation across the Medical RAG pipeline
Không nên đặt tên đơn giản là:
“Retrieval failure.”

Tên chính xác hơn:
“Insufficient stage-wise diagnosis of Medical RAG failures across retrieval, evidence sufficiency, reasoning, and generation.”

Tại sao?
Vì nó được support từ nhiều hướng:
Xiong
   ↓
retrieval / corpus / k / context

Ke
   ↓
document representation / retrieval / hallucination

Liu
   ↓
retrieval + evaluation bottleneck

Almanac
   ↓
evidence sufficiency + omission

Vladika
   ↓
question → retrieval → evidence → reasoning

Alkhalaf
   ↓
correct retrieval → incomplete generation

Các source hiện tại tạo thành một chuỗi evidence rất nhất quán.    AI Healthcare Failure Analysis    AI Healthcare Failure Analysis    AI Healthcare Failure Analysis
Câu hỏi nghiên cứu tiềm năng
When a Medical RAG system fails, which stage is responsible: retrieval, evidence sufficiency, reasoning, or generation?

Đây là câu hỏi researchable hơn rất nhiều so với:
“Can we make Medical RAG more accurate?”

CORE PROBLEM #2 — Phù hợp nhất để ghép với Problem #1
Correct retrieval but incorrect/incomplete generation
Đây là nhánh failure có evidence trực tiếp rất đẹp.
Relevant evidence retrieved
             ↓
       LLM receives it
             ↓
     Wrong / incomplete answer

Alkhalaf cung cấp evidence trực tiếp về trường hợp retrieval đúng nhưng output vẫn incomplete.    AI Healthcare Failure Analysis
Vladika cho thấy evidence đúng nhưng reasoning có thể đưa ra verdict sai.    AI Healthcare Failure Analysis
Almanac cho thấy grounding cải thiện factuality nhưng không giải quyết completeness.    AI Healthcare Failure Analysis
Câu hỏi nghiên cứu tiềm năng
Why does a Medical RAG system generate an incorrect or incomplete answer even when relevant evidence has been successfully retrieved?

Đây là câu hỏi rất mạnh vì nó đi thẳng vào RAG ≠ generation control.
D18. Core research direction đang hình thành
Hai problem trên thực ra có thể ghép thành một research theme duy nhất:
             MEDICAL RAG
                  │
                  ▼
            User Query
                  │
                  ▼
            RETRIEVAL
                  │
        ┌─────────┴─────────┐
        │                   │
     Wrong/Noise         Correct
        │                   │
        ▼                   ▼
 Retrieval Failure    Evidence Sufficiency
                            │
                       sufficient?
                       /       \
                     No         Yes
                     │           │
                     ▼           ▼
                  Failure     Reasoning
                                │
                            Generation
                                │
                       correct / incorrect

Research theme
Stage-wise Failure Analysis and Evidence-Grounded Reliability of Medical RAG Systems

Đây chưa phải tên project cuối cùng.
Nó là research direction được literature hiện tại hỗ trợ tốt nhất.
D19. Một phát hiện rất đáng chú ý về hướng nghiên cứu của bạn
Ban đầu rất dễ đi theo:
“Tôi muốn xây một Medical RAG chatbot.”

Nhưng literature hiện tại lại dẫn chúng ta đến một hướng khác:
“Trước khi xây một Medical RAG tốt hơn, hãy hiểu chính xác Medical RAG thất bại ở đâu.”

Và đây là điểm khiến project có khả năng có giá trị nghiên cứu hơn một chatbot RAG thông thường.
Thay vì:
Build RAG
→ compare accuracy
→ claim better

ta có thể đi theo:
Build controlled Medical RAG
          ↓
Introduce controlled failure conditions
          ↓
Measure retrieval
          ↓
Measure evidence sufficiency
          ↓
Measure reasoning
          ↓
Measure generation
          ↓
Map failure propagation

Đây là hệ quả trực tiếp của pattern mà các paper hiện tại đang để lộ, đặc biệt là Liu, Xiong, Vladika và Alkhalaf.    AI Healthcare Failure Analysis    AI Healthcare Failure Analysis    AI Healthcare Failure Analysis
D20. Nhưng chưa được phép kết luận “đề tài đã chốt”
Có 3 việc còn phải kiểm tra trước khi biến Core Problem thành thesis/project thật:
1. Novelty check
Phải tìm thêm primary papers hỏi đúng hoặc gần đúng:
retrieval correctness → evidence sufficiency → reasoning → generation failure attribution.

Đặc biệt cần kiểm tra các hướng RAG evaluation, faithfulness, groundedness, retrieval attribution, self-reflection, medical claim verification.
2. Existing benchmark check
Phải xem đã có benchmark/framework nào làm gần giống chưa.
Nếu đã có, research phải chuyển từ:
“build a framework”

sang có thể là:
“apply/extend/evaluate framework trong Medical RAG.”

3. Feasibility check
Với điều kiện project một học kỳ và GPU hạn chế, không nên tự train LLM.
Một hướng khả thi hơn là:
Existing LLM
+
Controlled Medical RAG
+
Controlled failure injection
+
Stage-wise evaluation
+
Error taxonomy

Điều này rất phù hợp với phạm vi nghiên cứu hiện tại.
Kết luận Part D
Sau toàn bộ 9 nguồn, hai candidate core problems mạnh nhất hiện tại là:
Core Problem 1: Medical RAG failure chưa được chẩn đoán theo từng stage một cách đầy đủ và nhất quán.

Core Problem 2: Relevant evidence có thể được retrieve thành công nhưng reasoning/generation vẫn tạo ra answer sai hoặc không đầy đủ.

Và hai vấn đề này có thể hợp nhất thành một research theme:
Stage-wise Failure Analysis of Medical RAG: From Retrieval and Evidence Sufficiency to Reasoning and Generation.

Đây mới là điểm mà tôi cho rằng research của bạn đã bắt đầu chuyển từ “tìm hiểu RAG” sang “có một câu hỏi nghiên cứu thực sự”.
Bước kế tiếp hợp lý nhất không phải xây hệ thống ngay, mà là Novelty / Prior-work Verification cho 2 core problems này: tìm các primary papers gần nhất để kiểm tra liệu chúng thực sự còn khoảng trống hay đã có phương pháp giải quyết.