# Individual Report: Lab 3 - Chatbot vs ReAct Agent

- **Student Name**: Trịnh Đức Anh
- **Student ID**: 2A202600499
- **Date**: 2026-04-06

---

## I. Technical Contribution (15 Points)

### Modules Implemented

| Module | File | Mô tả |
|:---|:---|:---|
| **LLM Provider Selection** | `src/main.py` | Hỗ trợ `--provider` CLI arg, interactive menu chọn OpenAI (GitHub Models), Gemini, Local Phi-3 |
| **OpenAI / Github Models** | `src/core/openai_provider.py` | Tích hợp thành công `gpt-4o` qua endpoint Github Models bằng Personal Access Token |
| **Agent v2 (Improved)** | `src/agent/agent.py` | Cải thiện system prompt, few-shot examples, parse error bailout |
| **Gemini Provider Fix** | `src/core/gemini_provider.py` | Fix deprecated SDK warning, model migration, error handling |
| **Local Provider Fix** | `src/core/local_provider.py` | Fix `n_ctx` memory issue gây crash context creation |
| **Metrics Enhancement** | `src/telemetry/metrics.py` | Thêm pricing cho `gemini-2.0-flash` model |
| **Unicode Fix** | `src/main.py` | Fix `UnicodeEncodeError` cho Vietnamese trên Windows |
| **Test Scenarios** | `test_scenarios.py` | Script test tự động so sánh Chatbot vs Agent |

### Code Highlights

**1. Provider Selection Interactive (main.py)**
```python
def select_provider_interactive():
    print("\n=== Chọn LLM Provider ===")
    print("  1) Google Gemini (gemini-2.0-flash) — nhanh, qua API")
    print("  2) Local Phi-3 (CPU)                — chậm, offline")
    print("  3) OpenAI / GitHub Models (gpt-4o)  — qua Azure OpenAI")
    while True:
        choice = input("\nNhập 1, 2 hoặc 3 (mặc định=3): ").strip()
        if choice == "1": return "google", "gemini-2.0-flash"
        if choice == "2": return "local", "Phi-3-mini-4k-instruct"
        if choice in {"", "3"}: return "openai", "gpt-4o"
```

**2. Agent v2 — Few-shot examples trong system prompt (agent.py)**
```python
# Thêm ví dụ cụ thể để LLM hiểu đúng format
"""
Ví dụ 1 (Tìm suất chiếu):
Thought: Người dùng muốn xem phim hành động gần Royal City, tôi cần tìm suất chiếu.
Action: recommend_showtimes({"location":"Royal City","genre":"action","seats":2,...})

Ví dụ 2 (Giữ ghế):
Thought: Đã có suất chiếu, tôi sẽ giữ ghế.
Action: hold_best_seats({"cinema_name":"CGV Vincom Royal City",...})
"""
```

**3. Consecutive Parse Error Bailout (agent.py)**
```python
consecutive_parse_errors += 1
if consecutive_parse_errors >= 3:
    logger.log_event("PARSE_ERROR_BAILOUT", {...})
    if len(content) > 20:
        return content  # Salvage useful content
    return "Xin lỗi, mình gặp lỗi khi xử lý."
```

### Documentation — Interaction với ReAct Loop

Hệ thống hoạt động theo vòng lặp ReAct:

```
User Input → System Prompt (with tools) → LLM Generate
    ↓
Parse Output → Action? → Execute Tool → Observation → Back to LLM
    ↓
Parse Output → Final Answer? → Return to User
    ↓
Parse Error? → Append error feedback to scratchpad → Retry
    ↓
Max steps? → Timeout message
```

Mỗi bước đều được log qua `IndustryLogger` với các event types: `AGENT_START`, `LLM_RESPONSE`, `TOOL_EXECUTED`, `HALLUCINATION_ERROR`, `JSON_PARSER_ERROR`, `TIMEOUT`, `AGENT_END`.

---

## II. Debugging Case Study (10 Points)

### Problem 1: `UnicodeEncodeError` — Vietnamese Characters Crash on Windows

- **Problem Description**: Agent hoạt động đúng logic nhưng crash khi `print()` kết quả tiếng Việt trên Windows console.
- **Log Source**:
```
UnicodeEncodeError: 'charmap' codec can't encode character '\u0111' in position 53: character maps to <undefined>
```
Ký tự `đ` (U+0111) và `Đ` (U+0110) không tồn tại trong bảng mã Windows cp1252.

- **Diagnosis**: Windows PowerShell mặc định dùng code page 1252 (Western European), không hỗ trợ ký tự tiếng Việt. Khi `print()` cố gắng encode chuỗi tiếng Việt → crash. Đây không phải lỗi LLM hay prompt, mà là lỗi **runtime environment**.

- **Solution**: Thêm vào đầu `main.py`:
```python
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
```

### Problem 2: `404 models/gemini-1.5-flash is not found` — Deprecated Model

- **Problem Description**: Agent gọi Gemini API thành công (API key hợp lệ) nhưng model `gemini-1.5-flash` không còn tồn tại.
- **Log Source**:
```json
{"event": "LLM_ERROR", "data": {"step": 1, "error": "[LLM Error] 404 models/gemini-1.5-flash is not found for API version v1beta"}}
```

- **Diagnosis**: Google đã deprecate model `gemini-1.5-flash` và thay bằng `gemini-2.0-flash`. Code cũ hardcode tên model cũ → 404.

- **Solution**: Dùng `genai.list_models()` để list models khả dụng, cập nhật tất cả references sang `gemini-2.0-flash`.

### Problem 3: `ValueError: Failed to create llama_context` — Local Model Memory

- **Problem Description**: Local Phi-3 model (2.4GB) load thành công nhưng fail khi tạo inference context.
- **Log Source**:
```
ValueError: Failed to create llama_context
```

- **Diagnosis**: `n_ctx=4096` yêu cầu ~162MB compute buffer. Trên một số máy có RAM hạn chế, hệ thống không cấp phát đủ memory cho context window 4096 tokens. Khi chạy với `n_ctx=2048`, model load và inference thành công (load time ~3.2s, inference ~2.6 tokens/s).

- **Solution**: Giảm `n_ctx` từ 4096 xuống 2048:
```python
def __init__(self, model_path, n_ctx=2048, ...):
```

### Problem 4: `429 Quota Exceeded` — Free Tier Rate Limiting

- **Problem Description**: Gemini API key hợp lệ nhưng free tier quota đã hết.
- **Log Source**:
```json
{"event": "LLM_ERROR", "data": {"error": "429 You exceeded your current quota... limit: 0, model: gemini-2.0-flash"}}
```

- **Diagnosis**: Free tier có giới hạn requests/day. Khi quota hết, API trả 429 với `limit: 0`. Agent detect lỗi nhờ error handling trong `GeminiProvider.generate()` và trả message phù hợp thay vì crash.

- **Solution**: Thêm error handling trong provider:
```python
try:
    response = self.model.generate_content(full_prompt)
except Exception as exc:
    return {"content": f"[LLM Error] {exc}", ...}
```

### Problem 5: `JSON_PARSER_ERROR` — LLM Hallucinated `Action` Format

- **Problem Description**: Ở các version đầu tiên (v1), hệ thống thường xuyên bị treo và báo lỗi Parse do LLM tự ý chế ra định dạng trả về thay vì xuất ra JSON chuẩn. Lỗi này trực tiếp ảnh hưởng tới **Token Efficiency** (do phải lặp lại nhiều lần tốn tokens) và **Loop count** (gây ra vòng Endless Loop không thể kết thúc để xuất ra Final Answer).
- **Log Source**:
```json
{"event": "JSON_PARSER_ERROR", "data": {"step": 2, "error": "Expecting property name enclosed in double quotes: line 1 column 2 (char 1)"}}
```

- **Diagnosis**: LLM có hiện tượng **Hallucination**, thay vì sinh ra cấu trúc `Action: tool_name({"arg": "value"})`, model (đặc biệt là bản Local Phi-3) lại sinh ra `Action: Tôi sẽ dùng tool recommend_showtimes`. Parser dùng Regex không thể extract được JSON và throw Exception. Nếu không xử lý, Agent sẽ bị "Endless Loop" mãi cho đến đoạn `TIMEOUT` (vượt `max_steps`).

- **Solution**:
  1. Cải tiến nội dung prompt (`system_prompt` v2) để làm mẫu định dạng vài example chuẩn xác (*Few-shot prompting*), giúp tiết kiệm prompt/completion tokens (tối ưu Token Efficiency) do LLM follow mượt hơn và giảm thiểu lỗi sinh chữ lạc đề.
  2. Implement cơ chế **Bailout / Guardrails**: Khi bộ đếm `consecutive_parse_errors` >= 3, hệ thống chủ động thu hồi context và ngắt vòng lặp (break loop). Điều này giữ cho Latency (thời gian trả phản hồi) ổn định dưới 2s thay vì bắt người dùng chờ vô vọng khi Parser bị fail liên tiếp.

---

## III. Personal Insights: Chatbot vs ReAct (10 Points)

### 1. Reasoning — Vai trò của `Thought` block

**Chatbot baseline** trả lời trực tiếp: khi được hỏi "Tìm phim hành động gần Royal City, 2 vé tối nay dưới 250k", chatbot chỉ có thể **mô tả** quy trình đặt vé mà không thực sự **thực hiện** nó. Chatbot bịa ra tên rạp, ghế, giá — tất cả là hallucination.

**ReAct Agent** với `Thought` block giúp agent:
- **Decompose** (phân tách) yêu cầu phức tạp thành các bước nhỏ
- **Plan** (lên kế hoạch) gọi tool nào trước, tool nào sau
- **Reflect** (phản hồi) dựa trên Observation thực tế từ tool

Ví dụ quy trình thực tế:
```
Thought: Cần tìm suất chiếu → Action: recommend_showtimes(...)
Observation: {CGV Royal City, Dune 19:00, 95k/vé}
Thought: Đã có kết quả, giữ ghế → Action: hold_best_seats(...)
Observation: {held: E5, E6, subtotal: 190k}
Thought: Áp mã giảm giá → Action: apply_best_promo(...)
Thought: Đã hoàn tất → Final Answer: Xác nhận booking
```

### 2. Reliability — Khi nào Agent tệ hơn Chatbot?

Agent perform **worse** trong các trường hợp:
- **Câu hỏi đơn giản**: "Phim là gì?" — Agent mất 3-4 bước suy luận và gọi tool không cần thiết, trong khi Chatbot trả lời ngay
- **Model yếu (Phi-3 local)**: Model 3.8B parameters thường không follow format ReAct đúng → parser error liên tục → timeout
- **Latency**: Agent cần 3-4 lần gọi LLM × latency mỗi lần → tổng latency cao hơn 3-4x so với chatbot (1 lần gọi)
- **Token cost**: Mỗi step tích lũy scratchpad dài hơn → prompt tokens tăng theo cấp số nhân

### 3. Observation — Ảnh hưởng của environment feedback

Observation là **điểm khác biệt then chốt** giữa Agent và Chatbot. Khi tool trả về dữ liệu thực:

```json
{"cinema_name": "CGV Vincom Royal City", "distance_km": 0.0, "movie_title": "Dune: Part Two", "score": 100.0}
```

Agent sử dụng observation để:
- **Grounding** (neo thực tế): Dùng đúng tên rạp, giá, ghế từ hệ thống thật
- **Decision making**: Chọn suất chiếu có score cao nhất
- **Error correction**: Nếu tool trả error → agent biết thử lại hoặc điều chỉnh

Chatbot không có observation → tất cả output đều dựa trên training data → dễ hallucinate.

---

## IV. Future Improvements (5 Points)

### Scalability
- **Async tool execution**: Dùng `asyncio` để gọi nhiều tools song song (ví dụ: kiểm tra giá + kiểm tra ghế cùng lúc)
- **Caching layer**: Cache kết quả tool (suất chiếu thường không đổi trong 5 phút) để giảm latency
- **Load balancing**: Fallback chain: Gemini → OpenAI → Local, tự động chuyển khi quota/rate limit

### Safety
- **Supervisor LLM**: Dùng một LLM nhỏ hơn để audit action trước khi execute (kiểm tra args hợp lệ, prevent injection)
- **Guardrails**: Max 6 steps (đã implement), budget cap, user confirmation trước khi thanh toán thật
- **Input sanitization**: Validate JSON args trước khi pass vào tool function

### Performance
- **Vector DB cho tool retrieval**: Khi có 50+ tools, dùng embedding similarity để chọn 3-5 tools phù hợp nhất cho mỗi query thay vì list tất cả trong system prompt
- **Streaming response**: Dùng `stream()` method để hiển thị Thought process realtime
- **Model optimization**: Quantize model nhỏ hơn (Q2_K) hoặc dùng ONNX runtime cho local inference nhanh hơn

---

> **Submitted by**: Dương Văn Hiệp (2A202600052)
> **Date**: 2026-04-06
