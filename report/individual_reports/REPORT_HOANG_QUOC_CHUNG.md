# Individual Report: Lab 3 - Chatbot vs ReAct Agent (Enhanced Version)

- **Student Name**: Hoàng Quốc Chung
- **Student ID**: 2A202600070
- **Date**: 2026-04-06

---

## I. Technical Contribution (15 Points)

Trong lab này, đóng góp chính của em tập trung vào 3 lớp kỹ thuật: orchestration agent, domain tools, và độ bền hệ thống khi chạy đa provider.

- **Modules Implemented / Improved**:
  - `src/agent/agent.py`: hoàn thiện vòng lặp ReAct, parser action/final answer, cơ chế bailout khi parse lỗi liên tiếp, telemetry event đầy đủ.
  - `src/tools/movie_booking_tools.py`: xây bộ tool đặt vé phim có logic chấm điểm recommendation, ghế ngồi, khuyến mãi.
  - `src/core/openai_provider.py`: tích hợp OpenAI và GitHub Models endpoint dựa trên loại API key.
  - `src/core/gemini_provider.py`: xử lý lỗi provider runtime theo pattern `[LLM Error] ...` để agent không văng exception thô.
  - `src/core/local_provider.py`: tối ưu tham số local inference theo hướng chạy CPU ổn định hơn.
  - `src/main.py`: thêm provider selection thực dụng cho CLI/interactive.
  - `test_scenarios.py`: tạo khung test so sánh chatbot và agent theo case có kiểm soát.

- **Code Highlights**:

```python
# src/agent/agent.py
if consecutive_parse_errors >= 3:
    logger.log_event("PARSE_ERROR_BAILOUT", {...})
    if len(content) > 20:
        return content
    return "Xin lỗi, mình gặp lỗi khi xử lý. Bạn thử lại nhé."
```

Ý nghĩa: hệ thống không chờ timeout vô ích khi model liên tục vi phạm format, giúp giảm latency tail và tránh vòng lặp vô hạn.

```python
# src/tools/movie_booking_tools.py
score = movie["hot"] * 10 - distance * 6 + time_bonus(showtime, preferred_time)
```

Ý nghĩa: recommendation không random, có thước đo định lượng rõ ràng giữa chất lượng phim, khoảng cách và khung giờ.

```python
# src/core/openai_provider.py
if self.api_key and self.api_key.startswith("github_pat_"):
    self.client = OpenAI(api_key=self.api_key, base_url="https://models.inference.ai.azure.com")
```

Ý nghĩa: dùng cùng một provider class cho 2 backend mà không phải đổi luồng phía agent.

- **Documentation (ReAct interaction)**:
  - Input người dùng đi qua `system_prompt` + `scratchpad`.
  - LLM sinh `Action` hoặc `Final Answer`.
  - Nếu `Action`, agent gọi tool và đưa `Observation` ngược lại prompt vòng sau.
  - Nếu format sai, agent tự nhắc lại ràng buộc output.
  - Nếu lỗi lặp lại vượt ngưỡng, hệ thống chủ động dừng an toàn.

---

## II. Debugging Case Study (10 Points)

### Case: Local model drift format gây sai grounding ở bước kết luận

- **Problem Description**:
  - Với provider local, model trả output lẫn nhiều block (Action + ví dụ + Final Answer) trong cùng một phản hồi.
  - Kết quả cuối chứa phim ngoài dataset nội bộ.

- **Log/Test Source**:
  - `test_results/results_local_20260406_142403.json`
  - Trong `history[0].llm_output` của mode `agent`, output chứa `The Matrix Resurrections` dù không có trong `MOVIES`.

- **Diagnosis**:
  - Prompt hiện tại giàu few-shot giúp model mạnh tuân thủ tốt, nhưng có thể làm model local nhỏ “echo” ví dụ.
  - Parser hiện đã robust ở mức parse JSON/action, nhưng chưa có lớp hậu kiểm consistency giữa `Final Answer` và `Observation`.
  - Vì vậy agent vẫn có thể trả lời “đúng format” nhưng sai dữ liệu nghiệp vụ.

- **Solution đã làm**:
  - Thêm guardrail `PARSE_ERROR_BAILOUT` để tránh loop kéo dài khi format lỗi liên tiếp.
  - Ghi telemetry chi tiết (`LLM_RESPONSE`, `JSON_PARSER_ERROR`, `TOOL_EXECUTED`) để truy nguyên gốc lỗi.

- **Solution đề xuất tiếp theo (cải tiến)**:
  - Tách profile prompt theo provider: prompt ngắn hơn cho local.
  - Chỉ chấp nhận `Final Answer` nếu entity chính (phim/rạp/suất chiếu) xuất hiện trong observation gần nhất.
  - Bổ sung schema + semantic validator trước khi trả câu trả lời cuối.

---

## III. Personal Insights: Chatbot vs ReAct (10 Points)

1. **Reasoning**:
`Thought` block tạo ra lợi thế lớn ở bài toán có thứ tự nghiệp vụ. Chatbot có thể nói trôi chảy nhưng không “thực thi” quy trình; ReAct biến nhiệm vụ thành chuỗi hành động kiểm chứng được.

2. **Reliability — Khi nào Agent tệ hơn Chatbot?**:
Agent kém chatbot ở các câu hỏi đơn giản, hoặc khi model yếu không giữ được output contract. Khi đó chi phí token cao hơn nhưng chất lượng không tăng tương ứng.

3. **Observation**:
Observation là cơ chế chống hallucination hiệu quả nhất. Nếu không dùng observation để ràng buộc bước kế tiếp, agent sẽ quay về hành vi chatbot thuần ngôn ngữ và dễ bịa dữ liệu.

Kết luận cá nhân: giá trị thật của ReAct không nằm ở “suy nghĩ nhiều hơn”, mà nằm ở vòng phản hồi với môi trường và khả năng kiểm tra được từng bước.

---

## IV. Future Improvements (5 Points)

- **Scalability**:
  - Thêm fallback runtime tự động giữa provider theo loại lỗi (`quota`, `timeout`, `provider down`).
  - Thêm cache cho tool results ngắn hạn để giảm số lần gọi lặp.

- **Safety**:
  - Dùng JSON Schema/Pydantic để kiểm tra args trước khi execute tool.
  - Thêm validator cho `Final Answer` dựa trên observation để chặn grounding lỗi.

- **Performance**:
  - Rút gọn scratchpad bằng cơ chế summarization theo bước để giảm prompt growth.
  - Phân tách prompt strategy theo từng provider thay vì dùng một prompt cố định cho mọi model.

---

> **Submitted by**: Hoàng Quốc Chung (2A202600070)
> **Date**: 2026-04-06
