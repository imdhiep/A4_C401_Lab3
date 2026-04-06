import ast
import json
import re
import time
from typing import Any, Dict, List, Optional

from src.core.llm_provider import LLMProvider
from src.telemetry.logger import logger
from src.telemetry.metrics import tracker


class ReActAgent:
    """A ReAct-style agent with telemetry, error tracking, and tool execution."""

    def __init__(self, llm: LLMProvider, tools: List[Dict[str, Any]], max_steps: int = 5):
        self.llm = llm
        self.tools = tools
        self.max_steps = max_steps
        self.history: List[Dict[str, Any]] = []
        self.tool_map = {tool["name"]: tool["func"] for tool in tools}

    def get_system_prompt(self) -> str:
        tool_descriptions = "\n".join(
            [f"- {tool['name']}: {tool['description']}" for tool in self.tools]
        )

        return f"""
        Bạn là AI agent đặt vé xem phim tại Việt Nam theo phong cách ReAct.

        Mục tiêu:
        1. Hiểu yêu cầu người dùng.
        2. Dùng tool để tìm suất chiếu phù hợp.
        3. Nếu đã có lựa chọn tốt, giữ ghế đẹp.
        4. Nếu đã có tổng tiền, áp mã giảm giá.
        5. Kết thúc bằng câu trả lời xác nhận thanh toán.

        Danh sách tools:
        {tool_descriptions}

        Bạn chỉ được trả về đúng 1 trong 2 dạng sau:

        Thought: <một câu suy nghĩ ngắn>
        Action: tool_name({{"arg1":"value","arg2":123}})

        hoặc

        Thought: <một câu suy nghĩ ngắn>
        Final Answer: <câu trả lời cuối bằng tiếng Việt>

        Ví dụ 1 (Tìm suất chiếu):
        Thought: Người dùng muốn xem phim hành động gần Royal City, tôi cần tìm suất chiếu.
        Action: recommend_showtimes({{"location":"Royal City","genre":"action","seats":2,"budget_k":250,"preferred_time":"evening","max_results":5}})

        Ví dụ 2 (Giữ ghế):
        Thought: Đã có suất chiếu Captain America tại CGV Royal City lúc 19:00, tôi sẽ giữ ghế.
        Action: hold_best_seats({{"cinema_name":"CGV Vincom Royal City","movie_title":"Captain America: Brave New World","showtime":"19:00","seats":2,"price_per_seat_k":95,"preference":"center"}})

        Ví dụ 3 (Áp mã giảm giá):
        Thought: Đã giữ ghế thành công, tổng 190k. Tôi sẽ áp mã giảm giá.
        Action: apply_best_promo({{"total_vnd":190000,"is_student":false,"is_member":true,"payment_method":"momo"}})

        Ví dụ 4 (Kết thúc):
        Thought: Đã hoàn tất tất cả bước. Tôi sẽ tóm tắt kết quả.
        Final Answer: Tôi đã chọn 2 ghế E5, E6 tại CGV Vincom Royal City, phim Captain America lúc 19:00. Tổng sau giảm giá MEMBER10: 171,000đ. Bạn có đồng ý thanh toán không?

        Quy tắc bắt buộc:
        - Chỉ gọi đúng tool có trong danh sách.
        - Args trong Action phải là JSON object hợp lệ.
        - Mỗi lần chỉ gọi 1 tool.
        - Không bịa Observation.
        - Khi đã có đủ dữ liệu thì phải trả Final Answer, không lặp vô hạn.
        - Với bài toán đặt vé phim, quy trình thường là:
        recommend_showtimes -> hold_best_seats -> apply_best_promo -> Final Answer
        """.strip()

    def _build_prompt(self, user_input: str, scratchpad: str) -> str:
        if scratchpad.strip():
            return f"""
            Yêu cầu người dùng:
            {user_input}

            Lịch sử suy luận:
            {scratchpad}

            Hãy tạo bước tiếp theo.
            Chỉ trả về Thought + Action hoặc Thought + Final Answer.
            """.strip()

        return f"""
        Yêu cầu người dùng:
        {user_input}

        Hãy bắt đầu với bước hợp lý nhất.
        Chỉ trả về Thought + Action hoặc Thought + Final Answer.
        """.strip()

    def run(self, user_input: str) -> str:
        # Clear history for new session
        self.history = []

        logger.log_event("AGENT_START", {
            "input": user_input,
            "model": self.llm.model_name,
            "max_steps": self.max_steps,
            "available_tools": list(self.tool_map.keys()),
        })

        scratchpad = ""
        started_at = time.perf_counter()
        consecutive_parse_errors = 0

        for step in range(1, self.max_steps + 1):
            prompt = self._build_prompt(user_input, scratchpad)
            result = self.llm.generate(prompt, system_prompt=self.get_system_prompt())

            tracker.track_request(
                provider=result.get("provider", "unknown"),
                model=self.llm.model_name,
                usage=result.get("usage", {}),
                latency_ms=result.get("latency_ms", 0),
            )

            content = (result.get("content") or "").strip()
            logger.log_event("LLM_RESPONSE", {
                "step": step,
                "content": content,
                "usage": result.get("usage", {}),
                "latency_ms": result.get("latency_ms", 0),
                "provider": result.get("provider", "unknown"),
            })

            self.history.append({
                "step": step,
                "llm_output": content,
            })

            # Check for LLM errors
            if content.startswith("[LLM Error]"):
                logger.log_event("LLM_ERROR", {"step": step, "error": content})
                return f"Lỗi hệ thống: {content}"

            action = self._parse_action(content)
            if action:
                consecutive_parse_errors = 0  # Reset on successful parse
                tool_name = action["tool"]
                if tool_name not in self.tool_map:
                    logger.log_event("HALLUCINATION_ERROR", {
                        "step": step,
                        "tool": tool_name,
                        "content": content,
                    })
                    scratchpad += (
                        f"\n{content}\n"
                        f"Observation: Tool '{tool_name}' không tồn tại. "
                        "Hãy chỉ dùng tool trong danh sách.\n"
                    )
                    continue

                tool_started = time.perf_counter()
                observation = self._execute_tool(tool_name, action.get("args", {}))
                tool_latency_ms = int((time.perf_counter() - tool_started) * 1000)

                logger.log_event("TOOL_EXECUTED", {
                    "step": step,
                    "tool": tool_name,
                    "args": action.get("args", {}),
                    "tool_latency_ms": tool_latency_ms,
                    "observation": observation,
                })

                scratchpad += f"\n{content}\nObservation: {observation}\n"
                continue

            final_answer = self._parse_final_answer(content)
            if final_answer:
                total_duration_ms = int((time.perf_counter() - started_at) * 1000)
                logger.log_event("AGENT_END", {
                    "status": "success",
                    "steps": step,
                    "total_duration_ms": total_duration_ms,
                    "final_answer": final_answer,
                })
                return final_answer

            # Parse error — track consecutive failures
            consecutive_parse_errors += 1
            logger.log_event("JSON_PARSER_ERROR", {
                "step": step,
                "content": content,
                "consecutive_errors": consecutive_parse_errors,
            })

            if consecutive_parse_errors >= 3:
                logger.log_event("PARSE_ERROR_BAILOUT", {
                    "step": step,
                    "consecutive_errors": consecutive_parse_errors,
                })
                # Try to salvage a useful answer from the content
                if len(content) > 20:
                    return content
                return "Xin lỗi, mình gặp lỗi khi xử lý. Bạn thử lại nhé."

            scratchpad += (
                f"\n{content}\n"
                "Observation: Format lỗi. Bạn phải trả về đúng Thought + Action "
                "hoặc Thought + Final Answer.\n"
            )

        total_duration_ms = int((time.perf_counter() - started_at) * 1000)
        logger.log_event("TIMEOUT", {
            "steps": self.max_steps,
            "total_duration_ms": total_duration_ms,
            "history": self.history,
        })
        return (
            "Mình đã dừng do vượt quá số bước suy luận. "
            "Bạn thử nhập ngắn gọn hơn, ví dụ: "
            "'Tìm phim hành động gần Royal City, 2 vé tối nay dưới 250k'."
        )

    def _parse_final_answer(self, text: str) -> Optional[str]:
        match = re.search(r"Final Answer\s*:\s*(.*)", text, flags=re.IGNORECASE | re.DOTALL)
        if not match:
            return None
        return match.group(1).strip()

    def _parse_action(self, text: str) -> Optional[Dict[str, Any]]:
        action_match = re.search(r"Action\s*:\s*(.*)", text, flags=re.IGNORECASE | re.DOTALL)
        if not action_match:
            return None

        action_text = action_match.group(1).strip()

        # Handle case where action text might contain Final Answer (multi-line capture)
        # Only take text up to first newline that starts with a keyword
        lines = action_text.split("\n")
        clean_lines = []
        for line in lines:
            stripped = line.strip()
            if stripped.lower().startswith("final answer") or stripped.lower().startswith("observation"):
                break
            clean_lines.append(line)
        action_text = "\n".join(clean_lines).strip()

        json_blob = self._extract_balanced_json(action_text)
        if action_text.startswith("{") and json_blob:
            payload = self._safe_load_mapping(json_blob)
            if payload and "tool" in payload:
                return {"tool": payload["tool"], "args": payload.get("args", {})}

        call_match = re.match(r"([a-zA-Z_][a-zA-Z0-9_]*)\s*\((.*)\)\s*$", action_text, flags=re.DOTALL)
        if not call_match:
            return None

        tool_name = call_match.group(1).strip()
        raw_args = call_match.group(2).strip()

        if not raw_args:
            return {"tool": tool_name, "args": {}}

        if raw_args.startswith("{"):
            args_blob = self._extract_balanced_json(raw_args)
            if not args_blob:
                return None
            parsed_args = self._safe_load_mapping(args_blob)
            if parsed_args is None:
                return None
            return {"tool": tool_name, "args": parsed_args}

        return None

    def _extract_balanced_json(self, text: str) -> Optional[str]:
        depth = 0
        in_string = False
        escape = False

        for index, char in enumerate(text):
            if char == '"' and not escape:
                in_string = not in_string

            if char == "\\" and not escape:
                escape = True
            else:
                escape = False

            if in_string:
                continue

            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    return text[:index + 1]

        return None

    def _safe_load_mapping(self, text: str) -> Optional[Dict[str, Any]]:
        try:
            payload = json.loads(text)
        except Exception:
            try:
                payload = ast.literal_eval(text)
            except Exception:
                return None

        if not isinstance(payload, dict):
            return None
        return payload

    def _execute_tool(self, tool_name: str, args: Dict[str, Any]) -> str:
        tool_fn = self.tool_map.get(tool_name)
        if tool_fn is None:
            return json.dumps({"error": f"Tool {tool_name} not found."}, ensure_ascii=False)

        try:
            result = tool_fn(**args)
            return json.dumps(result, ensure_ascii=False)
        except TypeError as exc:
            logger.log_event("TOOL_ARGUMENT_ERROR", {
                "tool": tool_name,
                "args": args,
                "error": str(exc),
            })
            return json.dumps({"error": str(exc)}, ensure_ascii=False)
        except Exception as exc:
            logger.error(f"Tool execution failed: {tool_name} - {exc}", exc_info=True)
            return json.dumps({"error": str(exc)}, ensure_ascii=False)