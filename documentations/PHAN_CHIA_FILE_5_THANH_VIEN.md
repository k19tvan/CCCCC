# Phân chia chính xác file cho 5 thành viên và 5 feature branch

> Lập ngày **05/11/2026**, dựa trên file `D:\#NAM_3\SE373\DOAN\ke_hoach_task_va_cau_truc_code_issue_agent.md` và code thực tế trong `nhom-20/`.
>
> **Mục đích:** chia bản code đã có thành 5 phần để mỗi người commit/push trên một feature branch; sau khi merge đủ 5 phần, nội dung code phải bằng bản nguồn. Đây là phân công trách nhiệm bàn giao, không phải kết luận ai đã viết từng dòng code trong lịch sử.
>
> Kế hoạch nguồn dùng tên **Người 1–5**, chưa có họ tên thật. Tài liệu này giữ nguyên các số đó. Kế hoạch nguồn là **Phase 1**; không tự lấy các task Phase 2 để phân lại code hiện tại.

## 1. Kết luận cần đọc trước khi giao cho AI

**“Lọc file” nghĩa là chỉ đưa file thuộc người đó vào commit/PR. Không xóa file của người khác khỏi bản code đầy đủ rồi commit việc xóa.**

Git merge theo thay đổi so với commit chung, không ghép các thư mục như thao tác copy. Nếu tạo cả 5 nhánh từ bản code đầy đủ rồi mỗi người xóa file ngoài phần mình, các commit xóa có thể làm mất code khi merge.

Với trạng thái repo đã kiểm tra, cách tách phù hợp là:

1. Giữ bản code đầy đủ ở commit nguồn `5d3f73a0c0b9906c9b9abc509144446bd908aaf4`.
2. Tạo **cả 5 feature branch từ cùng commit nền** `6f8427a9eaa85060c7791b07eebd6391d45e97a9`.
3. Trên từng branch, chỉ lấy đúng đường dẫn thuộc người đó từ commit nguồn; giữ nguyên vị trí tương đối và nội dung.
4. File đã có ở commit nền vẫn được giữ nguyên nếu không thuộc người đó. Ví dụ `README.md` có sẵn trên nền: Người 2–5 vẫn thấy nó nhưng không sửa/commit nó.
5. Kiểm tra bản ghép đủ 5 phần, rồi mở 5 PR về `main`. Không merge thẳng `dev` vào `main` để thay thế quá trình chia 5 phần.

**Không hứa từng feature branch chạy độc lập.** Các module đang import lẫn nhau, nên một nhánh chỉ có phần của một người có thể chưa import/chạy test được. Bản ghép đủ 5 phần phải được kiểm thử. Việc giữ cấu trúc thư mục không tự giải quyết các dependency còn thiếu trên nhánh riêng.

## 2. Mốc dữ liệu đã đối chiếu

| Hạng mục | Giá trị tại lúc lập tài liệu |
|---|---|
| Thư mục nguồn | `D:\#NAM_3\SE373\DOAN\nhom-20` |
| Branch đang mở | `dev` |
| HEAD / bản `origin/dev` lưu tại máy | `5d3f73a0c0b9906c9b9abc509144446bd908aaf4` |
| Bản `origin/main` lưu tại máy | `6f8427a9eaa85060c7791b07eebd6391d45e97a9` |
| Nội dung commit nền | Chỉ có `README.md` |
| File đang được Git theo dõi | **114 file** |
| File có thay đổi/chưa theo dõi trước khi tạo hướng dẫn này | **0** |
| File bỏ qua bởi Git tại lúc quét | **1.019**: 64 cache Python, 6 file `frontend/dist/`, 949 file `frontend/node_modules/` |
| File hướng dẫn mới | `PHAN_CHIA_FILE_5_THANH_VIEN.md`, giao **Người 1** |

Thông tin `origin/*` trên đây là remote-tracking ref ở máy, **chưa phải kết quả fetch mới từ server**. Trước khi tạo nhánh, fetch rồi kiểm tra lại `origin/main`. Nếu commit đã khác mốc nền, dừng quy trình tự động để tính lại phần chênh lệch; không reset/force-push `main` về mốc cũ.

Tổng bàn giao: **114 file nguồn + 1 file hướng dẫn = 115 file**. File hướng dẫn mới không nằm trong commit nguồn nên được copy riêng vào nhánh Người 1.

## 3. Bảng phân công và ranh giới trách nhiệm

| Thành viên | Feature branch | File nguồn | Nhiệm vụ |
|---|---|---:|---|
| Người 1 | `feature/foundation` | 30 | Config, contract, trace chung, che secret, orchestrator, package backend, tài liệu tổng và fixture dùng chung |
| Người 2 | `feature/agent-llm` | 11 | LLM client/FakeLLM, issue agent graph, prompt, retry, conversation agent, trace tool trong agent |
| Người 3 | `feature/tools-skill` | 21 | Registry, list/read/search, gather_context, repo fixture, test và tài liệu giao diện tool |
| Người 4 | `feature/validation` | 9 | Quy tắc ticket, schema, affected files, route hợp lệ, validator hook |
| Người 5 | `feature/app-evals` | 43 | CLI/API, React frontend, bộ eval, demo, dữ liệu mẫu, test ứng dụng và end-to-end |

Người 1 nhận thêm file hướng dẫn này, nên nhánh Người 1 có **31 đường dẫn bàn giao**. Số file không phản ánh độ khó/công sức; phân chia theo nhiệm vụ trong kế hoạch gốc.

### 3.1. Đối chiếu tên trong kế hoạch với code thật

| Tên/phần trong kế hoạch | Vị trí thực tế và owner duy nhất |
|---|---|
| `src/issue_agent/contracts.py`, `config.py`, `logging_config.py` | `backend/src/agent_system/core/` tương ứng — Người 1 |
| `orchestrator.py` ghi phối hợp Người 1 + 2 | Toàn bộ `backend/src/agent_system/orchestrator/` — **Người 1**; Người 2 góp ý nhưng không commit cùng file |
| `agent/loop.py` | `backend/src/agent_system/subagents/issue_agent/graph.py` — Người 2 |
| `llm/client.py` | `backend/src/agent_system/core/llm/client.py` — Người 2 |
| `tools/`, `skills/gather_context.py` | Nằm dưới `backend/src/agent_system/subagents/issue_agent/` — Người 3 |
| `validation/affected_files.py` | Chức năng nằm ở `backend/src/agent_system/subagents/issue_agent/validation/path_guard.py` — Người 4 |
| `app/` | `backend/src/agent_system/interfaces/` và `frontend/` — Người 5 |
| `evals/`, `tests/`, `examples/` | Nằm dưới `backend/`; owner xác định theo từng file ở manifest |

Không tạo thêm thư mục theo tên cũ chỉ để giống bản kế hoạch.

### 3.2. Các file dễ bị chia trùng — chốt một người duy nhất

| File/nhóm file | Owner | Cách xử lý |
|---|---|---|
| `backend/src/agent_system/core/contracts.py` | Người 1 | Tất cả được đọc/import; chỉ Người 1 bàn giao/sửa schema chung |
| `backend/src/agent_system/orchestrator/router.py` | Người 1 | Chọn issue agent hay conversation agent; khác routing của ticket |
| `backend/src/agent_system/orchestrator/offline.py` | Người 1 | Fake assistant tổng cho orchestration; giữ cùng nhóm orchestrator |
| `backend/src/agent_system/subagents/issue_agent/offline_llm.py` | Người 2 | Fake triage LLM bên trong issue agent |
| `backend/src/agent_system/subagents/conversation_agent/agent.py` | Người 2 | Agent hội thoại dùng LLM, không phải giao diện frontend |
| `backend/src/agent_system/subagents/issue_agent/ticket_rules.py` | Người 4 | Kiểm tra liên trường và thông điệp route; dù nằm ngoài thư mục validation vẫn thuộc Người 4 |
| `backend/src/agent_system/subagents/issue_agent/tracing.py` | Người 2 | Proxy trace các tool call trong vòng agent; không phải owner của implementation tool |
| `backend/src/agent_system/core/trace.py` và `core/redaction.py` | Người 1 | Tiện ích dùng chung của nhiều module |
| `backend/tests/conftest.py` | Người 1 | Fixture chung, có import registry của Người 3 |
| Các file `backend/tests/fixtures/sample_repo/` | Người 3 | Repo dữ liệu để tools/eval dùng; mọi file được liệt kê riêng bên dưới |
| `backend/examples/test_search_code.py` | Người 3 | Script thử riêng search tool; ngoại lệ trong thư mục examples |
| `backend/examples/demo_run.py` | Người 5 | Demo tool/skill trên sample project; không tự coi đây là demo toàn pipeline |
| `docs/decisions/01_tool_interface_agreement.md` | Người 3 | ADR về tool interface; Người 1 và 2 phối hợp review |
| `docs/evidence/live_eval_2026-10-27.csv` | Người 5 | Evidence đã được chọn để lưu; khác báo cáo tự sinh trong evals/reports |
| `backend/pyproject.toml` và `backend/uv.lock` | Người 1 | Dependency/package backend; đi cùng nhau |
| `frontend/package.json` và `frontend/package-lock.json` | Người 5 | Dependency frontend; đi cùng nhau |
| `backend/.env_example` | Người 1 | Đây là **tên thật**, có dấu gạch dưới; không đổi thành `.env.example` trong đợt tách |
| `frontend/.env.example` | Người 5 | Template cấu hình frontend, giữ đúng tên |
| `docs/2026-10-06_de_tai_01_coding_agent.md` và bản ở `documentations/` | Người 1 | Giữ cả hai file đang được quản lý; không tự gộp/xóa bản trùng |
| Từng file `__init__.py` | Theo manifest | File package có owner cụ thể; không thêm bản sao vào cả 5 nhánh |

**Khoảng lệch đã thấy:** README hướng dẫn copy `backend/.env.example` nhưng tên tracked hiện là `backend/.env_example`. Repo cũng chưa có `.github/workflows/ci.yml`, `requirements.txt` hay `docs/architecture.md` như cấu trúc đề xuất. Đây không phải file bị bỏ sót trong danh sách. Người 1 xử lý việc bổ sung/đổi tên bằng thay đổi riêng sau khi tách; không tự tạo file giả trong đợt sao chép này.

## 4. Manifest chính thức — danh sách chính xác từng file

Quy ước cho AI/script:

- Các đường dẫn trong `owners["N"].files` tính từ gốc `nhom-20/`, dùng dấu `/`.
- Danh sách là **allowlist chính xác**, không phải ví dụ. Không suy rộng quyền sở hữu cả thư mục từ một file.
- Mỗi file nguồn xuất hiện **đúng một lần**. Không có owner “chung”, “1 + 2”, hoặc “tùy người”.
- `guide_file` là file mới duy nhất ngoài 114 file nguồn; chỉ `guide_owner` commit file đó.
- Không tự gán file mới chưa có trong manifest. Báo đường dẫn để cập nhật phân công trước.
- Nếu danh sách trong mô tả và manifest bị lệch sau này, dừng để đối chiếu; không tự chọn theo phỏng đoán.

<!-- OWNERSHIP_MANIFEST_START -->
```json
{
  "schema_version": 1,
  "snapshot_date": "2026-11-05",
  "repo_root": "nhom-20",
  "source_commit": "5d3f73a0c0b9906c9b9abc509144446bd908aaf4",
  "base_commit": "6f8427a9eaa85060c7791b07eebd6391d45e97a9",
  "tracked_file_count": 114,
  "guide_file": "PHAN_CHIA_FILE_5_THANH_VIEN.md",
  "guide_owner": "1",
  "owners": {
    "1": {
      "name": "Người 1 — Nền tảng, contract và tích hợp",
      "branch": "feature/foundation",
      "file_count": 30,
      "files": [
        ".gitignore",
        "README.md",
        "backend/.env_example",
        "backend/README.md",
        "backend/__init__.py",
        "backend/pyproject.toml",
        "backend/src/agent_system/__init__.py",
        "backend/src/agent_system/core/__init__.py",
        "backend/src/agent_system/core/config.py",
        "backend/src/agent_system/core/contracts.py",
        "backend/src/agent_system/core/logging_config.py",
        "backend/src/agent_system/core/redaction.py",
        "backend/src/agent_system/core/trace.py",
        "backend/src/agent_system/orchestrator/__init__.py",
        "backend/src/agent_system/orchestrator/graph.py",
        "backend/src/agent_system/orchestrator/offline.py",
        "backend/src/agent_system/orchestrator/process.py",
        "backend/src/agent_system/orchestrator/router.py",
        "backend/src/agent_system/subagents/__init__.py",
        "backend/src/agent_system/subagents/base.py",
        "backend/tests/__init__.py",
        "backend/tests/conftest.py",
        "backend/tests/unit/__init__.py",
        "backend/tests/unit/test_chat_orchestrator.py",
        "backend/tests/unit/test_orchestrator.py",
        "backend/tests/unit/test_redaction.py",
        "backend/uv.lock",
        "docs/2026-10-06_de_tai_01_coding_agent.md",
        "documentations/2026-10-06_de_tai_01_coding_agent.md",
        "figures/phase1_pipeline.jpg"
      ]
    },
    "2": {
      "name": "Người 2 — LLM và agent",
      "branch": "feature/agent-llm",
      "file_count": 11,
      "files": [
        "backend/src/agent_system/core/llm/__init__.py",
        "backend/src/agent_system/core/llm/client.py",
        "backend/src/agent_system/subagents/conversation_agent/__init__.py",
        "backend/src/agent_system/subagents/conversation_agent/agent.py",
        "backend/src/agent_system/subagents/issue_agent/__init__.py",
        "backend/src/agent_system/subagents/issue_agent/graph.py",
        "backend/src/agent_system/subagents/issue_agent/offline_llm.py",
        "backend/src/agent_system/subagents/issue_agent/prompts.py",
        "backend/src/agent_system/subagents/issue_agent/tracing.py",
        "backend/tests/unit/test_issue_agent.py",
        "backend/tests/unit/test_llm_client.py"
      ]
    },
    "3": {
      "name": "Người 3 — Repository tools và skill",
      "branch": "feature/tools-skill",
      "file_count": 21,
      "files": [
        "backend/examples/test_search_code.py",
        "backend/src/agent_system/subagents/issue_agent/skills/__init__.py",
        "backend/src/agent_system/subagents/issue_agent/skills/gather_context.py",
        "backend/src/agent_system/subagents/issue_agent/tools/__init__.py",
        "backend/src/agent_system/subagents/issue_agent/tools/list_files.py",
        "backend/src/agent_system/subagents/issue_agent/tools/read_file.py",
        "backend/src/agent_system/subagents/issue_agent/tools/registry.py",
        "backend/src/agent_system/subagents/issue_agent/tools/search_code.py",
        "backend/tests/fixtures/sample_repo/README.md",
        "backend/tests/fixtures/sample_repo/__init__.py",
        "backend/tests/fixtures/sample_repo/calculator.py",
        "backend/tests/fixtures/sample_repo/subdir/deep_module.py",
        "backend/tests/fixtures/sample_repo/user_service.py",
        "backend/tests/fixtures/sample_repo/utils/__init__.py",
        "backend/tests/fixtures/sample_repo/utils/helpers.py",
        "backend/tests/unit/test_gather_context.py",
        "backend/tests/unit/test_list_files.py",
        "backend/tests/unit/test_read_file.py",
        "backend/tests/unit/test_registry.py",
        "backend/tests/unit/test_search_code.py",
        "docs/decisions/01_tool_interface_agreement.md"
      ]
    },
    "4": {
      "name": "Người 4 — Validation và routing của ticket",
      "branch": "feature/validation",
      "file_count": 9,
      "files": [
        "backend/src/agent_system/subagents/issue_agent/ticket_rules.py",
        "backend/src/agent_system/subagents/issue_agent/validation/__init__.py",
        "backend/src/agent_system/subagents/issue_agent/validation/issue_schema.py",
        "backend/src/agent_system/subagents/issue_agent/validation/path_guard.py",
        "backend/src/agent_system/subagents/issue_agent/validation/route_guard.py",
        "backend/src/agent_system/subagents/issue_agent/validation/routing.py",
        "backend/src/agent_system/subagents/issue_agent/validation/schema_guard.py",
        "backend/src/agent_system/subagents/issue_agent/validation/ticket_guard.py",
        "backend/tests/unit/test_validation.py"
      ]
    },
    "5": {
      "name": "Người 5 — App, frontend, demo và eval",
      "branch": "feature/app-evals",
      "file_count": 43,
      "files": [
        "backend/evals/README.md",
        "backend/evals/__init__.py",
        "backend/evals/cases.jsonl",
        "backend/evals/evaluator.py",
        "backend/evals/faults.py",
        "backend/evals/run_evals.py",
        "backend/evals/runner.py",
        "backend/examples/demo_run.py",
        "backend/examples/sample_issues.json",
        "backend/examples/sample_project/README.md",
        "backend/examples/sample_project/config.py",
        "backend/examples/sample_project/models/order.py",
        "backend/examples/sample_project/services/order_service.py",
        "backend/examples/sample_project/services/payment_service.py",
        "backend/src/agent_system/interfaces/__init__.py",
        "backend/src/agent_system/interfaces/adapter.py",
        "backend/src/agent_system/interfaces/cli.py",
        "backend/src/agent_system/interfaces/server.py",
        "backend/tests/integration/__init__.py",
        "backend/tests/integration/test_end_to_end.py",
        "backend/tests/unit/test_cli.py",
        "backend/tests/unit/test_evals.py",
        "backend/tests/unit/test_server.py",
        "docs/evidence/live_eval_2026-10-27.csv",
        "frontend/.env.example",
        "frontend/README.md",
        "frontend/index.html",
        "frontend/package-lock.json",
        "frontend/package.json",
        "frontend/src/App.tsx",
        "frontend/src/api.ts",
        "frontend/src/components/ChatView.tsx",
        "frontend/src/components/Sidebar.tsx",
        "frontend/src/components/TicketCard.tsx",
        "frontend/src/components/ui.tsx",
        "frontend/src/fonts/exo2-latin-ext.woff2",
        "frontend/src/fonts/exo2-latin.woff2",
        "frontend/src/fonts/exo2-vietnamese.woff2",
        "frontend/src/main.tsx",
        "frontend/src/styles.css",
        "frontend/src/types.ts",
        "frontend/tsconfig.json",
        "frontend/vite.config.ts"
      ]
    }
  }
}
```
<!-- OWNERSHIP_MANIFEST_END -->

## 5. File được dùng chung, file không đưa lên nhánh

**Quyền đọc/import không đồng nghĩa quyền commit.** Người 2 cần contract Người 1, tools Người 3 và rules Người 4; Người 5 cần backend của các người khác. Có thể đọc/test trên bản đầy đủ hoặc bản tích hợp. Không copy thêm dependency của người khác vào commit của mình.

| Loại | Xử lý khi tách |
|---|---|
| `.git/` | Metadata của Git; không copy thủ công, không xóa |
| `__pycache__/`, `*.pyc`, `.pytest_cache/`, coverage | Không đưa vào feature commit |
| `frontend/node_modules/`, `frontend/dist/` | Không đưa vào feature commit; cài/build lại từ source và lockfile |
| `.venv/`, `venv/`, `*.egg-info/`, `build/` | Không đưa vào feature commit |
| `.env`, file secret/config cá nhân | Không đưa vào commit; không dùng để thay template |
| `backend/evals/reports/` | Báo cáo phát sinh, không đưa vào commit tách; evidence được duyệt đã có đường dẫn riêng trong manifest |
| File không có trong manifest và không thuộc nhóm bỏ qua | Báo rõ đường dẫn; không tự xóa hoặc phân cho người bất kỳ |

Giữ các font `.woff2`, hình pipeline `.jpg`, lockfile và CSV evidence trong manifest: đây là tài nguyên dự án cần bàn giao, dù không phải source code dạng text.

Git chỉ lưu file, không lưu thư mục rỗng. Thư mục cha sẽ được dựng theo các đường dẫn file; không cần thêm `.gitkeep` ngoài manifest.
