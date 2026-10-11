import gradio as gr
import spaces
import requests

from agent import GaiaAgent
from gaia_tools import (
    reverse_string,
    sum_spreadsheet_column,
)


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PROVIDER = "huggingface_hub"

GAIA_API = "https://agents-course-unit4-scoring.hf.space"


# ============================================================
# GPU TEST
# ============================================================

@spaces.GPU
def gpu_test():
    return "✓ GPU function detected successfully."


# ============================================================
# MODEL ROUTER TEST
# ============================================================

def test_router(question):
    """Test question type detection."""
    
    try:
        from agent import detect_question_type, GEMINI_MODELS
        
        if not question or not question.strip():
            return "ERROR: Enter a question."
        
        question_type = detect_question_type(question)
        
        result = (
            f"Question Type: {question_type}\n"
            f"Models: {GEMINI_MODELS}"
        )
        
        return result
    
    except Exception as e:
        return f"ERROR: {e}"


# ============================================================
# GET GAIA QUESTIONS
# ============================================================

def get_gaia_questions():
    """Fetch GAIA questions from the API."""
    
    try:
        response = requests.get(
            f"{GAIA_API}/questions",
            timeout=30
        )
        response.raise_for_status()
        return response.json()
    
    except Exception as e:
        raise RuntimeError(f"Failed to fetch GAIA questions: {e}")


# ============================================================
# FIND TASK
# ============================================================

def find_task(task_id):
    """Find a specific GAIA task by ID."""
    
    try:
        questions = get_gaia_questions()

        # Case 1: List of tasks
        if isinstance(questions, list):
            for task in questions:
                if not isinstance(task, dict):
                    continue

                current_id = (
                    task.get("task_id")
                    or task.get("id")
                )

                if str(current_id) == str(task_id):
                    question = (
                        task.get("Question")
                        or task.get("question")
                        or ""
                    )
                    file_name = (
                        task.get("file_name")
                        or task.get("filename")
                        or task.get("file")
                        or ""
                    )
                    return question, file_name

        # Case 2: Dictionary of tasks
        if isinstance(questions, dict):
            # Subcase: Direct key lookup
            if task_id in questions:
                task = questions[task_id]
                if isinstance(task, dict):
                    question = (
                        task.get("Question")
                        or task.get("question")
                        or ""
                    )
                    file_name = (
                        task.get("file_name")
                        or task.get("filename")
                        or task.get("file")
                        or ""
                    )
                    return question, file_name

            # Subcase: Dictionary containing a list
            for key in ["questions", "data", "tasks", "results"]:
                items = questions.get(key)

                if not isinstance(items, list):
                    continue

                for task in items:
                    if not isinstance(task, dict):
                        continue

                    current_id = (
                        task.get("task_id")
                        or task.get("id")
                    )

                    if str(current_id) == str(task_id):
                        question = (
                            task.get("Question")
                            or task.get("question")
                            or ""
                        )
                        file_name = (
                            task.get("file_name")
                            or task.get("filename")
                            or task.get("file")
                            or ""
                        )
                        return question, file_name

        return "", ""

    except Exception as e:
        raise RuntimeError(f"Could not retrieve GAIA task: {e}")


# ============================================================
# LOAD TASK AUTOMATICALLY
# ============================================================

def load_task(task_id):
    """Load a GAIA task and display its question."""
    
    if not task_id or not task_id.strip():
        return (
            "",
            "",
            "❌ ERROR: Enter a GAIA Task ID."
        )

    try:
        question, file_name = find_task(
            task_id.strip()
        )

        if not question:
            return (
                "",
                "",
                "❌ ERROR: Task ID was not found in the GAIA API."
            )

        if file_name:
            status = (
                "✓ Task loaded successfully\n"
                f"Attachment: {file_name}"
            )
        else:
            status = (
                "✓ Task loaded successfully\n"
                "Attachment: None"
            )

        return (
            question,
            file_name,
            status
        )

    except Exception as e:
        return (
            "",
            "",
            f"❌ ERROR: {e}"
        )


# ============================================================
# SHOW ALL QUESTIONS
# ============================================================

def show_questions():
    """Display all GAIA questions available."""
    
    try:
        questions = get_gaia_questions()
        return str(questions)
    except Exception as e:
        return f"ERROR: {e}"


# ============================================================
# REVERSE TEST
# ============================================================

def test_reverse():
    """Test the reverse_string tool."""
    
    try:
        return reverse_string(
            "tfel"
        )
    except Exception as e:
        return f"ERROR: {e}"


# ============================================================
# EXCEL TEST
# ============================================================

def test_excel_sum(
    file_path,
    column_name
):
    """Test the sum_spreadsheet_column tool."""
    
    try:
        if not file_path:
            return "ERROR: Excel/CSV path is required."

        if not column_name:
            return "ERROR: Column name is required."

        return sum_spreadsheet_column(
            file_path,
            column_name
        )

    except Exception as e:
        return f"ERROR: {e}"


# ============================================================
# RUN GAIA AGENT
# ============================================================

def run_agent(
    task_id,
    question,
    file_name
):
    """
    Run GAIA Agent on a question.
    Uses dynamic model selection via router.
    """
    
    try:
        if not task_id:
            return "❌ ERROR: Task ID is required."

        if not question:
            return (
                "❌ ERROR: Question is empty. "
                "Use 'Load Task' first."
            )

        # Initialize agent with question for dynamic model selection
        agent = GaiaAgent(
            question=question
        )

        answer = agent.run(
            task_id=task_id,
            question=question,
            file_name=file_name or None,
        )

        return answer

    except Exception as e:
        return (
            "❌ ERROR:\n"
            + str(e)
        )


# ============================================================
# RUN ALL QUESTIONS + SUBMIT TO THE COURSE SCORER
# ============================================================

import json
import os
import tempfile
import time

ANSWERS_FILE = os.path.join(tempfile.gettempdir(), "gaia_answers.json")
SPACE_ID = os.environ.get("SPACE_ID", "jesuslayer/gaia-agent-eva")
PAUSE_BETWEEN_QUESTIONS = 5   # seconds, to stay under the per-minute quota
RETRY_WAIT = 25               # seconds before retrying a question that errored


def _load_saved_answers():
    try:
        with open(ANSWERS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []


def _save_answers(rows):
    with open(ANSWERS_FILE, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)


def _answers_table(rows):
    return [
        [i + 1, r["task_id"][:8], (r.get("question") or "")[:90], r["submitted_answer"]]
        for i, r in enumerate(rows)
    ]


def _solve_one(task_id, question, file_name):
    agent = GaiaAgent(question=question)
    return str(agent.run(task_id=task_id, question=question, file_name=file_name or None))


def run_all_questions(progress=gr.Progress()):
    """Run the agent on every GAIA question, saving answers as they come."""
    try:
        questions = get_gaia_questions()
    except Exception as e:
        return f"❌ ERROR fetching questions: {e}", []
    if not isinstance(questions, list) or not questions:
        return "❌ ERROR: unexpected questions format", []

    rows = []
    total = len(questions)
    for i, q in enumerate(questions):
        task_id = q.get("task_id")
        question = q.get("question") or ""
        file_name = q.get("file_name") or None
        progress(i / total, desc=f"Question {i + 1}/{total}")

        try:
            answer = _solve_one(task_id, question, file_name)
        except Exception as e:
            answer = f"ERROR: {e}"

        if answer.startswith("ERROR"):
            print(f"[RUN-ALL] {task_id[:8]} errored, retrying in {RETRY_WAIT}s")
            time.sleep(RETRY_WAIT)
            try:
                answer = _solve_one(task_id, question, file_name)
            except Exception as e:
                answer = f"ERROR: {e}"

        rows.append({"task_id": task_id, "question": question, "submitted_answer": answer})
        _save_answers(rows)
        print(f"[RUN-ALL] {i + 1}/{total} {task_id[:8]} -> {answer[:80]}")
        time.sleep(PAUSE_BETWEEN_QUESTIONS)

    errors = sum(1 for r in rows if r["submitted_answer"].startswith("ERROR"))
    status = (
        f"✓ Finished {len(rows)} questions ({errors} with errors). "
        f"Answers saved. Review the table, then click Submit answers."
    )
    return status, _answers_table(rows)


def show_saved_answers():
    rows = _load_saved_answers()
    if not rows:
        return "No saved answers yet. Click 'Run all questions' first.", []
    return f"{len(rows)} saved answers.", _answers_table(rows)


def submit_answers(username):
    """POST the saved answers to the course scorer."""
    username = (username or "").strip()
    if not username:
        return "❌ ERROR: Hugging Face username is required."
    rows = _load_saved_answers()
    if not rows:
        return "❌ ERROR: no saved answers. Click 'Run all questions' first."

    payload = {
        "username": username,
        "agent_code": f"https://huggingface.co/spaces/{SPACE_ID}/tree/main",
        "answers": [
            {"task_id": r["task_id"], "submitted_answer": r["submitted_answer"]}
            for r in rows
        ],
    }
    try:
        r = requests.post(f"{GAIA_API}/submit", json=payload, timeout=120)
        r.raise_for_status()
        d = r.json()
        return (
            "✓ Submission successful!\n"
            f"User: {d.get('username')}\n"
            f"Score: {d.get('score')}% "
            f"({d.get('correct_count')}/{d.get('total_attempted')} correct)\n"
            f"Message: {d.get('message', '')}"
        )
    except requests.exceptions.HTTPError as e:
        detail = ""
        try:
            detail = e.response.json().get("detail", e.response.text[:300])
        except Exception:
            detail = e.response.text[:300] if e.response is not None else ""
        return f"❌ Submission failed: {e}\n{detail}"
    except Exception as e:
        return f"❌ Submission failed: {e}"


# ============================================================
# UI
# ============================================================

with gr.Blocks(
    title="GAIA Agent - Dynamic Model Router"
) as demo:

    gr.Markdown(
        """
# GAIA Agent with Dynamic Model Router
**LangGraph + Gemini (Flash-Lite pool with automatic fallback)**

The agent is a LangGraph loop (assistant ⇄ tools) with tools for Wikipedia, web search,
YouTube transcripts, spreadsheets, Python execution and audio transcription (Whisper).
The question type is detected to choose a specialised system prompt.

Enter a GAIA Task ID and the system will retrieve the question automatically.
"""
    )

    # ========================================================
    # SYSTEM TAB
    # ========================================================

    with gr.Tab("System"):

        gr.Markdown("### System Tests")

        gpu_button = gr.Button(
            "Test GPU",
            variant="secondary"
        )
        gpu_output = gr.Textbox(
            label="GPU Result",
            interactive=False
        )
        gpu_button.click(
            gpu_test,
            outputs=gpu_output
        )

    # ========================================================
    # ROUTER TAB
    # ========================================================

    with gr.Tab("Router"):

        gr.Markdown(
            "### Test Question Type Detection"
        )

        router_input = gr.Textbox(
            label="Question",
            lines=5,
            placeholder="Enter a GAIA question to see which model will be used..."
        )

        router_button = gr.Button(
            "Detect Question Type",
            variant="secondary"
        )

        router_output = gr.Textbox(
            label="Router Decision",
            interactive=False,
            lines=5
        )

        router_button.click(
            test_router,
            inputs=router_input,
            outputs=router_output
        )

    # ========================================================
    # TOOLS TAB
    # ========================================================

    with gr.Tab("Tools"):

        gr.Markdown("### Reverse String")
        gr.Markdown("`tfel` → `left`")

        reverse_button = gr.Button(
            "Test Reverse",
            variant="secondary"
        )
        reverse_output = gr.Textbox(
            label="Reverse Result",
            interactive=False
        )
        reverse_button.click(
            test_reverse,
            outputs=reverse_output
        )

        gr.Markdown("---")

        gr.Markdown("### Excel / CSV Column Sum")

        excel_path = gr.Textbox(
            label="Excel/CSV path",
            placeholder="/tmp/example.xlsx"
        )

        column_name = gr.Textbox(
            label="Column name",
            placeholder="Burgers"
        )

        sum_button = gr.Button(
            "Sum Column",
            variant="secondary"
        )

        sum_output = gr.Textbox(
            label="Sum Result",
            interactive=False
        )

        sum_button.click(
            test_excel_sum,
            inputs=[excel_path, column_name],
            outputs=sum_output
        )

    # ========================================================
    # GAIA DATASET TAB
    # ========================================================

    with gr.Tab("GAIA Dataset"):

        gr.Markdown("### GAIA Validation Questions")

        question_button = gr.Button(
            "Get GAIA Questions",
            variant="secondary"
        )

        questions_output = gr.Textbox(
            label="Questions",
            lines=20,
            interactive=False
        )

        question_button.click(
            show_questions,
            outputs=questions_output
        )

    # ========================================================
    # AGENT TAB
    # ========================================================

    with gr.Tab("Agent"):

        gr.Markdown(
            """
### Run GAIA Agent

1. Enter the Task ID
2. Click **Load Task**
3. The question and attachment are retrieved automatically
4. Click **Run Agent** to solve
"""
        )

        task_id = gr.Textbox(
            label="GAIA Task ID",
            placeholder="2d83110e-a098-4ebb-9987-066c06fa42d0"
        )

        load_button = gr.Button(
            "Load Task",
            variant="secondary"
        )

        task_status = gr.Textbox(
            label="Task Status",
            interactive=False,
            lines=3
        )

        question = gr.Textbox(
            label="GAIA Question",
            lines=8,
            interactive=False
        )

        file_name = gr.Textbox(
            label="Attachment",
            interactive=False
        )

        load_button.click(
            load_task,
            inputs=[task_id],
            outputs=[question, file_name, task_status]
        )

        gr.Markdown("---")

        run_button = gr.Button(
            "Run Agent",
            variant="primary"
        )

        answer = gr.Textbox(
            label="Final Answer",
            lines=4,
            interactive=False
        )

        run_button.click(
            run_agent,
            inputs=[task_id, question, file_name],
            outputs=answer
        )


    # ========================================================
    # SUBMIT TAB
    # ========================================================

    with gr.Tab("Submit"):

        gr.Markdown(
            """
### Run all questions and submit to the leaderboard

1. Click **Run all questions** (20 questions, about 20-30 minutes). Answers are saved as they come.
2. Review the table. **Show saved answers** reloads it without running again.
3. Enter your Hugging Face username and click **Submit answers**.
"""
        )

        run_all_button = gr.Button(
            "Run all questions",
            variant="primary"
        )

        run_all_status = gr.Textbox(
            label="Status",
            interactive=False,
            lines=3
        )

        answers_table = gr.Dataframe(
            headers=["#", "Task", "Question", "Answer"],
            interactive=False
        )

        run_all_button.click(
            run_all_questions,
            outputs=[run_all_status, answers_table]
        )

        reload_button = gr.Button(
            "Show saved answers",
            variant="secondary"
        )

        reload_button.click(
            show_saved_answers,
            outputs=[run_all_status, answers_table]
        )

        gr.Markdown("---")

        hf_username = gr.Textbox(
            label="Hugging Face username",
            value="jesuslayer"
        )

        submit_button = gr.Button(
            "Submit answers",
            variant="primary"
        )

        submit_output = gr.Textbox(
            label="Submission result",
            interactive=False,
            lines=6
        )

        submit_button.click(
            submit_answers,
            inputs=[hf_username],
            outputs=submit_output
        )

# ============================================================
# LAUNCH
# ============================================================

if __name__ == "__main__":
    demo.launch()