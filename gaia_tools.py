import os
import subprocess
import tempfile
import requests
import pandas as pd
import time

from smolagents import tool


# ============================================================
# CONFIGURATION
# ============================================================

GAIA_API = "https://agents-course-unit4-scoring.hf.space"
MAX_RETRIES = 3
RETRY_DELAY = 2  # seconds
TIMEOUT = 120


# ============================================================
# INTERNAL: DOWNLOAD GAIA ATTACHMENT
# ============================================================

def download_gaia_file(task_id: str, file_name: str | None = None) -> str:
    """
    Download the attachment associated with a GAIA task.
    
    Args:
        task_id: GAIA task ID.
        file_name: Optional original filename for the attachment.
    
    Returns:
        Local path of the downloaded attachment.
    
    Raises:
        RuntimeError: If download fails after retries.
    """

    cache_dir = os.path.join(
        tempfile.gettempdir(),
        "gaia_files"
    )

    os.makedirs(
        cache_dir,
        exist_ok=True
    )

    if file_name:
        safe_name = os.path.basename(file_name)
    else:
        safe_name = f"{task_id}.bin"

    local_path = os.path.join(
        cache_dir,
        safe_name
    )

    # Use cached file if it already exists
    if os.path.exists(local_path):
        print(f"[CACHE] Using cached file: {local_path}")
        return local_path

    url = f"{GAIA_API}/files/{task_id}"
    
    print(f"[DOWNLOAD] Fetching: {url}")

    # Retry logic
    for attempt in range(MAX_RETRIES):
        try:
            response = requests.get(
                url,
                timeout=TIMEOUT
            )
            response.raise_for_status()

            # Try to get filename from response headers
            if not file_name:
                content_disposition = response.headers.get(
                    "content-disposition",
                    ""
                )

                if "filename=" in content_disposition:
                    detected_name = (
                        content_disposition
                        .split("filename=", 1)[1]
                        .strip()
                        .strip('"')
                    )

                    if detected_name:
                        safe_name = os.path.basename(detected_name)
                        local_path = os.path.join(
                            cache_dir,
                            safe_name
                        )

            # Write file
            with open(
                local_path,
                "wb"
            ) as f:
                f.write(response.content)

            print(f"[DOWNLOAD] ✓ Success: {local_path} ({len(response.content)} bytes)")
            return local_path

        except Exception as e:
            print(f"[DOWNLOAD] Attempt {attempt + 1}/{MAX_RETRIES} failed: {e}")
            
            if attempt < MAX_RETRIES - 1:
                print(f"[DOWNLOAD] Retrying in {RETRY_DELAY}s...")
                time.sleep(RETRY_DELAY)
            else:
                raise RuntimeError(
                    f"Failed to download file after {MAX_RETRIES} attempts: {e}"
                )


# ============================================================
# TOOL: DOWNLOAD TASK FILE
# ============================================================

@tool
def download_task_file(task_id: str) -> str:
    """
    Download the attachment associated with a GAIA task.
    
    Args:
        task_id: GAIA task ID.
    
    Returns:
        Local path to the downloaded attachment.
    
    Raises:
        RuntimeError: If download fails.
    """

    try:
        return download_gaia_file(task_id)
    except Exception as e:
        return f"ERROR: Failed to download file: {e}"


# ============================================================
# TOOL: READ TEXT FILE
# ============================================================

@tool
def read_text_file(file_path: str) -> str:
    """
    Read a text-based file.
    
    Args:
        file_path: Local path to the text file.
    
    Returns:
        The contents of the file (up to 40000 characters).
    """

    try:
        if not os.path.exists(file_path):
            return f"ERROR: File not found: {file_path}"

        with open(
            file_path,
            "r",
            encoding="utf-8",
            errors="ignore"
        ) as f:
            content = f.read()

        print(f"[READ] Read {len(content)} characters from {file_path}")

        return content[:40000]

    except Exception as e:
        return f"ERROR: Failed to read file: {e}"


# ============================================================
# TOOL: READ SPREADSHEET
# ============================================================

@tool
def read_spreadsheet(file_path: str) -> str:
    """
    Read an Excel or CSV spreadsheet.
    
    Args:
        file_path: Local path to the spreadsheet.
    
    Returns:
        The contents of all spreadsheet sheets.
    """

    try:
        if not os.path.exists(file_path):
            return f"ERROR: File not found: {file_path}"

        extension = os.path.splitext(
            file_path
        )[1].lower()

        if extension == ".csv":
            df = pd.read_csv(
                file_path
            )

            result = df.to_string(index=False)
            print(f"[SPREADSHEET] CSV: {len(df)} rows × {len(df.columns)} columns")
            return result

        elif extension in [".xlsx", ".xls"]:
            sheets = pd.read_excel(
                file_path,
                sheet_name=None
            )

            output = []

            for sheet_name, df in sheets.items():
                output.append(
                    f"SHEET: {sheet_name}\n"
                    f"ROWS: {len(df)}\n"
                    f"COLUMNS: {list(df.columns)}\n\n"
                    f"{df.to_string(index=False)}"
                )

            result = "\n\n".join(output)[:40000]
            print(f"[SPREADSHEET] Excel: {len(sheets)} sheets")
            return result

        else:
            return f"ERROR: Unsupported spreadsheet format: {extension}"

    except Exception as e:
        return f"ERROR: Failed to read spreadsheet: {e}"


# ============================================================
# TOOL: RUN PYTHON FILE
# ============================================================

@tool
def run_python_file(file_path: str) -> str:
    """
    Execute a Python file and return its output.
    
    Args:
        file_path: Local path to the Python file.
    
    Returns:
        Standard output and error output from the program.
    """

    try:
        if not os.path.exists(file_path):
            return f"ERROR: File not found: {file_path}"

        result = subprocess.run(
            ["python", file_path],
            capture_output=True,
            text=True,
            timeout=60
        )

        output = result.stdout

        if result.stderr:
            output += (
                "\nSTDERR:\n"
                + result.stderr
            )

        print(f"[PYTHON] Executed {file_path}")
        print(f"[PYTHON] Return code: {result.returncode}")

        return output[:40000]

    except subprocess.TimeoutExpired:
        return "ERROR: Python execution timed out (60 seconds)"
    except Exception as e:
        return f"ERROR: Failed to execute Python file: {e}"


# ============================================================
# TOOL: REVERSE STRING
# ============================================================

@tool
def reverse_string(text: str) -> str:
    """
    Reverse a string.
    
    Args:
        text: The string to reverse.
    
    Returns:
        The reversed string.
    """

    try:
        result = text[::-1]
        print(f"[REVERSE] Input length: {len(text)}")
        print(f"[REVERSE] Output: {result[:100]}")
        return result
    except Exception as e:
        return f"ERROR: Failed to reverse string: {e}"


# ============================================================
# TOOL: SUM SPREADSHEET COLUMN
# ============================================================

@tool
def sum_spreadsheet_column(
    file_path: str,
    column_name: str
) -> str:
    """
    Calculate the exact sum of a spreadsheet column.
    
    Args:
        file_path: Local path to the Excel or CSV spreadsheet.
        column_name: Exact name of the column to sum.
    
    Returns:
        The calculated sum as a string.
    """

    try:
        if not os.path.exists(file_path):
            return f"ERROR: File not found: {file_path}"

        extension = os.path.splitext(
            file_path
        )[1].lower()

        if extension == ".csv":
            df = pd.read_csv(file_path)

        elif extension in [".xlsx", ".xls"]:
            df = pd.read_excel(file_path)

        else:
            return f"ERROR: Unsupported file format: {extension}"

        # Validate column exists
        if column_name not in df.columns:
            available = ", ".join(df.columns)
            return (
                f"ERROR: Column '{column_name}' not found. "
                f"Available columns: {available}"
            )

        # Convert to numeric and sum
        numeric_values = pd.to_numeric(
            df[column_name],
            errors="coerce"
        )

        # Count how many values were converted
        valid_count = numeric_values.notna().sum()
        total_count = len(numeric_values)

        total = numeric_values.sum()

        print(
            f"[SUM] Column '{column_name}': "
            f"{valid_count}/{total_count} values summed = {total}"
        )

        return str(int(total)) if total == int(total) else str(total)

    except Exception as e:
        return f"ERROR: Failed to sum column: {e}"


# ============================================================
# TOOL: INSPECT SPREADSHEET
# ============================================================

@tool
def inspect_spreadsheet(file_path: str) -> str:
    """
    Inspect the structure of a spreadsheet.
    
    Args:
        file_path: Local path to the Excel or CSV spreadsheet.
    
    Returns:
        Sheet names, column names, row counts, and sample rows.
    """

    try:
        if not os.path.exists(file_path):
            return f"ERROR: File not found: {file_path}"

        extension = os.path.splitext(
            file_path
        )[1].lower()

        if extension == ".csv":
            df = pd.read_csv(file_path)

            result = (
                f"TYPE: CSV\n"
                f"ROWS: {len(df)}\n"
                f"COLUMNS: {list(df.columns)}\n\n"
                f"{df.head(10).to_string(index=False)}"
            )

            print(f"[INSPECT] CSV: {len(df)} rows × {len(df.columns)} cols")
            return result

        elif extension in [".xlsx", ".xls"]:
            sheets = pd.read_excel(
                file_path,
                sheet_name=None
            )

            output = []
            output.append(
                f"SHEETS: {list(sheets.keys())}"
            )

            for sheet_name, df in sheets.items():
                output.append(
                    f"\n--- SHEET: {sheet_name} ---\n"
                    f"COLUMNS: {list(df.columns)}\n"
                    f"ROWS: {len(df)}\n\n"
                    f"{df.head(10).to_string(index=False)}"
                )

            result = "\n".join(output)[:40000]
            print(f"[INSPECT] Excel: {len(sheets)} sheets")
            return result

        else:
            return f"ERROR: Unsupported spreadsheet format: {extension}"

    except Exception as e:
        return f"ERROR: Failed to inspect spreadsheet: {e}"


# ============================================================
# TOOL: TRANSCRIBE AUDIO
# ============================================================

@tool
def transcribe_audio(file_path: str) -> str:
    """
    Transcribe an audio file using OpenAI Whisper.
    
    Args:
        file_path: Local path to the audio file (MP3, WAV, M4A, etc).
    
    Returns:
        Transcribed text from the audio.
    """

    try:
        if not os.path.exists(file_path):
            return f"ERROR: File not found: {file_path}"

        # Import Whisper
        try:
            import whisper
        except ImportError:
            return (
                "ERROR: Whisper not installed. "
                "Install with: pip install openai-whisper"
            )

        print(f"[WHISPER] Loading model...")
        
        # Use base model (good balance of speed/accuracy)
        model = whisper.load_model("base")
        
        print(f"[WHISPER] Transcribing: {file_path}")
        
        result = model.transcribe(
            file_path,
            language="en"
        )

        transcription = result["text"].strip()
        
        print(f"[WHISPER] ✓ Transcription complete ({len(transcription)} chars)")

        return transcription

    except Exception as e:
        return f"ERROR: Failed to transcribe audio: {e}"


# ============================================================
# BUILD TOOL LIST
# ============================================================

def build_gaia_tools():
    """
    Build and return the list of GAIA tools.
    
    Returns:
        List of tool functions for smolagents.
    """
    
    return [
        download_task_file,
        read_text_file,
        read_spreadsheet,
        run_python_file,
        reverse_string,
        sum_spreadsheet_column,
        inspect_spreadsheet,
        transcribe_audio,
    ]