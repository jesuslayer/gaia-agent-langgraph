 GAIA Agent: LangGraph + Gemini

An AI agent that answers questions from the GAIA benchmark by searching the web, reading Wikipedia, watching YouTube transcripts, opening spreadsheets and running Python. Built for the Hugging Face Agents Course (Unit 4).

**Result: 70% (14/20 correct)**, above the 30% required to pass.**

Live Space: https://huggingface.co/spaces/jesuslayer/gaia-agent-eva

## How it works

The agent is a LangGraph loop: the model decides, calls a tool, reads the result, and decides again until it can give a final answer.

```
START -> assistant <-> tools -> final answer
```

- **Brain:** Google Gemini through the free API (`langchain-google-genai`).
- **Model pool with automatic fallback:** it starts with the Flash-Lite models (15 requests/min, ~500/day) and falls back to larger Flash models if one runs out of quota (429) or is overloaded (503).
- **Question router:** detects the question type and picks a specialised system prompt.
- **Answer cleaning:** GAIA uses exact match, so the final answer is stripped of extra words and formatted strictly.

### Tools

| Tool | Use |
|---|---|
| Wikipedia search and page (with "as of date") | Facts that must not be guessed |
| Web search and page visit | Anything outside Wikipedia |
| YouTube transcripts | Video questions |
| Spreadsheet inspect, read and sum | Excel attachments |
| Python file and code execution | Calculations, scripts |
| Whisper (base, CPU) | Audio attachments |
| Reverse string | Reversed-text puzzles |

## What I learned (the real story)

1. **First attempt: Qwen2.5-Coder, then Kimi K2** through the Hugging Face Inference API. I alternated them, but they often did not return answers.
2. **The hidden problem was ZeroGPU.** `@spaces.GPU(duration=50)` kills any call after 50 seconds, and an agent doing web searches takes longer. The result was a generic "Error" with no clear cause.
3. **Switching to Google's free Gemini API** removed the GPU dependency. The LLM now runs on Google's servers, so the agent runs freely.
4. **Quota is a design problem.** The strongest free model allowed only 20 requests a day. A pool of models with fallback solved it.
5. **Never let the model guess.** A hallucinated answer (a dinosaur article) was fixed by giving the agent Wikipedia tools and a "never guess" rule.
6. **Attachments:** the course API returned 404 for files, so the agent falls back to the GAIA dataset on the Hub.

## Known limits

- No vision yet: the chess-position and bird-video questions fail.
- Some multi-step web searches (sports statistics, grant numbers) are still unreliable.

## Run it

```bash
pip install -r requirements.txt
export GEMINI_API_KEY=your_free_key  
export HF_TOKEN=your_token            # optional, for GAIA dataset attachments
python app.py
```

## Stack

Python, LangGraph, LangChain, Google Gemini API, Gradio, smolagents tools, Whisper, Hugging Face Spaces.

## Next steps

- Add image input with Gemini vision.
- Log which model answered each question.
- Improve search for statistics questions.
