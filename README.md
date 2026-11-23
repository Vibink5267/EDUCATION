# Extension Request Sorter

Console tool: an LLM extracts course, days, reason and a summary from each student
extension request. Python checks the facts, protects privacy and applies the policy.
The tool only recommends; faculty make every final decision.

## Setup
1. `py -m pip install -r requirements.txt`
2. Set the Windows User Environment Variable `GROQ_API_KEY` before running.
Alternatively, copy `.env.example` to `.env` and add your Groq API key.
Never share or commit the real API key.
3. `python main.py`

## Files
- `requests.csv`, `courses.csv`: inputs
- `system_prompt.txt`: 5-role prompt (Authority, Exemplar, Constraint, Rubric, Metadata)
- `results.csv`: every request, no original text
- `review_queue.csv`: not within policy, with original text (course faculty only)
- `summary.txt`: counts for the HOD, no request text

## Policy (edit constants at the top of `main.py`)
1. Course or days missing -> `needs_more_info`
2. Days <= 3 and reason medical/family/technical -> `within_policy`
3. Everything else -> `needs_faculty_review`

## Groq configuration
- Provider: Groq
- Default model: `qwen/qwen3.8-27b`
- API key environment variable: `GROQ_API_KEY`
- Optional model override: `GROQ_MODEL`

If you previously set `GROQ_MODEL` to an old model, update it to `qwen/qwen3.8-27b` or remove that variable.
