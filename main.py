"""Extension Request Sorter.

The LLM only EXTRACTS details (course, days, reason, summary).
Python checks the facts, protects privacy and applies the policy.
The tool never approves or rejects anything: faculty decide.

Run:  python main.py
"""

import csv
import json
import os
import re
import sys
import time
from collections import Counter

# ---------------------------------------------------------------------------
# Settings and policy constants (the department can change these easily)
# ---------------------------------------------------------------------------
MAX_DAYS_WITHIN_POLICY = 3
AUTO_REASONS = ("medical", "family", "technical")   # reasons that can be within policy

VALID_REASONS = ("medical", "family", "technical", "event", "workload", "other")
MIN_DAYS = 1
MAX_DAYS = 30
MAX_SUMMARY_WORDS = 15
MAX_RETRIES = 1                                      # one retry, then fall back

WITHIN_POLICY = "within_policy"
NEEDS_FACULTY_REVIEW = "needs_faculty_review"
NEEDS_MORE_INFO = "needs_more_info"

MODEL = os.getenv("GROQ_MODEL", "qwen/qwen3.8-27b")  # active Groq model
REQUESTS_FILE = "requests.csv"
COURSES_FILE = "courses.csv"
PROMPT_FILE = "system_prompt.txt"
RESULTS_FILE = "results.csv"
REVIEW_FILE = "review_queue.csv"
SUMMARY_FILE = "summary.txt"

# Used when the LLM fails twice: nothing is known, so it goes to needs_more_info.
FALLBACK_REPLY = {
    "course": None,
    "days": None,
    "reason": "other",
    "summary": "Details could not be extracted automatically",
}

# Illness / treatment words that must never appear in a summary or report.
# Matched at the start of a word, so "treat" also catches "treatment".
SENSITIVE_PREFIXES = [
    "fever", "dengue", "covid", "corona", "malaria", "typhoid", "jaundice",
    "flu", "cold", "cough", "viral", "infect", "cancer", "chemo", "tumor",
    "tumour", "diabet", "asthma", "migraine", "headache", "anxiety",
    "depress", "panic", "mental", "therap", "psych", "counsel", "surgery",
    "surgical", "operat", "treat", "medicat", "diagnos", "injur", "fractur",
    "accident", "hospital", "admitted", "poison", "pregnan", "dialysis",
]
# The tool recommends only, so these words must never appear either.
DECISION_PREFIXES = ["approv", "reject", "grant", "deni"]

# Words a student might use for a number of days.
NUMBER_WORDS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
    "seven": 7, "eight": 8, "nine": 9, "ten": 10, "fifteen": 15,
    "twenty": 20, "thirty": 30,
}

# Counts for the cost and latency note.
STATS = {"calls": 0, "seconds": 0.0, "input_tokens": 0, "output_tokens": 0}
_client = None


# ---------------------------------------------------------------------------
# Loading files
# ---------------------------------------------------------------------------
def load_env_file(path=".env"):
    """Read KEY=value lines from .env so the API key is never inside the code."""
    if not os.path.exists(path):
        return
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                os.environ.setdefault(key.strip(), value.strip().strip('"'))


def load_courses(path):
    """Return a dictionary like {"CS101": "Programming in Python"}."""
    courses = {}
    try:
        with open(path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                courses[row["code"].strip().upper()] = row["name"].strip()
    except FileNotFoundError:
        sys.exit(f"Cannot find {path}")
    return courses


def load_requests(path):
    """Return a list of dictionaries with keys id, date, request."""
    rows = []
    try:
        with open(path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                rows.append({
                    "id": row["id"].strip(),
                    "date": row["date"].strip(),
                    "request": row["request"].strip(),
                })
    except FileNotFoundError:
        sys.exit(f"Cannot find {path}")
    return rows


def load_system_prompt(path, courses):
    """Read the prompt file and insert the real course codes."""
    try:
        with open(path, encoding="utf-8") as f:
            text = f.read()
    except FileNotFoundError:
        sys.exit(f"Cannot find {path}")
    return text.replace("{COURSE_LIST}", ", ".join(courses))


# ---------------------------------------------------------------------------
# LLM call (the only function that depends on the LLM provider)
# ---------------------------------------------------------------------------
def call_llm(system_prompt, request_text):
    """Send one request to Groq and return its raw text reply."""
    global _client

    if _client is None:
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError(
                "GROQ_API_KEY is not set. Add it to Windows Environment Variables "
                "or to a local .env file."
            )

        try:
            from groq import Groq
        except ImportError as error:
            raise RuntimeError(
                "The Groq package is not installed. Run: py -m pip install groq"
            ) from error

        _client = Groq(api_key=api_key)

    start = time.time()
    response = _client.chat.completions.create(
        model=MODEL,
        max_tokens=400,
        temperature=0,
        messages=[
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": f"<request>\n{request_text}\n</request>",
            },
        ],
    )

    STATS["calls"] += 1
    STATS["seconds"] += time.time() - start

    usage = getattr(response, "usage", None)
    if usage is not None:
        STATS["input_tokens"] += getattr(usage, "prompt_tokens", 0) or 0
        STATS["output_tokens"] += getattr(usage, "completion_tokens", 0) or 0

    if not response.choices:
        raise RuntimeError("Groq returned an empty response.")

    message = response.choices[0].message
    raw_text = getattr(message, "content", None)
    if not isinstance(raw_text, str) or not raw_text.strip():
        raise RuntimeError("Groq returned an empty message.")

    return raw_text.strip()


def parse_json(text):
    """Turn the LLM reply into a dictionary, or return None if it is not JSON."""
    text = text.strip()
    text = re.sub(r"^```(?:json)?|```$", "", text).strip()   # remove code fences
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    start, end = text.find("{"), text.rfind("}")             # JSON hidden in extra text
    if start != -1 and end > start:
        try:
            return json.loads(text[start:end + 1])
        except json.JSONDecodeError:
            return None
    return None


# ---------------------------------------------------------------------------
# Checking the LLM reply
# ---------------------------------------------------------------------------
def is_valid(data):
    """True only if the reply has the exact shape and allowed values."""
    if not isinstance(data, dict):
        return False
    if set(data.keys()) != {"course", "days", "reason", "summary"}:
        return False

    course, days = data["course"], data["days"]
    if course is not None and not isinstance(course, str):
        return False
    if days is not None:
        if isinstance(days, bool) or not isinstance(days, int):
            return False
        if days < MIN_DAYS or days > MAX_DAYS:
            return False
    if data["reason"] not in VALID_REASONS:
        return False

    summary = data["summary"]
    if not isinstance(summary, str) or not summary.strip():
        return False
    if len(summary.split()) > MAX_SUMMARY_WORDS:
        return False
    return True


def classify(request_text, system_prompt):
    """Ask the LLM, check the reply, retry once, then fall back.

    Returns (reply_dictionary, used_fallback).
    """
    for attempt in range(1 + MAX_RETRIES):
        try:
            data = parse_json(call_llm(system_prompt, request_text))
            if is_valid(data):
                return data, False
            print(f"    attempt {attempt + 1}: reply was not valid")
        except Exception as error:          # network error, rate limit, etc.
            print(f"    attempt {attempt + 1}: LLM call failed ({type(error).__name__})")
    return dict(FALLBACK_REPLY), True


# ---------------------------------------------------------------------------
# Python fact checks and privacy check
# ---------------------------------------------------------------------------
def check_course(course, request_text, courses):
    """Keep the course only if it is in courses.csv AND written in the request.

    If the request mentions more than one known course, it is unclear which
    one is meant, so the course is dropped.
    """
    if course is None:
        return None
    mentioned = {code for code in courses
                 if re.search(rf"\b{re.escape(code)}\b", request_text, re.IGNORECASE)}
    code = course.strip().upper()
    if len(mentioned) == 1 and code in mentioned:
        return code
    return None


def find_written_days(request_text):
    """Return the set of day counts the student wrote, e.g. '2 days', 'two more days'."""
    words = "|".join(NUMBER_WORDS)
    pattern = rf"\b(\d+|{words})[\s-]+(?:(?:more|extra|additional)\s+)?days?\b"
    found = set()
    for match in re.finditer(pattern, request_text, re.IGNORECASE):
        token = match.group(1).lower()
        found.add(int(token) if token.isdigit() else NUMBER_WORDS[token])
    return found


def check_days(days, request_text):
    """Keep days only if the student wrote that number next to the word 'day(s)'."""
    if days is None:
        return None
    return days if days in find_written_days(request_text) else None


def is_summary_safe(summary, courses):
    """False if the summary names an illness/treatment, a possible person's name,
    or uses decision words like 'approved'."""
    lower = summary.lower()
    for prefix in SENSITIVE_PREFIXES + DECISION_PREFIXES:
        if re.search(rf"\b{prefix}", lower):
            return False
    # A capital letter in the middle of a summary may be a name. Course codes are fine.
    words = re.findall(r"[A-Za-z0-9]+", summary)
    for word in words[1:]:
        if word.upper() in courses:
            continue
        if word[0].isupper():
            return False
    return True


def build_neutral_summary(course, days, reason):
    """A safe summary built only from the checked facts."""
    if days is None:
        text = "Requests an extension"
    else:
        text = f"Requests {days} day" + ("" if days == 1 else "s")
    if course:
        text += f" for {course}"
    return f"{text}, {reason} reason"


# ---------------------------------------------------------------------------
# Policy (applied in Python, never by the LLM)
# ---------------------------------------------------------------------------
def recommend(course, days, reason):
    """Apply the department extension policy. This is a recommendation only."""
    if course is None or days is None:
        return NEEDS_MORE_INFO
    if days <= MAX_DAYS_WITHIN_POLICY and reason in AUTO_REASONS:
        return WITHIN_POLICY
    return NEEDS_FACULTY_REVIEW


# ---------------------------------------------------------------------------
# One request from start to finish
# ---------------------------------------------------------------------------
def process_request(row, courses, system_prompt):
    reply, used_fallback = classify(row["request"], system_prompt)

    course = check_course(reply["course"], row["request"], courses)
    days = check_days(reply["days"], row["request"])
    reason = reply["reason"]

    summary = reply["summary"]
    facts_changed = (course != reply["course"]) or (days != reply["days"])
    if used_fallback or facts_changed or not is_summary_safe(summary, courses):
        summary = build_neutral_summary(course, days, reason)

    return {
        "id": row["id"],
        "date": row["date"],
        "course": course,
        "days": days,
        "reason": reason,
        "recommendation": recommend(course, days, reason),
        "summary": summary,
        "request": row["request"],      # only ever written to review_queue.csv
        "fallback": used_fallback,
    }


# ---------------------------------------------------------------------------
# Output files
# ---------------------------------------------------------------------------
def write_reports(results):
    """Write results.csv, review_queue.csv and summary.txt."""
    shown = ["id", "date", "course", "days", "reason", "recommendation", "summary"]

    # 1. results.csv: every request, NO original text
    with open(RESULTS_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=shown, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)

    # 2. review_queue.csv: not within policy, WITH original text (course faculty only)
    queue = [r for r in results if r["recommendation"] != WITHIN_POLICY]
    with open(REVIEW_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=shown + ["request"], extrasaction="ignore")
        writer.writeheader()
        writer.writerows(queue)

    # 3. summary.txt: counts only, NO request text (for the HOD)
    by_rec = Counter(r["recommendation"] for r in results)
    by_reason = Counter(r["reason"] for r in results)
    by_course = Counter(r["course"] or "none" for r in results)
    fallbacks = sum(1 for r in results if r["fallback"])

    lines = [
        "EXTENSION REQUEST SUMMARY",
        "Recommendations only. Faculty make every final decision.",
        "",
        f"Total requests: {len(results)}",
        f"LLM fallbacks used: {fallbacks}",
        "",
        "By recommendation:",
    ]
    for label in (WITHIN_POLICY, NEEDS_FACULTY_REVIEW, NEEDS_MORE_INFO):
        lines.append(f"  {label}: {by_rec.get(label, 0)}")
    lines += ["", "By reason:"]
    for label in VALID_REASONS:
        lines.append(f"  {label}: {by_reason.get(label, 0)}")
    lines += ["", "By course:"]
    for label, count in sorted(by_course.items()):
        lines.append(f"  {label}: {count}")

    with open(SUMMARY_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    load_env_file()
    courses = load_courses(COURSES_FILE)
    rows = load_requests(REQUESTS_FILE)
    system_prompt = load_system_prompt(PROMPT_FILE, courses)

    print(f"Processing {len(rows)} requests...")
    results = []
    for row in rows:
        result = process_request(row, courses, system_prompt)
        results.append(result)
        print(f"  {row['id']}: {result['recommendation']}")

    write_reports(results)
    print(f"\nWrote {RESULTS_FILE}, {REVIEW_FILE}, {SUMMARY_FILE}")

    if STATS["calls"]:
        average = STATS["seconds"] / STATS["calls"]
        print(f"API calls: {STATS['calls']}  |  average {average:.2f}s per call  |  "
              f"tokens in/out: {STATS['input_tokens']}/{STATS['output_tokens']}")


if __name__ == "__main__":
    main()
