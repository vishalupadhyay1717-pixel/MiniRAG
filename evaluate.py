"""
evaluate.py
-----------
Standalone retrieval evaluation script.

This measures RETRIEVAL quality only (does the right chunk get found?),
not generation quality - grading free-text LLM answers automatically
is out of scope for a minor project and would be misleading to present
as "scientific". Instead we check whether the expected (source, page)
pair shows up in the Top-1 / Top-3 results for each labeled question.

Run independently of the GUI:

    python evaluate.py

Requires the index to already be built (run app.py -> Build Index first).
"""

import json
import os

import retriever

EVAL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "evaluation")
QUESTIONS_FILE = os.path.join(EVAL_DIR, "questions.json")


def load_questions() -> list[dict]:
    if not os.path.exists(QUESTIONS_FILE):
        raise FileNotFoundError(f"Evaluation questions not found at {QUESTIONS_FILE}")
    with open(QUESTIONS_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def _matches(result: dict, expected_source: str, expected_page: int) -> bool:
    return result["source"] == expected_source and result["page"] == expected_page


def evaluate() -> None:
    if not retriever.index_exists():
        print("No index found. Please build the index first by running app.py.")
        return

    index, chunks = retriever.load_index()
    questions = load_questions()

    top1_hits = 0
    top3_hits = 0
    per_question_report = []

    for q in questions:
        results = retriever.search(index, chunks, q["question"], top_k=3)

        top1_ok = bool(results) and _matches(results[0], q["expected_source"], q["expected_page"])
        top3_ok = any(_matches(r, q["expected_source"], q["expected_page"]) for r in results)

        top1_hits += int(top1_ok)
        top3_hits += int(top3_ok)

        per_question_report.append((q["question"], top1_ok, top3_ok))

    total = len(questions)
    top1_acc = (top1_hits / total * 100) if total else 0.0
    top3_acc = (top3_hits / total * 100) if total else 0.0

    print("========================================")
    print("MiniRAG Retrieval Evaluation")
    print("========================================")
    for question, top1_ok, top3_ok in per_question_report:
        mark1 = "OK" if top1_ok else "--"
        mark3 = "OK" if top3_ok else "--"
        print(f"[Top1:{mark1}  Top3:{mark3}]  {question}")
    print("----------------------------------------")
    print(f"Total Questions: {total}")
    print(f"Top-1 Accuracy: {top1_acc:.1f}%")
    print(f"Top-3 Accuracy: {top3_acc:.1f}%")
    print("========================================")


if __name__ == "__main__":
    evaluate()
