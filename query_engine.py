"""
query_engine.py
AI Data Analyst question router.

Pandas = calculations on the real dataframe
Qwen    = reasoning and explanation
Hybrid  = Pandas calculates, Qwen explains
"""

import ast
import re

import numpy as np
import pandas as pd

from ollama_client import ask_ollama


# -------------------------------------------------------------------
# Safety configuration
# -------------------------------------------------------------------

BLOCKED_CALL_NAMES = {
    "eval",
    "open",
    "compile",
    "__import__",
    "input",
    "getattr",
    "setattr",
    "delattr",
    "exec",
}

ALLOWED_IMPORT_MODULES = {
    "pandas",
    "numpy",
}


# -------------------------------------------------------------------
# Question routing
# -------------------------------------------------------------------

REASONING_TRIGGERS = [
    "why",
    "recommend",
    "recommendation",
    "suggest",
    "insight",
    "insights",
    "should we",
    "should i",
    "advice",
    "advise",
    "opinion",
    "think about",
    "improve",
    "strategy",
    "explain",
    "what does this mean",
    "interpret",
]

COMPUTE_TRIGGERS = [
    "total",
    "sum",
    "average",
    "avg",
    "mean",
    "count",
    "how many",
    "top",
    "highest",
    "lowest",
    "maximum",
    "max",
    "minimum",
    "min",
    "median",
    "percentage",
    "percent",
    "ratio",
    "number of",
    "trend",
    "pattern",
]


def _classify_question(question: str) -> str:
    """
    Return:
        reasoning
        compute
        hybrid
    """

    q = question.lower()

    word_boundary_needed = {
        "top",
        "min",
        "max",
        "avg",
        "count",
    }

    def has_any(triggers):
        for trigger in triggers:
            if trigger in word_boundary_needed:
                if re.search(r"\b" + re.escape(trigger) + r"\b", q):
                    return True
            elif trigger in q:
                return True

        return False

    has_reasoning = has_any(REASONING_TRIGGERS)
    has_compute = has_any(COMPUTE_TRIGGERS)

    if has_reasoning and has_compute:
        return "hybrid"

    if has_reasoning:
        return "reasoning"

    return "compute"


# -------------------------------------------------------------------
# Generated-code safety checking
# -------------------------------------------------------------------

def _is_safe_code(code: str) -> bool:
    """
    Check model-generated Python before execution.

    Only pandas/numpy imports are allowed.
    Dangerous built-ins and dunder access are rejected.
    """

    try:
        tree = ast.parse(code, mode="exec")
    except SyntaxError:
        return False

    for node in ast.walk(tree):

        # import pandas / numpy only
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name not in ALLOWED_IMPORT_MODULES:
                    return False

        elif isinstance(node, ast.ImportFrom):
            if node.module not in ALLOWED_IMPORT_MODULES:
                return False

        # Reject dunder names
        elif isinstance(node, ast.Name):
            if node.id.startswith("__"):
                return False

        # Reject dunder attributes
        elif isinstance(node, ast.Attribute):
            if node.attr.startswith("__"):
                return False

        # Reject dangerous function calls
        elif isinstance(node, ast.Call):
            fn = node.func

            if isinstance(fn, ast.Name):
                if fn.id in BLOCKED_CALL_NAMES:
                    return False

            elif isinstance(fn, ast.Attribute):
                if fn.attr in BLOCKED_CALL_NAMES:
                    return False

    return True


def _extract_code_block(raw: str) -> str:
    """
    Extract Python code if Qwen accidentally returns markdown fences
    or a small amount of explanatory text.
    """

    if not raw:
        return ""

    text = raw.strip()

    fence_match = re.search(
        r"```(?:python)?\s*(.*?)```",
        text,
        re.DOTALL | re.IGNORECASE,
    )

    if fence_match:
        text = fence_match.group(1).strip()

    lines = []

    for line in text.splitlines():
        stripped = line.strip()

        if not stripped:
            continue

        if stripped.lower().startswith(
            (
                "here",
                "this ",
                "note:",
                "explanation",
                "the code",
            )
        ):
            continue

        lines.append(line)

    return "\n".join(lines).strip()


# -------------------------------------------------------------------
# Result formatting
# -------------------------------------------------------------------

def _format_result(result):
    """
    Convert pandas/numpy results into readable output.

    Prevents scientific notation and giant Series dumps.
    """

    if isinstance(result, (int, np.integer)):
        return f"{int(result):,}"

    if isinstance(result, (float, np.floating)):
        value = float(result)

        if np.isnan(value):
            return "NaN"

        if value.is_integer():
            return f"{value:,.0f}"

        return f"{value:,.2f}"

    if isinstance(result, pd.Series):

        if len(result) == 1:
            return _format_result(result.iloc[0])

        top = result.head(10)

        formatted = "\n".join(
            f"- {idx}: {_format_result(value)}"
            for idx, value in top.items()
        )

        if len(result) > 10:
            formatted += f"\n- ... ({len(result)} rows total)"

        return formatted

    if isinstance(result, pd.DataFrame):

        if result.shape == (1, 1):
            return _format_result(result.iloc[0, 0])

        return result.head(10).to_string()

    return str(result)


# -------------------------------------------------------------------
# Pandas computation
# -------------------------------------------------------------------

def _generate_pandas_code(
    question: str,
    df: pd.DataFrame,
    model: str,
) -> str:

    cols_info = ", ".join(
        f"{column} ({df[column].dtype})"
        for column in df.columns
    )

    prompt = f"""
You are a Python code generator.

Output ONLY executable Python code.
No markdown.
No explanations.
No comments.

DataFrame variable:
df

Columns:
{cols_info}

STRICT RULES:

1. Always assign the final answer to a variable named result.

2. For TOTAL, SUM, AVERAGE, COUNT, MEAN, MEDIAN or another
   single aggregate question, result MUST be one scalar value.

3. Do not return a Series for a single aggregate.

4. For TOP N or category breakdown questions, a small Series/DataFrame
   is allowed.

5. Use only df, pandas and numpy.

Examples:

Question:
What is the total revenue?

Code:
result = df["revenue"].sum()

Question:
What is the average order value?

Code:
result = df["order_value"].mean()

Question:
Top 5 products by revenue

Code:
result = (
    df.groupby("product")["revenue"]
    .sum()
    .sort_values(ascending=False)
    .head(5)
)

Question:
Which country has the highest revenue?

Code:
result = (
    df.groupby("country")["revenue"]
    .sum()
    .idxmax()
)

Now generate code for:

{question}

Output ONLY Python code.
"""

    raw = ask_ollama(
        prompt,
        model=model,
        temperature=0.0,
    )

    return _extract_code_block(raw)


def _answer_with_pandas(
    question: str,
    df: pd.DataFrame,
    model: str,
) -> str:

    try:
        code = _generate_pandas_code(
            question,
            df,
            model,
        )
    except RuntimeError as e:
        return str(e)

    if not code:
        return (
            "The model didn't return any code. "
            "Try rephrasing your question more simply."
        )

    if not _is_safe_code(code):
        return (
            "The generated code failed the safety check, "
            "so I won't execute it.\n\n"
            f"```python\n{code}\n```"
        )

    # Only expose the objects the generated code needs.
    safe_globals = {
        "__builtins__": {},
        "pd": pd,
        "np": np,
    }

    local_ns = {
        "df": df,
        "pd": pd,
        "np": np,
        "result": None,
    }

    try:
        # Deliberately execute only after AST safety validation.
        exec(code, safe_globals, local_ns)

        result = local_ns.get("result")

    except Exception as e:
        return (
            f"The generated code failed to run: `{e}`\n\n"
            f"**Generated code:**\n"
            f"```python\n{code}\n```"
        )

    if result is None:
        return (
            "The code ran but didn't produce a result.\n\n"
            f"**Generated code:**\n"
            f"```python\n{code}\n```"
        )

    formatted = _format_result(result)

    explain_prompt = f"""
The user asked:

"{question}"

The exact computed answer from the real dataset is:

{formatted}

Write one short, clear answer.

Use the numbers EXACTLY as provided.
Do not recompute.
Do not round differently.
Do not use scientific notation.

If this is a breakdown, use a short bullet list.
"""

    try:
        explanation = ask_ollama(
            explain_prompt,
            model=model,
            temperature=0.2,
        )

        return explanation if explanation else formatted

    except RuntimeError:
        return formatted


# -------------------------------------------------------------------
# Reasoning path
# -------------------------------------------------------------------

def _answer_with_reasoning(
    question: str,
    context_summary: str,
    model: str,
) -> str:

    prompt = f"""
You are a senior business data analyst.

Use the dataset summary below to answer the question.

Focus on reasoning, interpretation and practical insight.
Do not invent numbers that are not present in the summary.

Keep the answer concise:
3-5 sentences or a short bullet list.

Dataset summary:

{context_summary}

Question:

{question}

Answer:
"""

    try:
        return ask_ollama(
            prompt,
            model=model,
            temperature=0.4,
        )

    except RuntimeError as e:
        return str(e)


# -------------------------------------------------------------------
# PUBLIC API
# -------------------------------------------------------------------

def answer_question(
    question: str,
    df: pd.DataFrame,
    context_summary: str,
    model: str,
) -> str:
    """
    Main entry point used by app.py.

    compute   -> Pandas
    reasoning -> Qwen
    hybrid    -> Pandas + Qwen
    """

    category = _classify_question(question)

    if category == "reasoning":
        return _answer_with_reasoning(
            question,
            context_summary,
            model,
        )

    if category == "hybrid":

        computed = _answer_with_pandas(
            question,
            df,
            model,
        )

        reasoning_prompt = f"""
The user asked:

"{question}"

A computation on the real dataset produced:

{computed}

Dataset summary:

{context_summary}

Explain what the result means in 2-4 concise sentences.

Do not change the computed number.
"""

        try:
            return ask_ollama(
                reasoning_prompt,
                model=model,
                temperature=0.4,
            )

        except RuntimeError:
            return computed

    return _answer_with_pandas(
        question,
        df,
        model,
    )


# -------------------------------------------------------------------
# Backward compatibility
# -------------------------------------------------------------------

def answer_with_real_data(
    question: str,
    df: pd.DataFrame,
    model: str,
) -> str:

    return _answer_with_pandas(
        question,
        df,
        model,
    )
