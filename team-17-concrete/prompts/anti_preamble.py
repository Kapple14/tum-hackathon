"""
Anti-preamble prompt templates for Allplan-style direct answers.

These prompts enforce the answer style used in the dual evaluator notebook:
- Answers start directly with helpful verbs or the term being defined
- Responses stay concise (1-3 sentences, max ~50 words)
- No introductory fluff or meta commentary
"""

from __future__ import annotations

from typing import Sequence

from llama_index.core import PromptTemplate


ALLPLAN_QA_TEMPLATE = PromptTemplate(
    (
        "You are a technical assistant for Allplan 2020. Answer based on the manual context.\n\n"
        "CONTEXT:\n"
        "{context_str}\n\n"
        "QUESTION: {query_str}\n\n"
        "CRITICAL: Answer directly without preamble. DO NOT start with:\n"
        "❌ 'Allplan 2020 supports...'\n"
        "❌ 'In Allplan, you can...'\n"
        "❌ 'The two primary workflows are...'\n"
        "❌ 'Allplan offers...'\n"
        "❌ 'There are...'\n\n"
        "CORRECT EXAMPLES (start immediately with the answer):\n"
        "✅ 'You can select the edit tool first and then the elements, or select the elements first and then the tool.'\n"
        "✅ 'To expand or collapse all areas, select and hold Ctrl+Shift while double-clicking within the name line.'\n"
        "✅ 'The status bar displays information such as the reference scale, unit of length, coordinate tracking, country, drawing type, angle, and memory allocation percentage.'\n\n"
        "ANSWER RULES:\n"
        "1. Start immediately with: 'You can', 'You must', 'To [action]', '[Term] is', or a direct statement\n"
        "2. NO introductory phrases or meta-commentary\n"
        "3. Length: 1-3 sentences (25-50 words maximum)\n"
        "4. End with a period (not ...)\n"
        "5. Use exact Allplan terminology\n\n"
        "ANSWER (start directly, no preamble):"
    )
)


ALLPLAN_REFINE_TEMPLATE = PromptTemplate(
    (
        "QUESTION: {query_str}\n"
        "EXISTING ANSWER: {existing_answer}\n"
        "NEW CONTEXT: {context_msg}\n\n"
        "Refine if the new context helps. NO preamble. Start directly with the answer.\n"
        "Maintain 1-3 sentences (max 50 words).\n\n"
        "REFINED ANSWER:"
    )
)


ALLPLAN_PROCEDURE_TEMPLATE = PromptTemplate(
    (
        "CONTEXT: {context_str}\n\n"
        "QUESTION: {query_str}\n\n"
        "CRITICAL: Do NOT start with 'Allplan supports...', 'The procedure is...', or 'There are X steps...'\n\n"
        "START DIRECTLY WITH:\n"
        "✅ 'To [action], [step 1], then [step 2], and [step 3].'\n"
        "✅ 'You can [action] by [method 1], then [method 2].'\n\n"
        "EXAMPLES:\n"
        "- 'To sort documents, click the column title to sort in ascending order, then click it again for descending order.'\n"
        "- 'You can move documents using drag-and-drop by selecting them, then holding the Shift key while dragging to the destination folder.'\n\n"
        "RULES:\n"
        "- Start with 'To' or 'You can'\n"
        "- 1-3 sentences (max 50 words)\n"
        "- No preamble or introduction\n"
        "- End sentences with a period\n\n"
        "ANSWER (start with 'To' or 'You can'):"
    )
)


ALLPLAN_DEFINITION_TEMPLATE = PromptTemplate(
    (
        "CONTEXT: {context_str}\n\n"
        "QUESTION: {query_str}\n\n"
        "CRITICAL: Do NOT start with 'In Allplan,...', 'Allplan defines...', or 'The definition is...'\n\n"
        "START DIRECTLY WITH THE TERM:\n"
        "✅ 'A SmartPart is a parametric Allplan CAD object...'\n"
        "✅ 'The current or active drawing file is the one in which you draw...'\n"
        "✅ '[Term] is [definition]. [Purpose].'\n\n"
        "RULES:\n"
        "- Start with the term being defined (article + term + 'is')\n"
        "- 1-2 sentences (max 45 words)\n"
        "- No preamble\n"
        "- End sentences with a period\n\n"
        "ANSWER (start with '[Term] is'):"
    )
)


ALLPLAN_LOCATION_TEMPLATE = PromptTemplate(
    (
        "CONTEXT: {context_str}\n\n"
        "QUESTION: {query_str}\n\n"
        "CRITICAL: Do NOT start with 'In Allplan,...', 'Allplan provides...', or 'The location is...'\n\n"
        "START DIRECTLY:\n"
        "✅ 'You can access these tools from [location].'\n"
        "✅ '[Feature] is located in [Menu - Tab - Option].'\n"
        "✅ 'The middle of the title bar displays [content].'\n\n"
        "RULES:\n"
        "- Start with 'You can', '[Feature] is located', or a direct description\n"
        "- 1-2 sentences (max 45 words)\n"
        "- No preamble\n"
        "- End sentences with a period\n\n"
        "ANSWER (no preamble):"
    )
)


ALLPLAN_LIST_TEMPLATE = PromptTemplate(
    (
        "CONTEXT: {context_str}\n\n"
        "QUESTION: {query_str}\n\n"
        "CRITICAL: For list questions, answer DIRECTLY with the list.\n\n"
        "DO NOT START WITH:\n"
        "❌ 'Allplan 2020 supports the following...'\n"
        "❌ 'The types are...'\n"
        "❌ 'There are X primary...'\n\n"
        "START DIRECTLY:\n"
        "✅ 'The five predefined structural levels are Site, Structure, Building, Story, and Substory.'\n"
        "✅ 'Two predefined sorting criteria are \"Sort by building structure\" and \"Sort by drawing file.\"'\n"
        "✅ 'Allplan 2020 supports DXF, DWG, DGN, PDF, IFC, CINEMA 4D, SketchUp, Rhino, STL, VRML, and XML formats.'\n\n"
        "For 'What are the X...' questions:\n"
        "- Start with: 'The [number] [items] are [list].'\n"
        "- OR: '[Items] include [list].'\n\n"
        "For 'two primary' questions:\n"
        "- Start with: 'You can either [option 1], or [option 2].'\n"
        "- OR: 'The two [items] are [item 1] and [item 2].'\n\n"
        "RULES:\n"
        "- 1-2 sentences (max 50 words)\n"
        "- No preamble\n"
        "- Natural list with commas and 'and'\n"
        "- End sentences with a period\n\n"
        "ANSWER (start directly with the list or 'You can either'):"
    )
)


PROMPT_RULES: Sequence[tuple[Sequence[str], PromptTemplate]] = (
    (
        (
            "two primary",
            "two main",
            "three primary",
            "three main",
            "what are the two",
            "what are the three",
            "what types",
            "what formats",
            "what options",
            "what settings",
            "which",
            "list",
        ),
        ALLPLAN_LIST_TEMPLATE,
    ),
    (
        ("how do i", "how can i", "how to", "steps to", "procedure", "process for", "how does"),
        ALLPLAN_PROCEDURE_TEMPLATE,
    ),
    (
        ("what is", "what are", "define", "explain", "meaning of", "definition of", "what does"),
        ALLPLAN_DEFINITION_TEMPLATE,
    ),
    (
        ("where", "location", "find", "access", "which menu", "which dialog", "which tab"),
        ALLPLAN_LOCATION_TEMPLATE,
    ),
)


def select_prompt_template(question: str) -> PromptTemplate:
    """Select the best prompt template for the incoming question."""
    question_lower = question.lower()

    for keywords, template in PROMPT_RULES:
        if any(phrase in question_lower for phrase in keywords):
            return template

    return ALLPLAN_QA_TEMPLATE


__all__ = [
    "ALLPLAN_QA_TEMPLATE",
    "ALLPLAN_REFINE_TEMPLATE",
    "ALLPLAN_PROCEDURE_TEMPLATE",
    "ALLPLAN_DEFINITION_TEMPLATE",
    "ALLPLAN_LOCATION_TEMPLATE",
    "ALLPLAN_LIST_TEMPLATE",
    "select_prompt_template",
]

