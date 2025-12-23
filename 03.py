# graph.py
from __future__ import annotations
from typing import TypedDict, List, Literal, Dict, Any
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from gmail_tools import fetch_recent_emails, label_email, delete_email

class Email(TypedDict):
    id: str
    subject: str
    from_: str
    snippet: str
    label: Literal["unknown", "spam", "ham", "unsure"]

class GraphState(TypedDict):
    emails: List[Email]
    index: int

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

def fetch_emails_node(state: GraphState) -> GraphState:
    raw = fetch_recent_emails(max_results=20)
    emails: List[Email] = []
    for m in raw:
        emails.append(
            {
                "id": m["id"],
                "subject": m["subject"],
                "from_:": m["from"],
                "snippet": m["snippet"],
                "label": "unknown",
            }
        )
    return {"emails": emails, "index": 0}

def classify_email_node(state: GraphState) -> GraphState:
    i = state["index"]
    if i >= len(state["emails"]):
        return {}
    email = state["emails"][i]
    prompt = (
        "You are a spam filter.\n\n"
        f"From: {email['from_:']}\n"
        f"Subject: {email['subject']}\n"
        f"Body: {email['snippet']}\n\n"
        "Classify this email strictly as one of: spam, ham, unsure.\n"
        "Return ONLY the label word."
    )
    resp = llm.invoke(prompt)
    label_raw = resp.content.strip().lower()
    if "spam" in label_raw and "ham" not in label_raw:
        label: Literal["spam","ham","unsure"] = "spam"
    elif "ham" in label_raw:
        label = "ham"
    else:
        label = "unsure"
    state["emails"][i]["label"] = label
    return {"emails": state["emails"]}

def apply_label_node(state: GraphState) -> GraphState:
    i = state["index"]
    if i >= len(state["emails"]):
        return {}
    email = state["emails"][i]
    if email["label"] == "spam":
        label_email(email["id"], ["AI_SPAM_REVIEW"])
    elif email["label"] == "ham":
        label_email(email["id"], ["AI_HAM"])
    else:
        label_email(email["id"], ["AI_UNSURE"])
    state["index"] = i + 1
    return {"index": state["index"], "emails": state["emails"]}

def route_after_apply(state: GraphState):
    if state["index"] >= len(state["emails"]):
        return "done"
    return "classify"

def delete_confirmed_spam_node(state: GraphState) -> GraphState:
    for email in state["emails"]:
        if email["label"] == "spam":
            # at this point you already reviewed label AI_SPAM_REVIEW in Gmail
            delete_email(email["id"])
    return state

graph = StateGraph(GraphState)
graph.add_node("fetch_emails", fetch_emails_node)
graph.add_node("classify_email", classify_email_node)
graph.add_node("apply_label", apply_label_node)
graph.add_node("delete_spam", delete_confirmed_spam_node)

graph.add_edge(START, "fetch_emails")
graph.add_edge("fetch_emails", "classify_email")
graph.add_edge("classify_email", "apply_label")
graph.add_conditional_edges(
    "apply_label",
    route_after_apply,
    {"classify": "classify_email", "done": "delete_spam"},
)
graph.add_edge("delete_spam", END)

app = graph.compile()

if __name__ == "__main__":
    init_state: GraphState = {"emails": [], "index": 0}
    result = app.invoke(init_state)
    print("Done. Processed emails:", len(result["emails"]))
