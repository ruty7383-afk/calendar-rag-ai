"""Local Streamlit chat using the existing Calendar RAG engine."""

from io import StringIO
import os
from threading import RLock

import httpx
import streamlit as st

import main as rag


class MissingAPIKeyError(ValueError):
    pass


@st.cache_resource(show_spinner=False)
def get_backend():
    """Keep one engine in server memory; serialize access across browser sessions."""
    rag.load_dotenv(rag.BASE_DIR / ".env")
    api_key = os.getenv("GOOGLE_API_KEY", "").strip()
    if not api_key:
        raise MissingAPIKeyError()
    engine = rag.build_query_engine(api_key)
    if engine is None:
        raise RuntimeError("No events available")
    return engine, RLock()


def answer_question(question):
    engine, lock = get_backend()
    with lock:
        response = rag.supported_response(engine.query(question))
        sources = StringIO()
        rag.print_sources(response, file=sources)
        return {
            "role": "assistant",
            "content": str(response),
            "sources": sources.getvalue().strip(),
            "error": False,
        }


def error_message(error):
    """Expose fixed messages only, never raw exceptions or credentials."""
    if isinstance(error, MissingAPIKeyError):
        return "חסר GOOGLE_API_KEY. יש להגדיר אותו בקובץ .env שבשורש הפרויקט ולהפעיל מחדש את האפליקציה."
    if isinstance(error, rag.MetadataConsistencyError):
        return "לא ניתן להציג את התשובה: פרטי האירועים לא תאמו למאגר. אפשר לנסח את השאלה מחדש."
    code = getattr(error, "code", None)
    messages = {
        400: "Google דחתה את הבקשה. יש לבדוק את הגדרות הגישה לשירות.",
        401: "האימות מול Google נכשל. יש לבדוק את המפתח המקומי.",
        403: "אין הרשאה להשתמש בשירות Google. יש לבדוק את הרשאות המפתח.",
        404: "שירות המודל המבוקש אינו זמין כרגע.",
        429: "מכסת הבקשות של Google מוצתה. יש להמתין לחידוש המכסה ולנסות מאוחר יותר.",
        503: "שירות Google אינו זמין כרגע. אפשר לנסות מאוחר יותר.",
        504: "תם זמן ההמתנה לתשובת Google. אפשר לנסות מאוחר יותר.",
    }
    if isinstance(code, int) and code in messages:
        return f"{messages[code]} (קוד {code})"
    if isinstance(error, (httpx.TransportError, TimeoutError, ConnectionError)):
        return "אירעה שגיאת רשת בתקשורת עם Google. יש לבדוק את החיבור ולנסות מאוחר יותר."
    return "לא ניתן להשלים את הבקשה כרגע. יש לבדוק את החיבור ואת זמינות שירות Google."


def render_message(message):
    with st.chat_message(message["role"]):
        if message.get("error"):
            st.error(message["content"])
        else:
            st.markdown(message["content"])
        if message.get("sources"):
            st.text(message["sources"])


def main():
    st.set_page_config(page_title="עוזר לוח השנה שלי", page_icon="📅", layout="centered")
    st.markdown(
        """<style>
        [data-testid="stAppViewContainer"], [data-testid="stChatMessage"],
        [data-testid="stChatMessageContent"], [data-testid="stChatInput"] {
            direction: rtl; text-align: right;
        }
        [data-testid="stMarkdownContainer"] p { white-space: pre-wrap; }
        [data-testid="stChatInput"] textarea { direction: rtl; text-align: right; }
        [data-testid="stText"] { direction: rtl; text-align: right; white-space: pre-wrap; }
        </style>""",
        unsafe_allow_html=True,
    )
    st.title("עוזר לוח השנה שלי")
    st.caption("שאלו בעברית וקבלו תשובה עם המקורות התומכים בה.")

    if "messages" not in st.session_state:
        st.session_state.messages = []
    for message in st.session_state.messages:
        render_message(message)

    question = st.chat_input("כתבו שאלה על לוח השנה...")
    if question and question.strip():
        question = question.strip()
        user_message = {"role": "user", "content": question}
        st.session_state.messages.append(user_message)
        render_message(user_message)
        with st.spinner("מחפש תשובה..."):
            try:
                reply = answer_question(question)
            except Exception as error:
                reply = {
                    "role": "assistant", "content": error_message(error),
                    "sources": "", "error": True,
                }
        st.session_state.messages.append(reply)
        render_message(reply)


if __name__ == "__main__":
    main()
