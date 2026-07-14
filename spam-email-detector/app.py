import re
import joblib
import numpy as np
import streamlit as st

st.set_page_config(page_title="Spam Detector", page_icon="📧", layout="centered")

st.title("📧 Spam Email Detector")
st.write("Paste an email below and check if it's spam or not. (Model: Naive Bayes on Spambase dataset)")

st.divider()

# these are the 48 words used in the spambase dataset
words = ["make", "address", "all", "3d", "our", "over", "remove", "internet",
         "order", "mail", "receive", "will", "people", "report", "addresses",
         "free", "business", "email", "you", "credit", "your", "font", "000",
         "money", "hp", "hpl", "george", "650", "lab", "labs", "telnet", "857",
         "data", "415", "85", "technology", "1999", "parts", "pm", "direct",
         "cs", "meeting", "original", "project", "re", "edu", "table", "conference"]

chars = [";", "(", "[", "!", "$", "#"]


def get_features(text):
    text_lower = text.lower()
    all_words = re.findall(r"[a-zA-Z0-9$]+", text_lower)
    n_words = len(all_words) if len(all_words) > 0 else 1
    n_chars = len(text) if len(text) > 0 else 1

    word_freq = []
    for w in words:
        count = all_words.count(w)
        word_freq.append(100 * count / n_words)

    char_freq = []
    for c in chars:
        char_freq.append(100 * text.count(c) / n_chars)

    caps = re.findall(r"[A-Z]+", text)
    if len(caps) == 0:
        cap_avg, cap_max, cap_total = 0, 0, 0
    else:
        lengths = [len(c) for c in caps]
        cap_avg = sum(lengths) / len(lengths)
        cap_max = max(lengths)
        cap_total = sum(lengths)

    features = word_freq + char_freq + [cap_avg, cap_max, cap_total]
    return np.array(features).reshape(1, -1), all_words


# ---- sidebar with model info ----
with st.sidebar:
    st.header("ℹ️ About")
    st.write("This app uses a **Naive Bayes** model trained on the "
             "UCI Spambase dataset (57 features) to classify emails.")
    st.write("**Features used:**")
    st.write("- Frequency of 48 common spam words")
    st.write("- Frequency of 6 symbols (!, $, # etc.)")
    st.write("- Capital letter run-length stats")
    st.caption("Made by MD Farid Khan")

# ---- sample buttons ----
st.write("**Try a sample:**")
c1, c2, c3 = st.columns(3)

if "email_text" not in st.session_state:
    st.session_state.email_text = ""

with c1:
    if st.button("🚨 Spam example", use_container_width=True):
        st.session_state.email_text = (
            "CONGRATULATIONS! You have WON $1,000,000 in our lottery!!! "
            "Click here NOW to claim your FREE prize. Send your credit "
            "card information immediately. 100% FREE, order now!!!"
        )

with c2:
    if st.button("✅ Normal example", use_container_width=True):
        st.session_state.email_text = (
            "Hi George, please find attached the project report for "
            "tomorrow's meeting. Let me know if you have questions "
            "before the conference call. Thanks, Sarah"
        )

with c3:
    if st.button("🧹 Clear", use_container_width=True):
        st.session_state.email_text = ""

email_text = st.text_area("Email text", height=200, key="email_text")

check_btn = st.button("🔍 Check Spam", type="primary", use_container_width=True)

st.divider()

if check_btn:
    if email_text.strip() == "":
        st.warning("Please paste some text first!")
    else:
        model = joblib.load("spam_model.pkl")
        features, all_words = get_features(email_text)
        result = model.predict(features)[0]

        # result box
        if result == 1:
            st.markdown(
                "<div style='background-color:#3a1414; padding:20px; "
                "border-radius:10px; border:1px solid #ff5a5a;'>"
                "<h3 style='color:#ff5a5a; margin:0;'>🚨 This looks like SPAM</h3>"
                "</div>", unsafe_allow_html=True
            )
        else:
            st.markdown(
                "<div style='background-color:#122b18; padding:20px; "
                "border-radius:10px; border:1px solid #3fb950;'>"
                "<h3 style='color:#3fb950; margin:0;'>✅ This looks like a normal email</h3>"
                "</div>", unsafe_allow_html=True
            )

        st.write("")

        # probability
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(features)[0]
            col1, col2 = st.columns(2)
            col1.metric("Spam probability", f"{proba[1]*100:.1f}%")
            col2.metric("Not spam probability", f"{proba[0]*100:.1f}%")
            st.progress(proba[1])

        # show which spam words were found in the text (nice extra detail)
        found_words = [w for w in words if w in all_words]
        if found_words:
            st.write("**Spam-related words found in text:**")
            st.write(", ".join(found_words))
        else:
            st.write("No common spam words found in this text.")