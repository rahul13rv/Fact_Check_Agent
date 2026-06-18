import streamlit as st
import PyPDF2
import requests
import json
import os
from dotenv import load_dotenv
from tavily import TavilyClient

load_dotenv()

TAVILY_KEY = os.getenv("TAVILY_API_KEY", "")
OPENROUTER_KEY = os.getenv("OPENROUTER_API_KEY", "")

try:
    TAVILY_KEY = st.secrets["TAVILY_API_KEY"]
    OPENROUTER_KEY = st.secrets["OPENROUTER_API_KEY"]
except:
    pass


st.set_page_config(page_title="Fact-Check Agent", page_icon="🔍", layout="wide")
st.title("🔍 The Fact-Check Agent")
st.write("Upload a marketing PDF to extract claims, cross-reference against live web data, and flag inaccuracies.")

missing = [k for k, v in {"TAVILY_API_KEY": TAVILY_KEY, "OPENROUTER_API_KEY": OPENROUTER_KEY}.items()
           if not v or "your_" in v]
if missing:
    st.warning(f"⚠️ Missing API keys: **{', '.join(missing)}**. Set them in `.env`.")

st.divider()


def read_pdf(pdf_file):
    reader = PyPDF2.PdfReader(pdf_file)
    return "\n".join(p.extract_text() or "" for p in reader.pages)


def llm(prompt):
    resp = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers={"Authorization": f"Bearer {OPENROUTER_KEY}", "Content-Type": "application/json"},
        json={
            "model": "nex-agi/nex-n2-pro:free",
            "messages": [{"role": "user", "content": prompt}],
            "reasoning": {"enabled": True},
        },
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


def web_search(query):
    return TavilyClient(TAVILY_KEY).search(query=query, search_depth="advanced")


def parse_json(raw):
    s = raw.strip()
    if s.startswith("```json"):
        s = s[7:]
    elif s.startswith("```"):
        s = s[3:]
    if s.endswith("```"):
        s = s[:-3]
    return json.loads(s.strip())


EXTRACT_PROMPT = """\
Analyze this marketing document text. Pull out the top 5 most specific, \
verifiable claims — stats, percentages, financial figures, dates, named facts. \
Skip subjective fluff. If fewer than 5 exist, return only those. \
Respond with ONLY raw JSON, no markdown: {{"claims": ["claim1", "claim2"]}}

Text:
{text}"""

VERIFY_PROMPT = """\
Verify this claim using the search results below. \
Classify as "Verified", "Inaccurate", or "False". \
If inaccurate, give the corrected fact. Include up to 2 source URLs. \
Respond with ONLY raw JSON:
{{{{"verdict":"...","correct_fact":"...","evidence":"...","sources":["url1"]}}}}

Claim: "{claim}"

Search results:
{context}"""


uploaded = st.file_uploader("📤 Upload PDF Document", type="pdf")

if uploaded and st.button("🚀 Start Fact-Checking", type="primary", disabled=bool(missing)):

    with st.spinner("Reading PDF..."):
        text = read_pdf(uploaded)

    if not text.strip():
        st.error("Could not extract text from this PDF.")
        st.stop()

    st.info(f"📄 Extracted {len(text)} characters.")

    with st.spinner("Extracting claims..."):
        try:
            raw = llm(EXTRACT_PROMPT.format(text=text[:8000]))
            claims = parse_json(raw).get("claims", [])
        except json.JSONDecodeError:
            st.error("LLM returned unparseable JSON.")
            with st.expander("Raw response"):
                st.code(raw)
            st.stop()
        except Exception as e:
            st.error(f"Claim extraction failed: {e}")
            st.stop()

    if not claims:
        st.warning("No checkable claims found.")
        st.stop()

    st.success(f"Found {len(claims)} claims to verify!")
    st.divider()

    results = []
    bar = st.progress(0, text="Verifying claims...")

    for i, claim in enumerate(claims):
        with st.spinner(f"Checking ({i+1}/{len(claims)}): {claim[:60]}..."):
            try:
                hits = web_search(claim)
                context = "\n".join(
                    f"URL: {r.get('url')}\n{r.get('content')}\n"
                    for r in hits.get("results", [])
                )
                verdict = parse_json(llm(VERIFY_PROMPT.format(claim=claim, context=context)))
                verdict["original_claim"] = claim
                results.append(verdict)
            except Exception as e:
                results.append({
                    "original_claim": claim, "verdict": "Error",
                    "correct_fact": str(e), "evidence": "", "sources": [],
                })
        bar.progress((i + 1) / len(claims))

    bar.empty()
    st.divider()

    st.header("📊 Fact-Check Report")

    verdicts = [r.get("verdict", "Error") for r in results]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("✅ Verified", verdicts.count("Verified"))
    c2.metric("⚠️ Inaccurate", verdicts.count("Inaccurate"))
    c3.metric("❌ False", verdicts.count("False"))
    c4.metric("🚫 Errors", verdicts.count("Error"))
    st.divider()

    icons = {"Verified": "✅", "Inaccurate": "⚠️", "False": "❌", "Error": "🚫"}

    for r in results:
        v = r.get("verdict", "Error")
        st.subheader(f"{icons.get(v, '❓')} {v}")
        st.write(f"**Claim:** {r.get('original_claim')}")
        if v == "Inaccurate":
            st.write(f"**Correct Fact:** {r.get('correct_fact')}")
        st.write(f"**Evidence:** {r.get('evidence')}")
        for url in r.get("sources", []):
            if url:
                st.markdown(f"- [{url}]({url})")
        st.divider()
