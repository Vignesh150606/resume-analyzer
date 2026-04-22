import streamlit as st
import requests

# ── Page Configuration ────────────────────────────────────────
st.set_page_config(
    page_title="AI Resume Analyzer",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Constants ─────────────────────────────────────────────────
API_URL = "http://localhost:8080"

# ── Sidebar ───────────────────────────────────────────────────
with st.sidebar:
    st.header("💡 How It Works")
    st.markdown("""
    1. **Upload** your resume as PDF
    2. **Paste** the job description
    3. **Click** Analyze Resume
    4. Get your **match score**, **skill gaps**, and **AI suggestions**
    """)
    st.markdown("---")
    st.header("📌 Tips")
    st.markdown("""
    - Use a **digital PDF** (not scanned)
    - Paste the **full** job description
    - A score above **65** means good match
    - Add **missing keywords** to your resume
    - Tailor your resume for **each job**
    """)
    st.markdown("---")
    st.caption("Built with FastAPI + Sentence Transformers + Google Gemini")

# ── Custom CSS ────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 1rem 0;
        background: linear-gradient(90deg, #1a1a2e, #16213e);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .skill-match {
        background-color: #d4edda;
        padding: 0.3rem 0.6rem;
        border-radius: 15px;
        margin: 0.2rem;
        display: inline-block;
        font-size: 0.85rem;
        color: #155724;
    }
    .skill-missing {
        background-color: #f8d7da;
        padding: 0.3rem 0.6rem;
        border-radius: 15px;
        margin: 0.2rem;
        display: inline-block;
        font-size: 0.85rem;
        color: #721c24;
    }
    .skill-extra {
        background-color: #cce5ff;
        padding: 0.3rem 0.6rem;
        border-radius: 15px;
        margin: 0.2rem;
        display: inline-block;
        font-size: 0.85rem;
        color: #004085;
    }
    .suggestion-high {
        border-left: 4px solid #dc3545;
        padding: 0.8rem;
        margin: 0.5rem 0;
        background: #fff5f5;
        border-radius: 0 8px 8px 0;
    }
    .suggestion-medium {
        border-left: 4px solid #ffc107;
        padding: 0.8rem;
        margin: 0.5rem 0;
        background: #fffdf0;
        border-radius: 0 8px 8px 0;
    }
    .suggestion-low {
        border-left: 4px solid #28a745;
        padding: 0.8rem;
        margin: 0.5rem 0;
        background: #f0fff4;
        border-radius: 0 8px 8px 0;
    }
</style>
""", unsafe_allow_html=True)


# ── display_results DEFINED BEFORE USE ────────────────────────
def display_results(data: dict):
    """Render all analysis results in a clean, organized layout."""

    st.header("📊 Analysis Results")

    final = data["final_score"]
    grade = final["grade"]

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Overall Score", f"{final['final_score']}/100")
    with col2:
        st.metric("Grade", f"{grade} — {final['label']}")
    with col3:
        st.metric("Keyword Match", f"{data['skill_comparison']['match_percentage']}%")
    with col4:
        st.metric("Semantic Match", f"{round(data['semantic_score'] * 100, 1)}%")

    st.progress(float(final["final_score"]) / 100)

    # Score breakdown
    with st.expander("📈 Score Breakdown Details"):
        breakdown = final.get("breakdown", {})
        if breakdown:
            st.markdown("**How your score was calculated:**")
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                kw = breakdown.get("keyword_match_contribution", 0)
                st.metric("Keyword Match (40%)", f"{kw}/40")
                st.progress(float(kw) / 40 if kw <= 40 else 1.0)
            with col_b:
                sem = breakdown.get("semantic_contribution", 0)
                st.metric("Semantic Match (40%)", f"{sem}/40")
                st.progress(float(sem) / 40 if sem <= 40 else 1.0)
            with col_c:
                sec = breakdown.get("section_contribution", 0)
                st.metric("Section Score (20%)", f"{sec}/20")
                st.progress(float(sec) / 20 if sec <= 20 else 1.0)

    st.markdown("---")

    left, right = st.columns([1, 1], gap="large")

    with left:
        st.subheader("🎯 Skill Analysis")

        matched = data["skill_comparison"]["matched_skills"]
        missing = data["skill_comparison"]["missing_skills"]
        extra = data["skill_comparison"]["extra_skills"]

        if matched:
            st.markdown("**✅ Skills matched with JD:**")
            skills_html = " ".join([
                f'<span class="skill-match">{s}</span>' for s in matched
            ])
            st.markdown(skills_html, unsafe_allow_html=True)

        if missing:
            st.markdown("**❌ Skills missing from resume:**")
            skills_html = " ".join([
                f'<span class="skill-missing">{s}</span>' for s in missing
            ])
            st.markdown(skills_html, unsafe_allow_html=True)

        if extra:
            st.markdown("**➕ Additional skills you have:**")
            skills_html = " ".join([
                f'<span class="skill-extra">{s}</span>' for s in extra
            ])
            st.markdown(skills_html, unsafe_allow_html=True)

        st.markdown("---")
        st.subheader("📑 Section Relevance")
        for section, score in sorted(
            data["section_scores"].items(),
            key=lambda x: x[1],
            reverse=True
        ):
            st.markdown(f"**{section.title()}**")
            st.progress(float(score), text=f"{round(score * 100, 1)}%")

    with right:
        llm = data["llm_analysis"]

        st.subheader("🤖 AI Assessment")

        prob = llm["hiring_probability"].upper()
        prob_icon = {"HIGH": "🟢", "MEDIUM": "🟡", "LOW": "🔴"}.get(prob, "⚪")
        st.markdown(f"**Shortlisting Probability:** {prob_icon} {prob}")
        st.info(llm["overall_assessment"])

        st.markdown("**💪 Your Strengths:**")
        for strength in llm["top_strengths"]:
            st.markdown(f"✅ {strength}")

        gaps = llm.get("critical_gaps", [])
        if gaps and gaps != ["No critical gaps detected"]:
            st.markdown("**⚠️ Critical Gaps:**")
            for gap in gaps:
                st.markdown(f"❌ {gap}")

        st.markdown("---")
        st.subheader("💡 Improvement Suggestions")

        for suggestion in llm["improvement_suggestions"]:
            priority = suggestion.get("priority", "medium").lower()
            section = suggestion.get("section", "General")
            issue = suggestion.get("issue", "")
            fix = suggestion.get("suggestion", "")

            if not section and not issue and not fix:
                continue

            css_class = f"suggestion-{priority}"
            priority_icon = {"high": "🔴", "medium": "🟡", "low": "🟢"}.get(
                priority, "⚪"
            )
            content_lines = [f"<strong>{priority_icon} {section}</strong>"]
            if issue:
                content_lines.append(f"<em>Issue:</em> {issue}")
            if fix:
                content_lines.append(f"<em>Fix:</em> {fix}")

            st.markdown(f"""
<div class="{css_class}">
    {'<br>'.join(content_lines)}
</div>
""", unsafe_allow_html=True)

    st.markdown("---")

    st.subheader("🔍 ATS Keywords to Add")
    ats_keywords = llm.get("ats_keywords_to_add", [])
    if ats_keywords:
        keywords_html = " ".join([
            f'<span class="skill-missing">{kw}</span>'
            for kw in ats_keywords
        ])
        st.markdown(keywords_html, unsafe_allow_html=True)
        st.caption("Add these exact keywords to pass ATS filters")
    else:
        st.success("✅ Your resume contains the key ATS keywords for this role")

    st.markdown("---")

    st.subheader("✍️ AI-Suggested Professional Summary")
    st.markdown(f"> {llm['rewritten_summary']}")
    st.caption("Copy and customize this for your resume")

    st.markdown("---")

    with st.expander("🔧 View Raw JSON Response"):
        st.json(data)


# ── Header ────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>📄 AI Resume Analyzer</h1>
    <p>Upload your resume and paste a job description to get AI-powered analysis</p>
</div>
""", unsafe_allow_html=True)


# ── Input Section ─────────────────────────────────────────────
col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.subheader("📎 Upload Resume")
    uploaded_file = st.file_uploader(
        "Choose your resume PDF",
        type=["pdf"],
        help="Upload a digital PDF (not scanned)"
    )
    if uploaded_file:
        st.success(
            f"✅ Uploaded: {uploaded_file.name} "
            f"({uploaded_file.size / 1024:.1f} KB)"
        )

with col2:
    st.subheader("📋 Job Description")
    job_description = st.text_area(
        "Paste the full job description here",
        height=200,
        placeholder="We are looking for a Python Backend Developer..."
    )
    jd_word_count = len(job_description.split()) if job_description else 0
    st.caption(f"{jd_word_count} words")


# ── Analyze Button ────────────────────────────────────────────
st.markdown("---")
_, col_btn, _ = st.columns([1, 1, 1])

with col_btn:
    analyze_clicked = st.button(
        "🔍 Analyze Resume",
        type="primary",
        use_container_width=True,
        disabled=(uploaded_file is None or len(job_description.strip()) < 50)
    )

if uploaded_file is None:
    st.info("👆 Upload a PDF resume to get started")
elif len(job_description.strip()) < 50:
    st.warning("⚠️ Please paste a job description (minimum 50 characters)")


# ── Run Analysis ──────────────────────────────────────────────
if analyze_clicked:
    with st.spinner("🤖 Analyzing your resume... This takes 15-30 seconds"):
        try:
            response = requests.post(
                f"{API_URL}/analyze",
                files={
                    "resume": (
                        uploaded_file.name,
                        uploaded_file.getvalue(),
                        "application/pdf"
                    )
                },
                data={"job_description": job_description},
                timeout=120
            )

            if response.status_code == 200:
                data = response.json()
                st.success("✅ Analysis complete!")
                st.markdown("---")
                display_results(data)
            else:
                error_detail = response.json().get("detail", "Unknown error")
                st.error(f"❌ Analysis failed: {error_detail}")

        except requests.exceptions.ConnectionError:
            st.error(
                "❌ Cannot connect to backend. "
                "Make sure the FastAPI server is running on port 8080."
            )
        except requests.exceptions.Timeout:
            st.error("❌ Request timed out. Please try again.")
        except Exception as e:
            st.error(f"❌ Unexpected error: {str(e)}")