import streamlit as st
import pandas as pd
import joblib
from pathlib import Path
import math
import numpy as np
from rag_explainer import build_rag_explanation

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Money Follows",
    page_icon="💰",
    layout="wide",
)

MODEL_PATH = Path(__file__).parent / "money_follows_model.pkl"


@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file not found: {MODEL_PATH.name}. "
            "Place money_follows_model.pkl in the same folder as app.py."
        )
    return joblib.load(MODEL_PATH)


try:
    model = load_model()
except Exception as exc:
    st.error("Money Follows could not load its trained model.")
    st.code(str(exc))
    st.stop()



def student_decision_pathway(college_tier: str, entrance_score: float, loan_amount: float) -> str:
    """Return the project's illustrative student-loan pathway."""
    amount = float(loan_amount)
    score = float(entrance_score)

    if college_tier == "Tier 1":
        if score < 40 or amount > 9_000_000:
            return "Detailed Assessment"
        if score >= 80 and amount <= 9_000_000:
            return "Easy Approval"
        if score >= 60 and amount <= 8_000_000:
            return "Standard Approval"
        if score >= 40 and amount <= 5_000_000:
            return "Additional Assessment"
        return "Detailed Assessment"

    if college_tier == "Tier 2":
        if amount > 2_050_000:
            return "Detailed Assessment"
        if score >= 75:
            return "Easy Approval"
        if score >= 50:
            return "Standard Approval"
        return "Additional Assessment"

    # Tier 3
    if amount > 1_000_000:
        return "Detailed Assessment"
    if score >= 60:
        return "Easy Approval" if amount <= 550_000 else "Standard Approval"
    if score >= 40:
        return "Easy Approval" if amount <= 550_000 else "Additional Assessment"
    return "Additional Assessment"


# ---------------- PREMIUM DARK THEME ----------------
st.markdown(
    """
    <style>
    :root { --mf-lime:#d7ff45; --mf-gold:#c9a64b; --mf-muted:#a5a5a5; }
    .stApp {
      background: radial-gradient(ellipse at 50% 15%, rgba(176,220,20,.12), transparent 38%),
                  radial-gradient(ellipse at 50% 0%, rgba(215,255,69,.08), transparent 52%), #050605;
      color:#f7f7f2;
    }
    [data-testid="stHeader"] { background:rgba(5,6,5,.85); }
    h1,h2,h3,h4,p,label,span,div { color:inherit; }
    .block-container { max-width: 1240px; padding-top: 1.2rem; padding-bottom: 3rem; }
    .mf-nav { display:flex; align-items:center; justify-content:space-between; gap:1rem;
      padding:16px 22px; border:1px solid rgba(215,255,69,.18); border-radius:18px;
      background:rgba(12,13,11,.88); box-shadow:0 10px 40px rgba(0,0,0,.28); margin-bottom:42px; }
    .mf-brand { font-size:1.35rem; font-weight:800; letter-spacing:-.04em; color:#f8f8f0; }
    .mf-brand span { color:var(--mf-lime); }
    .mf-navnote { color:#b8b8b0; font-size:.92rem; }
    .mf-hero { text-align:center; padding:35px 12px 20px; }
    .mf-pill { display:inline-block; border:1px solid rgba(215,255,69,.48); color:var(--mf-lime);
      border-radius:999px; padding:8px 16px; font-size:.85rem; letter-spacing:.04em;
      background:rgba(215,255,69,.07); box-shadow:0 0 24px rgba(215,255,69,.10); }
    .mf-title { font-size:clamp(2.6rem,5.8vw,5.1rem); line-height:1.06; font-weight:800;
      letter-spacing:-.055em; margin:24px auto 18px; max-width:950px; color:#f7f7f2; }
    .mf-title em { color:var(--mf-lime); font-style:normal; text-shadow:0 0 30px rgba(215,255,69,.18); }
    .mf-copy { color:#b8b9b1; font-size:1.08rem; line-height:1.75; max-width:700px; margin:0 auto 26px; }
    .mf-arc { height:70px; max-width:900px; margin:5px auto 20px; border-top:4px solid var(--mf-lime);
      border-radius:50% 50% 0 0 / 100% 100% 0 0; filter:drop-shadow(0 0 8px rgba(215,255,69,.9))
      drop-shadow(0 0 28px rgba(215,255,69,.55)); opacity:.95; }
    .mf-card { background:linear-gradient(145deg,rgba(19,21,16,.97),rgba(8,9,8,.98));
      border:1px solid rgba(215,255,69,.24); border-radius:22px; padding:28px;
      box-shadow:0 18px 65px rgba(0,0,0,.35), inset 0 1px 0 rgba(255,255,255,.025); }
    [data-testid="stForm"] { background:rgba(15,17,13,.96); border:1px solid rgba(215,255,69,.25);
      border-radius:18px; padding:22px; }
    input,textarea,[data-baseweb="select"] > div,[data-baseweb="input"] > div {
      background:#151713 !important; color:#f7f7f2 !important; border-color:#373b2d !important; }
    [data-testid="stNumberInput"] button { color:var(--mf-lime) !important; }
    div.stButton > button, div.stFormSubmitButton > button {
      background:var(--mf-lime); color:#090a06; border:1px solid #e8ff91; border-radius:999px;
      padding:.72rem 1.4rem; font-weight:800; min-height:48px; box-shadow:0 0 22px rgba(215,255,69,.20);
      transition:all .2s ease; }
    div.stButton > button:hover, div.stFormSubmitButton > button:hover {
      background:#e6ff83; color:#050605; border-color:#efffae; transform:translateY(-1px);
      box-shadow:0 0 32px rgba(215,255,69,.35); }
    [data-testid="stMetric"], [data-testid="stExpander"] { background:#10120e; border:1px solid #303426;
      border-radius:16px; padding:14px; }
    [data-testid="stMetricValue"] { color:var(--mf-lime) !important; }
    [data-testid="stAlert"] { border-radius:14px; }
    hr { border-color:#2c3023; }
    .mf-footer { text-align:center; color:#77796f; font-size:.82rem; margin-top:42px; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------- SESSION STATE ----------------
if "user_name" not in st.session_state:
    st.session_state.user_name = ""
if "page" not in st.session_state:
    st.session_state.page = "welcome"

# ---------------- BRAND HEADER ----------------
st.markdown(
    "<div class='mf-nav'><div class='mf-brand'>◈ <span>Money</span> Follows</div>"
    "<div class='mf-navnote'>AI-Powered Loan Assessment&nbsp;&nbsp; · &nbsp;&nbsp;Clearer decisions</div></div>",
    unsafe_allow_html=True,
)

# ---------------- WELCOME / LANDING PAGE ----------------
if st.session_state.page == "welcome":
    st.markdown("<div class='mf-hero'>", unsafe_allow_html=True)
    st.markdown("<div class='mf-pill'>✦ &nbsp; AI-POWERED LOAN ASSESSMENT</div>", unsafe_allow_html=True)
    st.markdown("<div class='mf-arc'></div>", unsafe_allow_html=True)
    st.markdown("<div class='mf-title'>Make your next move<br>with <em>confidence.</em></div>", unsafe_allow_html=True)
    st.markdown("<div class='mf-copy'>A clearer way to understand your loan options. Start with your name and explore a tailored assessment experience.</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
    left, center, right = st.columns([1, 1.05, 1])
    with center:
        #st.markdown("<div class='mf-card'>", unsafe_allow_html=True)
        st.markdown("### Welcome! 👋")
        st.markdown("Enter your name to begin your loan assessment.")
        with st.form("welcome_form"):
            name = st.text_input("Your name", placeholder="Enter your name", label_visibility="visible")
            start = st.form_submit_button("Get Started  →", use_container_width=True)
        st.markdown("<div style='color:#8f9384;font-size:.82rem;text-align:center;margin-top:10px;'>A simple start. A clearer next step.</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
        if start:
            if name.strip():
                st.session_state.user_name = name.strip()
                st.session_state.page = "dashboard"
                st.rerun()
            else:
                st.warning("Please enter your name to continue.")
    st.markdown("<div class='mf-footer'>Money Follows · Created by Side Kicks - A B6 Group<br>Avaneesh · Deepak · Deeksha · Vishitha</div>", unsafe_allow_html=True)

# ---------------- LOAN DASHBOARD ----------------
elif st.session_state.page == "dashboard":
    st.markdown(f"<div class='mf-hero' style='padding:8px 0 18px;'><div class='mf-pill'>YOUR ASSESSMENT SPACE</div><div class='mf-title' style='font-size:clamp(2rem,4vw,3.4rem);'>Welcome back, <em>{st.session_state.user_name}</em></div><div class='mf-copy'>Share a few details to explore your loan assessment.</div></div>", unsafe_allow_html=True)

    loan_type_display = st.selectbox(
        "Select loan type",
        [
            "🏠 Home Loan",
            "💳 Personal Loan",
            "🎓 Education Loan",
            "🏢 Business Loan",
            "🚗 Vehicle Loan",
        ],
    )

    with st.form("loan_application"):
        st.subheader("1. Applicant Details")
        col1, col2 = st.columns(2)

        is_student_loan = loan_type_display == "🎓 Education Loan"

        with col1:
            age = st.number_input("Age", min_value=18, max_value=100, value=25)
            if not is_student_loan:
                monthly_income = st.number_input(
                    "Monthly Income (₹)", min_value=0, value=50000, step=1000
                )
            else:
                monthly_income = None
            if not is_student_loan:
                employment_years = st.number_input(
                    "Years of Employment", min_value=0.0, max_value=60.0,
                    value=2.0, step=0.5
                )
            else:
                # Not collected for the student-loan pathway.
                employment_years = 0.0

        with col2:
            if not is_student_loan:
                cibil_score = st.number_input(
                    "CIBIL Score", min_value=300, max_value=900, value=750
                )
            else:
                cibil_score = None
            existing_debt = st.number_input(
                "Existing Monthly Loan Payments (₹)", min_value=0,
                value=5000, step=500
            )
            missed_payments = st.number_input(
                "Missed Payments in Recent Years", min_value=0,
                max_value=100, value=0
            )

        # Education-specific inputs appear only for student loans.
        entrance_score = None
        college_tier = None
        if is_student_loan:
            st.subheader("Education Details")
            edu_col1, edu_col2 = st.columns(2)
            with edu_col1:
                entrance_score = st.number_input(
                    "Entrance Exam Score (%)", min_value=0.0, max_value=100.0,
                    value=75.0, step=1.0
                )
            with edu_col2:
                college_tier = st.selectbox(
                    "College Tier", ["Tier 1", "Tier 2", "Tier 3"]
                )

        st.subheader("2. Loan Details")
        col3, col4 = st.columns(2)
        with col3:
            loan_amount = st.number_input(
                "Requested Loan Amount (₹)", min_value=1000,
                value=500000, step=10000
            )
            annual_interest = st.number_input(
                "Annual Interest Rate (%)", min_value=0.1,
                max_value=40.0, value=10.0, step=0.5
            )
        with col4:
            tenure_years = st.number_input(
                "Repayment Tenure (Years)", min_value=1,
                max_value=40, value=5
            )
            down_payment = st.number_input(
                "Down Payment / Own Contribution (₹)", min_value=0,
                value=100000, step=10000
            )

        st.subheader("3. Additional Details")
        asset_value = None
        annual_revenue = 0.0
        annual_profit = 0.0
        education_cost = 0.0
        financial_support = 0.0

        if loan_type_display == "🏠 Home Loan":
            asset_value = st.number_input(
                "Property Value (₹)", min_value=1, value=800000, step=50000
            )
        elif loan_type_display == "🚗 Vehicle Loan":
            asset_value = st.number_input(
                "Vehicle Value (₹)", min_value=1, value=800000, step=50000
            )
        elif loan_type_display == "🎓 Education Loan":
            education_cost = st.number_input(
                "Total Education Cost (₹)", min_value=0,
                value=1000000, step=50000
            )
            financial_support = st.number_input(
                "Scholarships / Family Contribution (₹)", min_value=0,
                value=200000, step=10000
            )
        elif loan_type_display == "🏢 Business Loan":
            annual_revenue = st.number_input(
                "Annual Business Revenue (₹)", min_value=0,
                value=1200000, step=50000
            )
            annual_profit = st.number_input(
                "Annual Business Profit (₹)", min_value=0,
                value=200000, step=10000
            )

        submitted = st.form_submit_button(
            "Assess Application", use_container_width=True
        )

    if submitted:
        errors = []
        if not is_student_loan and monthly_income <= 0:
            errors.append("Monthly income must be greater than zero.")
        if loan_amount <= 0:
            errors.append("Loan amount must be greater than zero.")
        if tenure_years <= 0:
            errors.append("Repayment tenure must be greater than zero.")
        if loan_type_display in ("🏠 Home Loan", "🚗 Vehicle Loan"):
            if asset_value is None or asset_value <= 0:
                errors.append("Enter a valid property or vehicle value.")
        if loan_type_display == "🎓 Education Loan" and financial_support > education_cost:
            errors.append("Financial support cannot exceed education cost.")

        if errors:
            for message in errors:
                st.error(message)
        else:
            principal = float(loan_amount)
            monthly_rate = float(annual_interest) / 1200
            months = int(tenure_years) * 12
            if monthly_rate == 0:
                emi = principal / months
            else:
                growth = (1 + monthly_rate) ** months
                emi = principal * monthly_rate * growth / (growth - 1)

            total_monthly_debt = float(existing_debt) + emi
            if is_student_loan:
                # Income and CIBIL are intentionally not requested for student loans.
                dti = float("nan")
                cibil_category = "Not collected for student loans"
            else:
                dti = total_monthly_debt / float(monthly_income) * 100
                if cibil_score < 680:
                    cibil_category = "Needs Improvement"
                elif cibil_score <= 730:
                    cibil_category = "Fair"
                elif cibil_score <= 770:
                    cibil_category = "Good"
                elif cibil_score <= 790:
                    cibil_category = "Very Good"
                else:
                    cibil_category = "Excellent"

            ltv = (
                float(loan_amount) / float(asset_value) * 100
                if asset_value is not None and asset_value > 0
                else float("nan")
            )

            st.divider()
            st.header("Application Results")

            if is_student_loan:
                pathway = student_decision_pathway(
                    college_tier, entrance_score, loan_amount
                )
                st.subheader("Education Loan Decision Pathway")
                if pathway == "Easy Approval":
                    st.success(f"Rule-based pathway: **{pathway}**")
                elif pathway == "Standard Approval":
                    st.info(f"Rule-based pathway: **{pathway}**")
                elif pathway == "Additional Assessment":
                    st.warning(f"Rule-based pathway: **{pathway}**")
                else:
                    st.error(f"Rule-based pathway: **{pathway}**")
                st.caption("Based on college tier, entrance score, and requested loan amount.")

            metric1, metric2, metric3 = st.columns(3)
            metric1.metric("Estimated Monthly EMI", f"₹{emi:,.2f}")
            if is_student_loan:
                metric2.metric("Entrance Exam Score", f"{entrance_score:.1f}%")
                metric3.metric("College Tier", college_tier)
            else:
                metric2.metric("Debt-to-Income Ratio", f"{dti:.1f}%")
                metric3.metric("CIBIL Band", cibil_category)

            with st.expander("View financial summary", expanded=True):
                summary1, summary2 = st.columns(2)
                with summary1:
                    st.write(f"**Applicant:** {st.session_state.user_name}")
                    st.write(f"**Loan type:** {loan_type_display}")
                    st.write(f"**Requested amount:** ₹{loan_amount:,.0f}")
                    st.write(f"**Annual interest rate:** {annual_interest:.1f}%")
                with summary2:
                    st.write(f"**Existing monthly debt:** ₹{existing_debt:,.0f}")
                    st.write(f"**Total monthly debt after loan:** ₹{total_monthly_debt:,.2f}")
                    st.write(f"**Repayment tenure:** {tenure_years} years")
                    if not math.isnan(ltv):
                        st.write(f"**Loan-to-value ratio:** {ltv:.1f}%")

            # Match the training dataset's loan-type labels.
            model_loan_type = {
                "🏠 Home Loan": "Home Loan",
                "💳 Personal Loan": "Personal Loan",
                "🎓 Education Loan": "Student Loan",
                "🏢 Business Loan": "Business Loan",
                "🚗 Vehicle Loan": "Vehicle Loan",
            }[loan_type_display]

            # Match the exact feature names from the saved model.
            model_input = pd.DataFrame([{
                "loan_type": model_loan_type,
                "age": age,
                "monthly_income": float(monthly_income) if monthly_income is not None else np.nan,
                "employment_years": employment_years,
                "cibil_score": float(cibil_score) if cibil_score is not None else np.nan,
                "existing_monthly_debt": existing_debt,
                "loan_amount": loan_amount,
                "annual_interest_rate": annual_interest,
                "tenure_years": tenure_years,
                "missed_payments": missed_payments,
                "asset_value": (
                    float(asset_value) if asset_value is not None else float("nan")
                ),
                "annual_business_revenue": annual_revenue,
                "annual_business_profit": annual_profit,
                "education_cost": education_cost,
                "financial_support": financial_support,
                "entrance_score": float(entrance_score) if entrance_score is not None else np.nan,
                "college_tier": college_tier if college_tier is not None else np.nan,
                "emi": emi,
                "dti": dti,
                "ltv": ltv,
            }])

            st.markdown("## Assessment Result")
            try:
                prediction = int(model.predict(model_input)[0])
                class_labels = list(model.classes_)
                if hasattr(model, "predict_proba") and 1 in class_labels:
                    probabilities = model.predict_proba(model_input)[0]
                    approved_score = float(probabilities[class_labels.index(1)]) * 100
                else:
                    approved_score = None
            except Exception as exc:
                st.error("The model could not process these inputs.")
                st.code(str(exc))
                st.stop()

            # High-visibility result card and compact financial snapshot.
            result_col, score_col = st.columns([1.5, 1])
            with result_col:
                if prediction == 1:
                    st.success("### Model Prediction: Prediction Scale")
                else:
                    st.error("### Model Prediction: Prediction Scale")
            with score_col:
                if approved_score is not None:
                    st.metric("Model Score — Approved Class", f"{approved_score:.1f}%")
                    st.progress(min(max(approved_score / 100, 0.0), 1.0))
                else:
                    st.metric("Model Score", "Unavailable")

            st.caption("Approved-class score: the model's estimated probability for the Approved class.")

            st.subheader("Financial Snapshot")
            snap1, snap2, snap3, snap4 = st.columns(4)
            snap1.metric("Monthly EMI", f"₹{emi:,.0f}")
            if is_student_loan:
                snap2.metric("Entrance Score", f"{entrance_score:.1f}%")
                snap3.metric("College Tier", college_tier)
            else:
                snap2.metric("Debt-to-Income", f"{dti:.1f}%")
                snap3.metric("CIBIL Score", f"{cibil_score}")
            snap4.metric("Missed Payments", f"{missed_payments}")

            st.subheader("Decision Explanation")
            rag_context = {
                "loan_type": model_loan_type,
                "prediction": prediction,
                "approved_score": approved_score,
                "loan_amount": float(loan_amount),
                "missed_payments": int(missed_payments),
                "employment_years": float(employment_years),
                "cibil_score": None if cibil_score is None else float(cibil_score),
                "dti": None if is_student_loan else float(dti),
                "college_tier": college_tier,
                "entrance_score": None if entrance_score is None else float(entrance_score),
                "pathway": pathway if is_student_loan else None,
                "education_cost": float(education_cost) if is_student_loan else None,
                "financial_support": float(financial_support) if is_student_loan else None,
            }
            rag_result = build_rag_explanation(rag_context)
            st.write(rag_result["explanation"])
            with st.expander("Knowledge retrieved for this explanation"):
                for item in rag_result["sources"]:
                    st.markdown(f"**{item['title']}**")
                    st.write(item["text"])

            with st.container(border=True):
                st.markdown("**Applicant financial profile**")
                factor_col1, factor_col2 = st.columns(2)
                with factor_col1:
                    if is_student_loan:
                        st.write(f"**Entrance exam score:** {entrance_score:.1f}%")
                        st.write(f"**College tier:** {college_tier}")
                        st.write(f"**Decision pathway:** {pathway}")
                    else:
                        if cibil_score >= 680:
                            st.success("CIBIL score is at or above the 680-point reference threshold.")
                        else:
                            st.warning("CIBIL score is below the 680-point reference threshold.")
                        if dti <= 40:
                            st.success("Debt-to-income ratio is at or below the 40% reference threshold.")
                        else:
                            st.warning("Debt-to-income ratio is above the 40% reference threshold.")
                with factor_col2:
                    if is_student_loan:
                        st.write(f"**Requested loan amount:** ₹{loan_amount:,.0f}")
                        remaining_cost = max(float(education_cost) - float(financial_support), 0)
                        st.write(f"**Education funding gap:** ₹{remaining_cost:,.0f}")
                        st.caption("The education pathway uses the college tier, entrance score, and requested amount.")
                    else:
                        if missed_payments <= 2:
                            st.success("Reported missed payments are at or below 2.")
                        else:
                            st.warning("Reported missed payments are above 2.")
                        if employment_years >= 2:
                            st.success("Employment history is at least 2 years.")
                        else:
                            st.warning("Employment history is less than 2 years.")

            st.subheader("Loan-Specific Context")
            if loan_type_display == "🏠 Home Loan":
                st.write(
                    f"**Property value:** ₹{asset_value:,.0f}  |  "
                    f"**Loan-to-value ratio:** {ltv:.1f}%"
                )
                st.write(
                    "The application includes the entered property value and the "
                    "requested loan amount. A higher loan-to-value ratio means the "
                    "requested loan is larger relative to the property value."
                )
            elif loan_type_display == "🚗 Vehicle Loan":
                st.write(
                    f"**Vehicle value:** ₹{asset_value:,.0f}  |  "
                    f"**Loan-to-value ratio:** {ltv:.1f}%"
                )
                st.write(
                    "The application includes the entered vehicle value and requested "
                    "loan amount. The ratio compares the loan with the vehicle value."
                )
            elif loan_type_display == "💳 Personal Loan":
                st.write(
                    f"**Requested amount:** ₹{loan_amount:,.0f}  |  "
                    f"**Monthly income:** ₹{monthly_income:,.0f}"
                )
                st.write(
                    "Review the requested amount alongside monthly income, existing "
                    "debt, estimated EMI, credit score, and employment history."
                )
            elif loan_type_display == "🎓 Education Loan":
                st.write(f"**Entrance exam score:** {entrance_score:.1f}%")
                st.write(f"**College tier:** {college_tier}")
                st.write(f"**Education cost:** ₹{education_cost:,.0f}")
                st.write(f"**Scholarships / family contribution:** ₹{financial_support:,.0f}")
                remaining_cost = max(float(education_cost) - float(financial_support), 0)
                st.write(f"**Education cost after entered support:** ₹{remaining_cost:,.0f}")
                st.write(
                    "These amounts summarize the education cost and the support entered."
                )
            elif loan_type_display == "🏢 Business Loan":
                st.write(f"**Annual business revenue:** ₹{annual_revenue:,.0f}")
                st.write(f"**Annual business profit:** ₹{annual_profit:,.0f}")
                if annual_revenue > 0:
                    st.write(f"**Entered profit margin:** {(annual_profit / annual_revenue) * 100:.1f}%")
                else:
                    st.write("**Entered profit margin:** Not available (revenue is zero).")
                st.write(
                    "Revenue and profit are applicant-entered context for the business "
                    "application; they are not independently verified financial records."
                )

            with st.expander("How to read this result"):
                st.write("The decision pathway and model prediction are separate assessment views, so their results may differ.")
                st.write("The approved-class score shows the model's estimated likelihood for the approved class.")

    st.markdown("<div class='mf-footer'>Money Follows · Side Kicks B6 Group · Avaneesh · Deepak · Deeksha · Vishitha</div>", unsafe_allow_html=True)
    if st.button("← Back to Welcome Page"):
        st.session_state.page = "welcome"
        st.rerun()
