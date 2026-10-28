"""
Money Follows lightweight RAG explanation helper.

This free version retrieves relevant project rules using TF-IDF (scikit-learn)
and assembles a grounded explanation from those retrieved passages. It does
not call a paid API or claim to be an LLM-generated response.
"""
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

KNOWLEDGE_BASE = [
    {
        "title": "How to interpret the model prediction",
        "text": "The Random Forest model predicts the approved or rejected class from the application fields it was trained on. Its approved-class score is the model's estimated probability for the approved class, not a guarantee of a real lender decision."
    },
    {
        "title": "Education loan pathway inputs",
        "text": "The Money Follows education loan pathway uses college tier, entrance exam score, and requested loan amount. The pathway is a separate rule-based assessment view and can differ from the Random Forest model prediction."
    },
    {
        "title": "Tier 1 education pathway",
        "text": "For Tier 1 education applications, this project's configured pathway sends scores below 40 percent or requests above ₹90 lakh to Detailed Assessment. Scores of at least 80 percent and amounts up to ₹90 lakh use Easy Approval. Scores of at least 60 percent and amounts up to ₹80 lakh use Standard Approval. Remaining cases with scores at least 40 percent and amounts up to ₹50 lakh use Additional Assessment; other cases use Detailed Assessment."
    },
    {
        "title": "Tier 2 education pathway",
        "text": "For Tier 2 education applications, requests above ₹20.5 lakh use Detailed Assessment. For amounts within that limit, scores of at least 75 percent use Easy Approval, scores of at least 50 percent use Standard Approval, and lower scores use Additional Assessment."
    },
    {
        "title": "Tier 3 education pathway",
        "text": "For Tier 3 education applications, requests above ₹10 lakh use Detailed Assessment. For requests up to ₹10 lakh and scores at least 60 percent, amounts up to ₹5.5 lakh use Easy Approval and higher amounts use Standard Approval. For scores from 40 percent to below 60 percent, amounts up to ₹5.5 lakh use Easy Approval and higher amounts use Additional Assessment. Scores below 40 percent use Additional Assessment."
    },
    {
        "title": "Student education funding gap",
        "text": "The education funding gap is the entered education cost minus scholarships or family contribution, with the result not below zero. These values summarize information entered by the applicant."
    },
    {
        "title": "Credit score context",
        "text": "For non-student applications, the app displays reference bands for CIBIL score: below 680 Needs Improvement, 680 to 730 Fair, 731 to 770 Good, 771 to 790 Very Good, and above 790 Excellent. These are the project's display bands."
    },
    {
        "title": "Debt-to-income context",
        "text": "For non-student applications, the app compares the debt-to-income ratio with a 40 percent reference threshold. A ratio at or below 40 percent is displayed as at or below the reference threshold; above 40 percent is displayed as above the reference threshold."
    },
    {
        "title": "Missed payment context",
        "text": "The app highlights whether reported missed payments are at or below 2 or above 2. This is a contextual reference used in the displayed explanation."
    },
    {
        "title": "Employment history context",
        "text": "The app highlights whether reported employment history is at least 2 years or less than 2 years. This is a contextual reference used in the displayed explanation."
    },
    {
        "title": "Loan amount and EMI",
        "text": "The estimated monthly EMI is calculated from the requested principal, annual interest rate, and repayment tenure entered by the applicant. Existing monthly loan payments are added to the estimated EMI to show total monthly debt after the new loan."
    },
    {
        "title": "Synthetic training data",
        "text": "The project model was trained on a synthetic dataset. Test-set accuracy describes performance against held-out examples from that dataset and should not be interpreted as proof of accuracy on real loan applications."
    },
]

_VECTOR_TEXTS = [f"{item['title']}. {item['text']}" for item in KNOWLEDGE_BASE]
_VECTORIZER = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
_MATRIX = _VECTORIZER.fit_transform(_VECTOR_TEXTS)


def _query_for(context):
    pieces = [
        f"loan type {context.get('loan_type')}",
        "approved prediction" if context.get("prediction") == 1 else "rejected prediction",
        f"loan amount {context.get('loan_amount')}",
        f"missed payments {context.get('missed_payments')}",
        f"employment years {context.get('employment_years')}",
    ]
    if context.get("loan_type") == "Student Loan":
        pieces.extend([
            "education student loan pathway",
            f"college tier {context.get('college_tier')}",
            f"entrance score {context.get('entrance_score')}",
            f"decision pathway {context.get('pathway')}",
            "education funding gap scholarships family contribution",
        ])
    else:
        pieces.extend([
            f"CIBIL credit score {context.get('cibil_score')}",
            f"debt to income ratio {context.get('dti')}",
            "credit score context debt to income context",
        ])
    return " ".join(str(x) for x in pieces if x is not None)


def build_rag_explanation(context, top_k=3):
    """Retrieve relevant passages and build a concise explanation."""
    query_vector = _VECTORIZER.transform([_query_for(context)])
    similarities = cosine_similarity(query_vector, _MATRIX).ravel()
    ranked = similarities.argsort()[::-1]
    sources = [KNOWLEDGE_BASE[int(i)] for i in ranked[:top_k] if similarities[i] > 0]

    outcome = "Likely Approved" if context.get("prediction") == 1 else "Likely Rejected"
    parts = [
        f"The Random Forest model's result for this application is **{outcome}**."
    ]
    score = context.get("approved_score")
    if score is not None:
        parts.append(
            f"The model's approved-class score is **{score:.1f}%**; this is the model's estimated probability for the approved class, not a guaranteed lender decision."
        )

    if context.get("loan_type") == "Student Loan":
        parts.append(
            f"The separate education pathway is **{context.get('pathway')}** for {context.get('college_tier')} with an entrance score of {float(context.get('entrance_score')):.1f}% and a requested amount of ₹{float(context.get('loan_amount')):,.0f}."
        )
        cost = context.get("education_cost")
        support = context.get("financial_support")
        if cost is not None and support is not None:
            gap = max(float(cost) - float(support), 0)
            parts.append(
                f"The entered education cost after scholarships/family contribution is **₹{gap:,.0f}**."
            )
        parts.append(
            "The rule-based education pathway and the Random Forest prediction are separate assessments, so they may differ."
        )
    else:
        cibil = context.get("cibil_score")
        dti = context.get("dti")
        if cibil is not None:
            if cibil < 680:
                parts.append("The entered CIBIL score is below the app's 680-point reference threshold.")
            else:
                parts.append("The entered CIBIL score is at or above the app's 680-point reference threshold.")
        if dti is not None:
            if dti <= 40:
                parts.append("The debt-to-income ratio is at or below the app's 40% reference threshold.")
            else:
                parts.append("The debt-to-income ratio is above the app's 40% reference threshold.")
        if context.get("missed_payments", 0) <= 2:
            parts.append("Reported missed payments are at or below the app's 2-payment reference.")
        else:
            parts.append("Reported missed payments are above the app's 2-payment reference.")

    parts.append(
        "This explanation is based on the application details and the project knowledge passages retrieved below."
    )
    return {"explanation": "\n\n".join(parts), "sources": sources}
