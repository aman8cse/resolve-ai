import requests
import streamlit as st


API_URL = "http://127.0.0.1:8000"


st.set_page_config(
    page_title="ResolveAI",
    page_icon="🤖",
    layout="wide",
)


st.title("ResolveAI")
st.caption("AI-powered customer support assistant")


message = st.text_area(
    "Customer message",
    placeholder="e.g. My package says delivered but I never received it.",
    height=130,
)


if st.button("Analyze", type="primary", use_container_width=True):

    if not message.strip():
        st.warning("Please enter a customer message.")
        st.stop()

    try:
        response = requests.post(
            f"{API_URL}/analyze",
            json={"message": message},
            timeout=120,
        )

        response.raise_for_status()
        result = response.json()

    except requests.exceptions.RequestException as e:
        st.error(f"Could not connect to ResolveAI API: {e}")
        st.stop()

    st.divider()

    # -------------------------
    # Summary
    # -------------------------

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Intent",
            result["intent"].replace("_", " ").title(),
        )

    with col2:
        confidence = result.get("intent_confidence")

        if confidence is not None:
            st.metric(
                "Intent Confidence",
                f"{confidence:.1%}",
            )
        else:
            st.metric("Intent Confidence", "N/A")

    with col3:
        st.metric(
            "Decision",
            result["decision"],
        )

    # -------------------------
    # Generated response
    # -------------------------

    st.subheader("Suggested Response")

    st.info(result["reply"])

    # -------------------------
    # Historical cases
    # -------------------------

    st.subheader("Similar Historical Cases")

    for i, case in enumerate(result["cases"], start=1):

        with st.expander(
            f"Case {i}  •  Similarity {case['similarity']:.3f}"
        ):
            st.markdown("**Customer**")
            st.write(case["customer"])

            st.markdown("**Historical Amazon Response**")
            st.write(case["response"])

    # -------------------------
    # Decision reason
    # -------------------------

    st.subheader("Decision Reason")
    st.write(result["reason"])

    # -------------------------
    # Latency
    # -------------------------

    with st.expander("System Latency"):
        latency = result["latency"]

        st.write(
            f"Classifier: {latency['classifier_ms']:.2f} ms"
        )
        st.write(
            f"Retrieval: {latency['retrieval_ms']:.2f} ms"
        )
        st.write(
            f"Generation: {latency['generation_ms']:.2f} ms"
        )
        st.write(
            f"Escalation: {latency['escalation_ms']:.2f} ms"
        )
        st.write(
            f"Total: {latency['total_ms']:.2f} ms"
        )