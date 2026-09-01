"""
Streamlit dashboard for Agentic Data Science: Automated Road Risk Analytics.

Run with: streamlit run app.py

Layout:
  Sidebar    upload a CSV or use the bundled sample dataset, plus filters
  Tab 1      overview metrics, model performance, risk map
  Tab 2      ranked road segment table with drill down
  Tab 3      AI report generation and municipal email dispatch
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import plotly.express as px
import streamlit as st

from src.config import (
    MODEL_PATH,
    RAW_DATA_PATH,
    RISK_LABEL_COLORS,
    SCALER_PATH,
)
from src.data_processing import DataQualityAgent, prepare_features
from src.email_dispatch import MunicipalEmailDispatcher
from src.report_generator import RoadRiskReportAgent
from src.risk_model import CompositeRiskEngine, RiskPredictionModel, load_metrics
from src.config import METRICS_PATH

st.set_page_config(
    page_title="Automated Road Risk Analytics",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource(show_spinner=False)
def load_model_resources():
    model = RiskPredictionModel()
    model.load(MODEL_PATH, SCALER_PATH)
    metrics = load_metrics(METRICS_PATH) if os.path.exists(METRICS_PATH) else {}
    return model, metrics


@st.cache_data(show_spinner=False)
def load_default_dataset():
    return pd.read_csv(RAW_DATA_PATH)


@st.cache_data(show_spinner=True)
def run_pipeline(df: pd.DataFrame):
    enriched, feature_matrix = prepare_features(df)

    quality_agent = DataQualityAgent()
    quality_agent.fit(enriched)
    anomaly_flags = quality_agent.flag_anomalies(enriched)
    enriched = enriched.copy()
    enriched["is_anomalous"] = anomaly_flags.values

    model, metrics = load_model_resources()
    probabilities = model.predict_proba(feature_matrix)

    engine = CompositeRiskEngine()
    scored = engine.score_batch(enriched, probabilities)
    return scored, metrics


def render_sidebar():
    st.sidebar.title("Road Risk Analytics")
    st.sidebar.caption("Agentic Data Science Capstone, Batch 2023CP-28")

    uploaded_file = st.sidebar.file_uploader(
        "Upload road sensor CSV", type=["csv"], help="Must include the same columns as the sample dataset."
    )

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.sidebar.success(f"Loaded {len(df)} uploaded records.")
    else:
        df = load_default_dataset()
        st.sidebar.info(f"Using bundled sample dataset ({len(df)} records).")

    st.sidebar.markdown("---")
    city_options = sorted(df["city"].dropna().unique().tolist())
    selected_cities = st.sidebar.multiselect("City", options=city_options, default=city_options)

    min_score, max_score = st.sidebar.slider(
        "Composite risk score range", min_value=0, max_value=100, value=(0, 100)
    )

    return df, selected_cities, min_score, max_score


def render_overview_tab(scored: pd.DataFrame, metrics: dict):
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Segments analyzed", f"{len(scored):,}")
    col2.metric("Critical risk segments", int((scored["risk_tier"] == "Critical").sum()))
    col3.metric("Average risk score", f"{scored['composite_risk_score'].mean():.1f}")
    col4.metric("Anomalous readings", int(scored.get("is_anomalous", pd.Series(dtype=bool)).sum()))

    st.markdown("### Model performance (held out test set)")
    if metrics:
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Accuracy", f"{metrics.get('accuracy', 0)}%")
        m2.metric("Precision", f"{metrics.get('precision', 0)}%")
        m3.metric("Recall", f"{metrics.get('recall', 0)}%")
        m4.metric("F1 Score", f"{metrics.get('f1_score', 0)}%")
        m5.metric("ROC AUC", f"{metrics.get('roc_auc', 0)}")

        importance = metrics.get("feature_importance", {})
        if importance:
            importance_df = pd.DataFrame(
                {"feature": list(importance.keys()), "importance": list(importance.values())}
            ).sort_values("importance", ascending=True)
            fig = px.bar(
                importance_df,
                x="importance",
                y="feature",
                orientation="h",
                title="What the model actually learned to weight",
            )
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("No training metrics found. Run train_model.py first.")

    st.markdown("### Geographic risk map")
    map_df = scored.dropna(subset=["latitude", "longitude"])
    if len(map_df) > 0:
        fig_map = px.scatter_mapbox(
            map_df,
            lat="latitude",
            lon="longitude",
            color="risk_tier",
            size="composite_risk_score",
            hover_name="road_name",
            hover_data=["city", "ward", "composite_risk_score", "risk_tier"],
            color_discrete_map=RISK_LABEL_COLORS,
            zoom=3.6,
            height=560,
            mapbox_style="carto-positron",
        )
        fig_map.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})
        st.plotly_chart(fig_map, use_container_width=True)

    col_left, col_right = st.columns(2)
    with col_left:
        tier_counts = scored["risk_tier"].value_counts().reindex(["Low", "Medium", "High", "Critical"]).fillna(0)
        fig_pie = px.pie(
            names=tier_counts.index,
            values=tier_counts.values,
            title="Risk tier distribution",
            color=tier_counts.index,
            color_discrete_map=RISK_LABEL_COLORS,
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    with col_right:
        city_avg = scored.groupby("city")["composite_risk_score"].mean().sort_values(ascending=False)
        fig_city = px.bar(
            x=city_avg.index,
            y=city_avg.values,
            title="Average composite risk score by city",
            labels={"x": "City", "y": "Average risk score"},
        )
        st.plotly_chart(fig_city, use_container_width=True)


def render_segments_tab(scored: pd.DataFrame):
    st.markdown("### Ranked road segments")
    display_cols = [
        "record_id",
        "city",
        "ward",
        "road_name",
        "composite_risk_score",
        "risk_tier",
        "rainfall_mm",
        "road_age_years",
        "traffic_density_vph",
        "drainage_quality_score",
        "top_factors",
    ]
    display_cols = [c for c in display_cols if c in scored.columns]
    ranked = scored.sort_values("composite_risk_score", ascending=False)[display_cols]
    st.dataframe(ranked, use_container_width=True, height=460)

    st.download_button(
        "Download scored dataset as CSV",
        data=ranked.to_csv(index=False).encode("utf-8"),
        file_name="scored_road_segments.csv",
        mime="text/csv",
    )


def render_report_tab(scored: pd.DataFrame):
    st.markdown("### AI agent report generation and municipal dispatch")

    ranked = scored.sort_values("composite_risk_score", ascending=False).reset_index(drop=True)
    options = [
        f"{row.road_name}, {row.ward}, {row.city} (score {row.composite_risk_score})"
        for row in ranked.itertuples()
    ]

    if not options:
        st.info("No records available to report on.")
        return

    selected_idx = st.selectbox("Select a road segment", options=range(len(options)), format_func=lambda i: options[i])
    record = ranked.iloc[selected_idx].to_dict()

    agent = RoadRiskReportAgent(prefer_local_llm=True)

    if "generated_report" not in st.session_state:
        st.session_state.generated_report = {}

    if st.button("Generate agentic report", type="primary"):
        report = agent.generate_report(record)
        st.session_state.generated_report[selected_idx] = report

    report = st.session_state.generated_report.get(selected_idx)

    if report:
        st.text_input("Subject", value=report.subject, key=f"subject_{selected_idx}")
        st.text_area("Report body", value=report.body, height=280, key=f"body_{selected_idx}")
        badge = "Local LLM" if report.backend_used == "local_llm" else "Rule based template agent"
        st.caption(f"Drafted by: {badge} | Priority: {report.priority} | Timeline: {report.timeline}")

        st.markdown("#### Dispatch to civic contact")
        dry_run = st.checkbox("Dry run (recommended, does not send a real email)", value=True)
        override_email = st.text_input("Override recipient email (optional)", value="")

        if st.button("Send report"):
            dispatcher = MunicipalEmailDispatcher(dry_run=dry_run)
            result = dispatcher.send_report(
                city=record["city"],
                ward=record["ward"],
                subject=st.session_state[f"subject_{selected_idx}"],
                body=st.session_state[f"body_{selected_idx}"],
                override_recipient=override_email or None,
            )
            if result.success:
                st.success(f"{result.message} Recipient: {result.recipient}")
            else:
                st.error(result.message)

    st.markdown("---")
    st.markdown("#### Dispatch history")
    dispatcher = MunicipalEmailDispatcher(dry_run=True)
    history = dispatcher.read_dispatch_log()
    if history:
        st.dataframe(pd.DataFrame(history[-20:][::-1]), use_container_width=True)
    else:
        st.caption("No dispatch attempts logged yet.")


def main():
    df, selected_cities, min_score, max_score = render_sidebar()

    if not selected_cities:
        st.warning("Select at least one city from the sidebar to see results.")
        return

    filtered_df = df[df["city"].isin(selected_cities)].reset_index(drop=True)

    if len(filtered_df) == 0:
        st.warning("No records match the current filters.")
        return

    scored, metrics = run_pipeline(filtered_df)
    scored = scored[
        (scored["composite_risk_score"] >= min_score) & (scored["composite_risk_score"] <= max_score)
    ]

    st.title("Agentic Data Science: Automated Road Risk Analytics")
    st.caption(
        "Ingests road sensor data, predicts deterioration risk, and drafts prioritized "
        "maintenance reports for civic engineers across Indian cities."
    )

    tab1, tab2, tab3 = st.tabs(["Overview", "Road Segments", "AI Reports and Dispatch"])
    with tab1:
        render_overview_tab(scored, metrics)
    with tab2:
        render_segments_tab(scored)
    with tab3:
        render_report_tab(scored)


if __name__ == "__main__":
    main()
