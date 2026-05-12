"""CryptoLOB — interactive dashboard for INDENG 242B final project.

Run locally:
    streamlit run streamlit_app.py

Deploy: push this file (plus dashboard/, .streamlit/, requirements.txt) to
GitHub and connect the repo at https://share.streamlit.io.

Pages
-----
1. Overview — hero, abstract, headline numbers.
2. Dataset — feature schema, mid-price, label distribution.
3. Model Performance — RMSE / R^2 / DirAcc / AUC table + predicted-vs-actual scatter.
4. Backtest Sandbox — interactive: user adjusts signal threshold and cost_bps,
   dashboard recomputes NAV / Sharpe / MaxDD for every selected model.
5. About — team, methodology, repo link.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from dashboard.data_io import (
    FALLBACK_REGRESSION,
    MODEL_FAMILIES,
    fallback_predictions_demo,
    load_dataset_metadata,
    load_predictions_panel,
    load_regression_metrics,
)
from dashboard.backtest import (
    HORIZON_STEPS,
    PERIODS_PER_YEAR,
    SAMPLE_INTERVAL_S,
    backtest_strategy,
    buy_and_hold,
    max_drawdown_series,
)

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="CryptoLOB · INDENG 242B",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

REPO_URL = "https://github.com/242B-group/INDENG242B-Project"

# ---------------------------------------------------------------------------
# Sidebar — global navigation + project info
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 📈 CryptoLOB Dashboard")
    st.caption("INDENG 1/242B Final Project · Spring 2026")
    page = st.radio(
        "Navigate",
        [
            "🏠 Overview",
            "📊 Dataset",
            "🤖 Model Performance",
            "🧪 Backtest Sandbox",
            "ℹ️ About",
        ],
        label_visibility="collapsed",
    )
    st.divider()
    st.caption(
        f"[GitHub repo]({REPO_URL})  ·  "
        "Wish Wang · Yijun Gu · Arthur Fang · Haorui Zhang"
    )

# ---------------------------------------------------------------------------
# Cached loaders
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def get_metadata():
    return load_dataset_metadata()


@st.cache_data(show_spinner=False)
def get_regression_metrics():
    return load_regression_metrics()


@st.cache_data(show_spinner=False)
def get_predictions_panel():
    panel = load_predictions_panel()
    if panel is None or panel.empty:
        panel, was_demo = fallback_predictions_demo(), True
    else:
        was_demo = False
    return panel, was_demo


metadata = get_metadata()
metrics_df = get_regression_metrics()
panel, panel_is_demo = get_predictions_panel()

PRED_COLS = [c for c in panel.columns if c.startswith("pred_")]
AVAILABLE_FAMILIES = [c.replace("pred_", "") for c in PRED_COLS]


# ===========================================================================
# Page: Overview
# ===========================================================================
def page_overview():
    st.title("📈 Predicting BTC/USDT mid-price moves from limit-order-book data")
    st.markdown(
        """
        **An interactive dashboard for our INDENG 242B final project.**
        We collected ~4 hours of live Binance order-book snapshots, built a
        60×82 feature tensor per sample, and benchmarked six model families
        from ARIMA to a bidirectional GRU with Bahdanau attention. Every model
        ended up with negative R² on the held-out test set, and a
        cost-aware backtest shows that **no strategy — including buy-and-hold —
        made money on this 50-minute test window**. This dashboard lets you
        interrogate that result yourself.
        """
    )

    st.divider()

    # Headline cards
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Order-book snapshots", f"{metadata.get('raw_rows', 14577):,}")
    c2.metric("Feature dimensions", f"{metadata.get('num_features', 82)}/step")
    c3.metric("Sequence length", f"{metadata.get('lookback_window', 60)} steps")
    c4.metric("Prediction horizon", f"{metadata.get('prediction_horizon', 10)} sec")

    st.divider()

    st.subheader("Pipeline at a glance")
    st.markdown(
        """
        ```
        Binance WebSocket  →  20-level LOB @ 1 Hz  →  60×82 windows
                                                             ↓
            ARIMA · RandomForest · RNN · LSTM · CNN-LSTM · GRU+Attention
                                                             ↓
                       MSE + Adam + early-stop + top-K ensemble
                                                             ↓
                Regression / classification metrics  +  PnL backtest
        ```
        """
    )

    st.subheader("Headline test-set numbers")
    show_df = metrics_df.copy()
    if "RMSE_x1e4" in show_df.columns:
        show_df = show_df.rename(
            columns={
                "RMSE_x1e4": "RMSE (×10⁻⁴)",
                "MAE_x1e4": "MAE (×10⁻⁴)",
                "R2": "R²",
                "DirAcc": "Dir. Acc.",
                "AUC": "AUC",
            }
        )
        for c in ["Dir. Acc."]:
            if c in show_df.columns:
                show_df[c] = show_df[c].map(lambda v: f"{v:.1%}")
    st.dataframe(show_df, use_container_width=True, hide_index=True)

    st.info(
        "👉 Jump to **🧪 Backtest Sandbox** to vary the signal threshold and "
        "transaction cost in real time and see how each model's PnL responds."
    )


# ===========================================================================
# Page: Dataset
# ===========================================================================
def page_dataset():
    st.title("📊 Dataset")

    st.markdown(
        """
        We subscribe to the public **Binance WebSocket stream
        `btcusdt@depth20@100ms`**, which delivers the top-20 bids and asks
        every 100 ms. We sample one snapshot per second, giving a
        single-source-of-truth tensor of LOB state for ~4 hours of live
        trading. The dataset is fully reproducible from the repo:
        `python -m src.collect_orderbook` then `python -m src.build_dataset`.
        """
    )

    c1, c2, c3 = st.columns(3)
    splits = metadata.get("split_sizes", {})
    c1.metric("Train", f"{splits.get('train', 0):,}")
    c2.metric("Val", f"{splits.get('val', 0):,}")
    c3.metric("Test", f"{splits.get('test', 0):,}")

    st.subheader("Feature schema (82 per timestep)")
    schema = pd.DataFrame(
        [
            {"Group": "bid_price_1..20", "Count": 20, "Description": "20 deepest bid prices"},
            {"Group": "bid_size_1..20", "Count": 20, "Description": "20 deepest bid sizes"},
            {"Group": "ask_price_1..20", "Count": 20, "Description": "20 deepest ask prices"},
            {"Group": "ask_size_1..20", "Count": 20, "Description": "20 deepest ask sizes"},
            {"Group": "spread", "Count": 1, "Description": "best_ask − best_bid"},
            {"Group": "order_book_imbalance_20", "Count": 1, "Description": "(Σbid − Σask)/(Σbid + Σask)"},
        ]
    )
    st.dataframe(schema, use_container_width=True, hide_index=True)
    st.caption(
        "`mid_price` and `timestamp` are intentionally excluded from X. "
        "`mid_price` is only used to construct the label "
        "`y_t = log(mid_{t+H} / mid_t)` so the model cannot trivially read it."
    )

    st.subheader("Test-set mid-price trajectory")
    if "mid_price_t" in panel.columns and len(panel) > 0:
        plot_df = panel.copy()
        plot_df["t"] = pd.to_datetime(plot_df["timestamp_ms"], unit="ms")
        fig = px.line(
            plot_df, x="t", y="mid_price_t",
            labels={"t": "Time", "mid_price_t": "BTC/USDT mid-price"},
            title="BTC/USDT mid-price over the held-out test window",
        )
        fig.update_layout(height=380, margin=dict(l=20, r=20, t=50, b=20))
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Realized 10-step log-return distribution")
        fig2 = px.histogram(
            plot_df, x="realized_log_return", nbins=80,
            labels={"realized_log_return": "log-return (10 s ahead)"},
            title="Realized log-returns on the test set",
        )
        fig2.update_layout(height=320, margin=dict(l=20, r=20, t=50, b=20))
        st.plotly_chart(fig2, use_container_width=True)

        c1, c2, c3 = st.columns(3)
        rr = plot_df["realized_log_return"].dropna()
        c1.metric("Mean", f"{rr.mean():.2e}")
        c2.metric("Std", f"{rr.std():.2e}")
        c3.metric("|drift|", f"{abs((np.exp(rr.sum()) - 1)):.3%}")
    else:
        st.info("No test-set context CSV found — see README for `export_dashboard_data.py`.")


# ===========================================================================
# Page: Model performance
# ===========================================================================
def page_models():
    st.title("🤖 Model performance on the held-out test set")

    st.markdown(
        """
        Six model families are trained on identical inputs and labels.
        Metrics below are computed after inverse-transforming predictions
        back to the raw log-return scale, so the numbers are directly
        comparable across families.
        """
    )

    show_df = metrics_df.copy()
    if "RMSE_x1e4" in show_df.columns:
        show_df = show_df.rename(
            columns={
                "RMSE_x1e4": "RMSE (×10⁻⁴)",
                "MAE_x1e4": "MAE (×10⁻⁴)",
                "R2": "R²",
                "DirAcc": "Dir. Acc.",
                "AUC": "AUC",
            }
        )
        if "Dir. Acc." in show_df.columns:
            show_df["Dir. Acc."] = show_df["Dir. Acc."].map(lambda v: f"{v:.1%}")
    st.dataframe(show_df, use_container_width=True, hide_index=True)

    st.warning(
        "**Every model achieves negative R²** — i.e. worse than predicting the "
        "test-set mean. ARIMA is the least negative (essentially mean-prediction); "
        "deeper models are progressively more negative, consistent with overfitting "
        "on a ~10k-sample training set drawn from a non-stationary regime."
    )

    st.divider()

    st.subheader("Predicted vs. realized log-return")
    if not PRED_COLS:
        st.info("No prediction CSVs found. Commit `replication/results/*_predictions.csv` to populate this view.")
        return

    family = st.selectbox(
        "Choose a model family",
        AVAILABLE_FAMILIES,
        format_func=lambda f: MODEL_FAMILIES.get(f, (f, ""))[0],
    )
    pred_col = f"pred_{family}"

    df = panel[["realized_log_return", pred_col]].dropna()
    if df.empty:
        st.warning("No data for this model.")
        return

    lim = float(max(df["realized_log_return"].abs().max(), df[pred_col].abs().max())) * 1.05

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["realized_log_return"], y=df[pred_col],
        mode="markers",
        marker=dict(size=4, opacity=0.5, color=MODEL_FAMILIES.get(family, ("", "#3498db"))[1]),
        name="(realized, predicted)",
    ))
    fig.add_trace(go.Scatter(
        x=[-lim, lim], y=[-lim, lim],
        mode="lines",
        line=dict(color="red", dash="dash"),
        name="y = x",
    ))
    fig.update_layout(
        title=f"{MODEL_FAMILIES.get(family, (family, ''))[0]} — predicted vs realized 10-s log-return",
        xaxis_title="Realized log-return",
        yaxis_title="Predicted log-return",
        height=480,
        margin=dict(l=20, r=20, t=60, b=20),
    )
    fig.update_xaxes(range=[-lim, lim])
    fig.update_yaxes(range=[-lim, lim])
    st.plotly_chart(fig, use_container_width=True)

    rmse = float(np.sqrt(np.mean((df[pred_col] - df["realized_log_return"]) ** 2)))
    bias = float((df[pred_col] - df["realized_log_return"]).mean())
    diracc = float(np.mean(np.sign(df[pred_col]) == np.sign(df["realized_log_return"])))
    c1, c2, c3 = st.columns(3)
    c1.metric("RMSE", f"{rmse:.2e}")
    c2.metric("Mean signed error", f"{bias:.2e}")
    c3.metric("Directional accuracy", f"{diracc:.1%}")

    st.caption(
        "Predictions cluster tightly around zero with effectively no slope "
        "against the realized axis — the visual signature of negative R²."
    )


# ===========================================================================
# Page: Backtest Sandbox — the headline interactive feature
# ===========================================================================
def page_backtest():
    st.title("🧪 Backtest Sandbox")
    st.markdown(
        """
        Each model produces a predicted 10-s log-return $\\hat y_t$.
        We convert it into a position with
        $\\text{pos}_t = \\operatorname{sign}(\\hat y_t)\\,\\mathbb{1}\\{|\\hat y_t| > \\tau\\}$,
        rebalance every 10 s (non-overlapping bets), and charge `cost_bps`
        on $|\\Delta\\text{position}|$. **Drag the two sliders below to see how
        each model's PnL responds.** This is the live version of Table 3 in
        the report.
        """
    )

    if not PRED_COLS:
        st.error("No prediction CSVs found. Commit `replication/results/*_predictions.csv` or run "
                 "`python -m replication.export_dashboard_data`.")
        return
    if panel_is_demo:
        st.warning("Showing a tiny demo dataset — commit the real predictions CSV for full results.")

    # --- Sandbox controls -------------------------------------------------
    with st.container(border=True):
        col_a, col_b = st.columns([1, 1])
        with col_a:
            tau = st.slider(
                "Signal threshold τ (log-return units)",
                min_value=0.0, max_value=5e-4,
                value=5e-5, step=1e-5, format="%.5f",
                help="Trade only when |ŷ| > τ. The report value (validation-calibrated) is 5e-5.",
            )
        with col_b:
            cost_bps = st.slider(
                "Round-trip cost (bps per |Δposition|)",
                min_value=0.0, max_value=5.0, value=1.0, step=0.25,
                help="The report uses 1 bp. Set to 0 to see model alpha without costs.",
            )

        col_c, col_d = st.columns([2, 1])
        with col_c:
            chosen = st.multiselect(
                "Models to compare",
                AVAILABLE_FAMILIES,
                default=AVAILABLE_FAMILIES,
                format_func=lambda f: MODEL_FAMILIES.get(f, (f, ""))[0],
            )
        with col_d:
            include_bh = st.toggle("Include Buy & Hold", value=True)

    if not chosen:
        st.info("Select at least one model.")
        return

    realized = panel["realized_log_return"].to_numpy()
    ts = pd.to_datetime(panel["timestamp_ms"], unit="ms")

    # Run backtests
    summaries = []
    nav_curves = {}
    dd_curves = {}

    for fam in chosen:
        pred = panel[f"pred_{fam}"].to_numpy()
        mask = ~np.isnan(pred) & ~np.isnan(realized)
        result = backtest_strategy(
            pred[mask], realized[mask], tau, cost_bps,
        )
        summaries.append({
            "Model": MODEL_FAMILIES.get(fam, (fam, ""))[0],
            "Cum. Return": result["cum_return"],
            "Ann. Vol.": result["ann_vol"],
            "Sharpe": result["sharpe"],
            "Max DD": result["max_dd"],
            "Win Rate": result["win_rate"],
            "Pos. fraction": result["position_fraction"],
        })
        nav_curves[fam] = (ts[mask].to_numpy(), result["nav"])
        dd_curves[fam] = (ts[mask].to_numpy(), result["drawdown"])

    if include_bh:
        bh = buy_and_hold(realized)
        summaries.append({
            "Model": "Buy & Hold",
            "Cum. Return": bh["cum_return"],
            "Ann. Vol.": bh["ann_vol"],
            "Sharpe": bh["sharpe"],
            "Max DD": bh["max_dd"],
            "Win Rate": bh["win_rate"],
            "Pos. fraction": 1.0,
        })
        nav_curves["buy_and_hold"] = (ts.to_numpy(), bh["nav"])
        dd_curves["buy_and_hold"] = (ts.to_numpy(), bh["drawdown"])

    # --- Summary table ----------------------------------------------------
    st.subheader("Live summary")
    summary_df = pd.DataFrame(summaries)
    fmt = summary_df.copy()
    fmt["Cum. Return"] = fmt["Cum. Return"].map(lambda v: f"{v:.3%}")
    fmt["Ann. Vol."] = fmt["Ann. Vol."].map(lambda v: f"{v:.1%}")
    fmt["Sharpe"] = fmt["Sharpe"].map(lambda v: f"{v:.1f}")
    fmt["Max DD"] = fmt["Max DD"].map(lambda v: f"{v:.3%}")
    fmt["Win Rate"] = fmt["Win Rate"].map(lambda v: f"{v:.1%}")
    fmt["Pos. fraction"] = fmt["Pos. fraction"].map(lambda v: f"{v:.1%}")
    st.dataframe(fmt, use_container_width=True, hide_index=True)
    st.caption(
        f"Sharpe annualized at periods/year = {PERIODS_PER_YEAR:,.0f} "
        f"(10-s rebalance, 24/7 market). Cost charged per |Δposition| in bps."
    )

    # --- NAV plot ---------------------------------------------------------
    st.subheader("Net asset value (initial = 1.0)")
    fig_nav = go.Figure()
    for fam, (t, nav) in nav_curves.items():
        label = "Buy & Hold" if fam == "buy_and_hold" else MODEL_FAMILIES.get(fam, (fam, ""))[0]
        color = "#1f77b4" if fam == "buy_and_hold" else MODEL_FAMILIES.get(fam, ("", "#888"))[1]
        fig_nav.add_trace(go.Scatter(x=t, y=nav, mode="lines", name=label,
                                     line=dict(color=color, width=2)))
    fig_nav.add_hline(y=1.0, line=dict(color="gray", dash="dot"), opacity=0.4)
    fig_nav.update_layout(
        height=420, margin=dict(l=20, r=20, t=30, b=20),
        xaxis_title="Time", yaxis_title="NAV",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig_nav, use_container_width=True)

    # --- Drawdown plot ----------------------------------------------------
    st.subheader("Drawdown")
    fig_dd = go.Figure()
    for fam, (t, dd) in dd_curves.items():
        label = "Buy & Hold" if fam == "buy_and_hold" else MODEL_FAMILIES.get(fam, (fam, ""))[0]
        color = "#1f77b4" if fam == "buy_and_hold" else MODEL_FAMILIES.get(fam, ("", "#888"))[1]
        fig_dd.add_trace(go.Scatter(x=t, y=dd, mode="lines", name=label,
                                    line=dict(color=color, width=2)))
    fig_dd.update_layout(
        height=320, margin=dict(l=20, r=20, t=20, b=20),
        xaxis_title="Time", yaxis_title="Drawdown",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig_dd, use_container_width=True)

    st.divider()
    with st.expander("💡 Things to try"):
        st.markdown(
            """
            - **Set cost to 0 bps.** Does *any* model now beat buy-and-hold? (The
              honest answer: no — confirming the negative R² in Table 2 is the
              binding constraint, not the friction.)
            - **Crank τ very high (≥ 2e-4).** Models stop trading entirely. NAV
              flatlines at 1.0. This is exactly the regime ARIMA already sits in
              under the default settings.
            - **Compare ARIMA vs the deep models at cost = 0 bps, τ = 0.** ARIMA
              still has the highest Sharpe — a simple temporal mean baseline
              beats every neural network on this dataset.
            """
        )


# ===========================================================================
# Page: About
# ===========================================================================
def page_about():
    st.title("ℹ️ About this project")

    st.markdown(
        f"""
        **Team:** Wish Wang, Yijun Gu, Arthur Fang, Haorui Zhang — UC Berkeley, Spring 2026.

        **Course:** INDENG 1/242B — Machine Learning and Data Analytics II.

        **Repository:** [{REPO_URL}]({REPO_URL})

        ### What this dashboard is
        An interactive companion to our [final report](https://github.com/242B-group/INDENG242B-Project).
        Page 4 (**Backtest Sandbox**) is the meaningful-interaction component
        from the project rubric: it lets anyone — without reading our code —
        verify how every model's PnL responds to the two parameters that
        matter most for a high-frequency strategy (signal threshold and
        transaction cost).

        ### Methodology recap

        - **Task** — regression on $y_t = \\log(\\text{{mid}}_{{t+10}} / \\text{{mid}}_t)$
          from a 60×82 LOB window.
        - **Models** — ARIMA, Random Forest (4920-d flattened), Stacked RNN,
          Stacked LSTM, Causal-CNN → LSTM, Bidirectional GRU + Bahdanau
          attention.
        - **Training** — Adam + MSE, dropout, early stopping (patience 5),
          grid search over hidden / layers / dropout / lr / batch size,
          top-K = 5 ensemble.
        - **Advanced component** — Bahdanau additive attention over the
          bidirectional-GRU encoder hidden states.
        - **Backtest** — non-overlapping 10-s bets, signed-threshold position
          rule, configurable per-turnover cost, Sharpe annualized to the
          24/7 crypto calendar.

        ### Honest reading of the results
        Every model has **negative R²** and the cost-aware backtest loses
        money — but so does buy-and-hold on this 50-minute window. The deep
        models lose more because they *also* pay frictional costs to make
        bets that turn out to be uncorrelated with the realized signal. We
        attribute this to data quantity (4 hours ≈ 10k samples is well below
        FI-2010-scale benchmarks) and non-stationarity rather than to any
        architectural flaw.

        ### AI disclosure
        We used Anthropic Claude as a coding assistant for the pipeline,
        backtest, and this dashboard. All AI-generated suggestions were
        inspected, modified, and validated. The framing, methodology
        choices, experimental design, and written analysis are our own.
        """
    )


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------
ROUTES = {
    "🏠 Overview": page_overview,
    "📊 Dataset": page_dataset,
    "🤖 Model Performance": page_models,
    "🧪 Backtest Sandbox": page_backtest,
    "ℹ️ About": page_about,
}

ROUTES[page]()
