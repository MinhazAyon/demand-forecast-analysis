
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from statsmodels.tsa.stattools import acf
from pytorch_forecasting import TimeSeriesDataSet

# -- For easy excess, I do column Mapping
PID = 'product_code'
WID = 'warehouse'
DATE = 'date'
TARGET = 'demand'

# -- PLOTS STYLE CONFIGURATION
sns.set_context("notebook", font_scale=1.1)
plt.style.use('seaborn-v0_8-whitegrid')
MAIN_COLOR = "#2C3E50"
ACCENT_COLOR = "#FF7F50"
GAP_COLOR = "#FF5C5C"
# Warehouse Level colors
COLORS = ['#FF7F50', '#2ECC71', '#3498DB', '#9B59B6']

# -- HELPER FUNCTIONS

def actual_vs_predicted(diags):
    plt.figure(figsize=(12, 8), facecolor="#FAFAFA")

    # Data
    tf = diags.dropna(subset=["actual", "p50"]).copy()
    tf = tf[(tf["actual"] > 0) & (tf["p50"] > 0)]

    # Hexbin
    hb = plt.hexbin(
        np.log1p(tf["p50"]), np.log1p(tf["actual"]), gridsize=60,
        cmap="mako", bins="log", mincnt=1
    )

    max_val = max(np.log1p(tf["actual"].max()), np.log1p(tf["p50"].max()))
    plt.plot([0, max_val], [0, max_val], color='red', linestyle="--", lw=2)

    plt.xlabel("log(1 + P50)")
    plt.ylabel("log(1 + Actual)")
    plt.title("Actual vs P50 Forecast (Log–Log Density)",fontweight="bold")
    cb = plt.colorbar(hb)
    cb.set_label("Log Density")

    plt.grid(True, linestyle=":", alpha=0.4)
    plt.tight_layout()
    plt.show()

def residual_plots(diags):
    # Unique Segmnets
    segments = diags['segment'].unique()

    # Plot Features
    fig, axes = plt.subplots(len(segments), 2, figsize=(22, 4 * len(segments)), gridspec_kw={'width_ratios': [2, 1]}, sharex=True)

    # Plot
    for i, seg in enumerate(segments):
        seg_df = diags[diags.segment == seg].copy()
        seg_df['residual'] = seg_df['actual'] - seg_df['p50']

        # LEFT: Residual scatter
        sns.scatterplot(
            data=seg_df, x='p50', y='residual', s=10, color=MAIN_COLOR, ax=axes[i, 0], hue='warehouse'
        )
        axes[i, 0].axhline(0, color='red', linestyle='--', lw=1.5)
        axes[i, 0].set_title(f"Segment {seg}: Residuals vs Prediction", fontsize=14)
        axes[i, 0].set_ylabel("Residuals", fontsize=12)
        axes[i, 0].set_ylabel("Residuals", fontsize=12)

        # RIGHT: Residual distribution
        mean_res = seg_df['residual'].mean()
        sns.histplot(
            seg_df['residual'], kde=True, ax=axes[i, 1], color=ACCENT_COLOR
        )
        axes[i, 1].axvline(mean_res, color='black', linestyle=':', lw=1.5)
        axes[i, 1].set_title(f"Segment {seg}: Error Distribution", fontsize=14)
        axes[i, 1].set_xlabel("Forecasted value (p50)", fontsize=12)
        axes[i, 1].set_ylabel("Frequency", fontsize=12)

    # Main title
    plt.suptitle("Residual Analysis and Prediction Bias", fontsize=20, fontweight='bold', y=1.01)

    plt.tight_layout()
    plt.show()

def qcc_plot(diags):
    plt.figure(figsize=(10, 6), facecolor="#FAFAFA")

    # Metrics
    quantiles = [0.5, 0.8, 0.9]
    labels = ["P50", "P80", "P90"]

    # Calculate what percentage of actual values fell below each prediction band
    observed = [
        (diags['actual'] <= diags['p50']).mean(),
        (diags['actual'] <= diags['p80']).mean(),
        (diags['actual'] <= diags['p90']).mean()
    ]

    # Visualization
    x = np.arange(len(labels))
    width = 0.35

    # Plot Expected vs Observed
    plt.bar(x - width/2, quantiles, width, label="Expected (Theoretical)", color=MAIN_COLOR, alpha=0.2)
    plt.bar(x + width/2, observed, width, label="Observed (Empirical)", color=ACCENT_COLOR)

    # Add percentage labels on top of the bars
    for i, v in enumerate(observed):
        plt.text(i + width/2, v + 0.02, f"{v:.1%}", ha="center", fontweight="bold", fontsize=11)

    # Formatting
    plt.xticks(x, labels, fontsize=11)
    plt.ylim(0, 1.1)
    plt.ylabel("Frequency (Actual ≤ Prediction Bound)", fontsize=12)
    plt.title("Quantile Calibration: Is the Model Reliable?", fontsize=14, fontweight="bold", pad=20)
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    plt.legend(frameon=True, facecolor='white')

    plt.tight_layout()
    plt.show()

def risk_driving_skus(forecast_df):
    fig, axes = plt.subplots(2, 2, figsize=(16, 10), facecolor="#FAFAFA")
    axes = axes.flatten()

    # Unique warehouses
    warehouses = sorted(forecast_df["warehouse"].unique())
    for ax, wh in zip(axes, warehouses):
        # Filter warehouse
        df_wh = forecast_df[forecast_df["warehouse"] == wh]
        # Get top SKUs
        top_skus = (df_wh.groupby(["product_code", "product_category"], as_index=False)
            .agg(safety_stock_units=("safety_stock_units", "sum"))
            .sort_values("safety_stock_units", ascending=False)
            .head(10)
            .sort_values("safety_stock_units", ascending=False))
        # Plot
        sns.barplot(
            data=top_skus, x="safety_stock_units", y="product_code",
            ax=ax, color=ACCENT_COLOR
        )

        # Labeling
        ax.set_title(f'Warehouse: {wh}', fontweight="bold")
        ax.set_xlabel("Safety Stock Units (P90 − P50)")
        ax.set_ylabel("")
        ax.grid(axis="x", alpha=0.3)

        # Annotate values
        for i, v in enumerate(top_skus["safety_stock_units"]): ax.text(v, i, f"{v:,.0f}", va="center", ha="left", fontsize=9)

    fig.suptitle("Top 10 Risk-Driving SKUs per Warehouse",fontsize=16,fontweight="bold", y=1.03)
    plt.tight_layout()
    plt.show()

def upside_risk_scale(forecast_df):
    plt.figure(figsize=(12, 8), facecolor="#FAFAFA")

    # Data
    forecast_df["upper_risk_90"] = forecast_df["p90"] - forecast_df["p50"]
    sns.scatterplot(
        data=forecast_df, x="p50", y="upper_risk_90", hue="segment", s=60, palette="tab10"
    )

    plt.xlabel("Predicted Demand (P50)")
    plt.ylabel("Upper Tail Risk (P90 − P50)")
    plt.title("Upside Risk Scaling by Business Segment", fontweight="bold")
    plt.legend(title="Segment", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.show()

def forecast_nx_mn(df, forecast_df):
    fig, axes = plt.subplots(9, 1, figsize=(18, 20), sharex=True)
    axes = axes.flatten()

    plot_idx = 0
    for seg in df.segment.unique():
        while True:
            filter_df = get_random(df_actual=df, df_forecast=forecast_df, segment_id=seg)
            if filter_df is False:
                continue
            else:
                break
        # Plotting data
        d_actual, d_forecast, S, W, P = filter_df
        # Plot
        plot_forecast(axes[plot_idx], d_actual, d_forecast,
            S, W, P, df.date.min(), bridge=True)
        # Increment
        plot_idx += 1

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=True)
    # Main Title
    fig.suptitle(
        "Demand Forecasts with Risk-Adjusted Decision Policy (Business Segments)",
        fontsize=16,
        fontweight="bold"
    )

    plt.tight_layout(rect=[0, 0.06, 1, 0.95])
    plt.show()

def diagnosis_frame(df, tft, training, segment_results):
     # -- Build dataset and dataloader
    validation = TimeSeriesDataSet.from_dataset(
        training, df, predict=False, stop_randomization=True
    )
    # -- Data loader
    val_dataloader = validation.to_dataloader(train=False, batch_size=256, num_workers=0)
    # -- Get predictions
    predictions = tft.predict(
        val_dataloader, mode="raw", return_x=True
    )
    # -- Actual and Predicted Values
    actual = predictions.x["decoder_target"]
    out = predictions.output["prediction"]

    # -- Get the forecasted dataframe
    x = predictions.x
    quantiles = tft.loss.quantiles
    batch_size, decoder_len, _ = out.shape

    diags = pd.DataFrame({
        "time_idx": x["decoder_time_idx"].reshape(-1).cpu().numpy(),
        "actual": actual.reshape(-1).cpu().numpy(),
        "product_code": x["groups"][:, 0].repeat_interleave(decoder_len).cpu().numpy(),
        "warehouse": x["groups"][:, 1].repeat_interleave(decoder_len).cpu().numpy(),
        "horizon": np.tile(np.arange(1, decoder_len + 1),batch_size),
    })
    # Inverse transform
    for i, q in enumerate(quantiles):
        diags[f"p{int(q*100):02d}"] = (out[:, :, i].reshape(-1).cpu().numpy())

    # Per Product Code and Warehouse data format
    diags = (diags.groupby(["product_code", "warehouse", "time_idx"]).median().reset_index())

    # Decode Product Code and Warehouse
    for col in ["product_code", "warehouse"]:
        enc = training.get_parameters()["categorical_encoders"][col]
        diags[col] = enc.inverse_transform(
            diags[col].astype(int).values
        )

    # -- Merging the segment data
    diags = diags.merge(segment_results[['product_code', 'warehouse', 'segment']], on=['product_code', 'warehouse'], how='inner')

    return diags, predictions

def sparcity(df):
    # Flag mapping function
    def demand_flag(n):
        if n <= 3:
            return 'one_off'
        elif n < 7:
            return 'sparse_intermediate'
        else:
            return 'normal'

    real_counts = (
        df.groupby(['product_code', 'warehouse'])['demand']
        .apply(lambda x: (x > 0).sum())
        .reset_index(name='non_zero_months')
    )
    # Sparsity Flag
    real_counts['sparsity_flag'] = real_counts['non_zero_months'].apply(demand_flag)

    # Merge into main dataset
    df = df.merge(
        real_counts[['product_code', 'warehouse', 'sparsity_flag']],
        on=['product_code', 'warehouse'],
        how='left'
    )

    return df

def fill_missing_dates(df):
    frames = []
    for (p, w), g in df.groupby(["product_code", "warehouse"]):
        start = g["first_sale_date"].iloc[0]
        end = df["date"].max()
        months = pd.date_range(start=start, end=end, freq="MS")
        tmp = pd.DataFrame({"product_code": p, "warehouse": w, "date": months})
        frames.append(tmp)

    full_grid = pd.concat(frames, ignore_index=True)
    # Bring category (static) back
    static_cols = df[["product_code", "warehouse", "product_category"]].drop_duplicates()
    # Merge
    full_grid = full_grid.merge(static_cols, on=["product_code", "warehouse"], how="left")
    # Merge actual demand and zero-fill
    df = full_grid.merge(
        df[["product_code", "warehouse", "date", "demand"]],
        on=["product_code", "warehouse", "date"],
        how="left"
    )
    # -- Fill NaN demand with 0
    df['demand'] = df['demand'].fillna(0)

    return df

def heterogeneity(df):
    sample_skus = df[PID].drop_duplicates().sample(30, random_state=70) # Taking 30 SKUs randomly

    # (Product x Time) Matrix
    pivot = (
        df[df[PID].isin(sample_skus)]
        .pivot_table(
            index=PID,
            columns=DATE,
            values=TARGET,
            aggfunc="sum"
        )
    )

    # Clean index/columns for visualization
    pivot.columns = [d.strftime('%Y-%m-%d') if hasattr(d, 'strftime') else d for d in pivot.columns]

    # Plot
    plt.figure(figsize=(16, 8), facecolor='#FAFAFA')
    ax = sns.heatmap(
        np.log1p(pivot),
        cmap="mako",
        cbar_kws={"label": "log(1 + Demand Volume)"}
    )

    # Label
    display_labels = [f"PR-{str(p).split('_')[-1]}" if isinstance(p, str) else f"PR_{p}" for p in pivot.index]
    ax.set_yticks(np.arange(len(display_labels)) + 0.5)
    ax.set_yticklabels(display_labels, fontsize=9)

    # X-axis: Only show every Nth label to prevent overlapping
    ticks = np.linspace(0, len(pivot.columns) - 1, 30, dtype=int)
    ax.set_xticks(ticks)
    ax.set_xticklabels([pivot.columns[i] for i in ticks], rotation=45, ha='right')

    plt.xlabel("Timeline (Date)")
    plt.ylabel("Product Sample")
    plt.title("Spatio-Temporal Demand Heterogeneity", fontsize=15, fontweight='bold', pad=15)

    plt.tight_layout()
    plt.show()
    
def volatile_trajectories(df):

    fig, axes = plt.subplots(2, 2, figsize=(20, 14), facecolor='#FAFAFA', sharex=True)
    axes = axes.flatten()

    # Metric Calculation Function
    def get_volatile_skus_per_warehouse(df, warehouse_id, top_n=5):
        # Filter for the specific warehouse
        wh_data = df[df[WID] == warehouse_id]

        # Calculate CV to identify the most volatile SKUs locally
        metrics = wh_data.groupby(PID)[TARGET].agg(['mean', 'std']).reset_index()
        metrics['cv'] = metrics['std'] / metrics['mean']

        # Get the IDs of the top N most volatile SKUs
        top_volatile = metrics.sort_values("cv", ascending=False).head(top_n)[PID].tolist()
        return wh_data[wh_data[PID].isin(top_volatile)]

    # Warehouses
    warehouses = df[WID].unique()
    for i, wh in enumerate(warehouses):
        ax = axes[i]
        plot_data = get_volatile_skus_per_warehouse(df, wh)

        # Plot each volatile SKU's timeline
        for sku in plot_data[PID].unique():
            sku_series = plot_data[plot_data[PID] == sku].sort_values(DATE)
            ax.plot(
                sku_series[DATE], sku_series[TARGET], alpha=0.7, linewidth=2,
                label=f"SKU {sku}"
            )

        # Formatting
        ax.set_title(f"Warehouse: {wh}", fontsize=14, fontweight='bold', pad=10)
        ax.set_ylabel("Demand Volume")
        ax.grid(True, linestyle='--', alpha=0.3)

        # Rotate dates for readability
        plt.setp(ax.get_xticklabels(), rotation=30)

    # Single Legend outside
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=5, bbox_to_anchor=(0.5, -0.05), fontsize=10)
    # Main title
    plt.suptitle("Heterogeneity: Raw Volatility Deep-Dive per Warehouse", fontsize=18, fontweight='bold', y=0.96)

    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    plt.show()


def heteroscedasticity_vis(df):
    fig, axes = plt.subplots(2, 2, figsize=(20, 14), facecolor='#FAFAFA', sharex=False, sharey=False)
    axes = axes.flatten()

    # Metric Calculation Function
    def get_warehouse_metrics(df, warehouse_id):
        # Filter for the specific warehouse
        wh_data = df[df[WID] == warehouse_id]

        # Calculate Mean and Std per SKU
        metrics = wh_data.groupby(PID)[TARGET].agg(['mean', 'std']).reset_index()
        return metrics.dropna()

    # Warehouses
    warehouses = df[WID].unique()

    for i, wh in enumerate(warehouses):
        ax = axes[i]
        m_data = get_warehouse_metrics(df, wh)

        # Regression plot to show the scaling relationship
        sns.regplot(
            data=m_data,x='mean', y='std', scatter_kws={'alpha':0.2, 'color':COLORS[i]},
            line_kws={'color':'#2C3E50', 'lw':3}, ax=ax
        )

        # Formatting
        ax.set_title(f"Warehouse: {wh}", fontsize=14, fontweight='bold', pad=10)
        ax.set_xlabel("Average Monthly Demand Volume", fontsize=11)
        ax.set_ylabel("Standard Deviation (Risk)", fontsize=11)
        ax.grid(True, alpha=0.1)

    # Main title
    plt.suptitle("Heteroscedasticity: Absolute Forecast Risk vs. Demand Volume", fontsize=18, fontweight='bold', y=0.96)

    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    plt.show()

def pareto_plot(df):
    fig, axes = plt.subplots(2, 2, figsize=(18, 12), facecolor='#FAFAFA', sharex=False, sharey=True)
    axes = axes.flatten()

    # Metric Calculation Function
    def get_warehouse_pareto_data(df, warehouse_id):
        # Filter for the specific warehouse
        wh_data = df[df[WID] == warehouse_id]

        # Calculate stats per SKU in this warehouse
        m = wh_data.groupby(PID)[TARGET].agg(['mean', 'std']).reset_index()
        m = m.sort_values('mean', ascending=False).dropna()

        # Calculate Cumulative Percentages
        m['cum_vol'] = m['mean'].cumsum() / m['mean'].sum() * 100
        m['cum_var'] = m['std'].cumsum() / m['std'].sum() * 100
        return m

    # Warwhouses
    warehouses = df[WID].unique()
    lines = []
    for i, wh in enumerate(warehouses):
        ax = axes[i]
        m_sort = get_warehouse_pareto_data(df, wh)
        x_axis = np.arange(len(m_sort))

        # Plot Volume and Risk
        l1, = ax.plot(x_axis, m_sort['cum_vol'], color=MAIN_COLOR, lw=3, label='Cumulative Volume')
        l2, = ax.plot(x_axis, m_sort['cum_var'], color=ACCENT_COLOR, lw=2.5, linestyle=':', label='Cumulative Risk (Std Dev)')

        # Fill the gap
        ax.fill_between(x_axis, m_sort['cum_vol'], m_sort['cum_var'], color=ACCENT_COLOR, alpha=0.1)

        # Store lines from the first plot for the legend
        if i == 0:
            lines = [l1, l2]

        ax.set_title(f"Warehouse: {wh}", fontsize=14, fontweight='bold')
        ax.set_ylabel("Cumulative (%)")
        ax.set_xlabel("SKU Count (Sorted by Mean Volume)")
        ax.grid(True, alpha=0.1)

    # Arrange Legend Outside Plots
    fig.legend(handles=lines, loc='lower center', ncol=2, bbox_to_anchor=(0.5, -0.05),
            frameon=True, facecolor='white', fontsize=12)

    # Main title
    plt.suptitle("Pareto: Volume Concentration vs. Uncertainty per Warehouse", fontsize=18, fontweight='bold', y=0.96)

    # Adjust layout
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    plt.show()

def temporal_memory(df):
    fig, axes = plt.subplots(2, 2, figsize=(18, 12), facecolor='#FAFAFA')
    axes = axes.flatten()

    # ACF Extraction Function
    def get_warehouse_acf_matrix(df, warehouse_id, top_n=20):
        # Filter for specific warehouse and sort by volume
        wh_data = df[df[WID] == warehouse_id]
        top_skus = wh_data.groupby(PID)[TARGET].mean().nlargest(top_n).index.tolist()

        acf_list = []
        for p in top_skus:
            series = wh_data[wh_data[PID] == p].sort_values(DATE)[TARGET].fillna(0)
            # Calculate ACF for 12 months; ensure series is long enough
            if len(series) > 12:
                res = acf(series, nlags=12, fft=True)
                acf_list.append(res)
            else:
                acf_list.append(np.zeros(13))

        return np.vstack(acf_list) if acf_list else np.zeros((top_n, 13))

    # Shared colorbar limits for direct comparison
    VMIN, VMAX = 0, 0.6
    warehouses = df[WID].unique()

    for i, wh in enumerate(warehouses):
        ax = axes[i]
        acf_matrix = get_warehouse_acf_matrix(df, wh)

        # cbar=False here because we will add one global colorbar at the end
        sns.heatmap(
            acf_matrix, cmap="mako", ax=ax, cbar=False, vmin=VMIN,
            vmax=VMAX,xticklabels=np.arange(0, 13)
        )

        ax.set_title(f"Warehouse: {wh}", fontsize=14, fontweight='bold')
        ax.set_xlabel("Lag (Months)")
        ax.set_ylabel("SKU Rank (by Volume)")

    # Arranging the Colorbar
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    sm = plt.cm.ScalarMappable(cmap="mako", norm=plt.Normalize(vmin=VMIN, vmax=VMAX))
    fig.colorbar(sm, cax=cbar_ax, label='Correlation Strength')

    plt.suptitle("Autocorrelation: Localized Seasonality vs. Randomness", fontsize=18, fontweight='bold', y=0.98)

    # Adjust layout
    plt.subplots_adjust(right=0.9, hspace=0.3, wspace=0.3)
    plt.show()

def sbc_segmentation(df):
    fig, axes = plt.subplots(2, 2, figsize=(16, 8), facecolor='#FAFAFA', sharex=True, sharey=True)
    axes = axes.flatten()

    # Metric Calculation Function
    def get_sbc_metrics(df, group_cols):
        # Standard stats
        metrics = df.groupby(group_cols)[TARGET].agg(['mean', 'std']).reset_index()
        metrics['cv2'] = (metrics['std'] / metrics['mean'])**2

        # ADI Calculation: total periods / periods with demand > 0
        adi_vals = df.groupby(group_cols).apply(
            lambda x: len(x) / (x[TARGET] > 0).sum() if (x[TARGET] > 0).any() else np.nan,
            include_groups=False
        ).reset_index(name='adi')

        return metrics.merge(adi_vals, on=group_cols).dropna()

    # Data
    wh_summary = get_sbc_metrics(df, [WID, PID])
    warehouses = wh_summary[WID].unique()

    for i, wh in enumerate(warehouses):
        ax = axes[i]
        data = wh_summary[wh_summary[WID] == wh]

        sns.scatterplot(data=data, x='adi', y='cv2', alpha=0.6, color=COLORS[i], ax=ax, s=40)

        # Threshold Lines
        ax.axhline(0.49, color='black', linestyle='--', alpha=0.3, lw=1.5)
        ax.axvline(1.32, color='black', linestyle='--', alpha=0.3, lw=1.5)

        # Title and Labels
        ax.set_title(f"Warehouse: {wh}", fontsize=14, fontweight='bold', pad=10)

        # Dynamic Quadrant Labels
        ylim = ax.get_ylim()[1]
        ax.text(1.05, 0.1, "Smooth", fontsize=10, weight='bold', color='#27ae60')
        ax.text(1.4, 0.1, "Intermittent", fontsize=10, weight='bold', color='#f39c12')
        ax.text(1.05, ylim*0.85, "Erratic", fontsize=10, weight='bold', color='#f39c12')
        ax.text(1.4, ylim*0.85, "Lumpy", fontsize=10, weight='bold', color='#e74c3c')

        # Axis formatting
        ax.set_xlabel("Average Demand Interval (ADI)")
        ax.set_ylabel("CV² (Volatility Index)")
        ax.grid(True, alpha=0.1)

    plt.yscale("log")
    plt.suptitle("Decentralized Demand: SBC Segmentation per Warehouse", fontsize=18, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.show()
    
def volatility_index(df):
    fig = plt.figure(figsize=(20, 14), facecolor='#FAFAFA')
    gs = fig.add_gridspec(2, 4, hspace=0.3)

    # Metric Calculation Function
    def get_cv_metrics(df, group_cols):
        # Calculate Mean, Std, and CV (Standard Deviation / Mean)
        metrics = df.groupby(group_cols)[TARGET].agg(['mean', 'std']).reset_index()
        metrics['cv'] = metrics['std'] / metrics['mean']
        return metrics.dropna()

    # Data
    global_cv = get_cv_metrics(df, PID)
    wh_cv = get_cv_metrics(df, [WID, PID])

    # Global CV Distribution
    ax_global = fig.add_subplot(gs[0, :])
    sns.histplot(
        global_cv['cv'], bins=100, kde=True, color='#2C3E50', edgecolor='white', ax=ax_global
    )

    # Global Formatting
    ax_global.axvline(1.0, color='#FF7F50', linestyle='--', lw=2.5, label='High Volatility Threshold (CV=1)')
    ax_global.set_title("Global CV Distribution", fontsize=16, fontweight='bold', pad=20)
    ax_global.set_xlabel("Volatility Index ($Std Dev / Mean$)")
    ax_global.set_ylabel("Frequency")
    ax_global.legend()
    ax_global.grid(True, ls="--", alpha=0.1)

    # Warehouse-Level CV Distribution
    warehouses = wh_cv[WID].unique()
    for i, wh in enumerate(warehouses):
        ax = fig.add_subplot(gs[1, i])
        data = wh_cv[wh_cv[WID] == wh]['cv']

        sns.histplot(
            data, bins=50, kde=True, color=COLORS[i], edgecolor='white', ax=ax
        )

        # Warehouse Formatting
        ax.axvline(1.0, color='black', linestyle='--', alpha=0.4, lw=1.5)
        ax.set_title(f"Warehouse: {wh}", fontsize=13, fontweight='bold')
        ax.set_xlabel("CV Index")
        ax.set_ylabel("Frequency")
        ax.grid(True, ls="--", alpha=0.1)

    # Main title
    plt.suptitle("Global vs. Warehouse-Level Volatility Index (CV) Distribution", fontsize=20, fontweight='bold', y=0.96)

    plt.tight_layout()
    plt.show()
    
def demand_intermittency(df):
    fig = plt.figure(figsize=(20, 14), facecolor='#FAFAFA')
    gs = fig.add_gridspec(2, 4, hspace=0.3)

    # Metric Calculation Function
    def get_zero_fractions(df, group_cols):
        # Calculate the fraction of periods where demand is exactly 0
        return df.assign(is_zero=(df[TARGET] == 0)).groupby(group_cols)["is_zero"].mean().reset_index()

    # Data
    global_zeros = get_zero_fractions(df, PID)
    wh_zeros = get_zero_fractions(df, [WID, PID])

    # Global Intermittency
    ax_global = fig.add_subplot(gs[0, :])
    sns.histplot(
        global_zeros["is_zero"], bins=100, kde=False, color=MAIN_COLOR, edgecolor='white', ax=ax_global
    )
    # Global Formating
    ax_global.set_yscale("log")
    ax_global.set_title("Global Intermittency", fontsize=16, fontweight='bold', pad=20)
    ax_global.set_xlabel("Zero-Demand Fraction", fontsize=12)
    ax_global.set_ylabel("Number of SKUs (Log Scale)", fontsize=12)
    ax_global.grid(True, ls="--", alpha=0.1)

    # Warehouse-Level Intermittency
    warehouses = wh_zeros[WID].unique()
    for i, wh in enumerate(warehouses):
        ax = fig.add_subplot(gs[1, i])
        data = wh_zeros[wh_zeros[WID] == wh]["is_zero"]

        sns.histplot(
            data, bins=30, kde=False, color=COLORS[i], edgecolor='white', ax=ax
        )
        # Warehouse Formatting
        ax.set_yscale("log")
        ax.set_title(f"Warehouse: {wh}", fontsize=13, fontweight='bold')
        ax.set_xlabel("Zero-Demand Fraction")
        ax.set_ylabel("Number of SKUs (Log Scale)")
        ax.grid(True, ls="--", alpha=0.1)

    # Main title
    plt.suptitle("Global vs. Warehouse-Level Intermittency: Fraction of Zero-Demand Periods", fontsize=20, fontweight='bold', y=0.96)

    plt.tight_layout()
    plt.show()

def perform_abc_xyz_segmentation(df:pd.DataFrame):
  """Segments demand behaviour into business categories (ABC/XYZ)"""
  # -- Aggregate demand by Product
  product_data = df.groupby(['product_code', 'warehouse']).agg(
      total_demand=('demand', 'sum'),
      std_demand=('demand', 'std'),
      mean_demand=('demand', 'mean')
  ).reset_index()

  # -- ABC ANALYSIS (Volume)
  product_data = product_data.sort_values(by='total_demand', ascending=False)
  product_data['cum_percentage'] = 100 * product_data['total_demand'].cumsum() / product_data['total_demand'].sum()

  def map_abc(perc):
      if perc <= 80: return 'A'
      if perc <= 95: return 'B'
      return 'C'

  product_data['ABC'] = product_data['cum_percentage'].apply(map_abc)

  # -- XYZ ANALYSIS (Volatility)
  # CV = Std Dev / Mean. Higher CV = Harder to forecast.
  product_data['CV'] = product_data['std_demand'] / product_data['mean_demand']

  def map_xyz(cv):
      if cv <= 0.5: return 'X'
      if cv <= 1.0: return 'Y'
      return 'Z'

  product_data['XYZ'] = product_data['CV'].apply(map_xyz)

  # Final Segment tag (e.g., AX, BY, CZ)
  product_data['segment'] = product_data['ABC'] + product_data['XYZ']

  return product_data.reset_index(drop=True)

def forecast_data(model, prediction_output):
  """Model Output"""
  quantiles = model.loss.quantiles
  _, horizon, n_quantiles = prediction_output.output.shape # [batch_size, prediction_horizon, n_quantiles]

  # Extract metadata (Time Index and Groups)
  # Reshape time_idx to match the flattened output later
  time_indices = prediction_output.x["decoder_time_idx"].cpu().numpy().flatten()

  # Repeat group identifiers for each step in the horizon
  groups = prediction_output.x["groups"].cpu().numpy()
  product_codes = np.repeat(groups[:, 0], horizon)
  warehouses = np.repeat(groups[:, 1], horizon)

  # Build the base DataFrame
  df = pd.DataFrame({
      "time_idx": time_indices,
      "product_code": product_codes,
      "warehouse": warehouses,
  })

  # Extract and Inverse-Transform Quantiles
  # Flatten the batch and horizon dimensions to align with the metadata
  raw_predictions = prediction_output.output.reshape(-1, n_quantiles).cpu().numpy()

  for i, q in enumerate(quantiles):
      col_name = f"p{int(q*100):02d}"
      # np.expm1 handles the log transformation: exp(x) - 1
      df[col_name] = np.expm1(raw_predictions[:, i])

  # Final Formatting
  # Standardizing 'demand' to a specific quantile (e.g., p90 for conservative stock)
  df["demand"] = df["p50"]
  df["log_demand"] = np.log1p(df["demand"])

  return df

def forecast_horizon(history, forecast_len, training, model):
  """To get the forecasted demand"""
  H = forecast_len
  step = 2
  history['month'] = history['date'].dt.month
  current_data = history.copy()
  all_forecasts = []

  for start in range(0, H, step):
      # build future rows (2 months ahead)
      future_rows = []

      for (p, w, c), g in current_data.groupby(["product_code", "warehouse", "product_category"]):
          g = g.sort_values("time_idx")
          last = g.iloc[-1]
          last_idx = last["time_idx"]
          last_month = last["month"]

          for h in range(1, step + 1):
              r = last.copy()
              r["time_idx"] = last_idx + h
              # unknown target
              r["log_demand"] = 0.0
              r["demand"] = 0.0
              # calendar
              m = ((last_month - 1 + h) % 12) + 1
              r["month_sin"] = np.sin(2*np.pi*m/12)
              r["month_cos"] = np.cos(2*np.pi*m/12)
              r["is_global_holiday_month"] = int(m in [11, 12])

              future_rows.append(r)

      future_df = pd.concat([current_data, pd.DataFrame(future_rows)], ignore_index=True)
      future_df[['is_global_holiday_month']] = future_df[['is_global_holiday_month']].astype('str')

      # Build dataset
      future_dataset = TimeSeriesDataSet.from_dataset(
          training, future_df, predict=True, stop_randomization=True
      )

      # Loader
      loader = future_dataset.to_dataloader(train=False, batch_size=128)

      # predict
      pred = model.predict(loader, mode="quantiles", return_x=True)
      result = forecast_data(model, pred)
      # Decoding
      for col in ["product_code", "warehouse"]:
          enc = training.get_parameters()["categorical_encoders"][col]
          result[col] = enc.inverse_transform(
              result[col].astype(int).values)

      # Full Dataset
      result_ = pd.DataFrame(future_rows).drop(columns=['log_demand', 'demand']).merge(result,
              on= ["time_idx", "product_code", "warehouse"], how='left', validate="many_to_one")

      # Store prediction
      all_forecasts.append(result_)

      # Append predictions as history
      current_data = pd.concat([current_data, result_.drop(columns=['p50', 'p80', 'p90'])], ignore_index=True)

  # final forecast
  forecast_df = pd.concat(all_forecasts)

  return forecast_df

def plot_forecast(ax, actual_df, forecast_df, segment_name, warehouse, product_code, start_date, bridge=False):
  """Plot total demand cycle of a product in a warehouse."""
  # Dates
  actual_dates = [pd.to_datetime(start_date) + pd.DateOffset(months=int(i))for i in actual_df["time_idx"]]
  forecast_dates = [pd.to_datetime(start_date) + pd.DateOffset(months=int(i))for i in forecast_df["time_idx"]]

  # Decision Demand
  buffer = 0.5 if 'X' in segment_name else 1.0
  decision_forecast = forecast_df["p50"] + buffer * (forecast_df["p80"] - forecast_df["p50"])

  # Actuals
  ax.plot(actual_dates, actual_df["demand"], color=MAIN_COLOR, lw=2, marker="o", label="Actual")

  # Uncertainty
  ax.fill_between(forecast_dates, forecast_df["p50"], forecast_df["p90"],
      color="#3498DB", alpha=0.15, label="P50–P90")

  # Baseline (p50)
  ax.plot(forecast_dates, forecast_df["p50"], color=GAP_COLOR, lw=1.2,
      linestyle=":", label="P50")

  # Decision forecast
  ax.plot(forecast_dates, decision_forecast, color="#E67E22", lw=2, marker="s", label="Decision")

  # --- Bridge
  if bridge:
      ax.plot([actual_dates[-1], forecast_dates[0]], [actual_df["demand"].iloc[-1], decision_forecast.iloc[0]], color="#E67E22",
          linestyle="--", alpha=0.6)
  # Lines
  ax.axvline(actual_dates[-1], color="black", linestyle="--", alpha=0.3)
  ax.axvspan(actual_dates[-1], forecast_dates[-1], color="#ECF0F1", alpha=0.4)

  ax.set_title(f"Segment: {segment_name} ||   Warehouse: {warehouse} ||   Product: {product_code}", fontsize=14, fontweight="bold")
  ax.grid(alpha=0.25)

  return ax

def get_random(df_actual: pd.DataFrame, df_forecast: pd.DataFrame, segment_id: str):
    """
    Extracts a random warehouse/product pair from a specific segment.
    """

    # Pre-filter by segment to reduce search space
    actual_s = df_actual[df_actual['segment'] == segment_id]
    if actual_s.empty:
        return False

    # Extract unique combinations efficiently
    unique_pairs = actual_s[['warehouse', 'product_code']].drop_duplicates()

    if unique_pairs.empty:
        return False

    # Select a random pair
    random_row = unique_pairs.sample(n=1).iloc[0]
    w_id, p_id = random_row['warehouse'], random_row['product_code']

    # Filter and sort subsets
    d_actual = actual_s[
        (actual_s["warehouse"] == w_id) &
        (actual_s["product_code"] == p_id)
    ].sort_values("time_idx")
    # Forecast sort
    d_forecast = df_forecast[
        (df_forecast['segment'] == segment_id) &
        (df_forecast["warehouse"] == w_id) &
        (df_forecast["product_code"] == p_id)
    ].sort_values("time_idx")

    return d_actual, d_forecast, segment_id, w_id, p_id

def missing_dates(df: pd.DataFrame):
    
    # Sum demand globally to identify timeline gaps
    daily_data = df.groupby('date')['demand'].sum().reset_index()

    # Create the full date range based on the dataset's min/max
    full_range = pd.date_range(start=df['date'].min(), end=df['date'].max(), freq='D').date

    # Reindex to find where days are missing
    df_reindexed = daily_data.set_index('date').reindex(full_range)
    df_reindexed['is_missing'] = df_reindexed['demand'].isna().astype(int)

    # Plotting
    plt.figure(figsize=(16, 3), facecolor='#FAFAFA')
    sns.heatmap(
        df_reindexed[['is_missing']].T,
        cbar=False,
        cmap=sns.color_palette([MAIN_COLOR, GAP_COLOR]),
        yticklabels=False
    )

    # Label
    plt.title('Global Timeline Continuity (Orange = Data Gaps)', fontsize=15, fontweight='bold', pad=15)
    plt.xlabel("Timeline Index (Days)", fontsize=12)
    plt.tight_layout()
    plt.show()

def demand_distribution(df):

    fig = plt.figure(figsize=(20, 14), facecolor='#FAFAFA')
    gs = fig.add_gridspec(2, 4, hspace=0.3) # 2 Rows, 4 Columns

    # Global Distribution
    ax_global = fig.add_subplot(gs[0, :])
    sns.histplot(
        df[TARGET], bins=100, kde=False,
        color= MAIN_COLOR, edgecolor='white',
        ax=ax_global
    )

    # Global Formatting
    ax_global.set_yscale("log")
    ax_global.set_title("Global Demand Distribution", fontsize=16, fontweight='bold', pad=20)
    ax_global.set_xlabel("Order Demand", fontsize=12)
    ax_global.set_ylabel("Frequency (Log Scale)", fontsize=12)
    ax_global.grid(True, which="both", ls="--", alpha=0.1)

    # Warehouse-Level Distributions
    warehouses = df[WID].unique()
    for i, wh in enumerate(warehouses):
        ax = fig.add_subplot(gs[1, i])
        wh_data = df[df[WID] == wh][TARGET]
        # Plot
        sns.histplot(
            wh_data, bins=50, kde=False, color=COLORS[i], edgecolor='white', ax=ax
        )

        # Warehouse Formatting
        ax.set_yscale("log")
        ax.set_title(f"Warehouse: {wh}", fontsize=13, fontweight='bold')
        ax.set_xlabel("Order Demand")
        ax.set_ylabel("Frequency (Log Scale)")
        ax.grid(True, which="both", ls="--", alpha=0.1)

    # Main title
    plt.suptitle("Global vs. Warehouse-Level Demand Distribution", fontsize=20, fontweight='bold', y=0.96)

    plt.tight_layout()
    plt.show()