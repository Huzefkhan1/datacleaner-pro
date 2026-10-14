import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from io import BytesIO

# ============ PAGE CONFIG ============
st.set_page_config(
    page_title="DataCleaner Pro",
    page_icon="🧹",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============ CUSTOM CSS ============
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1f77b4;
        margin-bottom: 0;
    }
    .sub-header {
        font-size: 1rem;
        color: #666;
        margin-top: 0;
    }
    .stButton > button {
        width: 100%;
        background-color: #1f77b4;
        color: white;
        font-weight: 600;
        padding: 0.75rem;
        border-radius: 0.5rem;
    }
    .stButton > button:hover {
        background-color: #165a8c;
    }
</style>
""", unsafe_allow_html=True)

# ============ HEADER ============
st.markdown('<p class="main-header">🧹 DataCleaner Pro</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Upload your file and get clean, analysis-ready data in seconds.</p>', unsafe_allow_html=True)
st.divider()


# ============ HELPER FUNCTIONS ============
def count_missing(d):
    """Count missing values including empty strings and NA variants"""
    total = 0
    for c in d.columns:
        total += int(d[c].isnull().sum())
        if d[c].dtype == 'object':
            s = d[c].astype(str).str.strip().str.lower()
            total += int(s.isin(['', 'na', 'n/a', 'null', 'none', 'nan', '-', '--']).sum())
    return total


def detect_issues(df):
    """Detect data quality issues in each column"""
    issues = {}
    
    for col in df.columns:
        col_issues = []
        col_lower = col.lower()
        
        # 1. Missing values
        missing_count = int(df[col].isnull().sum())
        empty_count = 0
        if df[col].dtype == 'object':
            s = df[col].astype(str).str.strip().str.lower()
            empty_count = int(s.isin(['', 'na', 'n/a', 'null', 'none', 'nan', '-', '--']).sum())
        
        total_missing = missing_count + empty_count
        if total_missing > 0:
            col_issues.append({
                'type': 'Missing Values',
                'count': int(total_missing),
                'severity': 'high' if total_missing > len(df) * 0.1 else 'medium'
            })
        
        # 2. Duplicates (only for ID-like columns)
        is_id_like = any(kw in col_lower for kw in ['id', 'code', 'key', 'number', 'no'])
        if is_id_like:
            dup_count = int(df[col].duplicated().sum())
            if dup_count > 0:
                col_issues.append({
                    'type': 'Duplicate Values (ID column)',
                    'count': dup_count,
                    'severity': 'high'
                })
        
        # 3. Extra whitespace
        if df[col].dtype == 'object':
            has_spaces = int(df[col].astype(str).str.contains(r'^\s+|\s+$', regex=True, na=False).sum())
            if has_spaces > 0:
                col_issues.append({
                    'type': 'Extra Whitespace',
                    'count': has_spaces,
                    'severity': 'low'
                })
        
        # 4. Negative values
        if df[col].dtype in ['int64', 'float64']:
            neg_count = int((df[col] < 0).sum())
            if neg_count > 0:
                col_issues.append({
                    'type': 'Negative Values',
                    'count': neg_count,
                    'severity': 'high'
                })
        
        # 5. Zero values in price/delivery/rating columns
        if df[col].dtype in ['int64', 'float64']:
            if any(kw in col_lower for kw in ['price', 'amount', 'cost', 'delivery', 'time']):
                zero_count = int((df[col] == 0).sum())
                if zero_count > 0:
                    col_issues.append({
                        'type': 'Zero Values (likely invalid)',
                        'count': zero_count,
                        'severity': 'high'
                    })
        
        # 6. Ratings out of range (0-5) — only for AVG ratings, not total
        is_avg_rating = any(kw in col_lower for kw in ['avg rating', 'average rating', 'avg_rating', 'rating_avg'])
        if df[col].dtype in ['int64', 'float64'] and is_avg_rating:
            out_of_range = int(((df[col] < 0) | (df[col] > 5)).sum())
            if out_of_range > 0:
                col_issues.append({
                    'type': 'Out of Range (0-5 expected)',
                    'count': out_of_range,
                    'severity': 'high'
                })
        
        # 7. Low cardinality
        if df[col].dtype in ['int64', 'float64'] and len(df) > 100:
            unique_ratio = df[col].nunique() / len(df)
            if 0 < unique_ratio < 0.01 and df[col].nunique() > 1:
                col_issues.append({
                    'type': 'Low Cardinality (possible categorical)',
                    'count': int(df[col].nunique()),
                    'severity': 'low'
                })
        
        # 8. Outliers (IQR)
        if df[col].dtype in ['int64', 'float64']:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            if IQR > 0:
                outliers = int(((df[col] < Q1 - 1.5*IQR) | (df[col] > Q3 + 1.5*IQR)).sum())
                if outliers > 0:
                    col_issues.append({
                        'type': 'Outliers (IQR)',
                        'count': outliers,
                        'severity': 'medium'
                    })
        
        if col_issues:
            issues[col] = col_issues
    
    return issues


def clean_dataframe(df, config):
    """Apply cleaning operations based on user configuration"""
    df_clean = df.copy()
    changes = []
    
    # 1. Clean column names
    if config.get('clean_column_names'):
        original = list(df_clean.columns)
        df_clean.columns = (df_clean.columns
                            .str.strip()
                            .str.lower()
                            .str.replace(' ', '_')
                            .str.replace(r'[^\w]', '', regex=True))
        changes.append(f"Standardized {len(original)} column names (lowercase, underscores)")
    
    # 2. Remove duplicate rows
    if config.get('remove_duplicates'):
        before = len(df_clean)
        df_clean = df_clean.drop_duplicates()
        removed = before - len(df_clean)
        if removed > 0:
            changes.append(f"Removed {removed:,} duplicate rows")
    
    # 3. Text cleaning
    for col in df_clean.select_dtypes(include='object').columns:
        if config.get('strip_spaces'):
            df_clean[col] = df_clean[col].astype(str).str.strip()
        
        if config.get('empty_to_nan'):
            df_clean[col] = df_clean[col].replace(['', 'NA', 'N/A', 'null', 'None', 'nan', '-', '--'], np.nan)
    
    # 4. Fix zero values FIRST (before missing handling)
    if config.get('fix_zeros'):
        for col in df_clean.select_dtypes(include=['int64', 'float64']).columns:
            col_lower = col.lower()
            if any(kw in col_lower for kw in ['price', 'amount', 'cost', 'delivery', 'time']):
                zero_count = int((df_clean[col] == 0).sum())
                if zero_count > 0:
                    df_clean[col] = df_clean[col].replace(0, np.nan)
                    changes.append(f"Converted {zero_count:,} zero values to NaN in '{col}'")
    
    # 5. Fix negative values
    if config.get('fix_negatives'):
        for col in df_clean.select_dtypes(include=['int64', 'float64']).columns:
            neg = int((df_clean[col] < 0).sum())
            if neg > 0:
                df_clean[col] = df_clean[col].clip(lower=0)
                changes.append(f"Corrected {neg:,} negative values in '{col}'")
    
    # 6. Handle missing values LAST (after all NaN creation)
    if config.get('handle_missing'):
        method = config.get('missing_method', 'drop')
        
        for col in df_clean.columns:
            missing_before = df_clean[col].isnull().sum()
            if missing_before == 0:
                continue
            
            if method == 'drop_rows':
                df_clean = df_clean.dropna(subset=[col])
            elif method == 'drop_cols':
                df_clean = df_clean.drop(columns=[col])
            elif method == 'mean' and df_clean[col].dtype in ['int64', 'float64']:
                df_clean[col] = df_clean[col].fillna(df_clean[col].mean())
            elif method == 'median' and df_clean[col].dtype in ['int64', 'float64']:
                df_clean[col] = df_clean[col].fillna(df_clean[col].median())
            elif method == 'mode':
                mode_val = df_clean[col].mode()
                if len(mode_val) > 0:
                    df_clean[col] = df_clean[col].fillna(mode_val[0])
            elif method == 'zero' and df_clean[col].dtype in ['int64', 'float64']:
                df_clean[col] = df_clean[col].fillna(0)
            
            filled = missing_before - df_clean[col].isnull().sum()
            if filled > 0:
                changes.append(f"Imputed {int(filled):,} missing values in '{col}' using {method}")
    
    return df_clean, changes


# ============ SIDEBAR ============
with st.sidebar:
    st.markdown("### ⚙️ DataCleaner Pro")
    st.markdown("---")
    st.markdown("#### How to use")
    st.markdown("""
    1. **Upload** your file
    2. **Review** detected issues
    3. **Configure** cleaning options
    4. **Download** clean data
    """)
    st.markdown("---")
    st.markdown("#### Supported Formats")
    st.markdown("• CSV (.csv)\n\n• Excel (.xlsx, .xls)")
    st.markdown("---")
    st.caption("Built with Streamlit • v1.0")


# ============ MAIN CONTENT ============
st.subheader("📁 Upload File")

uploaded_file = st.file_uploader(
    "Choose a CSV or Excel file",
    type=['csv', 'xlsx', 'xls'],
    help="Maximum file size: 200MB",
    label_visibility="collapsed"
)

if uploaded_file is not None:
    try:
        with st.spinner("Loading file..."):
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
        
        st.success(f"✓ File uploaded successfully: **{uploaded_file.name}**")
        
        # ============ OVERVIEW ============
        st.subheader("📊 Dataset Overview")
        
        total_missing = count_missing(df)
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Rows", f"{df.shape[0]:,}")
        col2.metric("Total Columns", f"{df.shape[1]:,}")
        col3.metric("Missing Values", f"{total_missing:,}")
        col4.metric("Duplicate Rows", f"{df.duplicated().sum():,}")
        
        st.divider()
        
        # ============ DATA QUALITY REPORT ============
        st.subheader("🔍 Data Quality Report")
        
        issues = detect_issues(df)
        
        if len(issues) == 0:
            st.success("✓ Excellent! Your dataset has no detected quality issues.")
        else:
            st.warning(f"⚠️ Detected issues in **{len(issues)}** column(s)")
            
            summary_data = []
            for col, col_issues in issues.items():
                for issue in col_issues:
                    summary_data.append({
                        'Column': col,
                        'Issue Type': issue['type'],
                        'Count': issue['count'],
                        'Severity': issue['severity'].upper()
                    })
            
            summary_df = pd.DataFrame(summary_data)
            st.dataframe(summary_df, use_container_width=True, hide_index=True)
            
            with st.expander("View Detailed Column-wise Issues"):
                for col, col_issues in issues.items():
                    st.markdown(f"**Column: `{col}`**")
                    for issue in col_issues:
                        icon = "🔴" if issue['severity'] == 'high' else "🟡" if issue['severity'] == 'medium' else "🟢"
                        st.markdown(f"  {icon} {issue['type']}: **{issue['count']:,}** affected values")
                    st.markdown("")
        
        st.divider()
        
        # ============ DATA PREVIEW ============
        st.subheader("👀 Data Preview")
        
        tab1, tab2 = st.tabs(["Data Sample", "Column Information"])
        
        with tab1:
            st.dataframe(df.head(20), use_container_width=True)
        
        with tab2:
            col_info = pd.DataFrame({
                'Column': df.columns,
                'Data Type': df.dtypes.astype(str).values,
                'Missing': df.isnull().sum().values,
                'Unique': df.nunique().values,
                'Sample Values': [str(df[c].dropna().head(2).tolist()) for c in df.columns]
            })
            st.dataframe(col_info, use_container_width=True, hide_index=True)
        
        st.divider()
        
        # ============ CLEANING CONFIGURATION ============
        st.subheader("🧹 Cleaning Configuration")
        st.markdown("Select the operations you want to apply:")
        
        col_a, col_b = st.columns(2)
        
        with col_a:
            st.markdown("**Basic Cleaning**")
            clean_cols = st.checkbox("Standardize column names (lowercase, no spaces)", value=True)
            remove_dups = st.checkbox("Remove duplicate rows", value=True)
            strip_sp = st.checkbox("Strip extra whitespace from text", value=True)
            empty_nan = st.checkbox("Convert empty strings to null values", value=False)
            fix_neg = st.checkbox("Correct negative values to zero", value=False)
            fix_zeros = st.checkbox("Convert zero values to null (price/delivery/rating)", value=False)
        
        with col_b:
            st.markdown("**Missing Values Handling**")
            handle_missing = st.checkbox("Handle missing values", value=True)
            
            missing_method = st.selectbox(
                "Imputation method:",
                options=['drop_rows', 'drop_cols', 'mean', 'median', 'mode', 'zero'],
                format_func=lambda x: {
                    'drop_rows': 'Drop rows with missing values',
                    'drop_cols': 'Drop columns with missing values',
                    'mean': 'Fill with mean (numeric only)',
                    'median': 'Fill with median (numeric only)',
                    'mode': 'Fill with mode (most frequent)',
                    'zero': 'Fill with zero (numeric only)'
                }[x]
            )
        
        st.markdown("")
        
        # ============ APPLY CLEANING ============
        if st.button("🚀 Clean Data", type="primary", use_container_width=True):
            
            config = {
                'clean_column_names': clean_cols,
                'remove_duplicates': remove_dups,
                'strip_spaces': strip_sp,
                'empty_to_nan': empty_nan,
                'fix_negatives': fix_neg,
                'fix_zeros': fix_zeros,
                'handle_missing': handle_missing,
                'missing_method': missing_method
            }
            
            with st.spinner("Cleaning data..."):
                df_clean, changes = clean_dataframe(df, config)
            
            st.success(f"✓ Cleaning completed — **{len(changes)}** operation(s) applied")
            
            with st.expander("Changes Applied", expanded=True):
                if changes:
                    for i, change in enumerate(changes, 1):
                        st.markdown(f"**{i}.** {change}")
                else:
                    st.info("No changes were applied.")
            
            st.divider()
            
            # ============ BEFORE vs AFTER ============
            st.subheader("⚖️ Before vs After Comparison")
            
            before_missing = count_missing(df)
            after_missing = count_missing(df_clean)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### ❌ Before Cleaning")
                st.metric("Rows", f"{df.shape[0]:,}")
                st.metric("Columns", f"{df.shape[1]:,}")
                st.metric("Missing Values", f"{before_missing:,}")
                st.metric("Duplicates", f"{df.duplicated().sum():,}")
            
            with col2:
                st.markdown("#### ✅ After Cleaning")
                st.metric("Rows", f"{df_clean.shape[0]:,}", 
                         delta=f"{df_clean.shape[0] - df.shape[0]:,}")
                st.metric("Columns", f"{df_clean.shape[1]:,}",
                         delta=f"{df_clean.shape[1] - df.shape[1]:,}")
                st.metric("Missing Values", f"{after_missing:,}",
                         delta=f"{after_missing - before_missing:,}")
                st.metric("Duplicates", f"{df_clean.duplicated().sum():,}",
                         delta=f"{df_clean.duplicated().sum() - df.duplicated().sum():,}")
            
            st.markdown("#### Preview of Cleaned Data")
            st.dataframe(df_clean.head(20), use_container_width=True)
            
            st.divider()
            
            # ============ DOWNLOAD ============
            st.subheader("💾 Download Cleaned Data")
            
            base_name = uploaded_file.name.rsplit('.', 1)[0]
            
            col1, col2 = st.columns(2)
            
            with col1:
                csv = df_clean.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download as CSV",
                    data=csv,
                    file_name=f"{base_name}_cleaned.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            
            with col2:
                buffer = BytesIO()
                with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                    df_clean.to_excel(writer, index=False, sheet_name='Cleaned Data')
                st.download_button(
                    label="📥 Download as Excel",
                    data=buffer.getvalue(),
                    file_name=f"{base_name}_cleaned.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True
                )
            
            # ============ VISUALIZATIONS ============
            st.divider()
            st.subheader("📈 Data Visualizations")
            
            numeric_cols = df_clean.select_dtypes(include=['int64', 'float64']).columns.tolist()
            cat_cols = df_clean.select_dtypes(include='object').columns.tolist()
            
            if len(numeric_cols) > 0 or len(cat_cols) > 0:
                tab1, tab2, tab3 = st.tabs(["Distribution", "Correlation", "Category Analysis"])
                
                with tab1:
                    if numeric_cols:
                        col_choice = st.selectbox("Select column:", numeric_cols, key='dist')
                        fig = px.histogram(df_clean, x=col_choice, 
                                          title=f"Distribution of {col_choice}",
                                          color_discrete_sequence=['#1f77b4'])
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("No numeric columns available for distribution analysis.")
                
                with tab2:
                    if len(numeric_cols) >= 2:
                        corr = df_clean[numeric_cols].corr()
                        fig = px.imshow(corr, text_auto=True, aspect="auto", 
                                       title="Correlation Heatmap", 
                                       color_continuous_scale='RdBu_r')
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("At least 2 numeric columns required for correlation analysis.")
                
                with tab3:
                    if cat_cols:
                        cat_choice = st.selectbox("Select column:", cat_cols, key='cat')
                        top_n = st.slider("Number of top values:", 5, 20, 10)
                        counts = df_clean[cat_choice].value_counts().head(top_n).reset_index()
                        counts.columns = [cat_choice, 'Count']
                        fig = px.bar(counts, x='Count', y=cat_choice, orientation='h',
                                    title=f"Top {top_n} values in {cat_choice}",
                                    color_discrete_sequence=['#1f77b4'])
                        fig.update_layout(yaxis={'categoryorder':'total ascending'})
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("No categorical columns available for analysis.")
            else:
                st.info("No columns available for visualization.")
            
            # ============ REPORT DOWNLOAD ============
            st.divider()
            st.subheader("📄 Cleaning Report")
            
            report = f"""# DataCleaner Pro — Cleaning Report

**Source File:** {uploaded_file.name}
**Generated:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

## Summary

| Metric | Before | After | Change |
|--------|--------|-------|--------|
| Rows | {df.shape[0]:,} | {df_clean.shape[0]:,} | {df_clean.shape[0] - df.shape[0]:,} |
| Columns | {df.shape[1]:,} | {df_clean.shape[1]:,} | {df_clean.shape[1] - df.shape[1]:,} |
| Missing Values | {before_missing:,} | {after_missing:,} | {after_missing - before_missing:,} |
| Duplicates | {df.duplicated().sum():,} | {df_clean.duplicated().sum():,} | {df_clean.duplicated().sum() - df.duplicated().sum():,} |

## Operations Applied

"""
            if changes:
                for i, change in enumerate(changes, 1):
                    report += f"{i}. {change}\n"
            else:
                report += "No operations were applied.\n"
            
            st.download_button(
                label="📥 Download Report (Markdown)",
                data=report,
                file_name=f"{base_name}_report.md",
                mime="text/markdown",
                use_container_width=True
            )
    
    except Exception as e:
        st.error(f"❌ Error processing file: {str(e)}")
        with st.expander("View Error Details"):
            st.exception(e)

else:
    st.info("👆 Upload a file above to begin")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### 📤 Upload")
        st.markdown("Upload your CSV or Excel file")
    
    with col2:
        st.markdown("### 🔍 Detect")
        st.markdown("Automatic issue detection")
    
    with col3:
        st.markdown("### 💾 Download")
        st.markdown("Get clean data instantly")
    
    with st.expander("ℹ️ About DataCleaner Pro"):
        st.markdown("""
        **DataCleaner Pro** automatically detects and fixes common data quality issues:
        
        - Missing values (NaN, empty strings, N/A variants)
        - Duplicate rows and values
        - Extra whitespace in text
        - Negative values in numeric columns
        - Zero values in price/delivery columns
        - Out-of-range ratings (0-5)
        - Statistical outliers (IQR method)
        - Low cardinality columns
        
        **Workflow:**
        1. Upload your dataset
        2. Review the quality report
        3. Select cleaning operations
        4. Download clean data and report
        
        **Supported formats:** CSV, XLSX, XLS
        """)