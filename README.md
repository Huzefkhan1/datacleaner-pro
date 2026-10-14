# 🧹 DataCleaner Pro

A Streamlit web app that automatically detects and fixes common data quality issues — upload messy CSV/Excel files and get clean, analysis-ready data in seconds.

**🌐 Live Demo:** [datacleaner-pro.streamlit.app](https://datacleaner-pro-k8rigfor3ckfyrcb8ujmdj.streamlit.app)

---

## 🎯 Features

### Auto Issue Detection (8 types)
- **Missing Values** — NaN, empty strings, N/A variants
- **Duplicate Values** — in ID-like columns
- **Extra Whitespace** — leading/trailing spaces in text
- **Negative Values** — in numeric columns
- **Zero Values** — invalid in price/delivery columns
- **Out-of-Range Ratings** — outside 0-5 range
- **Low Cardinality** — possible categorical disguised as number
- **Statistical Outliers** — using IQR method

### Smart Cleaning Pipeline
- Column name standardization (lowercase, underscores)
- Duplicate row removal
- Whitespace stripping
- Missing value imputation (mean / median / mode / zero / drop)
- Zero-to-NaN conversion for price/delivery columns
- Negative value correction

### Export & Reports
- Cleaned data as **CSV** or **Excel**
- Detailed cleaning report as **Markdown**
- Before vs After comparison metrics

### Interactive Visualizations
- Distribution histograms (Plotly)
- Correlation heatmaps
- Category analysis bar charts

---

## 🚀 How It Works

1. **Upload** your CSV or Excel file (up to 200MB)
2. **Review** auto-detected data quality issues
3. **Configure** cleaning operations from checkboxes
4. **Clean** with one click and download

---

## 🛠️ Tech Stack

| Tool | Purpose |
|------|---------|
| **Python 3.11** | Core language |
| **Streamlit** | Web interface |
| **Pandas** | Data manipulation |
| **NumPy** | Numeric operations |
| **Plotly** | Interactive charts |
| **OpenPyXL** | Excel read/write |

---

## 📁 Project Structure

datacleaner-pro/
├── app.py                 # Main Streamlit application
├── requirements.txt       # Python dependencies
├── README.md              # This file
├── .gitignore             # Git ignore rules
└── sample.csv             # Sample data

---

## 🏃 Run Locally

```bash
git clone https://github.com/Huzefkhan1/datacleaner-pro.git
cd datacleaner-pro
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
App will open at http://localhost:8501

🎓 Use Cases
Data Analysts — clean messy datasets before analysis

Data Scientists — preprocess data for ML pipelines

Business Analysts — validate Excel/CSV exports

Students — learn data quality best practices

🗺️ Roadmap
☑ File upload (CSV/Excel)
☑ 8 types of issue detection
☑ Configurable cleaning pipeline
☑ CSV/Excel export
☑ Cleaning report generation
☑ Interactive visualizations
□ Multi-file batch processing
□ Date parsing and normalization
□ Custom column transformation rules
□ Save & reuse cleaning presets

📄 License
MIT License — feel free to use and modify.

👨‍💻 Author
Huzef Khan

GitHub: @Huzefkhan1

⭐ If you find this useful, give it a star!

