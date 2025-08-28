import streamlit as st
from datetime import datetime, timedelta
import pandas as pd
import io
import json
import re
import requests
import os

from Config_file import logger
from Azure_OpenAI_Analyzer import AzureOpenAIAnalyzer
from Financial_Data_Fetcher import ( 
    FinancialDataFetcher, 
    display_financial_statements, 
    display_risk_analysis, 
    display_liquidity_analysis, 
    display_profitability_analysis, 
    display_cashflow_analysis, 
    generate_print_report,

    )
def main():
    """Main Streamlit app"""
    st.title("CreditIQ - Credit Compliance report with AI")
    # Input Section: Ticker and filing source choice
    col1, col2 = st.columns([1, 2])

    with col1:
        ticker = st.text_input("Enter Stock Ticker Symbol").upper()

    with col2:
        filing_source = st.radio(
            "Select 10-Q Filing Source:",
            ("Auto-fetch latest from SEC", "Upload custom file")
        )

    pdf_file = None
    raw_text = ""

    # Handle filing source
    analyzer = AzureOpenAIAnalyzer()  # construct once
    if filing_source == "Upload custom file":
        uploaded_file = st.file_uploader(
            "Upload 10-Q Filing", 
            type=["pdf", "docx", "txt"],
            help="Upload your 10-Q filing in PDF, Word, or Text format"
        )
        if uploaded_file:
            with st.spinner("Extracting text from uploaded document..."):
                raw_text = analyzer.extract_text_from_file(uploaded_file, filename=getattr(uploaded_file, "name", None), file_type=getattr(uploaded_file, "type", None))
            if raw_text:
                st.success("Document uploaded and processed successfully!")
    else:
        # Auto-fetch from SEC
        if ticker:
            try:
                # Load CIK mapping JSON (adjust path or add your own mapping as needed)
                mapping_path = "company_tickers_exchange.json"
                if os.path.exists(mapping_path):
                    with open(mapping_path, "r") as f:
                        CIK_dict = json.load(f)
                    CIK_df = pd.DataFrame(CIK_dict["data"], columns=CIK_dict["fields"])
                    CIK = CIK_df[CIK_df["ticker"] == ticker].cik.values[0]
                else:
                    # Minimal fallback: try SEC submissions search by ticker via known endpoint (requires CIK);
                    # Here we gracefully warn if mapping file is missing.
                    st.warning("Local CIK mapping not found. Please provide the mapping file or switch to 'Upload custom file'.")
                    CIK = None

                if CIK is not None:
                    # SEC headers and URLs
                    headers = {"User-Agent": "your.email@domain.com"}
                    submissions_url = f"https://data.sec.gov/submissions/CIK{str(CIK).zfill(10)}.json"
                    subs = requests.get(submissions_url, headers=headers).json()
                    recent = pd.DataFrame(subs["filings"]["recent"])
                    acc_num = recent[recent.form == "10-Q"].accessionNumber.values[0].replace("-", "")
                    doc_name = recent[recent.form == "10-Q"].primaryDocument.values[0]
                    html_url = f"https://www.sec.gov/Archives/edgar/data/{CIK}/{acc_num}/{doc_name}"
                    pdf_url = f"https://www.sec.gov/Archives/edgar/data/{CIK}/{acc_num}/{doc_name.replace('.htm', '.pdf')}"

                    pdf_response = requests.get(pdf_url, headers=headers)

                    if pdf_response.status_code == 200:
                        pdf_file = io.BytesIO(pdf_response.content)
                    else:
                        # Fallback to HTML -> PDF conversion (best-effort)
                        html_content = requests.get(html_url, headers=headers).content.decode("utf-8", errors="ignore")
                        html_content = re.sub(r'<img[^>]*>', '', html_content)
                        html_content = re.sub(r'<table[^>]*>.*?</table>', '', html_content, flags=re.DOTALL)
                        try:
                            from xhtml2pdf import pisa
                            pdf_buffer = io.BytesIO()
                            pisa.CreatePDF(io.StringIO(html_content), dest=pdf_buffer)
                            pdf_buffer.seek(0)
                            pdf_file = pdf_buffer
                        except Exception:
                            # If conversion not available, keep raw HTML text
                            raw_text = re.sub("<[^<]+?>", " ", html_content)

                    if pdf_file and not raw_text:
                        st.success(f"Latest 10-Q filing fetched for {ticker}")
            except Exception as e:
                st.error(f"Failed to fetch 10-Q from SEC: {e}")

    # Process if we have either uploaded text or fetched PDF
    if ticker and (raw_text or pdf_file):
        try:
            fetcher = FinancialDataFetcher(ticker)

            # Extract text from PDF if needed (pass explicit file_type to avoid .type error)
            if not raw_text and pdf_file:
                with st.spinner("Extracting text from 10-Q document..."):
                    raw_text = analyzer.extract_text_from_file(pdf_file, filename=f"{ticker}_10q.pdf", file_type="application/pdf")

            # If we still don't have text, stop early
            if not raw_text:
                st.error("Could not extract text from the 10-Q document.")
                return

            # Build vector index ONCE here so downstream analyses are not empty
            analyzer.vector_index = analyzer.create_vector_index(raw_text)

            # Fetch financial statements
            financial_data = fetcher.get_financial_statements()

            # Perform AI-driven analyses using vector embeddings
            with st.spinner("Analyzing risks using AI"):
                risk_analysis = analyzer.analyze_risks(raw_text)

            with st.spinner("Analyzing liquidity using AI"):
                liquidity_analysis = analyzer.analyze_liquidity(raw_text)

            with st.spinner("Analyzing profitability using AI"):
                profitability_analysis = analyzer.analyze_profitability(raw_text)

            with st.spinner("Analyzing cash flow using AI"):
                cashflow_analysis = analyzer.analyze_cashflow(raw_text)

            # Prepare HTML snippets for print
            # Financial tables
            bal_html = fetcher.format_financial_table(financial_data).to_html(index=False, escape=False, na_rep="")
            inc_html = fetcher.format_income_statement(financial_data).to_html(index=False, escape=False, na_rep="")
            cf_html = fetcher.format_cash_flow(financial_data).to_html(index=False, escape=False, na_rep="")
            tables_html = f"<h3>Balance Sheet</h3>{bal_html}<h3>Income Statement</h3>{inc_html}<h3>Cash Flow</h3>{cf_html}"

            # Main tabs for interactive display - Added Cash Flow tab
            tab1, tab2, tab3, tab4, tab5 = st.tabs(["Financial Statements", "Risk Analysis", "Liquidity", "Profitability", "Cash Flow"])

            with tab1:
                display_financial_statements(financial_data, ticker)

            with tab2:
                display_risk_analysis(risk_analysis)

            with tab3:
                display_liquidity_analysis(liquidity_analysis)

            with tab4:
                display_profitability_analysis(profitability_analysis)

            with tab5:
                display_cashflow_analysis(cashflow_analysis)

            # Generate full report for download
            st.markdown("---")
            st.subheader("📥 Export Report")

            # Download full HTML report
            all_html = {
                'tables': tables_html,
                'risk_analysis': risk_analysis.replace('\n', '<br>'),
                'liquidity_analysis': liquidity_analysis.replace('\n', '<br>'),
                'profitability_analysis': profitability_analysis.replace('\n', '<br>'),
                'cashflow_analysis': cashflow_analysis.replace('\n', '<br>')
            }
            report_html = generate_print_report(ticker, all_html)

            st.download_button(
                label="📊 Download Full Report (HTML)",
                data=report_html,
                file_name=f"{ticker}_financial_report_{datetime.now().strftime('%Y%m%d')}.html",
                mime="text/html"
            )

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            st.info("Please check your Azure OpenAI credentials and ensure the ticker symbol is valid.")
            logger.exception("Main app error:")
    else:
        if ticker:
            st.info("Please wait for the 10-Q filing to be fetched or upload a custom file.")
        else:
            st.info("Please enter a ticker symbol to proceed.")

if __name__ == "__main__":
    main()