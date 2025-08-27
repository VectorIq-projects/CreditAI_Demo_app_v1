import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import PyPDF2
import io
import base64
from typing import Dict, List, Optional, Tuple
import json
import re

# Azure OpenAI imports
from openai import AzureOpenAI
import tiktoken

# Configuration - Replace with your actual Azure OpenAI credentials
AZURE_OPENAI_KEY = "2f6e41aa534f49908feb01c6de771d6b"
AZURE_OPENAI_ENDPOINT = "https://ea-oai-sandbox.openai.azure.com/"
AZURE_OPENAI_DEPLOYMENT = "dev-gpt-4o"
AZURE_OPENAI_API_VERSION = "2024-02-01"

# Page configuration
st.set_page_config(
    page_title="Financial Statement Analyzer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main {
        padding: 0rem 1rem;
    }
    .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
        font-size: 1.1rem;
        font-weight: 500;
    }
    table {
        font-size: 0.9rem;
        width: 100%;
    }
    th {
        background-color: #f0f2f6;
        font-weight: bold;
        text-align: center;
    }
    td {
        text-align: right;
        padding: 8px;
    }
    .metric-header {
        background-color: #e3f2fd;
        padding: 10px;
        border-radius: 5px;
        margin: 10px 0;
    }
    .ai-summary {
        background-color: #f5f5f5;
        padding: 15px;
        border-radius: 8px;
        margin: 10px 0;
        border-left: 4px solid #1976d2;
    }
    .risk-card {
        background-color: #fff3e0;
        padding: 12px;
        border-radius: 6px;
        margin: 8px 0;
        border-left: 4px solid #ff9800;
    }
    @media print {
        .stButton {display: none;}
        .stFileUploader {display: none;}
        .stTextInput {display: none;}
    }
    .percentage-positive {
        color: #4caf50;
    }
    .percentage-negative {
        color: #f44336;
    }
</style>
""", unsafe_allow_html=True)

# (Classes AzureOpenAIAnalyzer and FinancialDataFetcher,
#  and functions display_financial_statements, display_risk_analysis,
#  display_liquidity_analysis, display_profitability_analysis)
# ... (existing code above remains unchanged)
class AzureOpenAIAnalyzer:
    """Handles Azure OpenAI document analysis"""
    
    def __init__(self):
        """Initialize Azure OpenAI client with hardcoded credentials"""
        try:
            self.client = AzureOpenAI(
                api_key=AZURE_OPENAI_KEY,
                api_version=AZURE_OPENAI_API_VERSION,
                azure_endpoint=AZURE_OPENAI_ENDPOINT
            )
            self.deployment_name = AZURE_OPENAI_DEPLOYMENT
            self.encoding = tiktoken.encoding_for_model("gpt-4")
        except Exception as e:
            st.error(f"Failed to initialize Azure OpenAI: {str(e)}")
            st.info("Please check your Azure OpenAI credentials in the code.")
            self.client = None
    
    def extract_text_from_pdf(self, pdf_file) -> str:
        """Extract text from uploaded PDF file"""
        try:
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            text = ""
            total_pages = len(pdf_reader.pages)
            
            # Show progress for large PDFs
            if total_pages > 10:
                progress_bar = st.progress(0, text=f"Reading PDF: 0/{total_pages} pages")
            
            for i, page in enumerate(pdf_reader.pages):
                text += page.extract_text() + "\n"
                
                if total_pages > 10:
                    progress_bar.progress((i + 1) / total_pages, 
                                        text=f"Reading PDF: {i+1}/{total_pages} pages")
            
            if total_pages > 10:
                progress_bar.empty()
            
            return text
        except Exception as e:
            st.error(f"Error reading PDF: {str(e)}")
            return ""
    
    def chunk_text(self, text: str, max_tokens: int = 6000) -> List[str]:
        """Split text into chunks for API processing"""
        sentences = text.split('. ')
        chunks = []
        current_chunk = []
        current_tokens = 0
        
        for sentence in sentences:
            sentence_tokens = len(self.encoding.encode(sentence + '. '))
            if current_tokens + sentence_tokens > max_tokens:
                chunks.append('. '.join(current_chunk) + '.')
                current_chunk = [sentence]
                current_tokens = sentence_tokens
            else:
                current_chunk.append(sentence)
                current_tokens += sentence_tokens
        
        if current_chunk:
            chunks.append('. '.join(current_chunk))
        
        return chunks
    
    def analyze_risks(self, text: str) -> Dict[str, List[str]]:
        """Analyze risks from 10-Q document"""
        if not self.client:
            return self._get_default_risk_response()
        
        prompt = """Analyze the following 10-Q filing excerpt and identify specific risks in these categories:
        1. Input Cost Risks - Material costs, supply chain, inflation impacts
        2. Operational Risks - Production, manufacturing, logistics challenges and their mitigation strategies
        3. Financial and Debt-Related Risks - Liquidity, credit, debt obligations
        4. Regulatory and Compliance Risks - Legal, regulatory requirements, compliance costs
        5. Strategic Implications - Market competition, strategic challenges, business model risks
        
        For each category, provide 2-3 specific risks mentioned in the text with brief descriptions.
        If a category has no relevant risks, use an empty array.
        
        Format the response as a JSON object with keys: input_cost_risks, operational_risks, financial_risks, regulatory_risks, strategic_risks
        Each key should contain an array of risk descriptions (max 100 words each).
        
        Text to analyze:
        {text}"""
        
        try:
            chunks = self.chunk_text(text)
            all_risks = {
                "input_cost_risks": [],
                "operational_risks": [],
                "financial_risks": [],
                "regulatory_risks": [],
                "strategic_risks": []
            }
            
            # Process first few chunks
            for i, chunk in enumerate(chunks[:3]):
                response = self.client.chat.completions.create(
                    model=self.deployment_name,
                    messages=[
                        {"role": "system", "content": "You are a financial analyst expert in risk analysis."},
                        {"role": "user", "content": prompt.format(text=chunk)}
                    ],
                    temperature=0.3,
                    max_tokens=1500,
                    response_format={"type": "json_object"}
                )
                
                chunk_risks = json.loads(response.choices[0].message.content)
                
                # Merge results
                for key in all_risks:
                    if key in chunk_risks:
                        all_risks[key].extend(chunk_risks[key])
            
            # Deduplicate and limit
            for key in all_risks:
                # Remove duplicates and limit to 3 items
                unique_risks = []
                seen = set()
                for risk in all_risks[key]:
                    risk_lower = risk.lower()
                    if risk_lower not in seen and len(risk_lower) > 20:  # Filter out very short responses
                        seen.add(risk_lower)
                        unique_risks.append(risk)
                all_risks[key] = unique_risks[:3]
            
            return all_risks
            
        except Exception as e:
            st.error(f"Error in risk analysis: {str(e)}")
            return self._get_default_risk_response()
    
    def analyze_liquidity(self, text: str) -> Dict[str, any]:
        """Analyze liquidity from 10-Q document"""
        if not self.client:
            return self._get_default_liquidity_response()
        
        prompt = """Analyze the liquidity position from this 10-Q filing excerpt. Focus on:
        1. Current cash position and cash equivalents
        2. Working capital status and trends
        3. Cash flow from operations
        4. Available credit facilities and unused borrowing capacity
        5. Debt maturities and obligations
        6. Management's liquidity assessment
        
        Provide specific numbers and percentages where mentioned.
        
        Format the response as a JSON object with keys:
        - summary: Overall liquidity assessment (max 150 words)
        - cash_position: Current cash and equivalents description
        - working_capital: Working capital analysis
        - cash_flow: Operating cash flow trends
        - credit_facilities: Available credit and borrowing capacity
        - key_metrics: Array of important liquidity metrics mentioned
        
        Text to analyze:
        {text}"""
        
        try:
            # Use relevant chunks
            chunks = self.chunk_text(text)
            relevant_text = ' '.join(chunks[:2])[:8000]
            
            response = self.client.chat.completions.create(
                model=self.deployment_name,
                messages=[
                    {"role": "system", "content": "You are a financial analyst expert in liquidity analysis."},
                    {"role": "user", "content": prompt.format(text=relevant_text)}
                ],
                temperature=0.3,
                max_tokens=1000,
                response_format={"type": "json_object"}
            )
            
            return json.loads(response.choices[0].message.content)
            
        except Exception as e:
            st.error(f"Error in liquidity analysis: {str(e)}")
            return self._get_default_liquidity_response()
    
    def analyze_profitability(self, text: str) -> Dict[str, any]:
        """Analyze profitability from 10-Q document"""
        if not self.client:
            return self._get_default_profitability_response()
        
        prompt = """Analyze the profitability from this 10-Q filing excerpt. Focus on:
        1. Revenue trends and growth rates
        2. Gross margin performance and factors affecting it
        3. Operating margin trends
        4. Net income and earnings per share
        5. Key profitability drivers and headwinds
        6. Segment performance if available
        
        Provide specific numbers and percentages where mentioned.
        
        Format the response as a JSON object with keys:
        - summary: Overall profitability assessment (max 150 words)
        - revenue_trends: Revenue performance and growth
        - margins: Gross and operating margin analysis
        - net_income: Net income and EPS performance
        - key_drivers: Array of main factors affecting profitability
        - outlook: Management's profitability outlook if mentioned
        
        Text to analyze:
        {text}"""
        
        try:
            chunks = self.chunk_text(text)
            relevant_text = ' '.join(chunks[:2])[:8000]
            
            response = self.client.chat.completions.create(
                model=self.deployment_name,
                messages=[
                    {"role": "system", "content": "You are a financial analyst expert in profitability analysis."},
                    {"role": "user", "content": prompt.format(text=relevant_text)}
                ],
                temperature=0.3,
                max_tokens=1000,
                response_format={"type": "json_object"}
            )
            
            return json.loads(response.choices[0].message.content)
            
        except Exception as e:
            st.error(f"Error in profitability analysis: {str(e)}")
            return self._get_default_profitability_response()
    
    def _get_default_risk_response(self) -> Dict:
        """Default response when AI analysis fails"""
        return {
            "input_cost_risks": ["AI analysis unavailable - please check Azure OpenAI configuration"],
            "operational_risks": ["AI analysis unavailable"],
            "financial_risks": ["AI analysis unavailable"],
            "regulatory_risks": ["AI analysis unavailable"],
            "strategic_risks": ["AI analysis unavailable"]
        }
    
    def _get_default_liquidity_response(self) -> Dict:
        """Default response when AI analysis fails"""
        return {
            "summary": "AI analysis unavailable - please check Azure OpenAI configuration",
            "cash_position": "N/A",
            "working_capital": "N/A",
            "cash_flow": "N/A",
            "credit_facilities": "N/A",
            "key_metrics": []
        }
    
    def _get_default_profitability_response(self) -> Dict:
        """Default response when AI analysis fails"""
        return {
            "summary": "AI analysis unavailable - please check Azure OpenAI configuration",
            "revenue_trends": "N/A",
            "margins": "N/A",
            "net_income": "N/A",
            "key_drivers": [],
            "outlook": "N/A"
        }

class FinancialDataFetcher:
    """Fetches financial data using yfinance"""
    
    def __init__(self, ticker: str):
        self.ticker = ticker.upper()
        self.stock = yf.Ticker(self.ticker)
        
    def get_financial_statements(self) -> Dict[str, pd.DataFrame]:
        """Fetch quarterly and annual financial statements"""
        try:
            with st.spinner(f"Fetching financial data for {self.ticker}..."):
                # Fetch data with error handling for each statement
                data = {}
                
                try:
                    data["quarterly_balance_sheet"] = self.stock.quarterly_balance_sheet
                except:
                    data["quarterly_balance_sheet"] = None
                
                try:
                    data["annual_balance_sheet"] = self.stock.balance_sheet
                except:
                    data["annual_balance_sheet"] = None
                
                try:
                    data["quarterly_income"] = self.stock.quarterly_income_stmt
                except:
                    data["quarterly_income"] = None
                
                try:
                    data["annual_income"] = self.stock.income_stmt
                except:
                    data["annual_income"] = None
                
                try:
                    data["quarterly_cashflow"] = self.stock.quarterly_cash_flow
                except:
                    data["quarterly_cashflow"] = None
                
                try:
                    data["annual_cashflow"] = self.stock.cash_flow
                except:
                    data["annual_cashflow"] = None
                
                # Check if we got any data
                if all(v is None for v in data.values()):
                    st.error(f"No financial data available for {self.ticker}")
                    return {}
                
                return data
                
        except Exception as e:
            st.error(f"Error fetching financial data: {str(e)}")
            return {}
    
    def format_financial_table(self, data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Format financial data into display table matching Excel structure"""
        if not data:
            return pd.DataFrame()
        
        # Create formatted table structure
        formatted_data = []
        
        # Balance Sheet Items
        balance_sheet_items = [
            ("BALANCE SHEET", None, "section_header"),
            ("Assets:", None, "subsection"),
            ("Cash and Equivalents", "Cash And Cash Equivalents", "item"),
            ("Short-Term Investments", "Other Short Term Investments", "item"),
            ("Accounts Receivable", "Accounts Receivable", "item"),
            ("Inventories", "Inventory", "item"),
            ("Current Assets", "Current Assets", "item"),
            ("Total Assets", "Total Assets", "item"),
            ("Working Capital", None, "calculated"),
            ("", None, "blank"),
            ("Liabilities:", None, "subsection"),
            ("Short-Term Debt", "Short Term Debt", "item"),
            ("Accounts Payable", "Accounts Payable", "item"),
            ("Current Liabilities", "Current Liabilities", "item"),
            ("Long-Term Debt", "Long Term Debt", "item"),
            ("Total Liabilities", "Total Liabilities Net Minority Interest", "item"),
            ("Net Worth (OE)", None, "calculated"),
            ("", None, "blank"),
            ("Ratios:", None, "subsection"),
            ("Current Ratio", None, "ratio"),
            ("Quick Ratio", None, "ratio"),
            ("Debt to Equity", None, "ratio"),
        ]
        
        # Process balance sheet
        for item_name, field_name, item_type in balance_sheet_items:
            row_data = {"Item": item_name}
            
            if item_type == "section_header":
                formatted_data.append(row_data)
                continue
            elif item_type == "subsection":
                formatted_data.append(row_data)
                continue
            elif item_type == "blank":
                formatted_data.append(row_data)
                continue
            
            # Add quarterly data (last 3 quarters)
            if "quarterly_balance_sheet" in data and data["quarterly_balance_sheet"] is not None:
                qbs = data["quarterly_balance_sheet"]
                if not qbs.empty:
                    # Try to find the field in the index
                    if field_name:
                        matching_fields = [idx for idx in qbs.index if field_name.lower() in str(idx).lower()]
                        if matching_fields:
                            field_to_use = matching_fields[0]
                            for i in range(min(3, len(qbs.columns))):
                                col_name = f"Q{i+1} {qbs.columns[i].strftime('%m/%d/%Y')}"
                                value = qbs.loc[field_to_use].iloc[i]
                                row_data[col_name] = self._format_value(value)
                                
                                # Add percentage change
                                if i > 0:
                                    prev_value = qbs.loc[field_to_use].iloc[i-1]
                                    pct_change = self._calculate_percentage_change(prev_value, value)
                                    row_data[f"Q{i+1} Δ%"] = pct_change
            
            # Add annual data (last 2 years)
            if "annual_balance_sheet" in data and data["annual_balance_sheet"] is not None:
                abs_data = data["annual_balance_sheet"]
                if not abs_data.empty:
                    if field_name:
                        matching_fields = [idx for idx in abs_data.index if field_name.lower() in str(idx).lower()]
                        if matching_fields:
                            field_to_use = matching_fields[0]
                            for i in range(min(2, len(abs_data.columns))):
                                col_name = f"FY {abs_data.columns[i].year}"
                                value = abs_data.loc[field_to_use].iloc[i]
                                row_data[col_name] = self._format_value(value)
                                
                                # Add percentage change
                                if i > 0:
                                    prev_value = abs_data.loc[field_to_use].iloc[i-1]
                                    pct_change = self._calculate_percentage_change(prev_value, value)
                                    row_data[f"FY{abs_data.columns[i].year} Δ%"] = pct_change
            
            # Calculate special items
            if item_name == "Working Capital":
                row_data = self._calculate_working_capital(data, row_data)
            elif item_name == "Net Worth (OE)":
                row_data = self._calculate_net_worth(data, row_data)
            elif item_name == "Current Ratio":
                row_data = self._calculate_current_ratio(data, row_data)
            elif item_name == "Quick Ratio":
                row_data = self._calculate_quick_ratio(data, row_data)
            elif item_name == "Debt to Equity":
                row_data = self._calculate_debt_to_equity(data, row_data)
            
            formatted_data.append(row_data)
        
        return pd.DataFrame(formatted_data)
    
    def format_income_statement(self, data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Format income statement data"""
        if not data:
            return pd.DataFrame()
        
        formatted_data = []
        
        income_items = [
            ("INCOME STATEMENT", None, "section_header"),
            ("Total Revenue", "Total Revenue", "item"),
            ("Cost of Revenue", "Cost Of Revenue", "item"),
            ("Gross Profit", "Gross Profit", "item"),
            ("Gross Margin %", None, "calculated"),
            ("Operating Expenses", "Operating Expense", "item"),
            ("Operating Income", "Operating Income", "item"),
            ("Operating Margin %", None, "calculated"),
            ("EBIT", "EBIT", "item"),
            ("Interest Expense", "Interest Expense", "item"),
            ("Tax", "Tax Provision", "item"),
            ("Net Income", "Net Income", "item"),
            ("Net Margin %", None, "calculated"),
            ("EPS Basic", "Basic EPS", "item"),
            ("EPS Diluted", "Diluted EPS", "item"),
        ]
        
        for item_name, field_name, item_type in income_items:
            row_data = {"Item": item_name}
            
            if item_type == "section_header":
                formatted_data.append(row_data)
                continue
            
            # Add quarterly data with percentage changes
            if "quarterly_income" in data and data["quarterly_income"] is not None:
                qi = data["quarterly_income"]
                if not qi.empty and field_name:
                    matching_fields = [idx for idx in qi.index if field_name.lower() in str(idx).lower()]
                    if matching_fields:
                        field_to_use = matching_fields[0]
                        for i in range(min(3, len(qi.columns))):
                            col_name = f"Q{i+1} {qi.columns[i].strftime('%m/%d/%Y')}"
                            value = qi.loc[field_to_use].iloc[i]
                            row_data[col_name] = self._format_value(value)
                            
                            # Add percentage change
                            if i > 0:
                                prev_value = qi.loc[field_to_use].iloc[i-1]
                                pct_change = self._calculate_percentage_change(prev_value, value)
                                row_data[f"Q{i+1} Δ%"] = pct_change
            
            # Add annual data with percentage changes
            if "annual_income" in data and data["annual_income"] is not None:
                ai = data["annual_income"]
                if not ai.empty and field_name:
                    matching_fields = [idx for idx in ai.index if field_name.lower() in str(idx).lower()]
                    if matching_fields:
                        field_to_use = matching_fields[0]
                        for i in range(min(2, len(ai.columns))):
                            col_name = f"FY {ai.columns[i].year}"
                            value = ai.loc[field_to_use].iloc[i]
                            row_data[col_name] = self._format_value(value)
                            
                            # Add percentage change
                            if i > 0:
                                prev_value = ai.loc[field_to_use].iloc[i-1]
                                pct_change = self._calculate_percentage_change(prev_value, value)
                                row_data[f"FY{ai.columns[i].year} Δ%"] = pct_change
            
            # Calculate margins
            if item_name == "Gross Margin %":
                row_data = self._calculate_gross_margin(data, row_data)
            elif item_name == "Operating Margin %":
                row_data = self._calculate_operating_margin(data, row_data)
            elif item_name == "Net Margin %":
                row_data = self._calculate_net_margin(data, row_data)
            
            formatted_data.append(row_data)
        
        return pd.DataFrame(formatted_data)
    
    def format_cash_flow(self, data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Format cash flow statement data"""
        if not data:
            return pd.DataFrame()
        
        formatted_data = []
        
        cash_flow_items = [
            ("CASH FLOW STATEMENT", None, "section_header"),
            ("Operating Activities:", None, "subsection"),
            ("Net Income", "Net Income", "item"),
            ("Depreciation", "Depreciation And Amortization", "item"),
            ("Working Capital Changes", "Change In Working Capital", "item"),
            ("Operating Cash Flow", "Operating Cash Flow", "item"),
            ("", None, "blank"),
            ("Investing Activities:", None, "subsection"),
            ("Capital Expenditures", "Capital Expenditure", "item"),
            ("Investments", "Net Investment Purchase And Sale", "item"),
            ("Investing Cash Flow", "Investing Cash Flow", "item"),
            ("", None, "blank"),
            ("Financing Activities:", None, "subsection"),
            ("Debt Repayment", "Net Issuance Payments Of Debt", "item"),
            ("Stock Repurchase", "Net Common Stock Issuance", "item"),
            ("Dividends", "Cash Dividends Paid", "item"),
            ("Financing Cash Flow", "Financing Cash Flow", "item"),
            ("", None, "blank"),
            ("Net Cash Flow", None, "calculated"),
            ("Free Cash Flow", "Free Cash Flow", "item"),
        ]
        
        for item_name, field_name, item_type in cash_flow_items:
            row_data = {"Item": item_name}
            
            if item_type in ["section_header", "subsection", "blank"]:
                formatted_data.append(row_data)
                continue
            
            # Add quarterly data
            if "quarterly_cashflow" in data and data["quarterly_cashflow"] is not None:
                qcf = data["quarterly_cashflow"]
                if not qcf.empty and field_name:
                    matching_fields = [idx for idx in qcf.index if field_name.lower() in str(idx).lower()]
                    if matching_fields:
                        field_to_use = matching_fields[0]
                        for i in range(min(3, len(qcf.columns))):
                            col_name = f"Q{i+1} {qcf.columns[i].strftime('%m/%d/%Y')}"
                            value = qcf.loc[field_to_use].iloc[i]
                            row_data[col_name] = self._format_value(value)
            
            # Add annual data
            if "annual_cashflow" in data and data["annual_cashflow"] is not None:
                acf = data["annual_cashflow"]
                if not acf.empty and field_name:
                    matching_fields = [idx for idx in acf.index if field_name.lower() in str(idx).lower()]
                    if matching_fields:
                        field_to_use = matching_fields[0]
                        for i in range(min(2, len(acf.columns))):
                            col_name = f"FY {acf.columns[i].year}"
                            value = acf.loc[field_to_use].iloc[i]
                            row_data[col_name] = self._format_value(value)
            
            # Calculate net cash flow
            if item_name == "Net Cash Flow":
                row_data = self._calculate_net_cash_flow(data, row_data)
            
            formatted_data.append(row_data)
        
        return pd.DataFrame(formatted_data)
    
    def _format_value(self, value) -> str:
        """Format financial values"""
        if pd.isna(value) or value is None:
            return "-"
        
        # Convert to thousands
        value = value / 1000
        
        if value < 0:
            return f"$({abs(value):,.0f})"
        else:
            return f"${value:,.0f}"
    
    def _calculate_percentage_change(self, old_value, new_value) -> str:
        """Calculate percentage change between two values"""
        if pd.isna(old_value) or pd.isna(new_value) or old_value == 0:
            return "-"
        
        pct_change = ((new_value - old_value) / abs(old_value)) * 100
        
        if pct_change > 0:
            return f'<span class="percentage-positive">+{pct_change:.1f}%</span>'
        else:
            return f'<span class="percentage-negative">{pct_change:.1f}%</span>'
    
    def _find_field_in_index(self, df: pd.DataFrame, field_name: str) -> Optional[str]:
        """Find matching field name in dataframe index"""
        if df is None or df.empty or not field_name:
            return None
        
        # Try exact match first
        if field_name in df.index:
            return field_name
        
        # Try case-insensitive match
        field_lower = field_name.lower()
        for idx in df.index:
            if str(idx).lower() == field_lower:
                return idx
        
        # Try partial match
        for idx in df.index:
            if field_lower in str(idx).lower():
                return idx
        
        return None
    
    def _calculate_working_capital(self, data: Dict, row_data: Dict) -> Dict:
        """Calculate working capital"""
        try:
            # Quarterly
            if "quarterly_balance_sheet" in data and data["quarterly_balance_sheet"] is not None:
                qbs = data["quarterly_balance_sheet"]
                if not qbs.empty:
                    ca_field = self._find_field_in_index(qbs, "Current Assets")
                    cl_field = self._find_field_in_index(qbs, "Current Liabilities")
                    
                    if ca_field and cl_field:
                        for i in range(min(3, len(qbs.columns))):
                            ca = qbs.loc[ca_field].iloc[i]
                            cl = qbs.loc[cl_field].iloc[i]
                            col_name = f"Q{i+1} {qbs.columns[i].strftime('%m/%d/%Y')}"
                            row_data[col_name] = self._format_value(ca - cl)
                            
                            # Add percentage change
                            if i > 0:
                                prev_ca = qbs.loc[ca_field].iloc[i-1]
                                prev_cl = qbs.loc[cl_field].iloc[i-1]
                                prev_wc = prev_ca - prev_cl
                                curr_wc = ca - cl
                                pct_change = self._calculate_percentage_change(prev_wc, curr_wc)
                                row_data[f"Q{i+1} Δ%"] = pct_change
            
            # Annual
            if "annual_balance_sheet" in data and data["annual_balance_sheet"] is not None:
                abs_data = data["annual_balance_sheet"]
                if not abs_data.empty:
                    ca_field = self._find_field_in_index(abs_data, "Current Assets")
                    cl_field = self._find_field_in_index(abs_data, "Current Liabilities")
                    
                    if ca_field and cl_field:
                        for i in range(min(2, len(abs_data.columns))):
                            ca = abs_data.loc[ca_field].iloc[i]
                            cl = abs_data.loc[cl_field].iloc[i]
                            col_name = f"FY {abs_data.columns[i].year}"
                            row_data[col_name] = self._format_value(ca - cl)
                            
                            # Add percentage change
                            if i > 0:
                                prev_ca = abs_data.loc[ca_field].iloc[i-1]
                                prev_cl = abs_data.loc[cl_field].iloc[i-1]
                                prev_wc = prev_ca - prev_cl
                                curr_wc = ca - cl
                                pct_change = self._calculate_percentage_change(prev_wc, curr_wc)
                                row_data[f"FY{abs_data.columns[i].year} Δ%"] = pct_change
        except Exception as e:
            st.warning(f"Could not calculate working capital: {str(e)}")
        
        return row_data
    
    def _calculate_net_worth(self, data: Dict, row_data: Dict) -> Dict:
        """Calculate net worth (Total Assets - Total Liabilities)"""
        try:
            # Quarterly
            if "quarterly_balance_sheet" in data and data["quarterly_balance_sheet"] is not None:
                qbs = data["quarterly_balance_sheet"]
                if not qbs.empty:
                    ta_field = self._find_field_in_index(qbs, "Total Assets")
                    tl_field = self._find_field_in_index(qbs, "Total Liabilities")
                    
                    if ta_field and tl_field:
                        for i in range(min(3, len(qbs.columns))):
                            ta = qbs.loc[ta_field].iloc[i]
                            tl = qbs.loc[tl_field].iloc[i]
                            col_name = f"Q{i+1} {qbs.columns[i].strftime('%m/%d/%Y')}"
                            row_data[col_name] = self._format_value(ta - tl)
                            
                            # Add percentage change
                            if i > 0:
                                prev_ta = qbs.loc[ta_field].iloc[i-1]
                                prev_tl = qbs.loc[tl_field].iloc[i-1]
                                prev_nw = prev_ta - prev_tl
                                curr_nw = ta - tl
                                pct_change = self._calculate_percentage_change(prev_nw, curr_nw)
                                row_data[f"Q{i+1} Δ%"] = pct_change
            
            # Annual
            if "annual_balance_sheet" in data and data["annual_balance_sheet"] is not None:
                abs_data = data["annual_balance_sheet"]
                if not abs_data.empty:
                    ta_field = self._find_field_in_index(abs_data, "Total Assets")
                    tl_field = self._find_field_in_index(abs_data, "Total Liabilities")
                    
                    if ta_field and tl_field:
                        for i in range(min(2, len(abs_data.columns))):
                            ta = abs_data.loc[ta_field].iloc[i]
                            tl = abs_data.loc[tl_field].iloc[i]
                            col_name = f"FY {abs_data.columns[i].year}"
                            row_data[col_name] = self._format_value(ta - tl)
                            
                            # Add percentage change
                            if i > 0:
                                prev_ta = abs_data.loc[ta_field].iloc[i-1]
                                prev_tl = abs_data.loc[tl_field].iloc[i-1]
                                prev_nw = prev_ta - prev_tl
                                curr_nw = ta - tl
                                pct_change = self._calculate_percentage_change(prev_nw, curr_nw)
                                row_data[f"FY{abs_data.columns[i].year} Δ%"] = pct_change
        except Exception as e:
            st.warning(f"Could not calculate net worth: {str(e)}")
        
        return row_data
    
    def _calculate_current_ratio(self, data: Dict, row_data: Dict) -> Dict:
        """Calculate current ratio"""
        try:
            # Quarterly
            if "quarterly_balance_sheet" in data and data["quarterly_balance_sheet"] is not None:
                qbs = data["quarterly_balance_sheet"]
                if not qbs.empty:
                    ca_field = self._find_field_in_index(qbs, "Current Assets")
                    cl_field = self._find_field_in_index(qbs, "Current Liabilities")
                    
                    if ca_field and cl_field:
                        for i in range(min(3, len(qbs.columns))):
                            ca = qbs.loc[ca_field].iloc[i]
                            cl = qbs.loc[cl_field].iloc[i]
                            if cl != 0 and not pd.isna(cl):
                                col_name = f"Q{i+1} {qbs.columns[i].strftime('%m/%d/%Y')}"
                                row_data[col_name] = f"{ca/cl:.2f}"
            
            # Annual
            if "annual_balance_sheet" in data and data["annual_balance_sheet"] is not None:
                abs_data = data["annual_balance_sheet"]
                if not abs_data.empty:
                    ca_field = self._find_field_in_index(abs_data, "Current Assets")
                    cl_field = self._find_field_in_index(abs_data, "Current Liabilities")
                    
                    if ca_field and cl_field:
                        for i in range(min(2, len(abs_data.columns))):
                            ca = abs_data.loc[ca_field].iloc[i]
                            cl = abs_data.loc[cl_field].iloc[i]
                            if cl != 0 and not pd.isna(cl):
                                col_name = f"FY {abs_data.columns[i].year}"
                                row_data[col_name] = f"{ca/cl:.2f}"
        except Exception as e:
            st.warning(f"Could not calculate current ratio: {str(e)}")
        
        return row_data
    
    def _calculate_quick_ratio(self, data: Dict, row_data: Dict) -> Dict:
        """Calculate quick ratio"""
        try:
            # Quarterly
            if "quarterly_balance_sheet" in data and data["quarterly_balance_sheet"] is not None:
                qbs = data["quarterly_balance_sheet"]
                if not qbs.empty:
                    ca_field = self._find_field_in_index(qbs, "Current Assets")
                    inv_field = self._find_field_in_index(qbs, "Inventory")
                    cl_field = self._find_field_in_index(qbs, "Current Liabilities")
                    
                    if ca_field and cl_field:
                        for i in range(min(3, len(qbs.columns))):
                            ca = qbs.loc[ca_field].iloc[i]
                            inv = qbs.loc[inv_field].iloc[i] if inv_field else 0
                            cl = qbs.loc[cl_field].iloc[i]
                            if cl != 0 and not pd.isna(cl):
                                col_name = f"Q{i+1} {qbs.columns[i].strftime('%m/%d/%Y')}"
                                row_data[col_name] = f"{(ca-inv)/cl:.2f}"
            
            # Annual
            if "annual_balance_sheet" in data and data["annual_balance_sheet"] is not None:
                abs_data = data["annual_balance_sheet"]
                if not abs_data.empty:
                    ca_field = self._find_field_in_index(abs_data, "Current Assets")
                    inv_field = self._find_field_in_index(abs_data, "Inventory")
                    cl_field = self._find_field_in_index(abs_data, "Current Liabilities")
                    
                    if ca_field and cl_field:
                        for i in range(min(2, len(abs_data.columns))):
                            ca = abs_data.loc[ca_field].iloc[i]
                            inv = abs_data.loc[inv_field].iloc[i] if inv_field else 0
                            cl = abs_data.loc[cl_field].iloc[i]
                            if cl != 0 and not pd.isna(cl):
                                col_name = f"FY {abs_data.columns[i].year}"
                                row_data[col_name] = f"{(ca-inv)/cl:.2f}"
        except Exception as e:
            st.warning(f"Could not calculate quick ratio: {str(e)}")
        
        return row_data
    
    def _calculate_debt_to_equity(self, data: Dict, row_data: Dict) -> Dict:
        """Calculate debt to equity ratio"""
        try:
            # Quarterly
            if "quarterly_balance_sheet" in data and data["quarterly_balance_sheet"] is not None:
                qbs = data["quarterly_balance_sheet"]
                if not qbs.empty:
                    tl_field = self._find_field_in_index(qbs, "Total Liabilities")
                    ta_field = self._find_field_in_index(qbs, "Total Assets")
                    
                    if tl_field and ta_field:
                        for i in range(min(3, len(qbs.columns))):
                            tl = qbs.loc[tl_field].iloc[i]
                            ta = qbs.loc[ta_field].iloc[i]
                            equity = ta - tl
                            if equity != 0 and not pd.isna(equity):
                                col_name = f"Q{i+1} {qbs.columns[i].strftime('%m/%d/%Y')}"
                                row_data[col_name] = f"{tl/equity:.2f}"
            
            # Annual
            if "annual_balance_sheet" in data and data["annual_balance_sheet"] is not None:
                abs_data = data["annual_balance_sheet"]
                if not abs_data.empty:
                    tl_field = self._find_field_in_index(abs_data, "Total Liabilities")
                    ta_field = self._find_field_in_index(abs_data, "Total Assets")
                    
                    if tl_field and ta_field:
                        for i in range(min(2, len(abs_data.columns))):
                            tl = abs_data.loc[tl_field].iloc[i]
                            ta = abs_data.loc[ta_field].iloc[i]
                            equity = ta - tl
                            if equity != 0 and not pd.isna(equity):
                                col_name = f"FY {abs_data.columns[i].year}"
                                row_data[col_name] = f"{tl/equity:.2f}"
        except Exception as e:
            st.warning(f"Could not calculate debt to equity: {str(e)}")
        
        return row_data
    
    def _calculate_gross_margin(self, data: Dict, row_data: Dict) -> Dict:
        """Calculate gross margin percentage"""
        try:
            # Quarterly
            if "quarterly_income" in data and data["quarterly_income"] is not None:
                qi = data["quarterly_income"]
                if not qi.empty:
                    rev_field = self._find_field_in_index(qi, "Total Revenue")
                    gp_field = self._find_field_in_index(qi, "Gross Profit")
                    
                    if rev_field and gp_field:
                        for i in range(min(3, len(qi.columns))):
                            revenue = qi.loc[rev_field].iloc[i]
                            gross_profit = qi.loc[gp_field].iloc[i]
                            if revenue != 0 and not pd.isna(revenue):
                                col_name = f"Q{i+1} {qi.columns[i].strftime('%m/%d/%Y')}"
                                row_data[col_name] = f"{(gross_profit/revenue)*100:.1f}%"
            
            # Annual
            if "annual_income" in data and data["annual_income"] is not None:
                ai = data["annual_income"]
                if not ai.empty:
                    rev_field = self._find_field_in_index(ai, "Total Revenue")
                    gp_field = self._find_field_in_index(ai, "Gross Profit")
                    
                    if rev_field and gp_field:
                        for i in range(min(2, len(ai.columns))):
                            revenue = ai.loc[rev_field].iloc[i]
                            gross_profit = ai.loc[gp_field].iloc[i]
                            if revenue != 0 and not pd.isna(revenue):
                                col_name = f"FY {ai.columns[i].year}"
                                row_data[col_name] = f"{(gross_profit/revenue)*100:.1f}%"
        except Exception as e:
            st.warning(f"Could not calculate gross margin: {str(e)}")
        
        return row_data
    
    def _calculate_operating_margin(self, data: Dict, row_data: Dict) -> Dict:
        """Calculate operating margin percentage"""
        try:
            # Quarterly
            if "quarterly_income" in data and data["quarterly_income"] is not None:
                qi = data["quarterly_income"]
                if not qi.empty:
                    rev_field = self._find_field_in_index(qi, "Total Revenue")
                    oi_field = self._find_field_in_index(qi, "Operating Income")
                    
                    if rev_field and oi_field:
                        for i in range(min(3, len(qi.columns))):
                            revenue = qi.loc[rev_field].iloc[i]
                            op_income = qi.loc[oi_field].iloc[i]
                            if revenue != 0 and not pd.isna(revenue):
                                col_name = f"Q{i+1} {qi.columns[i].strftime('%m/%d/%Y')}"
                                row_data[col_name] = f"{(op_income/revenue)*100:.1f}%"
            
            # Annual
            if "annual_income" in data and data["annual_income"] is not None:
                ai = data["annual_income"]
                if not ai.empty:
                    rev_field = self._find_field_in_index(ai, "Total Revenue")
                    oi_field = self._find_field_in_index(ai, "Operating Income")
                    
                    if rev_field and oi_field:
                        for i in range(min(2, len(ai.columns))):
                            revenue = ai.loc[rev_field].iloc[i]
                            op_income = ai.loc[oi_field].iloc[i]
                            if revenue != 0 and not pd.isna(revenue):
                                col_name = f"FY {ai.columns[i].year}"
                                row_data[col_name] = f"{(op_income/revenue)*100:.1f}%"
        except Exception as e:
            st.warning(f"Could not calculate operating margin: {str(e)}")
        
        return row_data
    
    def _calculate_net_margin(self, data: Dict, row_data: Dict) -> Dict:
        """Calculate net margin percentage"""
        try:
            # Quarterly
            if "quarterly_income" in data and data["quarterly_income"] is not None:
                qi = data["quarterly_income"]
                if not qi.empty:
                    rev_field = self._find_field_in_index(qi, "Total Revenue")
                    ni_field = self._find_field_in_index(qi, "Net Income")
                    
                    if rev_field and ni_field:
                        for i in range(min(3, len(qi.columns))):
                            revenue = qi.loc[rev_field].iloc[i]
                            net_income = qi.loc[ni_field].iloc[i]
                            if revenue != 0 and not pd.isna(revenue):
                                col_name = f"Q{i+1} {qi.columns[i].strftime('%m/%d/%Y')}"
                                row_data[col_name] = f"{(net_income/revenue)*100:.1f}%"
            
            # Annual
            if "annual_income" in data and data["annual_income"] is not None:
                ai = data["annual_income"]
                if not ai.empty:
                    rev_field = self._find_field_in_index(ai, "Total Revenue")
                    ni_field = self._find_field_in_index(ai, "Net Income")
                    
                    if rev_field and ni_field:
                        for i in range(min(2, len(ai.columns))):
                            revenue = ai.loc[rev_field].iloc[i]
                            net_income = ai.loc[ni_field].iloc[i]
                            if revenue != 0 and not pd.isna(revenue):
                                col_name = f"FY {ai.columns[i].year}"
                                row_data[col_name] = f"{(net_income/revenue)*100:.1f}%"
        except Exception as e:
            st.warning(f"Could not calculate net margin: {str(e)}")
        
        return row_data
    
    def _calculate_net_cash_flow(self, data: Dict, row_data: Dict) -> Dict:
        """Calculate net cash flow"""
        try:
            # Quarterly
            if "quarterly_cashflow" in data and data["quarterly_cashflow"] is not None:
                qcf = data["quarterly_cashflow"]
                if not qcf.empty:
                    ocf_field = self._find_field_in_index(qcf, "Operating Cash Flow")
                    icf_field = self._find_field_in_index(qcf, "Investing Cash Flow")
                    fcf_field = self._find_field_in_index(qcf, "Financing Cash Flow")
                    
                    for i in range(min(3, len(qcf.columns))):
                        ocf = qcf.loc[ocf_field].iloc[i] if ocf_field else 0
                        icf = qcf.loc[icf_field].iloc[i] if icf_field else 0
                        fcf = qcf.loc[fcf_field].iloc[i] if fcf_field else 0
                        col_name = f"Q{i+1} {qcf.columns[i].strftime('%m/%d/%Y')}"
                        row_data[col_name] = self._format_value(ocf + icf + fcf)
            
            # Annual
            if "annual_cashflow" in data and data["annual_cashflow"] is not None:
                acf = data["annual_cashflow"]
                if not acf.empty:
                    ocf_field = self._find_field_in_index(acf, "Operating Cash Flow")
                    icf_field = self._find_field_in_index(acf, "Investing Cash Flow")
                    fcf_field = self._find_field_in_index(acf, "Financing Cash Flow")
                    
                    for i in range(min(2, len(acf.columns))):
                        ocf = acf.loc[ocf_field].iloc[i] if ocf_field else 0
                        icf = acf.loc[icf_field].iloc[i] if icf_field else 0
                        fcf = acf.loc[fcf_field].iloc[i] if fcf_field else 0
                        col_name = f"FY {acf.columns[i].year}"
                        row_data[col_name] = self._format_value(ocf + icf + fcf)
        except Exception as e:
            st.warning(f"Could not calculate net cash flow: {str(e)}")
        
        return row_data

def display_financial_statements(financial_data: Dict[str, pd.DataFrame],ticker: str):
    """Display financial statements in tabular format"""
    st.header("📊 Financial Statements")
    st.caption("Data sourced from Yahoo Finance (yfinance)")
    
    if not financial_data:
        st.error("No financial data available. Please check the ticker symbol.")
        return
    
    # Create sub-tabs for different statements
    subtab1, subtab2, subtab3 = st.tabs(["Balance Sheet", "Income Statement", "Cash Flow"])
    
    with subtab1:
        st.subheader("Balance Sheet")
        st.caption("All values in thousands (000s)")
        
        # Get formatted balance sheet
        fetcher = FinancialDataFetcher(ticker)  # Ticker not needed for formatting
        balance_sheet = fetcher.format_financial_table(financial_data).fillna("")
        
        if not balance_sheet.empty:
            # Convert to HTML for better formatting
            html = balance_sheet.to_html(index=False, escape=False)
            
            # Apply custom styling
            html = html.replace('<table', '<table style="width:100%"')
            html = html.replace('<td>', '<td style="text-align:right; padding:8px;">')
            html = html.replace('<th>', '<th style="background-color:#f0f2f6; font-weight:bold; text-align:center; padding:8px;">')
            
            # Bold specific rows
            for item in ['BALANCE SHEET', 'Assets:', 'Liabilities:', 'Ratios:']:
                html = html.replace(f'<td style="text-align:right; padding:8px;">{item}</td>',
                                  f'<td style="text-align:left; padding:8px; font-weight:bold;">{item}</td>')
            
            st.markdown(html, unsafe_allow_html=True)
        else:
            st.info("Balance sheet data not available")
    
    with subtab2:
        st.subheader("Income Statement")
        st.caption("All values in thousands (000s)")
        
        income_statement = fetcher.format_income_statement(financial_data).fillna("")
        
        if not income_statement.empty:
            # Convert to HTML
            html = income_statement.to_html(index=False, escape=False)
            
            # Apply styling
            html = html.replace('<table', '<table style="width:100%"')
            html = html.replace('<td>', '<td style="text-align:right; padding:8px;">')
            html = html.replace('<th>', '<th style="background-color:#f0f2f6; font-weight:bold; text-align:center; padding:8px;">')
            
            # Bold header
            html = html.replace('<td style="text-align:right; padding:8px;">INCOME STATEMENT</td>',
                              '<td style="text-align:left; padding:8px; font-weight:bold;">INCOME STATEMENT</td>')
            
            st.markdown(html, unsafe_allow_html=True)
        else:
            st.info("Income statement data not available")
    
    with subtab3:
        st.subheader("Cash Flow Statement")
        st.caption("All values in thousands (000s)")
        
        cash_flow = fetcher.format_cash_flow(financial_data).fillna("")
        
        if not cash_flow.empty:
            # Convert to HTML
            html = cash_flow.to_html(index=False, escape=False)
            
            # Apply styling
            html = html.replace('<table', '<table style="width:100%"')
            html = html.replace('<td>', '<td style="text-align:right; padding:8px;">')
            html = html.replace('<th>', '<th style="background-color:#f0f2f6; font-weight:bold; text-align:center; padding:8px;">')
            
            # Bold headers
            for item in ['CASH FLOW STATEMENT', 'Operating Activities:', 'Investing Activities:', 'Financing Activities:']:
                html = html.replace(f'<td style="text-align:right; padding:8px;">{item}</td>',
                                  f'<td style="text-align:left; padding:8px; font-weight:bold;">{item}</td>')
            
            st.markdown(html, unsafe_allow_html=True)
        else:
            st.info("Cash flow data not available")

def display_risk_analysis(risk_data: Dict[str, List[str]]):
    """Display AI-analyzed risks"""
    st.header("⚠️ Risk Analysis")
    st.caption("AI-generated analysis from 10-Q document")
    
    risk_categories = [
        ("💰 Input Cost Risks", "input_cost_risks"),
        ("⚙️ Operational Risks & Mitigation", "operational_risks"),
        ("📉 Financial & Debt Risks", "financial_risks"),
        ("📋 Regulatory & Compliance Risks", "regulatory_risks"),
        ("🎯 Strategic Implications", "strategic_risks")
    ]
    
    for title, key in risk_categories:
        with st.expander(title, expanded=True):
            if risk_data.get(key) and any(risk for risk in risk_data[key] if risk and len(risk) > 20):
                for i, risk in enumerate(risk_data[key], 1):
                    if risk and len(risk) > 20:  # Only show substantial risks
                        st.markdown(f"""
                        <div class="risk-card">
                            <strong>Risk {i}:</strong> {risk}
                        </div>
                        """, unsafe_allow_html=True)
            else:
                st.info(f"No specific {title.lower()} identified in the document.")

def display_liquidity_analysis(liquidity_data: Dict[str, any]):
    """Display AI-analyzed liquidity position"""
    st.header("💧 Liquidity Analysis")
    st.caption("AI-generated analysis from 10-Q document")
    
    # Summary
    st.subheader("Executive Summary")
    st.markdown(f"""
    <div class="ai-summary">
        {liquidity_data.get("summary", "No summary available")}
    </div>
    """, unsafe_allow_html=True)
    
    # Key metrics in columns
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 💵 Cash Position")
        st.info(liquidity_data.get("cash_position", "N/A"))
        
        st.markdown("### 📊 Working Capital")
        st.info(liquidity_data.get("working_capital", "N/A"))
    
    with col2:
        st.markdown("### 💸 Cash Flow Trends")
        st.info(liquidity_data.get("cash_flow", "N/A"))
        
        st.markdown("### 🏦 Credit Facilities")
        st.info(liquidity_data.get("credit_facilities", "N/A"))
    
    # Key metrics
    if liquidity_data.get("key_metrics") and len(liquidity_data["key_metrics"]) > 0:
        st.markdown("### 📈 Key Liquidity Metrics")
        for metric in liquidity_data["key_metrics"]:
            if metric:  # Only show non-empty metrics
                st.markdown(f"• {metric}")

def display_profitability_analysis(profitability_data: Dict[str, any]):
    """Display AI-analyzed profitability"""
    st.header("📈 Profitability Analysis")
    st.caption("AI-generated analysis from 10-Q document")
    
    # Summary
    st.subheader("Executive Summary")
    st.markdown(f"""
    <div class="ai-summary">
        {profitability_data.get("summary", "No summary available")}
    </div>
    """, unsafe_allow_html=True)
    
    # Analysis sections
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📊 Revenue Trends")
        st.info(profitability_data.get("revenue_trends", "N/A"))
        
        st.markdown("### 💹 Net Income Performance")
        st.info(profitability_data.get("net_income", "N/A"))
    
    with col2:
        st.markdown("### 📉 Margin Analysis")
        st.info(profitability_data.get("margins", "N/A"))
        
        st.markdown("### 🔮 Management Outlook")
        st.info(profitability_data.get("outlook", "N/A"))
    
    # Key drivers
    if profitability_data.get("key_drivers") and len(profitability_data["key_drivers"]) > 0:
        st.markdown("### 🔑 Key Profitability Drivers")
        for driver in profitability_data["key_drivers"]:
            if driver:  # Only show non-empty drivers
                st.markdown(f"• {driver}")


def generate_print_report(ticker: str, all_data: Dict) -> str:
    """Generate HTML report for printing"""
    current_date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    # Build HTML with inline styles and JavaScript to trigger print when rendered
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Financial Analysis Report - {ticker}</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                margin: 40px;
                line-height: 1.6;
                color: #333;
            }}
            h1, h2, h3 {{
                color: #1976d2;
                margin-top: 20px;
            }}
            h1 {{
                border-bottom: 3px solid #1976d2;
                padding-bottom: 10px;
            }}
            table {{
                width: 100%;
                border-collapse: collapse;
                margin-bottom: 20px;
            }}
            th, td {{
                border: 1px solid #ddd;
                padding: 8px;
                text-align: right;
            }}
            th {{
                background-color: #f0f2f6;
                font-weight: bold;
            }}
            .section-header {{
                background-color: #e3f2fd;
                padding: 10px;
                margin-top: 20px;
                margin-bottom: 10px;
            }}
        </style>
    </head>
    <body>
        <h1>Financial Analysis Report - {ticker}</h1>
        <p><em>Generated on {current_date}</em></p>

        <div class="section-header">
            <h2>1. Financial Statements</h2>
        </div>
        <!-- Embed tables for Balance Sheet, Income Statement, Cash Flow -->
        {all_data['tables']}

        <div class="section-header">
            <h2>2. Risk Analysis</h2>
        </div>
        {all_data['risk_html']}

        <div class="section-header">
            <h2>3. Liquidity Analysis</h2>
        </div>
        {all_data['liquidity_html']}

        <div class="section-header">
            <h2>4. Profitability Analysis</h2>
        </div>
        {all_data['profitability_html']}

        <script>
            // Automatically open print dialog
            window.onload = function() {{ window.print(); }};
        </script>
    </body>
    </html>
    """
    return html


def main():
    """Main Streamlit app"""
    st.title("Financial Statement Analyzer with Artificial Intelligence")
    st.write("Use this app to fetch financial data and perform AI-driven analysis on 10-Q documents.")

    # Input Section: Ticker and PDF upload
    ticker = st.text_input("Enter Stock Ticker Symbol").upper()
    # pdf_file = st.file_uploader("Upload the Latest 10-Q PDF", type=["pdf"] )
    pdf_file = None  # will be set after fetching
    if ticker:
        try:
            # Load CIK mapping JSON (ensure this file is in your working directory)
            import requests, json
            with open("C:\\Users\\kumba\\Documents\\Streamlitapp_for_financial_analysis\\company_tickers_exchange.json", "r") as f:
                CIK_dict = json.load(f)
            CIK_df = pd.DataFrame(CIK_dict["data"], columns=CIK_dict["fields"])
            CIK = CIK_df[CIK_df["ticker"] == ticker].cik.values[0]

            # SEC headers and URLs
            headers = {"User-Agent": "your.email@domain.com"}
            submissions_url = f"https://data.sec.gov/submissions/CIK{str(CIK).zfill(10)}.json"
            subs = requests.get(submissions_url, headers=headers).json()
            recent = pd.DataFrame(subs["filings"]["recent"])
            acc_num = recent[recent.form == "10-Q"].accessionNumber.values[0].replace("-", "")
            doc_name = recent[recent.form == "10-Q"].primaryDocument.values[0]
            html_url = f"https://www.sec.gov/Archives/edgar/data/{CIK}/{acc_num}/{doc_name}"

        #     # Fetch HTML and convert to PDF bytes using xhtml2pdf (no external wkhtmltopdf needed)
        #     from xhtml2pdf import pisa
        #     pdf_buffer = io.BytesIO()
        #     # Convert HTML string to PDF
        #     html_content = requests.get(html_url, headers=headers).content.decode("utf-8")
        #     pisa.CreatePDF(io.StringIO(html_content), dest=pdf_buffer)
        #     pdf_buffer.seek(0)
        #     pdf_file = pdf_buffer
        # except Exception as e:
        #     st.error(f"Failed to fetch or convert the 10-Q: {e}")
            pdf_url = f"https://www.sec.gov/Archives/edgar/data/{CIK}/{acc_num}/{doc_name.replace('.htm', '.pdf')}"
        
            pdf_response = requests.get(pdf_url, headers=headers)
        
            if pdf_response.status_code == 200:
                pdf_file = io.BytesIO(pdf_response.content)
            else:
                # Fallback to HTML conversion with images removed
                html_content = requests.get(html_url, headers=headers).content.decode("utf-8")
                html_content = re.sub(r'<img[^>]*>', '', html_content)
                html_content = re.sub(r'<table[^>]*>.*?</table>', '', html_content, flags=re.DOTALL)
            
                from xhtml2pdf import pisa
                pdf_buffer = io.BytesIO()
                pisa.CreatePDF(io.StringIO(html_content), dest=pdf_buffer)
                pdf_buffer.seek(0)
                pdf_file = pdf_buffer
            
        except Exception as e:
            st.error(f"Failed to fetch or convert the 10-Q: {e}")
        #         import weasyprint
        
        #         html_content = requests.get(html_url, headers=headers).content.decode("utf-8")
        
        # # Create PDF using weasyprint with base URL for relative links
        #         base_url = f"https://www.sec.gov/Archives/edgar/data/{CIK}/{acc_num}/"
        #         pdf_bytes = weasyprint.HTML(
        #             string=html_content, 
        #             base_url=base_url
        #         ).write_pdf()
        
        #         pdf_file = io.BytesIO(pdf_bytes)
        
        # except Exception as e:
        #     st.error(f"Failed to fetch or convert the 10-Q: {e}")

    if ticker and pdf_file:
        try:
            # Initialize analyzer and fetcher
            analyzer = AzureOpenAIAnalyzer()
            fetcher = FinancialDataFetcher(ticker)

            # Extract text from PDF
            raw_text = analyzer.extract_text_from_pdf(pdf_file)

            # Fetch financial statements
            financial_data = fetcher.get_financial_statements()

            # Perform AI-driven analyses
            risk_data = analyzer.analyze_risks(raw_text)
            liquidity_data = analyzer.analyze_liquidity(raw_text)
            profitability_data = analyzer.analyze_profitability(raw_text)

            # Prepare HTML snippets for print
            # Financial tables
            bal_html = fetcher.format_financial_table(financial_data).to_html(index=False, escape=False,na_rep="")
            inc_html = fetcher.format_income_statement(financial_data).to_html(index=False, escape=False,na_rep="")
            cf_html = fetcher.format_cash_flow(financial_data).to_html(index=False, escape=False,na_rep="")
            tables_html = f"<h3>Balance Sheet</h3>{bal_html}<h3>Income Statement</h3>{inc_html}<h3>Cash Flow</h3>{cf_html}"

            # Risk HTML
            risk_items = ""
            for key, title in [("Financial And Debt-Related Risks", "Financial And Debt-Related Risks"), ("Debt Maturity", "Debt Maturity"),
                               ("Interest Expense", "Interest Expense"), ("Executive Summary", "Executive Summary")]:
                risk_items += f"<h4>{title}</h4><ul>"
                for r in risk_data.get(key, []):
                    risk_items += f"<li>{r}</li>"
                risk_items += "</ul>"
            risk_html = risk_items

            # Liquidity HTML
            liquidity_html = f"<p>{liquidity_data.get('summary','')}</p><ul>"
            for k in ['cash_position', 'working_capital', 'cash_flow', 'credit_facilities']:
                liquidity_html += f"<li><strong>{k.replace('_',' ').title()}:</strong> {liquidity_data.get(k,'')}</li>"
            liquidity_html += "</ul>"

            # Profitability HTML
            profitability_html = f"<p>{profitability_data.get('summary','')}</p><ul>"
            for k in ['revenue_trends', 'net_income', 'margins', 'outlook']:
                profitability_html += f"<li><strong>{k.replace('_',' ').title()}:</strong> {profitability_data.get(k,'')}</li>"
            profitability_html += "</ul>"

            # Main tabs for interactive display
            tab1, tab2, tab3, tab4 = st.tabs(["Financial Statements", "Risk Analysis", "Liquidity", "Profitability"])

            with tab1:
                display_financial_statements(financial_data,ticker)

            with tab2:
                display_risk_analysis(risk_data)

            with tab3:
                display_liquidity_analysis(liquidity_data)

            with tab4:
                display_profitability_analysis(profitability_data)

            # # Print functionality
            # if st.button("Print Full Report"):
            #     all_html = {
            #         'tables': tables_html,
            #         'risk_html': risk_html,
            #         'liquidity_html': liquidity_html,
            #         'profitability_html': profitability_html
            #     }
            #     report = generate_print_report(ticker, all_html)
            #     st.markdown(report, unsafe_allow_html=True)
            all_html = {
                'tables': tables_html,
                'risk_html': risk_html,
                'liquidity_html': liquidity_html,
                'profitability_html': profitability_html
            }
            report_html = generate_print_report(ticker, all_html)
            # Provide a download button for the HTML report
            st.download_button(
                label="Download Full Report as HTML",
                data=report_html,
                file_name=f"{ticker}_financial_report.html",
                mime="text/html"
            )

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
    else:
        st.info("Please enter a ticker symbol to proceed.")

if __name__ == "__main__":
    main()
