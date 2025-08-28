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
import logging
import docx2txt
import requests
import os
import mimetypes

# Azure OpenAI imports
from openai import AzureOpenAI
import tiktoken

# LlamaIndex imports for vector embeddings
from llama_index.core import Document, VectorStoreIndex, Settings
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.llms.azure_openai import AzureOpenAI as LlamaAzureOpenAI
from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding
from llama_index.core.indices.prompt_helper import PromptHelper
from llama_index.core.callbacks import CallbackManager, LlamaDebugHandler

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration - Replace with your actual Azure OpenAI credentials
AZURE_OPENAI_KEY = "2f6e41aa534f49908feb01c6de771d6b"
AZURE_OPENAI_ENDPOINT = "https://ea-oai-sandbox.openai.azure.com/"
AZURE_OPENAI_DEPLOYMENT = "dev-gpt-4o"
AZURE_OPENAI_EMBEDDING_DEPLOYMENT = "text-embedding-ada-002"
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
        line-height: 1.6;
    }
    .risk-card {
        background-color: #fff3e0;
        padding: 12px;
        border-radius: 6px;
        margin: 8px 0;
        border-left: 4px solid #ff9800;
        line-height: 1.6;
    }
    .profitability-paragraph {
        margin-bottom: 15px;
        line-height: 1.6;
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

class AzureOpenAIAnalyzer:
    """Handles Azure OpenAI document analysis with vector embeddings"""

    def __init__(self):
        """Initialize Azure OpenAI client with vector embedding support"""
        try:
            # Standard OpenAI client for direct API calls
            self.client = AzureOpenAI(
                api_key=AZURE_OPENAI_KEY,
                api_version=AZURE_OPENAI_API_VERSION,
                azure_endpoint=AZURE_OPENAI_ENDPOINT
            )
            self.deployment_name = AZURE_OPENAI_DEPLOYMENT
            # tiktoken lookup can fail on unknown model names; use a safe fallback
            try:
                self.encoding = tiktoken.encoding_for_model("gpt-4")
            except Exception:
                self.encoding = tiktoken.get_encoding("cl100k_base")

            # Initialize LlamaIndex components for vector embeddings
            self._initialize_vector_components()

        except Exception as e:
            st.error(f"Failed to initialize Azure OpenAI: {str(e)}")
            st.info("Please check your Azure OpenAI credentials in the code.")
            self.client = None
            self.vector_index = None

    def _initialize_vector_components(self):
        """Initialize LlamaIndex components for vector embeddings"""
        try:
            # Configure Azure OpenAI LLM for LlamaIndex
            self.llama_llm = LlamaAzureOpenAI(
                deployment_name=AZURE_OPENAI_DEPLOYMENT,
                api_key=AZURE_OPENAI_KEY,
                azure_endpoint=AZURE_OPENAI_ENDPOINT,
                api_version=AZURE_OPENAI_API_VERSION,
                temperature=0.1,
                max_tokens=3000
            )

            # Configure Azure OpenAI Embeddings
            self.embed_model = AzureOpenAIEmbedding(
                deployment_name=AZURE_OPENAI_EMBEDDING_DEPLOYMENT,
                api_key=AZURE_OPENAI_KEY,
                azure_endpoint=AZURE_OPENAI_ENDPOINT,
                api_version=AZURE_OPENAI_API_VERSION
            )

            # Configure prompt helper
            self.prompt_helper = PromptHelper(
                context_window=8192,
                num_output=3000,
                chunk_overlap_ratio=0.1
            )

            # Set up callback manager
            self.callback_manager = CallbackManager([
                LlamaDebugHandler(print_trace_on_end=False)
            ])

            # Apply global settings for LlamaIndex
            Settings.llm = self.llama_llm
            Settings.embed_model = self.embed_model
            Settings.prompt_helper = self.prompt_helper
            Settings.callback_manager = self.callback_manager

            # Node parser for chunking
            self.node_parser = SimpleNodeParser.from_defaults(
                chunk_size=512,
                chunk_overlap=50,
                include_metadata=True,
                include_prev_next_rel=True
            )

            self.vector_index = None
            logger.info("Vector components initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize vector components: {str(e)}")
            self.vector_index = None

    def _infer_mime_from_name(self, name: Optional[str]) -> Optional[str]:
        if not name:
            return None
        guess, _ = mimetypes.guess_type(name)
        return guess

    def extract_text_from_file(self, file, filename: Optional[str]=None, file_type: Optional[str]=None) -> str:
        """Extract text from uploaded file (PDF, Word, or Text).
           Handles Streamlit UploadedFile, file-like BytesIO, and raw bytes.
        """
        try:
            # Determine MIME/type
            ftype = file_type or getattr(file, "type", None) or self._infer_mime_from_name(getattr(file, "name", filename))

            # If we still don't know and it's BytesIO, default to PDF (our SEC fetch is PDF)
            if not ftype and isinstance(file, (io.BytesIO, bytes)):
                ftype = "application/pdf"

            # Normalize file-like object
            buffer = None
            if isinstance(file, io.BytesIO):
                buffer = file
                try:
                    buffer.seek(0)
                except Exception:
                    pass
            elif hasattr(file, "read"):  # Streamlit UploadedFile
                buffer = io.BytesIO(file.read())
                buffer.seek(0)
            elif isinstance(file, bytes):
                buffer = io.BytesIO(file)
                buffer.seek(0)
            else:
                return ""

            text = ""

            if ftype == "application/pdf":
                # PDF extraction
                try:
                    pdf_reader = PyPDF2.PdfReader(buffer)
                except Exception as pe:
                    # Some SEC files might not be pure PDFs; surface a helpful message
                    st.error(f"Unable to read PDF: {pe}")
                    return ""
                total_pages = len(pdf_reader.pages)
                if total_pages > 10:
                    progress_bar = st.progress(0, text=f"Reading PDF: 0/{total_pages} pages")
                for i, page in enumerate(pdf_reader.pages):
                    page_text = page.extract_text() or ""
                    text += page_text + "\n"
                    if total_pages > 10:
                        progress_bar.progress((i + 1) / total_pages, text=f"Reading PDF: {i+1}/{total_pages} pages")
                if total_pages > 10:
                    progress_bar.empty()

            elif ftype == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
                # Word document extraction
                # docx2txt expects a path or file-like; it works with BytesIO
                text = docx2txt.process(buffer)

            elif ftype == "text/plain":
                text = buffer.read().decode("utf-8", errors="ignore")

            else:
                st.error(f"Unsupported file type: {ftype or 'unknown'}")
                return ""

            return text.strip()
        except Exception as e:
            st.error(f"Error reading file: {str(e)}")
            return ""

    def create_vector_index(self, text: str) -> Optional[VectorStoreIndex]:
        """Create vector index from document text"""
        try:
            if not text or not text.strip():
                return None
            with st.spinner("Creating vector embeddings..."):
                # Create document
                document = Document(text=text)
                # Parse into nodes
                nodes = self.node_parser.get_nodes_from_documents([document])
                logger.info(f"Generated {len(nodes)} nodes from document")
                # Create vector index
                vector_index = VectorStoreIndex(nodes)
                logger.info("Vector index created successfully")
                return vector_index
        except Exception as e:
            logger.error(f"Failed to create vector index: {str(e)}")
            return None

    def query_vector_index(self, query: str, similarity_top_k: int = 3, max_retries: int = 3) -> str:
        """Query the vector index with retry logic"""
        if not self.vector_index:
            return ""

        try:
            query_engine = self.vector_index.as_query_engine(
                similarity_top_k=similarity_top_k,
                response_mode='compact'
            )

            for i in range(max_retries):
                try:
                    response = query_engine.query(query)
                    return str(response)
                except ValueError as e:
                    if 'context size' in str(e).lower() and i < max_retries - 1:
                        logger.warning(f"Context size issue, retry {i+1}")
                        query_engine = self.vector_index.as_query_engine(
                            similarity_top_k=max(1, similarity_top_k - i),
                            response_mode='compact'
                        )
                    else:
                        raise

            return "Unable to process query due to constraints"

        except Exception as e:
            logger.error(f"Error querying vector index: {str(e)}")
            return ""

    def analyze_risks(self, text: str) -> str:
        """Analyze risks from 10-Q document using vector embeddings"""
        if not self.client:
            return "AI analysis unavailable - please check Azure OpenAI configuration"

        # Ensure vector index is ready
        if not self.vector_index:
            self.vector_index = self.create_vector_index(text)
        if not self.vector_index:
            return "Unable to create vector index for analysis"

        # First get the currency scale
        currency_prompt = """From the given 10-Q filing, identify and return the currency unit or scale used for financial figures (e.g., "in millions", "in thousands")—typically found near the balance sheet or income statement headings."""
        
        # Risk analysis prompts (without Currency Scale as a section)
        risk_sections = {
            "Financial and Debt-Related Risks": """
            Role: "You are a financial analyst AI specializing in risk analysis using company 10-Q reports."
            Provide a comprehensive analysis in 2-3 sentences that includes: total debt, % of variable-rate debt, upcoming maturities; key debt facilities, hedging, covenants; and risks to liquidity, access to capital, or refinancing pressures. Write as a continuous narrative, not bullet points.
            """,
            "Debt maturity": """From the given 10-Q filing, provide a 2-3 sentence narrative covering debt maturity including total debt, near-term maturities, long-term maturities, interest rates, principal amount, carrying amount, and repayment or refinancing details. Include specific dollar amounts and dates where available. Write as continuous text, not bullet points.""",
            "Interest Expense": """Analyze the 10-Q filing and provide a 2-3 sentence summary about the company's debt interest expenses or interest rates. Include Net Interest expenses of that quarter and same quarter of previous year if available. Include specific dollar amounts and percentages. If not available, respond exactly: "This details are not provided in 10-Q filing". Write as continuous text, not bullet points.""",
            "Executive Summary": """Analyse how financial risk factors could negatively impact operations and cashflow presented in the latest 10-Q filing. Provide a 3-4 sentence narrative summary identifying how financial/debt/operational risks could negatively impact operations and cashflow. Include specific figures where applicable (e.g., $4,900 thousand or $4.9 million)."""
        }

        try:
            results = []
            
            # Get currency scale and format it as an inline sentence
            currency_response = self.query_vector_index(currency_prompt, similarity_top_k=3)
            if currency_response and currency_response.strip():
                # Format as a simple sentence without bold formatting
                results.append(f"The financial figures in the 10-Q filing are presented {currency_response.strip()}.")
                results.append("")
            
            # Process other risk sections (without bold formatting)
            for section_name, prompt in risk_sections.items():
                response = self.query_vector_index(prompt, similarity_top_k=3)
                if response:
                    results.append(f"**{section_name}:**")
                    results.append(response.strip())
                    results.append("")
                    
            return "\n".join(results) if results else "No risk-related details found in the provided filing text."
        except Exception as e:
            logger.error(f"Error in risk analysis: {str(e)}")
            return "Unable to complete risk analysis due to an error"

    def analyze_liquidity(self, text: str) -> str:
        """Analyze liquidity using prompts from Liquidity_summary_v2.py - formatted as bullets"""
        if not self.client:
            return "AI analysis unavailable - please check Azure OpenAI configuration"

        if not self.vector_index:
            self.vector_index = self.create_vector_index(text)
        if not self.vector_index:
            return "Unable to create vector index for analysis"

        prompts = {
            "cash_equivalents": """
                From this 10-Q filing, clearly identify and state the exact amount of cash and cash equivalents 
                the company has as of the quarter-end date. Provide the specific dollar amount in millions or 
                billions as stated in the filing. Include the exact date of the quarter-end.
            """,
            "liquidity_runway": """
                Based on management's disclosures in this 10-Q filing, determine whether the company's cash and 
                liquidity resources will last for the next 12 months. Look for management's assessment of their 
                liquidity position and any statements about their ability to fund operations.
            """,
            "credit_facilities": """
                From this 10-Q filing, provide information about the company's line of credit facilities. 
                Include: total facility amount available, current amount drawn or outstanding, and the 
                maturity or expiration date of the facility, details of any Credit Agreements.
            """,
            "going_concern": """
                From this 10-Q filing, determine if there is any mention of going concern issues or substantial 
                doubt about the company's ability to continue operations for the next 12 months. If no going 
                concern issues are disclosed, state this clearly.
            """
        }

        try:
            answers = {}
            for key, prompt in prompts.items():
                response = self.query_vector_index(prompt, similarity_top_k=3)
                answers[key] = response if response else "Information not available"

            # Format as bullet points following the sample format
            consolidated_prompt = f"""
            Based on the following information extracted from a 10-Q filing, create a bullet-point summary 
            formatted exactly as follows (use "- " prefix for each bullet point):

            1. Cash and Cash Equivalents: {answers.get('cash_equivalents', 'Not available')}
            2. 12-Month Liquidity Outlook: {answers.get('liquidity_runway', 'Not available')}
            3. Line of Credit Information: {answers.get('credit_facilities', 'Not available')}
            4. Going Concern Status: {answers.get('going_concern', 'Not available')}

            Create 4 bullet points, each starting with "- " that summarize:
            1. Current cash position and date
            2. Whether cash will last 12 months and management's expectations
            3. Credit facility details including amount, usage, and maturity
            4. Going concern status

            Use specific figures and dates where available. Format each as a complete sentence.
            """

            final_summary = self.query_vector_index(consolidated_prompt, similarity_top_k=2)
            
            # If the response doesn't have bullet points, format it ourselves
            if final_summary and not final_summary.startswith("- "):
                # Try to parse and format as bullets
                lines = final_summary.strip().split('\n')
                formatted_bullets = []
                for line in lines:
                    line = line.strip()
                    if line:
                        if not line.startswith("- "):
                            line = f"- {line}"
                        formatted_bullets.append(line)
                return "\n".join(formatted_bullets) if formatted_bullets else final_summary
            
            return final_summary if final_summary else "Unable to generate liquidity summary"
        except Exception as e:
            logger.error(f"Error in liquidity analysis: {str(e)}")
            return "Unable to complete liquidity analysis due to an error"

    def analyze_profitability(self, text: str) -> str:
        """Analyze profitability using prompts from profit_profile.py"""
        if not self.client:
            return "AI analysis unavailable - please check Azure OpenAI configuration"

        if not self.vector_index:
            self.vector_index = self.create_vector_index(text)
        if not self.vector_index:
            return "Unable to create vector index for analysis"

        prompt = """
        Your task is to analyse the company's profit growth based on the provided 10Q filing report.
        Write in a concise and analytical tone, like an investor earnings summary. Do not list bullet points—compose it as a narrative reflection using concrete financial figures and trends from the filing.
        Include actual figures with units (e.g., $ millions, x coverage).
        Provide 2-3 paragraphs focusing on the following aspects:

        1. Revenue Analysis: 
           - Identify total revenue and compare it to the previous quarter and the same quarter from the previous year.
           - Highlight any significant changes or trends in revenue streams.
           - Include actual figures with units (e.g., $ millions, x coverage).

        2. Operating Income:
           - Calculate and present the Operating income figures, comparing them to previous quarters and years.
           - Discuss the factors contributing to increases or decreases in Operating income.
           - Include actual figures with units (e.g., $ millions, x coverage).

        3. Profit Margins:
           - Analyze gross, operating, and net profit margins, providing both absolute figures and percentage changes.
           - Discuss any operational efficiencies or inefficiencies impacting margins.
           - Reference to liquidity or cash reserves if discussed in the context of financial strength or flexibility.
           - Include actual figures with units (e.g., $ millions, x coverage).

        4. Expenses Overview:
           - Break down major expenses reported in the 10Q and evaluate their impact on profit growth.
           - Highlight any significant cost-saving measures or areas of increased spending.
           - Include actual figures with units (e.g., $ millions, x coverage).

        5. Year-over-Year Growth:
           - Calculate year-over-year profit growth rates and discuss implications for future performance.
           - Include insights on how seasonal trends or external factors may influence these metrics.
           - Include actual figures with units (e.g., $ millions, x coverage).

        6. Forward-Looking Statements:
           - Summarize any management commentary or projections regarding future profit growth mentioned in the 10Q.
           - Analyze potential risks and opportunities that could affect future profitability.
           - Include actual figures with units (e.g., $ millions, x coverage).

        Write this as a continuous narrative of 2-3 paragraphs, not as bullet points or numbered sections.
        Each paragraph should be separated by a clear line break for readability.
        """

        try:
            response = self.query_vector_index(prompt, similarity_top_k=2)
            
            # Format with proper paragraph spacing
            if response:
                # Split into paragraphs and ensure proper spacing
                paragraphs = response.strip().split('\n\n')
                if len(paragraphs) == 1:
                    # If no double line breaks, try single line breaks
                    paragraphs = response.strip().split('\n')
                
                # Format each paragraph with proper spacing
                formatted_paragraphs = []
                for para in paragraphs:
                    para = para.strip()
                    if para and len(para) > 50:  # Only include substantial paragraphs
                        formatted_paragraphs.append(f'<p class="profitability-paragraph">{para}</p>')
                
                return ''.join(formatted_paragraphs) if formatted_paragraphs else response
            
            return "Unable to generate profitability analysis"
        except Exception as e:
            logger.error(f"Error in profitability analysis: {str(e)}")
            return "Unable to complete profitability analysis due to an error"

    def analyze_cashflow(self, text: str) -> str:
        """Analyze cash flow using prompts from Cashflow_summary_v1.py"""
        if not self.client:
            return "AI analysis unavailable - please check Azure OpenAI configuration"

        if not self.vector_index:
            self.vector_index = self.create_vector_index(text)
        if not self.vector_index:
            return "Unable to create vector index for analysis"

        prompt = """
        From the given 10-Q filing of a company, summarize the changes in the following cash flow activities in 3–4 sentences, focusing specifically on the *Cash Flow* section:

        * Cash flow from operating activities
        * Cash flow from investing activities
        * Cash flow from financing activities

        Include the starting amount, ending amount, comparison to the same period in the prior year, and explain the key reasons behind the changes.
        """

        try:
            response = self.query_vector_index(prompt, similarity_top_k=2)
            return response if response else "Unable to generate cash flow analysis"
        except Exception as e:
            logger.error(f"Error in cash flow analysis: {str(e)}")
            return "Unable to complete cash flow analysis due to an error"

    def _get_default_risk_response(self) -> Dict:
        """Default response when AI analysis fails"""
        return {
            "summary": "AI analysis unavailable - please check Azure OpenAI configuration"
        }

    def _get_default_liquidity_response(self) -> Dict:
        """Default response when AI analysis fails"""
        return {
            "summary": "AI analysis unavailable - please check Azure OpenAI configuration"
        }

    def _get_default_profitability_response(self) -> Dict:
        """Default response when AI analysis fails"""
        return {
            "summary": "AI analysis unavailable - please check Azure OpenAI configuration"
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

            if item_type in ("section_header", "subsection", "blank"):
                formatted_data.append(row_data)
                continue

            # Add last 3 quarters (labeled as Q, Q, Q)
            if "quarterly_balance_sheet" in data and data["quarterly_balance_sheet"] is not None:
                qbs = data["quarterly_balance_sheet"]
                if not qbs.empty and field_name:
                    matching_fields = [idx for idx in qbs.index if field_name.lower() in str(idx).lower()]
                    if matching_fields:
                        field_to_use = matching_fields[0]
                        for i in range(min(3, len(qbs.columns))):
                            col_name = f"Q {qbs.columns[i].strftime('%m/%d/%Y')}"
                            value = qbs.loc[field_to_use].iloc[i]
                            row_data[col_name] = self._format_value(value)
                            if i > 0:
                                prev_value = qbs.loc[field_to_use].iloc[i-1]
                                pct_change = self._calculate_percentage_change(prev_value, value)
                                row_data[f"Q Δ% {i}"] = pct_change

            # Add last 3 annual data
            if "annual_balance_sheet" in data and data["annual_balance_sheet"] is not None:
                abs_data = data["annual_balance_sheet"]
                if not abs_data.empty and field_name:
                    matching_fields = [idx for idx in abs_data.index if field_name.lower() in str(idx).lower()]
                    if matching_fields:
                        field_to_use = matching_fields[0]
                        for i in range(min(3, len(abs_data.columns))):
                            col_name = f"FY {abs_data.columns[i].year}"
                            value = abs_data.loc[field_to_use].iloc[i]
                            row_data[col_name] = self._format_value(value)
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

            # Add last 3 quarters with percentage changes
            if "quarterly_income" in data and data["quarterly_income"] is not None:
                qi = data["quarterly_income"]
                if not qi.empty and field_name:
                    matching_fields = [idx for idx in qi.index if field_name.lower() in str(idx).lower()]
                    if matching_fields:
                        field_to_use = matching_fields[0]
                        for i in range(min(3, len(qi.columns))):
                            col_name = f"Q {qi.columns[i].strftime('%m/%d/%Y')}"
                            value = qi.loc[field_to_use].iloc[i]
                            row_data[col_name] = self._format_value(value)
                            if i > 0:
                                prev_value = qi.loc[field_to_use].iloc[i-1]
                                pct_change = self._calculate_percentage_change(prev_value, value)
                                row_data[f"Q Δ% {i}"] = pct_change

            # Add last 3 annual data with percentage changes
            if "annual_income" in data and data["annual_income"] is not None:
                ai = data["annual_income"]
                if not ai.empty and field_name:
                    matching_fields = [idx for idx in ai.index if field_name.lower() in str(idx).lower()]
                    if matching_fields:
                        field_to_use = matching_fields[0]
                        for i in range(min(3, len(ai.columns))):
                            col_name = f"FY {ai.columns[i].year}"
                            value = ai.loc[field_to_use].iloc[i]
                            row_data[col_name] = self._format_value(value)
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

            # Add last 3 quarters
            if "quarterly_cashflow" in data and data["quarterly_cashflow"] is not None:
                qcf = data["quarterly_cashflow"]
                if not qcf.empty and field_name:
                    matching_fields = [idx for idx in qcf.index if field_name.lower() in str(idx).lower()]
                    if matching_fields:
                        field_to_use = matching_fields[0]
                        for i in range(min(3, len(qcf.columns))):
                            col_name = f"Q {qcf.columns[i].strftime('%m/%d/%Y')}"
                            value = qcf.loc[field_to_use].iloc[i]
                            row_data[col_name] = self._format_value(value)

            # Add last 3 annual data
            if "annual_cashflow" in data and data["annual_cashflow"] is not None:
                acf = data["annual_cashflow"]
                if not acf.empty and field_name:
                    matching_fields = [idx for idx in acf.index if field_name.lower() in str(idx).lower()]
                    if matching_fields:
                        field_to_use = matching_fields[0]
                        for i in range(min(3, len(acf.columns))):
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
        if field_name in df.index:
            return field_name
        field_lower = field_name.lower()
        for idx in df.index:
            if str(idx).lower() == field_lower:
                return idx
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
                            col_name = f"Q {qbs.columns[i].strftime('%m/%d/%Y')}"
                            row_data[col_name] = self._format_value(ca - cl)
            # Annual
            if "annual_balance_sheet" in data and data["annual_balance_sheet"] is not None:
                abs_data = data["annual_balance_sheet"]
                if not abs_data.empty:
                    ca_field = self._find_field_in_index(abs_data, "Current Assets")
                    cl_field = self._find_field_in_index(abs_data, "Current Liabilities")
                    if ca_field and cl_field:
                        for i in range(min(3, len(abs_data.columns))):
                            ca = abs_data.loc[ca_field].iloc[i]
                            cl = abs_data.loc[cl_field].iloc[i]
                            col_name = f"FY {abs_data.columns[i].year}"
                            row_data[col_name] = self._format_value(ca - cl)
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
                            col_name = f"Q {qbs.columns[i].strftime('%m/%d/%Y')}"
                            row_data[col_name] = self._format_value(ta - tl)
            # Annual
            if "annual_balance_sheet" in data and data["annual_balance_sheet"] is not None:
                abs_data = data["annual_balance_sheet"]
                if not abs_data.empty:
                    ta_field = self._find_field_in_index(abs_data, "Total Assets")
                    tl_field = self._find_field_in_index(abs_data, "Total Liabilities")
                    if ta_field and tl_field:
                        for i in range(min(3, len(abs_data.columns))):
                            ta = abs_data.loc[ta_field].iloc[i]
                            tl = abs_data.loc[tl_field].iloc[i]
                            col_name = f"FY {abs_data.columns[i].year}"
                            row_data[col_name] = self._format_value(ta - tl)
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
                                col_name = f"Q {qbs.columns[i].strftime('%m/%d/%Y')}"
                                row_data[col_name] = f"{ca/cl:.2f}"
            # Annual
            if "annual_balance_sheet" in data and data["annual_balance_sheet"] is not None:
                abs_data = data["annual_balance_sheet"]
                if not abs_data.empty:
                    ca_field = self._find_field_in_index(abs_data, "Current Assets")
                    cl_field = self._find_field_in_index(abs_data, "Current Liabilities")
                    if ca_field and cl_field:
                        for i in range(min(3, len(abs_data.columns))):
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
                                col_name = f"Q {qbs.columns[i].strftime('%m/%d/%Y')}"
                                row_data[col_name] = f"{(ca-inv)/cl:.2f}"
            # Annual
            if "annual_balance_sheet" in data and data["annual_balance_sheet"] is not None:
                abs_data = data["annual_balance_sheet"]
                if not abs_data.empty:
                    ca_field = self._find_field_in_index(abs_data, "Current Assets")
                    inv_field = self._find_field_in_index(abs_data, "Inventory")
                    cl_field = self._find_field_in_index(abs_data, "Current Liabilities")
                    if ca_field and cl_field:
                        for i in range(min(3, len(abs_data.columns))):
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
                                col_name = f"Q {qbs.columns[i].strftime('%m/%d/%Y')}"
                                row_data[col_name] = f"{tl/equity:.2f}"
            # Annual
            if "annual_balance_sheet" in data and data["annual_balance_sheet"] is not None:
                abs_data = data["annual_balance_sheet"]
                if not abs_data.empty:
                    tl_field = self._find_field_in_index(abs_data, "Total Liabilities")
                    ta_field = self._find_field_in_index(abs_data, "Total Assets")
                    if tl_field and ta_field:
                        for i in range(min(3, len(abs_data.columns))):
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
                                col_name = f"Q {qi.columns[i].strftime('%m/%d/%Y')}"
                                row_data[col_name] = f"{(gross_profit/revenue)*100:.1f}%"
            # Annual
            if "annual_income" in data and data["annual_income"] is not None:
                ai = data["annual_income"]
                if not ai.empty:
                    rev_field = self._find_field_in_index(ai, "Total Revenue")
                    gp_field = self._find_field_in_index(ai, "Gross Profit")
                    if rev_field and gp_field:
                        for i in range(min(3, len(ai.columns))):
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
                                col_name = f"Q {qi.columns[i].strftime('%m/%d/%Y')}"
                                row_data[col_name] = f"{(op_income/revenue)*100:.1f}%"
            # Annual
            if "annual_income" in data and data["annual_income"] is not None:
                ai = data["annual_income"]
                if not ai.empty:
                    rev_field = self._find_field_in_index(ai, "Total Revenue")
                    oi_field = self._find_field_in_index(ai, "Operating Income")
                    if rev_field and oi_field:
                        for i in range(min(3, len(ai.columns))):
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
                                col_name = f"Q {qi.columns[i].strftime('%m/%d/%Y')}"
                                row_data[col_name] = f"{(net_income/revenue)*100:.1f}%"
            # Annual
            if "annual_income" in data and data["annual_income"] is not None:
                ai = data["annual_income"]
                if not ai.empty:
                    rev_field = self._find_field_in_index(ai, "Total Revenue")
                    ni_field = self._find_field_in_index(ai, "Net Income")
                    if rev_field and ni_field:
                        for i in range(min(3, len(ai.columns))):
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
                        col_name = f"Q {qcf.columns[i].strftime('%m/%d/%Y')}"
                        row_data[col_name] = self._format_value(ocf + icf + fcf)
            # Annual
            if "annual_cashflow" in data and data["annual_cashflow"] is not None:
                acf = data["annual_cashflow"]
                if not acf.empty:
                    ocf_field = self._find_field_in_index(acf, "Operating Cash Flow")
                    icf_field = self._find_field_in_index(acf, "Investing Cash Flow")
                    fcf_field = self._find_field_in_index(acf, "Financing Cash Flow")
                    for i in range(min(3, len(acf.columns))):
                        ocf = acf.loc[ocf_field].iloc[i] if ocf_field else 0
                        icf = acf.loc[icf_field].iloc[i] if icf_field else 0
                        fcf = acf.loc[fcf_field].iloc[i] if fcf_field else 0
                        col_name = f"FY {acf.columns[i].year}"
                        row_data[col_name] = self._format_value(ocf + icf + fcf)
        except Exception as e:
            st.warning(f"Could not calculate net cash flow: {str(e)}")
        return row_data

def display_financial_statements(financial_data: Dict[str, pd.DataFrame], ticker: str):
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
        fetcher = FinancialDataFetcher(ticker)
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

def display_risk_analysis(risk_data: str):
    """Display AI-analyzed risks"""
    st.header("⚠️ Risk Analysis")
    st.caption("AI-generated analysis from 10-Q document")

    st.markdown(risk_data)

def display_liquidity_analysis(liquidity_data: str):
    """Display AI-analyzed liquidity position"""
    st.header("💧 Liquidity Analysis")
    st.caption("AI-generated analysis from 10-Q document")

    st.markdown(liquidity_data)

def display_profitability_analysis(profitability_data: str):
    """Display AI-analyzed profitability"""
    st.header("📈 Profitability Analysis")
    st.caption("AI-generated analysis from 10-Q document")

    st.markdown(f"""
    <div class="ai-summary">
        {profitability_data}
    </div>
    """, unsafe_allow_html=True)

def display_cashflow_analysis(cashflow_data: str):
    """Display AI-analyzed cash flow"""
    st.header("💰 Cash Flow Analysis")
    st.caption("AI-generated analysis from 10-Q document")

    st.markdown(f"""
    <div class="ai-summary">
        {cashflow_data}
    </div>
    """, unsafe_allow_html=True)

def generate_print_report(ticker: str, all_data: Dict) -> str:
    """Generate HTML report for printing"""
    current_date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    # Build HTML with inline styles
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
            .analysis-section {{
                background-color: #f8f9fa;
                padding: 15px;
                border-radius: 5px;
                border: 1px solid #dee2e6;
                margin: 10px 0;
            }}
        </style>
    </head>
    <body>
        <h1>Financial Analysis Report - {ticker}</h1>
        <p><em>Generated on {current_date}</em></p>
        <p><strong>Analysis Method: Artificial intelligence</strong></p>

        <div class="section-header">
            <h2>1. Financial Statements</h2>
        </div>
        {all_data['tables']}

        <div class="section-header">
            <h2>2. Risk Analysis</h2>
        </div>
        <div class="analysis-section">
            {all_data['risk_analysis']}
        </div>

        <div class="section-header">
            <h2>3. Liquidity Analysis</h2>
        </div>
        <div class="analysis-section">
            {all_data['liquidity_analysis']}
        </div>

        <div class="section-header">
            <h2>4. Profitability Analysis</h2>
        </div>
        <div class="analysis-section">
            {all_data['profitability_analysis']}
        </div>

        <div class="section-header">
            <h2>5. Cash Flow Analysis</h2>
        </div>
        <div class="analysis-section">
            {all_data['cashflow_analysis']}
        </div>
    </body>
    </html>
    """
    return html

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
            with st.spinner("Analyzing risks using AI with vector embeddings..."):
                risk_analysis = analyzer.analyze_risks(raw_text)

            with st.spinner("Analyzing liquidity using vector embeddings..."):
                liquidity_analysis = analyzer.analyze_liquidity(raw_text)

            with st.spinner("Analyzing profitability using vector embeddings..."):
                profitability_analysis = analyzer.analyze_profitability(raw_text)

            with st.spinner("Analyzing cash flow using vector embeddings..."):
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