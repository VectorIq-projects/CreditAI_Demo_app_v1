from Config_file import ( 
    AZURE_OPENAI_KEY, 
    AZURE_OPENAI_ENDPOINT, 
    AZURE_OPENAI_API_VERSION, 
    AZURE_OPENAI_DEPLOYMENT, 
    AZURE_OPENAI_EMBEDDING_DEPLOYMENT,
    logger
    )

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
                results.append(f"{currency_response.strip()}.")
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