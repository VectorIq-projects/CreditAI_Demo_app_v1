import streamlit as st
import logging



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
