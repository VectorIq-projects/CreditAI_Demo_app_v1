import streamlit as st
import pandas as pd
# import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
# import PyPDF2
# import io
# import base64
from typing import Dict, List, Optional, Tuple
# import json
# import re
# import logging
# import docx2txt
# import requests
# import os
# import mimetypes

# Azure OpenAI imports
# from openai import AzureOpenAI
# import tiktoken

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
