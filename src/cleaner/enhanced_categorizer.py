#!/usr/bin/env python3
"""
Enhanced Categorization Module
Advanced categorization using multiple techniques for improved accuracy.
"""

import pandas as pd
import numpy as np
import re
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
import logging
from difflib import SequenceMatcher

class EnhancedCategorizer:
    """
    Advanced categorization using multiple techniques:
    - Vendor database matching
    - Multi-dimensional analysis
    - Pattern recognition
    - Time-based analysis
    - Frequency analysis
    - Amount-based categorization
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Comprehensive vendor database with categories
        self.vendor_database = {
            # Software & Technology
            'software': {
                'vendors': [
                    'microsoft', 'adobe', 'google', 'salesforce', 'zoom', 'slack', 'dropbox',
                    'box', 'github', 'gitlab', 'atlassian', 'jira', 'confluence', 'trello',
                    'monday', 'notion', 'evernote', 'intuit', 'quickbooks', 'sage', 'xero',
                    'freshbooks', 'wave', 'zoho', 'hubspot', 'mailchimp', 'constant contact',
                    'surveymonkey', 'typeform', 'calendly', 'google ads', 'facebook ads',
                    'linkedin ads', 'twitter ads', 'instagram ads', 'youtube ads', 'bing ads',
                    'tiktok ads', 'snapchat ads', 'aws', 'google cloud', 'azure', 'digitalocean',
                    'heroku', 'vercel', 'netlify', 'cloudflare', 'akamai', 'fastly', 'cdn77',
                    'bunny cdn', 'keycdn', 'stackpath', 'discord', 'teams', 'meet', 'skype',
                    'whatsapp', 'telegram', 'signal', 'viber', 'wechat', 'line', 'kakao',
                    'snap', 'linkedin', 'facebook', 'twitter', 'instagram', 'tiktok', 'youtube',
                    'vimeo', 'dailymotion', 'twitch', 'reddit', 'pinterest', 'tumblr', 'medium',
                    'quora', 'spotify', 'apple music', 'pandora', 'soundcloud', 'tidal',
                    'amazon music', 'youtube music', 'deezer', 'napster', 'iheartradio',
                    'siriusxm', 'audible', 'scribd', 'kindle', 'kobo'
                ],
                'patterns': [r'software', r'license', r'subscription', r'cloud', r'saas', r'app'],
                'amount_ranges': [(10, 500), (50, 2000)],  # Typical software costs
                'frequency': 'monthly'  # Often recurring
            },
            
            # Office & Supplies
            'office': {
                'vendors': [
                    'staples', 'office depot', 'amazon', 'walmart', 'target', 'costco',
                    'best buy', 'dell', 'hp', 'lenovo', 'apple', 'samsung', 'canon', 'epson',
                    'brother', 'home depot', 'lowes', 'menards', 'ace hardware', 'true value',
                    'sherwin-williams', 'behr', 'benjamin moore', 'valspar', 'ppg'
                ],
                'patterns': [r'office', r'supplies', r'equipment', r'hardware', r'stationery'],
                'amount_ranges': [(5, 300), (20, 1000)],  # Office supply costs
                'frequency': 'variable'
            },
            
            # Travel & Transportation
            'travel': {
                'vendors': [
                    'uber', 'lyft', 'airbnb', 'expedia', 'booking', 'kayak', 'orbitz',
                    'priceline', 'marriott', 'hilton', 'hyatt', 'sheraton', 'westin',
                    'radisson', 'holiday inn', 'hotel', 'airline', 'delta', 'united',
                    'american airlines', 'southwest', 'jetblue', 'alaska', 'spirit',
                    'frontier', 'taxi', 'limo', 'rental car', 'hertz', 'avis', 'enterprise',
                    'budget', 'national', 'alamo', 'thrifty', 'dollar', 'zipcar'
                ],
                'patterns': [r'travel', r'transport', r'ride', r'hotel', r'airline', r'flight'],
                'amount_ranges': [(15, 500), (50, 2000)],  # Travel costs
                'frequency': 'variable'
            },
            
            # Utilities & Services
            'utilities': {
                'vendors': [
                    'verizon', 'at&t', 'comcast', 'spectrum', 'cox', 'centurylink', 'sprint',
                    't-mobile', 'dish', 'directv', 'netflix', 'spotify', 'hulu', 'disney',
                    'hbo', 'electric', 'gas', 'water', 'internet', 'phone', 'cable',
                    'satellite', 'security', 'alarm', 'monitoring'
                ],
                'patterns': [r'utility', r'bill', r'service', r'provider', r'company'],
                'amount_ranges': [(20, 200), (50, 500)],  # Utility costs
                'frequency': 'monthly'
            },
            
            # Food & Entertainment
            'food': {
                'vendors': [
                    'mcdonalds', 'subway', 'pizza hut', 'dominos', 'kfc', 'burger king',
                    'taco bell', 'chipotle', 'starbucks', 'dunkin', 'grubhub', 'doordash',
                    'uber eats', 'postmates', 'restaurant', 'cafe', 'diner', 'bar',
                    'pub', 'brewery', 'winery', 'liquor', 'grocery', 'supermarket'
                ],
                'patterns': [r'food', r'restaurant', r'cafe', r'dining', r'meal', r'lunch'],
                'amount_ranges': [(5, 100), (10, 200)],  # Food costs
                'frequency': 'variable'
            },
            
            # Professional Services
            'professional': {
                'vendors': [
                    'deloitte', 'pwc', 'ey', 'kpmg', 'mckinsey', 'bain', 'bcg', 'accenture',
                    'ibm', 'oracle', 'sap', 'cisco', 'intel', 'amd', 'lawyer', 'attorney',
                    'accountant', 'consultant', 'advisor', 'planner', 'therapist', 'doctor',
                    'dentist', 'chiropractor', 'massage', 'spa', 'salon', 'barber'
                ],
                'patterns': [r'professional', r'service', r'consulting', r'legal', r'medical'],
                'amount_ranges': [(100, 2000), (200, 5000)],  # Professional service costs
                'frequency': 'variable'
            },
            
            # Financial Services
            'financial': {
                'vendors': [
                    'jp morgan', 'chase', 'bank of america', 'wells fargo', 'citibank',
                    'us bank', 'pnc', 'capital one', 'amex', 'paypal', 'stripe', 'square',
                    'venmo', 'zelle', 'western union', 'moneygram', 'fidelity', 'vanguard',
                    'schwab', 'td ameritrade', 'etrade', 'robinhood', 'coinbase', 'kraken'
                ],
                'patterns': [r'bank', r'financial', r'credit', r'loan', r'investment'],
                'amount_ranges': [(10, 1000), (50, 10000)],  # Financial transaction costs
                'frequency': 'variable'
            },
            
            # Insurance & Healthcare
            'insurance': {
                'vendors': [
                    'aetna', 'blue cross', 'cigna', 'unitedhealth', 'humana', 'kaiser',
                    'anthem', 'metlife', 'state farm', 'allstate', 'geico', 'progressive',
                    'farmers', 'liberty mutual', 'nationwide', 'health', 'medical', 'dental',
                    'vision', 'pharmacy', 'cvs', 'walgreens', 'rite aid', 'pharmacy'
                ],
                'patterns': [r'insurance', r'health', r'medical', r'pharmacy', r'prescription'],
                'amount_ranges': [(20, 500), (50, 2000)],  # Insurance costs
                'frequency': 'monthly'
            },
            
            # Manufacturing & Industrial
            'manufacturing': {
                'vendors': [
                    'ge', '3m', 'caterpillar', 'deere', 'boeing', 'lockheed', 'raytheon',
                    'honeywell', 'emerson', 'rockwell', 'siemens', 'abb', 'schneider',
                    'mitsubishi', 'komatsu', 'hitachi', 'volvo', 'construction', 'industrial'
                ],
                'patterns': [r'manufacturing', r'industrial', r'construction', r'equipment'],
                'amount_ranges': [(500, 10000), (1000, 50000)],  # Manufacturing costs
                'frequency': 'variable'
            },
            
            # Energy & Utilities
            'energy': {
                'vendors': [
                    'exxon', 'chevron', 'shell', 'bp', 'conoco', 'occidental', 'marathon',
                    'valero', 'phillips', 'kinder', 'williams', 'enterprise', 'energy',
                    'oil', 'gas', 'electric', 'power', 'utility'
                ],
                'patterns': [r'energy', r'oil', r'gas', r'power', r'utility'],
                'amount_ranges': [(50, 1000), (100, 5000)],  # Energy costs
                'frequency': 'monthly'
            },
            
            # Retail & E-commerce
            'retail': {
                'vendors': [
                    'amazon', 'walmart', 'target', 'costco', 'best buy', 'kroger',
                    'albertsons', 'publix', 'safeway', 'winn-dixie', 'retail', 'store',
                    'shop', 'market', 'mall', 'outlet', 'discount'
                ],
                'patterns': [r'retail', r'store', r'shop', r'market', r'purchase'],
                'amount_ranges': [(10, 500), (25, 2000)],  # Retail costs
                'frequency': 'variable'
            }
        }
        
        # Amount-based categorization rules
        self.amount_categories = {
            'small_purchase': {'range': (0, 50), 'category': 'Small Purchase'},
            'medium_purchase': {'range': (50, 200), 'category': 'Medium Purchase'},
            'large_purchase': {'range': (200, 1000), 'category': 'Large Purchase'},
            'major_expense': {'range': (1000, 10000), 'category': 'Major Expense'},
            'capital_expense': {'range': (10000, float('inf')), 'category': 'Capital Expense'}
        }
        
        # Time-based patterns
        self.time_patterns = {
            'monthly': ['subscription', 'recurring', 'monthly', 'billing'],
            'quarterly': ['quarterly', 'quarter', 'q1', 'q2', 'q3', 'q4'],
            'annual': ['annual', 'yearly', 'year-end', 'renewal'],
            'one_time': ['one-time', 'setup', 'installation', 'initial']
        }
    
    def categorize_enhanced(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Enhanced categorization using multiple techniques.
        
        Args:
            df: Input DataFrame with vendor and amount columns
            
        Returns:
            DataFrame with enhanced categorization
        """
        self.logger.info("🏷️ Starting enhanced categorization...")
        
        # Ensure we have required columns
        if 'vendor' not in df.columns:
            self.logger.warning("⚠️ No vendor column found, using basic categorization")
            return self._basic_categorization(df)
        
        # Step 1: Vendor database matching
        df = self._vendor_database_categorization(df)
        self.logger.info(f"📊 Vendor database categorization: {df['category'].value_counts().to_dict()}")
        
        # Step 2: Multi-dimensional analysis
        df = self._multi_dimensional_categorization(df)
        self.logger.info(f"🔍 Multi-dimensional categorization: {df['category'].value_counts().to_dict()}")
        
        # Step 3: Amount-based categorization
        df = self._amount_based_categorization(df)
        self.logger.info(f"💰 Amount-based categorization: {df['category'].value_counts().to_dict()}")
        
        # Step 4: Time-based analysis
        df = self._time_based_categorization(df)
        self.logger.info(f"⏰ Time-based categorization: {df['category'].value_counts().to_dict()}")
        
        # Step 5: Frequency analysis
        df = self._frequency_based_categorization(df)
        self.logger.info(f"📈 Frequency-based categorization: {df['category'].value_counts().to_dict()}")
        
        # Step 6: Pattern recognition
        df = self._pattern_based_categorization(df)
        self.logger.info(f"🔍 Pattern-based categorization: {df['category'].value_counts().to_dict()}")
        
        # Step 7: Final consolidation and scoring
        df = self._consolidate_categories(df)
        self.logger.info(f"✅ Final categorization: {df['category'].value_counts().to_dict()}")
        
        return df
    
    def _vendor_database_categorization(self, df: pd.DataFrame) -> pd.DataFrame:
        """Categorize based on vendor database matching."""
        df['category'] = 'Other'  # Default category
        
        if 'vendor' not in df.columns:
            return df
        
        # Create vendor matching function
        def match_vendor(vendor):
            if pd.isna(vendor):
                return 'Other'
            
            vendor_lower = str(vendor).lower()
            
            # Check each category
            for category, data in self.vendor_database.items():
                # Check vendor names
                for vendor_name in data['vendors']:
                    if vendor_name in vendor_lower:
                        return category.title()
                
                # Check patterns
                for pattern in data['patterns']:
                    if re.search(pattern, vendor_lower, re.IGNORECASE):
                        return category.title()
            
            return 'Other'
        
        # Apply vendor matching
        df['category'] = df['vendor'].apply(match_vendor)
        
        return df
    
    def _multi_dimensional_categorization(self, df: pd.DataFrame) -> pd.DataFrame:
        """Categorize using multiple dimensions (vendor, amount, description)."""
        if 'amount' not in df.columns:
            return df
        
        # Create multi-dimensional scoring
        def score_category(row):
            vendor = str(row.get('vendor', '')).lower()
            amount = abs(row.get('amount', 0))
            description = str(row.get('memo', '')).lower()
            
            best_category = 'Other'
            best_score = 0
            
            for category, data in self.vendor_database.items():
                score = 0
                
                # Vendor matching score
                for vendor_name in data['vendors']:
                    if vendor_name in vendor:
                        score += 2
                        break
                
                # Pattern matching score
                for pattern in data['patterns']:
                    if re.search(pattern, vendor, re.IGNORECASE):
                        score += 1
                    if re.search(pattern, description, re.IGNORECASE):
                        score += 1
                
                # Amount range score
                for min_amt, max_amt in data['amount_ranges']:
                    if min_amt <= amount <= max_amt:
                        score += 1
                        break
                
                if score > best_score:
                    best_score = score
                    best_category = category.title()
            
            return best_category if best_score > 0 else 'Other'
        
        # Apply multi-dimensional categorization
        df['category'] = df.apply(score_category, axis=1)
        
        return df
    
    def _amount_based_categorization(self, df: pd.DataFrame) -> pd.DataFrame:
        """Categorize based on amount ranges."""
        if 'amount' not in df.columns:
            return df
        
        def categorize_by_amount(amount):
            amount = abs(amount)
            
            for category_name, category_data in self.amount_categories.items():
                min_amt, max_amt = category_data['range']
                if min_amt <= amount <= max_amt:
                    return category_data['category']
            
            return 'Other'
        
        # Only categorize if current category is 'Other'
        mask = df['category'] == 'Other'
        df.loc[mask, 'category'] = df.loc[mask, 'amount'].apply(categorize_by_amount)
        
        return df
    
    def _time_based_categorization(self, df: pd.DataFrame) -> pd.DataFrame:
        """Categorize based on time patterns and frequency."""
        if 'memo' not in df.columns:
            return df
        
        def analyze_time_patterns(memo):
            memo_lower = str(memo).lower()
            
            for frequency, patterns in self.time_patterns.items():
                for pattern in patterns:
                    if pattern in memo_lower:
                        if frequency == 'monthly':
                            return 'Recurring Monthly'
                        elif frequency == 'quarterly':
                            return 'Recurring Quarterly'
                        elif frequency == 'annual':
                            return 'Recurring Annual'
                        elif frequency == 'one_time':
                            return 'One-Time Expense'
            
            return None
        
        # Apply time-based categorization for 'Other' categories
        mask = df['category'] == 'Other'
        time_categories = df.loc[mask, 'memo'].apply(analyze_time_patterns)
        df.loc[mask & (time_categories.notna()), 'category'] = time_categories[mask & (time_categories.notna())]
        
        return df
    
    def _frequency_based_categorization(self, df: pd.DataFrame) -> pd.DataFrame:
        """Categorize based on vendor frequency and patterns."""
        if 'vendor' not in df.columns:
            return df
        
        # Analyze vendor frequency
        vendor_counts = df['vendor'].value_counts()
        
        def categorize_by_frequency(vendor):
            if pd.isna(vendor):
                return 'Other'
            
            count = vendor_counts.get(vendor, 0)
            
            if count > 10:
                return 'Frequent Vendor'
            elif count > 5:
                return 'Regular Vendor'
            elif count > 1:
                return 'Occasional Vendor'
            else:
                return 'One-Time Vendor'
        
        # Apply frequency categorization for 'Other' categories
        mask = df['category'] == 'Other'
        df.loc[mask, 'category'] = df.loc[mask, 'vendor'].apply(categorize_by_frequency)
        
        return df
    
    def _pattern_based_categorization(self, df: pd.DataFrame) -> pd.DataFrame:
        """Categorize based on text patterns in vendor and memo fields."""
        def analyze_patterns(row):
            vendor = str(row.get('vendor', '')).lower()
            memo = str(row.get('memo', '')).lower()
            combined_text = f"{vendor} {memo}"
            
            # Business type patterns
            if any(word in combined_text for word in ['inc', 'corp', 'llc', 'ltd', 'co']):
                return 'Business Services'
            
            # Industry patterns
            if any(word in combined_text for word in ['tech', 'software', 'digital', 'online']):
                return 'Technology'
            elif any(word in combined_text for word in ['food', 'restaurant', 'cafe', 'dining']):
                return 'Food & Dining'
            elif any(word in combined_text for word in ['travel', 'hotel', 'airline', 'transport']):
                return 'Travel & Transport'
            elif any(word in combined_text for word in ['health', 'medical', 'dental', 'pharmacy']):
                return 'Healthcare'
            elif any(word in combined_text for word in ['bank', 'financial', 'credit', 'loan']):
                return 'Financial Services'
            elif any(word in combined_text for word in ['insurance', 'coverage', 'policy']):
                return 'Insurance'
            elif any(word in combined_text for word in ['utility', 'electric', 'gas', 'water']):
                return 'Utilities'
            elif any(word in combined_text for word in ['office', 'supplies', 'equipment']):
                return 'Office & Supplies'
            
            return None
        
        # Apply pattern categorization for 'Other' categories
        mask = df['category'] == 'Other'
        pattern_categories = df.loc[mask].apply(analyze_patterns, axis=1)
        df.loc[mask & (pattern_categories.notna()), 'category'] = pattern_categories[mask & (pattern_categories.notna())]
        
        return df
    
    def _consolidate_categories(self, df: pd.DataFrame) -> pd.DataFrame:
        """Consolidate and clean up categories."""
        # Standardize category names
        category_mapping = {
            'Software': 'Technology',
            'Office': 'Office & Supplies',
            'Travel': 'Travel & Transport',
            'Utilities': 'Utilities',
            'Food': 'Food & Dining',
            'Professional': 'Professional Services',
            'Financial': 'Financial Services',
            'Insurance': 'Insurance',
            'Manufacturing': 'Manufacturing',
            'Energy': 'Energy & Utilities',
            'Retail': 'Retail & E-commerce'
        }
        
        df['category'] = df['category'].replace(category_mapping)
        
        # Ensure no empty categories
        df['category'] = df['category'].fillna('Other')
        
        return df
    
    def _basic_categorization(self, df: pd.DataFrame) -> pd.DataFrame:
        """Basic categorization as fallback."""
        if 'category' not in df.columns:
            df['category'] = 'Other'
        
        return df 