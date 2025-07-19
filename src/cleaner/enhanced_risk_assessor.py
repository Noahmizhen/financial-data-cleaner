#!/usr/bin/env python3
"""
Enhanced Risk Assessment Module
Advanced risk assessment using multiple techniques for improved accuracy.
Conservative thresholds with accuracy-first approach.
"""

import pandas as pd
import numpy as np
import re
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime, timedelta
import logging
from collections import defaultdict, Counter
import warnings

class RiskConfig:
    """Configuration for conservative risk assessment."""
    
    # Conservative thresholds
    AMOUNT_OUTLIER_THRESHOLD = 3.0  # 3 standard deviations
    FREQUENCY_ANOMALY_THRESHOLD = 2.5  # 2.5x normal frequency
    VENDOR_CONCENTRATION_THRESHOLD = 0.3  # 30% vendor concentration
    TEMPORAL_ANOMALY_THRESHOLD = 2.0  # 2x deviation from normal
    ROUND_NUMBER_THRESHOLD = 0.1  # 10% tolerance for round numbers
    
    # Accuracy-first settings
    MIN_CONFIDENCE_THRESHOLD = 0.8  # High confidence required
    MAX_PROCESSING_TIME = 300  # 5 minutes max
    DETAILED_ANALYSIS = True  # Comprehensive analysis
    
    # Risk scoring weights
    STATISTICAL_WEIGHT = 0.3
    VENDOR_WEIGHT = 0.25
    PATTERN_WEIGHT = 0.25
    TEMPORAL_WEIGHT = 0.2

class StatisticalRiskAnalyzer:
    """Analyze statistical anomalies in transaction data."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.config = RiskConfig()
    
    def analyze_amount_anomalies(self, df: pd.DataFrame) -> List[Dict]:
        """Detect unusual amount patterns and outliers."""
        risk_flags = []
        
        if 'amount' not in df.columns:
            return risk_flags
        
        try:
            amounts = df['amount'].abs()
            
            # Calculate statistical measures
            mean_amount = amounts.mean()
            std_amount = amounts.std()
            median_amount = amounts.median()
            
            # Detect outliers (3+ standard deviations)
            outlier_threshold = mean_amount + (self.config.AMOUNT_OUTLIER_THRESHOLD * std_amount)
            outliers = amounts[amounts > outlier_threshold]
            
            for idx, amount in outliers.items():
                risk_flags.append({
                    'type': 'amount_outlier',
                    'severity': 'HIGH',
                    'confidence': min(0.95, (amount - outlier_threshold) / std_amount),
                    'description': f'Unusual amount: ${amount:,.2f} (threshold: ${outlier_threshold:,.2f})',
                    'transaction_id': idx,
                    'amount': amount,
                    'vendor': df.loc[idx, 'vendor'] if 'vendor' in df.columns else 'Unknown',
                    'date': df.loc[idx, 'date'] if 'date' in df.columns else 'Unknown'
                })
            
            # Detect unusual amount distributions
            if len(amounts) > 10:
                # Check for unusual amount clustering
                amount_bins = pd.cut(amounts, bins=10)
                bin_counts = amount_bins.value_counts()
                
                # Flag if any bin has unusually high concentration
                expected_bin_count = len(amounts) / 10
                high_concentration_bins = bin_counts[bin_counts > expected_bin_count * 2]
                
                for bin_name, count in high_concentration_bins.items():
                    risk_flags.append({
                        'type': 'amount_clustering',
                        'severity': 'MEDIUM',
                        'confidence': 0.8,
                        'description': f'Unusual amount clustering: {count} transactions in range {bin_name}',
                        'bin_range': str(bin_name),
                        'count': count,
                        'expected': expected_bin_count
                    })
            
            # Detect round number patterns
            round_number_flags = self._detect_round_numbers(amounts, df)
            risk_flags.extend(round_number_flags)
            
        except Exception as e:
            self.logger.error(f"Amount anomaly analysis failed: {e}")
        
        return risk_flags
    
    def _detect_round_numbers(self, amounts: pd.Series, df: pd.DataFrame) -> List[Dict]:
        """Detect suspicious round number transactions."""
        risk_flags = []
        
        # Common round number patterns (more conservative)
        round_patterns = [
            (1000, 10000), # $1000, $2000, $3000, etc.
            (5000, 50000)  # $5000, $10000, $15000, etc.
        ]
        
        for base, max_amount in round_patterns:
            for multiplier in range(1, int(max_amount / base) + 1):
                round_amount = base * multiplier
                tolerance = round_amount * self.config.ROUND_NUMBER_THRESHOLD
                
                # Find transactions close to round numbers
                close_transactions = amounts[
                    (amounts >= round_amount - tolerance) & 
                    (amounts <= round_amount + tolerance)
                ]
                
                if len(close_transactions) > 0:
                    # Only flag if multiple transactions or very large amounts
                    if len(close_transactions) > 3 or round_amount > 5000:
                        for idx in close_transactions.index:
                            risk_flags.append({
                                'type': 'round_number',
                                'severity': 'MEDIUM',
                                'confidence': 0.7,
                                'description': f'Suspicious round number: ${close_transactions[idx]:,.2f} (close to ${round_amount:,.2f})',
                                'transaction_id': idx,
                                'amount': close_transactions[idx],
                                'target_round': round_amount,
                                'vendor': df.loc[idx, 'vendor'] if 'vendor' in df.columns else 'Unknown'
                            })
        
        return risk_flags
    
    def analyze_frequency_anomalies(self, df: pd.DataFrame) -> List[Dict]:
        """Detect unusual transaction frequency patterns."""
        risk_flags = []
        
        if 'date' not in df.columns:
            return risk_flags
        
        try:
            # Convert dates to datetime
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
            df = df.dropna(subset=['date'])
            
            if len(df) == 0:
                return risk_flags
            
            # Analyze daily frequency
            daily_counts = df.groupby(df['date'].dt.date).size()
            
            if len(daily_counts) > 0:
                mean_daily = daily_counts.mean()
                std_daily = daily_counts.std()
                
                # Detect days with unusually high transaction counts
                high_frequency_threshold = mean_daily + (self.config.FREQUENCY_ANOMALY_THRESHOLD * std_daily)
                high_frequency_days = daily_counts[daily_counts > high_frequency_threshold]
                
                for date, count in high_frequency_days.items():
                    risk_flags.append({
                        'type': 'frequency_anomaly',
                        'severity': 'HIGH',
                        'confidence': min(0.9, (count - high_frequency_threshold) / std_daily),
                        'description': f'Unusual transaction frequency: {count} transactions on {date} (avg: {mean_daily:.1f})',
                        'date': str(date),
                        'count': count,
                        'expected': mean_daily
                    })
            
            # Analyze vendor frequency
            if 'vendor' in df.columns:
                vendor_counts = df['vendor'].value_counts()
                mean_vendor_freq = vendor_counts.mean()
                std_vendor_freq = vendor_counts.std()
                
                # Detect vendors with unusually high frequency
                high_vendor_threshold = mean_vendor_freq + (self.config.FREQUENCY_ANOMALY_THRESHOLD * std_vendor_freq)
                high_frequency_vendors = vendor_counts[vendor_counts > high_vendor_threshold]
                
                for vendor, count in high_frequency_vendors.items():
                    risk_flags.append({
                        'type': 'vendor_frequency_anomaly',
                        'severity': 'MEDIUM',
                        'confidence': min(0.8, (count - high_vendor_threshold) / std_vendor_freq),
                        'description': f'Unusual vendor frequency: {vendor} has {count} transactions (avg: {mean_vendor_freq:.1f})',
                        'vendor': vendor,
                        'count': count,
                        'expected': mean_vendor_freq
                    })
            
        except Exception as e:
            self.logger.error(f"Frequency anomaly analysis failed: {e}")
        
        return risk_flags
    
    def analyze_temporal_anomalies(self, df: pd.DataFrame) -> List[Dict]:
        """Detect unusual temporal patterns."""
        risk_flags = []
        
        if 'date' not in df.columns:
            return risk_flags
        
        try:
            # Convert dates to datetime
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
            df = df.dropna(subset=['date'])
            
            if len(df) == 0:
                return risk_flags
            
            # Analyze day-of-week patterns
            df['day_of_week'] = df['date'].dt.day_name()
            day_counts = df['day_of_week'].value_counts()
            
            # Detect unusual day-of-week patterns
            if len(day_counts) > 0:
                mean_day_count = day_counts.mean()
                std_day_count = day_counts.std()
                
                # Flag unusual weekend activity
                weekend_days = ['Saturday', 'Sunday']
                for day in weekend_days:
                    if day in day_counts:
                        weekend_count = day_counts[day]
                        if weekend_count > mean_day_count * self.config.TEMPORAL_ANOMALY_THRESHOLD:
                            risk_flags.append({
                                'type': 'weekend_activity',
                                'severity': 'MEDIUM',
                                'confidence': 0.7,
                                'description': f'Unusual {day} activity: {weekend_count} transactions (avg: {mean_day_count:.1f})',
                                'day': day,
                                'count': weekend_count,
                                'expected': mean_day_count
                            })
            
            # Analyze time-of-day patterns (if available)
            if 'time' in df.columns or 'date' in df.columns:
                # Extract hour from date if time not available
                if 'time' not in df.columns:
                    df['hour'] = df['date'].dt.hour
                else:
                    df['hour'] = pd.to_datetime(df['time']).dt.hour
                
                # Detect unusual after-hours activity
                after_hours = df[df['hour'] >= 22]  # After 10 PM
                if len(after_hours) > 0:
                    after_hours_count = len(after_hours)
                    total_count = len(df)
                    after_hours_ratio = after_hours_count / total_count
                    
                    if after_hours_ratio > 0.1:  # More than 10% after hours
                        risk_flags.append({
                            'type': 'after_hours_activity',
                            'severity': 'MEDIUM',
                            'confidence': 0.8,
                            'description': f'Unusual after-hours activity: {after_hours_count} transactions after 10 PM ({after_hours_ratio:.1%} of total)',
                            'count': after_hours_count,
                            'ratio': after_hours_ratio
                        })
            
        except Exception as e:
            self.logger.error(f"Temporal anomaly analysis failed: {e}")
        
        return risk_flags

class VendorRiskAnalyzer:
    """Analyze vendor-related risks."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.config = RiskConfig()
        
        # Basic vendor risk database
        self.known_vendors = {
            'safe': [
                'microsoft', 'adobe', 'google', 'salesforce', 'zoom', 'slack',
                'staples', 'office depot', 'amazon', 'walmart', 'target',
                'uber', 'lyft', 'airbnb', 'marriott', 'hilton',
                'verizon', 'at&t', 'comcast', 'netflix', 'spotify',
                'mcdonalds', 'starbucks', 'chipotle', 'grubhub', 'doordash',
                'jp morgan', 'chase', 'bank of america', 'wells fargo',
                'aetna', 'blue cross', 'cigna', 'unitedhealth'
            ],
            'high_risk': [
                'unknown', 'test', 'sample', 'placeholder', 'temp',
                'cash', 'atm', 'withdrawal', 'deposit'
            ]
        }
    
    def assess_vendor_risk(self, df: pd.DataFrame) -> List[Dict]:
        """Assess vendor-related risks."""
        risk_flags = []
        
        if 'vendor' not in df.columns:
            return risk_flags
        
        try:
            # Analyze vendor concentration
            vendor_counts = df['vendor'].value_counts()
            total_transactions = len(df)
            
            # Detect high vendor concentration
            for vendor, count in vendor_counts.items():
                concentration = count / total_transactions
                
                if concentration > self.config.VENDOR_CONCENTRATION_THRESHOLD:
                    risk_flags.append({
                        'type': 'vendor_concentration',
                        'severity': 'HIGH',
                        'confidence': min(0.9, concentration),
                        'description': f'High vendor concentration: {vendor} represents {concentration:.1%} of transactions',
                        'vendor': vendor,
                        'count': count,
                        'concentration': concentration,
                        'total_transactions': total_transactions
                    })
            
            # Detect unknown vendors
            unknown_vendors = self._identify_unknown_vendors(df)
            for vendor_info in unknown_vendors:
                risk_flags.append({
                    'type': 'unknown_vendor',
                    'severity': 'MEDIUM',
                    'confidence': 0.8,
                    'description': f'Unknown vendor: {vendor_info["vendor"]} ({vendor_info["count"]} transactions)',
                    'vendor': vendor_info['vendor'],
                    'count': vendor_info['count'],
                    'total_amount': vendor_info['total_amount']
                })
            
            # Detect potentially high-risk vendors
            high_risk_flags = self._detect_high_risk_vendors(df)
            risk_flags.extend(high_risk_flags)
            
        except Exception as e:
            self.logger.error(f"Vendor risk assessment failed: {e}")
        
        return risk_flags
    
    def _identify_unknown_vendors(self, df: pd.DataFrame) -> List[Dict]:
        """Identify vendors not in our database."""
        unknown_vendors = []
        
        vendor_counts = df['vendor'].value_counts()
        
        for vendor, count in vendor_counts.items():
            vendor_lower = str(vendor).lower()
            
            # Check if vendor is in known database
            is_known = False
            for category, vendors in self.known_vendors.items():
                if any(known_vendor in vendor_lower for known_vendor in vendors):
                    is_known = True
                    break
            
            if not is_known:
                # Calculate total amount for this vendor
                vendor_transactions = df[df['vendor'] == vendor]
                total_amount = vendor_transactions['amount'].abs().sum() if 'amount' in df.columns else 0
                
                # Only flag if significant activity
                if count > 5 or total_amount > 1000:
                    unknown_vendors.append({
                        'vendor': vendor,
                        'count': count,
                        'total_amount': total_amount
                    })
        
        return unknown_vendors
    
    def _detect_high_risk_vendors(self, df: pd.DataFrame) -> List[Dict]:
        """Detect potentially high-risk vendors."""
        risk_flags = []
        
        vendor_counts = df['vendor'].value_counts()
        
        for vendor, count in vendor_counts.items():
            vendor_lower = str(vendor).lower()
            
            # Check against high-risk patterns
            for risk_pattern in self.known_vendors['high_risk']:
                if risk_pattern in vendor_lower:
                    risk_flags.append({
                        'type': 'high_risk_vendor',
                        'severity': 'HIGH',
                        'confidence': 0.9,
                        'description': f'High-risk vendor pattern detected: {vendor}',
                        'vendor': vendor,
                        'count': count,
                        'risk_pattern': risk_pattern
                    })
                    break
        
        return risk_flags

class PatternRiskAnalyzer:
    """Analyze pattern-based risks."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.config = RiskConfig()
    
    def detect_fraud_patterns(self, df: pd.DataFrame) -> List[Dict]:
        """Detect potential fraud patterns."""
        risk_flags = []
        
        try:
            # Detect sequential transaction patterns
            sequential_flags = self._detect_sequential_patterns(df)
            risk_flags.extend(sequential_flags)
            
            # Detect amount clustering patterns
            clustering_flags = self._detect_amount_clustering(df)
            risk_flags.extend(clustering_flags)
            
            # Detect vendor-amount patterns
            vendor_amount_flags = self._detect_vendor_amount_patterns(df)
            risk_flags.extend(vendor_amount_flags)
            
        except Exception as e:
            self.logger.error(f"Fraud pattern detection failed: {e}")
        
        return risk_flags
    
    def _detect_sequential_patterns(self, df: pd.DataFrame) -> List[Dict]:
        """Detect suspicious sequential transaction patterns."""
        risk_flags = []
        
        if 'date' not in df.columns or 'amount' not in df.columns:
            return risk_flags
        
        try:
            # Sort by date
            df_sorted = df.sort_values('date')
            
            # Look for identical amounts in sequence
            for i in range(len(df_sorted) - 1):
                current_amount = abs(df_sorted.iloc[i]['amount'])
                next_amount = abs(df_sorted.iloc[i + 1]['amount'])
                
                # Check for identical amounts within 24 hours
                current_date = pd.to_datetime(df_sorted.iloc[i]['date'])
                next_date = pd.to_datetime(df_sorted.iloc[i + 1]['date'])
                time_diff = (next_date - current_date).total_seconds() / 3600  # hours
                
                if current_amount == next_amount and time_diff <= 24:
                    risk_flags.append({
                        'type': 'sequential_identical_amounts',
                        'severity': 'HIGH',
                        'confidence': 0.8,
                        'description': f'Sequential identical amounts: ${current_amount:,.2f} within {time_diff:.1f} hours',
                        'amount': current_amount,
                        'time_diff_hours': time_diff,
                        'transaction_1': df_sorted.iloc[i]['vendor'] if 'vendor' in df.columns else 'Unknown',
                        'transaction_2': df_sorted.iloc[i + 1]['vendor'] if 'vendor' in df.columns else 'Unknown'
                    })
            
        except Exception as e:
            self.logger.error(f"Sequential pattern detection failed: {e}")
        
        return risk_flags
    
    def _detect_amount_clustering(self, df: pd.DataFrame) -> List[Dict]:
        """Detect suspicious amount clustering patterns."""
        risk_flags = []
        
        if 'amount' not in df.columns:
            return risk_flags
        
        try:
            amounts = df['amount'].abs()
            
            # Group amounts into clusters
            amount_clusters = defaultdict(list)
            cluster_tolerance = 0.05  # 5% tolerance
            
            for idx, amount in amounts.items():
                clustered = False
                for cluster_center in amount_clusters.keys():
                    if abs(amount - cluster_center) / cluster_center <= cluster_tolerance:
                        amount_clusters[cluster_center].append((idx, amount))
                        clustered = True
                        break
                
                if not clustered:
                    amount_clusters[amount].append((idx, amount))
            
            # Flag suspicious clusters
            for cluster_center, transactions in amount_clusters.items():
                if len(transactions) > 3:  # More than 3 similar amounts
                    risk_flags.append({
                        'type': 'amount_clustering',
                        'severity': 'MEDIUM',
                        'confidence': min(0.8, len(transactions) / 10),
                        'description': f'Amount clustering: {len(transactions)} transactions around ${cluster_center:,.2f}',
                        'cluster_center': cluster_center,
                        'transaction_count': len(transactions),
                        'transactions': transactions[:5]  # First 5 for reference
                    })
            
        except Exception as e:
            self.logger.error(f"Amount clustering detection failed: {e}")
        
        return risk_flags
    
    def _detect_vendor_amount_patterns(self, df: pd.DataFrame) -> List[Dict]:
        """Detect suspicious vendor-amount patterns."""
        risk_flags = []
        
        if 'vendor' not in df.columns or 'amount' not in df.columns:
            return risk_flags
        
        try:
            # Group by vendor and analyze amount patterns
            vendor_groups = df.groupby('vendor')
            
            for vendor, group in vendor_groups:
                if len(group) > 2:  # Only analyze vendors with multiple transactions
                    amounts = group['amount'].abs()
                    
                    # Check for unusual amount consistency
                    amount_std = amounts.std()
                    amount_mean = amounts.mean()
                    
                    if amount_std < amount_mean * 0.1:  # Very consistent amounts
                        risk_flags.append({
                            'type': 'vendor_amount_consistency',
                            'severity': 'MEDIUM',
                            'confidence': 0.7,
                            'description': f'Unusual amount consistency for {vendor}: {len(group)} transactions with low variance',
                            'vendor': vendor,
                            'transaction_count': len(group),
                            'mean_amount': amount_mean,
                            'std_amount': amount_std
                        })
            
        except Exception as e:
            self.logger.error(f"Vendor-amount pattern detection failed: {e}")
        
        return risk_flags

class RiskScorer:
    """Score and consolidate risk assessments."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.config = RiskConfig()
    
    def calculate_overall_risk_score(self, risk_flags: List[Dict]) -> Dict[str, Any]:
        """Calculate overall risk score and assessment."""
        if not risk_flags:
            return {
                'overall_risk_score': 0.0,
                'risk_level': 'LOW',
                'total_risk_flags': 0,
                'critical_risks': 0,
                'high_risks': 0,
                'medium_risks': 0,
                'low_risks': 0,
                'risk_categories': {},
                'recommendations': ['No significant risks detected'],
                'detailed_analysis': {}
            }
        
        # Categorize risks by severity
        critical_risks = [r for r in risk_flags if r['severity'] == 'CRITICAL']
        high_risks = [r for r in risk_flags if r['severity'] == 'HIGH']
        medium_risks = [r for r in risk_flags if r['severity'] == 'MEDIUM']
        low_risks = [r for r in risk_flags if r['severity'] == 'LOW']
        
        # Calculate weighted risk score
        total_score = (
            len(critical_risks) * 1.0 +
            len(high_risks) * 0.7 +
            len(medium_risks) * 0.4 +
            len(low_risks) * 0.1
        )
        
        max_possible_score = len(risk_flags) * 1.0
        overall_risk_score = min(1.0, total_score / max_possible_score) if max_possible_score > 0 else 0.0
        
        # Determine risk level
        if overall_risk_score >= 0.7:
            risk_level = 'CRITICAL'
        elif overall_risk_score >= 0.5:
            risk_level = 'HIGH'
        elif overall_risk_score >= 0.3:
            risk_level = 'MEDIUM'
        else:
            risk_level = 'LOW'
        
        # Group risks by category
        risk_categories = defaultdict(list)
        for risk in risk_flags:
            risk_categories[risk['type']].append(risk)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(risk_flags)
        
        return {
            'overall_risk_score': overall_risk_score,
            'risk_level': risk_level,
            'total_risk_flags': len(risk_flags),
            'critical_risks': len(critical_risks),
            'high_risks': len(high_risks),
            'medium_risks': len(medium_risks),
            'low_risks': len(low_risks),
            'risk_categories': dict(risk_categories),
            'recommendations': recommendations,
            'detailed_analysis': {
                'statistical_analysis': [r for r in risk_flags if 'amount' in r['type'] or 'frequency' in r['type'] or 'temporal' in r['type']],
                'vendor_analysis': [r for r in risk_flags if 'vendor' in r['type']],
                'pattern_analysis': [r for r in risk_flags if 'pattern' in r['type'] or 'clustering' in r['type'] or 'sequential' in r['type']]
            }
        }
    
    def _generate_recommendations(self, risk_flags: List[Dict]) -> List[str]:
        """Generate actionable recommendations based on risk flags."""
        recommendations = []
        
        # Group risks by type for better recommendations
        risk_types = defaultdict(list)
        for risk in risk_flags:
            risk_types[risk['type']].append(risk)
        
        # Generate specific recommendations
        if 'amount_outlier' in risk_types:
            count = len(risk_types['amount_outlier'])
            recommendations.append(f"Investigate {count} unusual amount transactions")
        
        if 'vendor_concentration' in risk_types:
            for risk in risk_types['vendor_concentration']:
                vendor = risk.get('vendor', 'Unknown')
                concentration = risk.get('concentration', 0)
                recommendations.append(f"Review vendor concentration risk with {vendor} ({concentration:.1%} of transactions)")
        
        if 'unknown_vendor' in risk_types:
            count = len(risk_types['unknown_vendor'])
            recommendations.append(f"Review {count} transactions with unknown vendors")
        
        if 'sequential_identical_amounts' in risk_types:
            count = len(risk_types['sequential_identical_amounts'])
            recommendations.append(f"Investigate {count} sequential identical amount transactions")
        
        if 'weekend_activity' in risk_types:
            count = len(risk_types['weekend_activity'])
            recommendations.append(f"Review {count} unusual weekend transactions")
        
        if 'after_hours_activity' in risk_types:
            count = len(risk_types['after_hours_activity'])
            recommendations.append(f"Review {count} after-hours transactions")
        
        # Add general recommendations if no specific ones
        if not recommendations:
            recommendations.append("Monitor transaction patterns for unusual activity")
        
        return recommendations

class EnhancedRiskAssessor:
    """
    Enhanced Risk Assessment using multiple techniques.
    Conservative thresholds with accuracy-first approach.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.config = RiskConfig()
        
        # Initialize analyzers
        self.statistical_analyzer = StatisticalRiskAnalyzer()
        self.vendor_analyzer = VendorRiskAnalyzer()
        self.pattern_analyzer = PatternRiskAnalyzer()
        self.risk_scorer = RiskScorer()
    
    def assess_risk_enhanced(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Comprehensive risk assessment using multiple techniques.
        
        Args:
            df: Input DataFrame with transaction data
            
        Returns:
            Comprehensive risk assessment report
        """
        self.logger.info("🔍 Starting enhanced risk assessment...")
        
        # Ensure we have required columns
        if 'amount' not in df.columns:
            self.logger.warning("⚠️ No amount column found, risk assessment limited")
        
        # Step 1: Statistical analysis
        self.logger.info("📊 Running statistical analysis...")
        statistical_risks = self.statistical_analyzer.analyze_amount_anomalies(df)
        statistical_risks.extend(self.statistical_analyzer.analyze_frequency_anomalies(df))
        statistical_risks.extend(self.statistical_analyzer.analyze_temporal_anomalies(df))
        
        # Step 2: Vendor analysis
        self.logger.info("🏢 Running vendor risk analysis...")
        vendor_risks = self.vendor_analyzer.assess_vendor_risk(df)
        
        # Step 3: Pattern analysis
        self.logger.info("🔍 Running pattern analysis...")
        pattern_risks = self.pattern_analyzer.detect_fraud_patterns(df)
        
        # Step 4: Consolidate all risks
        all_risks = statistical_risks + vendor_risks + pattern_risks
        
        # Step 5: Calculate overall risk assessment
        self.logger.info("📈 Calculating overall risk assessment...")
        risk_assessment = self.risk_scorer.calculate_overall_risk_score(all_risks)
        
        self.logger.info(f"✅ Risk assessment complete: {risk_assessment['risk_level']} risk level")
        
        return risk_assessment 