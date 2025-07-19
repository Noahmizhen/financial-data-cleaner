"""
Advanced Business Insights Engine
Analyzes spending patterns, vendor relationships, and cash flow analysis
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Any, Optional
import logging
from dataclasses import dataclass
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class SpendingPattern:
    """Represents a spending pattern analysis"""
    pattern_type: str
    description: str
    confidence: float
    impact_score: float
    recommendations: List[str]
    data_points: Dict[str, Any]

@dataclass
class VendorRelationship:
    """Represents vendor relationship analysis"""
    vendor_name: str
    total_spend: float
    transaction_count: int
    relationship_strength: float
    risk_level: str
    payment_pattern: str
    category_distribution: Dict[str, float]
    insights: List[str]

@dataclass
class CashFlowAnalysis:
    """Represents cash flow analysis results"""
    period: str
    net_cash_flow: float
    cash_inflow: float
    cash_outflow: float
    cash_flow_trend: str
    liquidity_ratio: float
    burn_rate: Optional[float]
    runway_months: Optional[float]
    insights: List[str]

@dataclass
class BusinessInsights:
    """Comprehensive business insights container"""
    spending_patterns: List[SpendingPattern]
    vendor_relationships: List[VendorRelationship]
    cash_flow_analysis: CashFlowAnalysis
    key_metrics: Dict[str, float]
    recommendations: List[str]
    risk_flags: List[str]
    opportunities: List[str]

class AdvancedBusinessInsightsEngine:
    """
    Advanced Business Insights Engine
    Provides comprehensive analysis of spending patterns, vendor relationships, and cash flow
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or self._get_default_config()
        self.analysis_cache = {}
        
    def _get_default_config(self) -> Dict:
        """Get default configuration for the insights engine"""
        return {
            'spending_analysis': {
                'min_pattern_confidence': 0.7,
                'significant_amount_threshold': 1000,
                'seasonal_analysis_months': 12,
                'trend_analysis_window': 30,
                'anomaly_detection_threshold': 2.0
            },
            'vendor_analysis': {
                'min_vendor_transactions': 3,
                'relationship_strength_threshold': 0.6,
                'high_risk_vendor_threshold': 0.8,
                'payment_terms_analysis': True,
                'vendor_categorization': True
            },
            'cash_flow_analysis': {
                'liquidity_threshold': 0.3,
                'burn_rate_calculation': True,
                'runway_analysis': True,
                'cash_flow_forecasting': True,
                'trend_analysis_periods': 6
            },
            'performance': {
                'enable_caching': True,
                'max_analysis_time': 30,
                'batch_processing_size': 1000
            }
        }
    
    def analyze_business_insights(self, df: pd.DataFrame) -> BusinessInsights:
        """
        Perform comprehensive business insights analysis
        
        Args:
            df: Cleaned financial data DataFrame
            
        Returns:
            BusinessInsights object with all analysis results
        """
        logger.info("Starting comprehensive business insights analysis")
        
        try:
            # Validate and prepare data
            df = self._prepare_data(df)
            
            # Perform core analyses
            spending_patterns = self._analyze_spending_patterns(df)
            vendor_relationships = self._analyze_vendor_relationships(df)
            cash_flow_analysis = self._analyze_cash_flow(df)
            
            # Generate key metrics
            key_metrics = self._calculate_key_metrics(df)
            
            # Generate recommendations and insights
            recommendations = self._generate_recommendations(
                spending_patterns, vendor_relationships, cash_flow_analysis, key_metrics
            )
            
            risk_flags = self._identify_risk_flags(
                spending_patterns, vendor_relationships, cash_flow_analysis
            )
            
            opportunities = self._identify_opportunities(
                spending_patterns, vendor_relationships, cash_flow_analysis, key_metrics
            )
            
            insights = BusinessInsights(
                spending_patterns=spending_patterns,
                vendor_relationships=vendor_relationships,
                cash_flow_analysis=cash_flow_analysis,
                key_metrics=key_metrics,
                recommendations=recommendations,
                risk_flags=risk_flags,
                opportunities=opportunities
            )
            
            logger.info(f"Business insights analysis completed successfully")
            return insights
            
        except Exception as e:
            logger.error(f"Error in business insights analysis: {str(e)}")
            raise
    
    def _prepare_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare and validate data for analysis"""
        logger.info("Preparing data for business insights analysis")
        
        # Ensure required columns exist
        required_columns = ['date', 'amount', 'description', 'category']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            logger.warning(f"Missing columns: {missing_columns}. Using available columns.")
        
        # Create vendor column if not present
        if 'vendor' not in df.columns:
            df['vendor'] = df['description'].apply(self._extract_vendor_from_description)
        
        # Ensure date column is datetime
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
            df = df.dropna(subset=['date'])
        
        # Ensure amount is numeric
        if 'amount' in df.columns:
            df['amount'] = pd.to_numeric(df['amount'], errors='coerce')
            df = df.dropna(subset=['amount'])
        
        # Add derived columns
        df['month'] = df['date'].dt.to_period('M')
        df['quarter'] = df['date'].dt.to_period('Q')
        df['year'] = df['date'].dt.year
        df['day_of_week'] = df['date'].dt.dayofweek
        df['is_weekend'] = df['day_of_week'].isin([5, 6])
        
        # Separate inflows and outflows
        df['is_inflow'] = df['amount'] > 0
        df['is_outflow'] = df['amount'] < 0
        df['abs_amount'] = abs(df['amount'])
        
        logger.info(f"Data prepared: {len(df)} transactions")
        return df
    
    def _extract_vendor_from_description(self, description: str) -> str:
        """Extract vendor name from transaction description"""
        if pd.isna(description):
            return "Unknown"
        
        # Simple vendor extraction - can be enhanced with NLP
        description = str(description).strip()
        
        # Common patterns
        if 'POS ' in description:
            return description.split('POS ')[-1].split()[0]
        elif 'PURCHASE ' in description:
            return description.split('PURCHASE ')[-1].split()[0]
        elif 'PAYMENT ' in description:
            return description.split('PAYMENT ')[-1].split()[0]
        else:
            # Take first word as vendor
            words = description.split()
            return words[0] if words else "Unknown"
    
    def _analyze_spending_patterns(self, df: pd.DataFrame) -> List[SpendingPattern]:
        """Analyze spending patterns using multiple techniques"""
        logger.info("Analyzing spending patterns")
        
        patterns = []
        
        # 1. Seasonal spending patterns
        seasonal_patterns = self._analyze_seasonal_patterns(df)
        patterns.extend(seasonal_patterns)
        
        # 2. Category spending patterns
        category_patterns = self._analyze_category_patterns(df)
        patterns.extend(category_patterns)
        
        # 3. Amount-based patterns
        amount_patterns = self._analyze_amount_patterns(df)
        patterns.extend(amount_patterns)
        
        # 4. Temporal patterns
        temporal_patterns = self._analyze_temporal_patterns(df)
        patterns.extend(temporal_patterns)
        
        # 5. Vendor concentration patterns
        vendor_patterns = self._analyze_vendor_concentration_patterns(df)
        patterns.extend(vendor_patterns)
        
        # 6. Anomaly detection
        anomaly_patterns = self._analyze_spending_anomalies(df)
        patterns.extend(anomaly_patterns)
        
        logger.info(f"Identified {len(patterns)} spending patterns")
        return patterns
    
    def _analyze_seasonal_patterns(self, df: pd.DataFrame) -> List[SpendingPattern]:
        """Analyze seasonal spending patterns"""
        patterns = []
        
        # Check if seasonal analysis is enabled
        if not self.config['spending_analysis'].get('seasonal_analysis_months', 12):
            return patterns
        
        try:
            # Monthly spending analysis
            monthly_spending = df.groupby('month')['abs_amount'].sum()
            
            if len(monthly_spending) >= 3:
                # Calculate seasonal variation
                mean_spending = monthly_spending.mean()
                std_spending = monthly_spending.std()
                
                # Find peak and low months
                peak_month = monthly_spending.idxmax()
                low_month = monthly_spending.idxmin()
                
                variation_coefficient = std_spending / mean_spending if mean_spending > 0 else 0
                
                if variation_coefficient > 0.3:
                    patterns.append(SpendingPattern(
                        pattern_type="Seasonal Variation",
                        description=f"Significant seasonal variation detected. Peak spending in {peak_month}, lowest in {low_month}",
                        confidence=min(0.9, 0.5 + variation_coefficient),
                        impact_score=variation_coefficient,
                        recommendations=[
                            "Plan cash flow for seasonal peaks",
                            "Negotiate better terms during low seasons",
                            "Consider seasonal budgeting"
                        ],
                        data_points={
                            'peak_month': str(peak_month),
                            'low_month': str(low_month),
                            'variation_coefficient': variation_coefficient,
                            'monthly_spending': monthly_spending.to_dict()
                        }
                    ))
        
        except Exception as e:
            logger.warning(f"Error in seasonal pattern analysis: {str(e)}")
        
        return patterns
    
    def _analyze_category_patterns(self, df: pd.DataFrame) -> List[SpendingPattern]:
        """Analyze spending patterns by category"""
        patterns = []
        
        try:
            if 'category' in df.columns:
                category_spending = df.groupby('category')['abs_amount'].agg(['sum', 'count', 'mean'])
                category_spending['percentage'] = category_spending['sum'] / category_spending['sum'].sum() * 100
                
                # Top spending categories
                top_categories = category_spending.nlargest(3, 'sum')
                
                for category, data in top_categories.iterrows():
                    patterns.append(SpendingPattern(
                        pattern_type="Category Concentration",
                        description=f"High concentration in {category}: {data['percentage']:.1f}% of total spending",
                        confidence=0.8,
                        impact_score=data['percentage'] / 100,
                        recommendations=[
                            f"Review {category} spending for optimization opportunities",
                            "Consider bulk purchasing for high-volume categories",
                            "Negotiate better rates with category suppliers"
                        ],
                        data_points={
                            'category': category,
                            'total_spend': data['sum'],
                            'transaction_count': data['count'],
                            'average_amount': data['mean'],
                            'percentage': data['percentage']
                        }
                    ))
        
        except Exception as e:
            logger.warning(f"Error in category pattern analysis: {str(e)}")
        
        return patterns
    
    def _analyze_amount_patterns(self, df: pd.DataFrame) -> List[SpendingPattern]:
        """Analyze patterns based on transaction amounts"""
        patterns = []
        
        try:
            amounts = df['abs_amount']
            
            # Round number analysis
            round_amounts = amounts[amounts % 100 == 0]
            round_percentage = len(round_amounts) / len(amounts) * 100
            
            if round_percentage > 20:
                patterns.append(SpendingPattern(
                    pattern_type="Round Number Spending",
                    description=f"High percentage of round number transactions: {round_percentage:.1f}%",
                    confidence=0.9,
                    impact_score=round_percentage / 100,
                    recommendations=[
                        "Review if round number payments are optimal",
                        "Consider more precise payment amounts",
                        "Analyze if this indicates estimation vs. actual costs"
                    ],
                    data_points={
                        'round_percentage': round_percentage,
                        'round_count': len(round_amounts),
                        'total_transactions': len(amounts)
                    }
                ))
            
            # Amount distribution analysis
            amount_quartiles = amounts.quantile([0.25, 0.5, 0.75])
            amount_skewness = stats.skew(amounts)
            
            if abs(amount_skewness) > 1.5:
                skew_direction = "right-skewed" if amount_skewness > 0 else "left-skewed"
                patterns.append(SpendingPattern(
                    pattern_type="Amount Distribution",
                    description=f"Spending distribution is {skew_direction} (skewness: {amount_skewness:.2f})",
                    confidence=0.7,
                    impact_score=abs(amount_skewness) / 3,
                    recommendations=[
                        "Analyze causes of spending distribution skew",
                        "Consider if large transactions need special attention",
                        "Review small transaction processing costs"
                    ],
                    data_points={
                        'skewness': amount_skewness,
                        'quartiles': amount_quartiles.to_dict(),
                        'mean': amounts.mean(),
                        'std': amounts.std()
                    }
                ))
        
        except Exception as e:
            logger.warning(f"Error in amount pattern analysis: {str(e)}")
        
        return patterns
    
    def _analyze_temporal_patterns(self, df: pd.DataFrame) -> List[SpendingPattern]:
        """Analyze temporal spending patterns"""
        patterns = []
        
        try:
            # Day of week patterns
            dow_spending = df.groupby('day_of_week')['abs_amount'].sum()
            dow_percentages = dow_spending / dow_spending.sum() * 100
            
            # Weekend vs weekday analysis
            weekday_spending = df[~df['is_weekend']]['abs_amount'].sum()
            weekend_spending = df[df['is_weekend']]['abs_amount'].sum()
            total_spending = weekday_spending + weekend_spending
            
            if total_spending > 0:
                weekend_percentage = weekend_spending / total_spending * 100
                
                if weekend_percentage > 30:
                    patterns.append(SpendingPattern(
                        pattern_type="Weekend Spending",
                        description=f"High weekend spending: {weekend_percentage:.1f}% of total",
                        confidence=0.8,
                        impact_score=weekend_percentage / 100,
                        recommendations=[
                            "Review weekend spending patterns",
                            "Consider if weekend transactions are necessary",
                            "Analyze weekend vs. weekday vendor differences"
                        ],
                        data_points={
                            'weekend_percentage': weekend_percentage,
                            'weekday_spending': weekday_spending,
                            'weekend_spending': weekend_spending,
                            'dow_distribution': dow_percentages.to_dict()
                        }
                    ))
            
            # Time-based trends
            if len(df) >= 30:
                daily_spending = df.groupby(df['date'].dt.date)['abs_amount'].sum()
                trend_coefficient = np.polyfit(range(len(daily_spending)), daily_spending.values, 1)[0]
                
                if abs(trend_coefficient) > daily_spending.std() * 0.1:
                    trend_direction = "increasing" if trend_coefficient > 0 else "decreasing"
                    patterns.append(SpendingPattern(
                        pattern_type="Spending Trend",
                        description=f"Spending is {trend_direction} over time",
                        confidence=0.7,
                        impact_score=abs(trend_coefficient) / daily_spending.mean() if daily_spending.mean() > 0 else 0,
                        recommendations=[
                            f"Monitor {trend_direction} spending trend",
                            "Analyze causes of trend",
                            "Plan for trend continuation"
                        ],
                        data_points={
                            'trend_coefficient': trend_coefficient,
                            'trend_direction': trend_direction,
                            'daily_spending_stats': {
                                'mean': daily_spending.mean(),
                                'std': daily_spending.std(),
                                'min': daily_spending.min(),
                                'max': daily_spending.max()
                            }
                        }
                    ))
        
        except Exception as e:
            logger.warning(f"Error in temporal pattern analysis: {str(e)}")
        
        return patterns
    
    def _analyze_vendor_concentration_patterns(self, df: pd.DataFrame) -> List[SpendingPattern]:
        """Analyze vendor concentration patterns"""
        patterns = []
        
        try:
            if 'vendor' in df.columns:
                vendor_spending = df.groupby('vendor')['abs_amount'].sum()
                total_spending = vendor_spending.sum()
                
                if total_spending > 0:
                    # Calculate concentration metrics
                    top_5_percentage = vendor_spending.nlargest(5).sum() / total_spending * 100
                    top_10_percentage = vendor_spending.nlargest(10).sum() / total_spending * 100
                    
                    # Herfindahl-Hirschman Index for concentration
                    hhi = ((vendor_spending / total_spending) ** 2).sum()
                    
                    if hhi > 0.25:  # High concentration
                        patterns.append(SpendingPattern(
                            pattern_type="Vendor Concentration",
                            description=f"High vendor concentration (HHI: {hhi:.3f}). Top 5 vendors: {top_5_percentage:.1f}%",
                            confidence=0.9,
                            impact_score=hhi,
                            recommendations=[
                                "Diversify vendor base to reduce concentration risk",
                                "Negotiate better terms with key vendors",
                                "Develop backup vendor relationships"
                            ],
                            data_points={
                                'hhi': hhi,
                                'top_5_percentage': top_5_percentage,
                                'top_10_percentage': top_10_percentage,
                                'vendor_count': len(vendor_spending),
                                'top_vendors': vendor_spending.nlargest(5).to_dict()
                            }
                        ))
        
        except Exception as e:
            logger.warning(f"Error in vendor concentration analysis: {str(e)}")
        
        return patterns
    
    def _analyze_spending_anomalies(self, df: pd.DataFrame) -> List[SpendingPattern]:
        """Detect spending anomalies"""
        patterns = []
        
        # Check if anomaly detection is enabled
        anomaly_threshold = self.config['spending_analysis'].get('anomaly_detection_threshold')
        if not anomaly_threshold:
            return patterns
        
        try:
            amounts = df['abs_amount']
            
            # Statistical anomaly detection
            mean_amount = amounts.mean()
            std_amount = amounts.std()
            
            if std_amount > 0:
                # Z-score based anomalies
                z_scores = np.abs((amounts - mean_amount) / std_amount)
                anomalies = amounts[z_scores > anomaly_threshold]
                
                if len(anomalies) > 0:
                    anomaly_percentage = len(anomalies) / len(amounts) * 100
                    
                    patterns.append(SpendingPattern(
                        pattern_type="Spending Anomalies",
                        description=f"Detected {len(anomalies)} anomalous transactions ({anomaly_percentage:.1f}%)",
                        confidence=0.8,
                        impact_score=anomaly_percentage / 100,
                        recommendations=[
                            "Review anomalous transactions for errors",
                            "Investigate unusual spending patterns",
                            "Set up alerts for large transactions"
                        ],
                        data_points={
                            'anomaly_count': len(anomalies),
                            'anomaly_percentage': anomaly_percentage,
                            'max_anomaly': anomalies.max(),
                            'min_anomaly': anomalies.min(),
                            'anomaly_threshold': mean_amount + anomaly_threshold * std_amount
                        }
                    ))
        
        except Exception as e:
            logger.warning(f"Error in anomaly detection: {str(e)}")
        
        return patterns
    
    def _analyze_vendor_relationships(self, df: pd.DataFrame) -> List[VendorRelationship]:
        """Analyze vendor relationships and patterns"""
        logger.info("Analyzing vendor relationships")
        
        relationships = []
        
        try:
            if 'vendor' in df.columns:
                vendor_data = df.groupby('vendor').agg({
                    'abs_amount': ['sum', 'count', 'mean', 'std'],
                    'date': ['min', 'max'],
                    'category': lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else 'Unknown'
                }).round(2)
                
                vendor_data.columns = ['total_spend', 'transaction_count', 'avg_amount', 'std_amount', 
                                     'first_transaction', 'last_transaction', 'primary_category']
                
                # Filter vendors with minimum transactions
                min_transactions = self.config['vendor_analysis']['min_vendor_transactions']
                significant_vendors = vendor_data[vendor_data['transaction_count'] >= min_transactions]
                
                for vendor, data in significant_vendors.iterrows():
                    relationship = self._analyze_single_vendor_relationship(df, vendor, data)
                    relationships.append(relationship)
                
                # Sort by total spend
                relationships.sort(key=lambda x: x.total_spend, reverse=True)
                
                logger.info(f"Analyzed {len(relationships)} vendor relationships")
        
        except Exception as e:
            logger.warning(f"Error in vendor relationship analysis: {str(e)}")
        
        return relationships
    
    def _analyze_single_vendor_relationship(self, df: pd.DataFrame, vendor: str, data: pd.Series) -> VendorRelationship:
        """Analyze relationship with a single vendor"""
        
        vendor_transactions = df[df['vendor'] == vendor]
        
        # Calculate relationship strength
        relationship_strength = min(1.0, data['transaction_count'] / 50)  # Normalize to 0-1
        
        # Determine risk level
        if data['total_spend'] > df['abs_amount'].sum() * 0.1:  # More than 10% of total spend
            risk_level = "High"
        elif data['total_spend'] > df['abs_amount'].sum() * 0.05:  # More than 5% of total spend
            risk_level = "Medium"
        else:
            risk_level = "Low"
        
        # Analyze payment pattern
        if data['transaction_count'] >= 5:
            payment_intervals = vendor_transactions['date'].sort_values().diff().dt.days
            avg_interval = payment_intervals.mean()
            
            if avg_interval <= 7:
                payment_pattern = "Frequent"
            elif avg_interval <= 30:
                payment_pattern = "Monthly"
            elif avg_interval <= 90:
                payment_pattern = "Quarterly"
            else:
                payment_pattern = "Infrequent"
        else:
            payment_pattern = "Insufficient Data"
        
        # Category distribution
        if 'category' in df.columns:
            category_dist = vendor_transactions['category'].value_counts(normalize=True).to_dict()
        else:
            category_dist = {"Unknown": 1.0}
        
        # Generate insights
        insights = []
        
        if data['transaction_count'] >= 10:
            # Trend analysis
            monthly_spending = vendor_transactions.groupby(vendor_transactions['date'].dt.to_period('M'))['abs_amount'].sum()
            if len(monthly_spending) >= 3:
                trend = np.polyfit(range(len(monthly_spending)), monthly_spending.values, 1)[0]
                if trend > 0:
                    insights.append("Increasing spending trend")
                elif trend < 0:
                    insights.append("Decreasing spending trend")
        
        if data['std_amount'] > data['avg_amount'] * 0.5:
            insights.append("High transaction amount variability")
        
        if data['total_spend'] > df['abs_amount'].sum() * 0.05:
            insights.append("Significant vendor concentration")
        
        return VendorRelationship(
            vendor_name=vendor,
            total_spend=data['total_spend'],
            transaction_count=data['transaction_count'],
            relationship_strength=relationship_strength,
            risk_level=risk_level,
            payment_pattern=payment_pattern,
            category_distribution=category_dist,
            insights=insights
        )
    
    def _analyze_cash_flow(self, df: pd.DataFrame) -> CashFlowAnalysis:
        """Analyze cash flow patterns and trends"""
        logger.info("Analyzing cash flow")
        
        try:
            # Separate inflows and outflows
            inflows = df[df['amount'] > 0]
            outflows = df[df['amount'] < 0]
            
            total_inflow = inflows['amount'].sum()
            total_outflow = abs(outflows['amount'].sum())
            net_cash_flow = total_inflow - total_outflow
            
            # Calculate liquidity ratio
            liquidity_ratio = total_inflow / total_outflow if total_outflow > 0 else float('inf')
            
            # Determine cash flow trend
            if len(df) >= 30:
                daily_net_flow = df.groupby(df['date'].dt.date)['amount'].sum()
                trend_coefficient = np.polyfit(range(len(daily_net_flow)), daily_net_flow.values, 1)[0]
                
                if trend_coefficient > 0:
                    cash_flow_trend = "Improving"
                elif trend_coefficient < 0:
                    cash_flow_trend = "Declining"
                else:
                    cash_flow_trend = "Stable"
            else:
                cash_flow_trend = "Insufficient Data"
            
            # Calculate burn rate and runway
            burn_rate = None
            runway_months = None
            
            if net_cash_flow < 0 and len(df) >= 30:
                # Calculate monthly burn rate
                monthly_outflow = outflows.groupby(outflows['date'].dt.to_period('M'))['abs_amount'].sum()
                avg_monthly_outflow = monthly_outflow.mean()
                
                if avg_monthly_outflow > 0:
                    burn_rate = avg_monthly_outflow
                    
                    # Estimate runway (assuming current cash position)
                    # This is a simplified calculation
                    current_cash = total_inflow - total_outflow
                    runway_months = current_cash / burn_rate if burn_rate > 0 else None
            
            # Generate insights
            insights = []
            
            if liquidity_ratio < 1.0:
                insights.append("Cash outflow exceeds inflow - monitor closely")
            elif liquidity_ratio < 1.2:
                insights.append("Low cash flow margin - consider cost optimization")
            else:
                insights.append("Healthy cash flow margin")
            
            if net_cash_flow < 0:
                insights.append("Negative net cash flow - review spending patterns")
            
            if burn_rate and runway_months and runway_months < 6:
                insights.append(f"Low cash runway ({runway_months:.1f} months) - urgent action needed")
            
            return CashFlowAnalysis(
                period=f"{df['date'].min().strftime('%Y-%m')} to {df['date'].max().strftime('%Y-%m')}",
                net_cash_flow=net_cash_flow,
                cash_inflow=total_inflow,
                cash_outflow=total_outflow,
                cash_flow_trend=cash_flow_trend,
                liquidity_ratio=liquidity_ratio,
                burn_rate=burn_rate,
                runway_months=runway_months,
                insights=insights
            )
        
        except Exception as e:
            logger.warning(f"Error in cash flow analysis: {str(e)}")
            return CashFlowAnalysis(
                period="Unknown",
                net_cash_flow=0,
                cash_inflow=0,
                cash_outflow=0,
                cash_flow_trend="Error",
                liquidity_ratio=0,
                burn_rate=None,
                runway_months=None,
                insights=["Error in cash flow analysis"]
            )
    
    def _calculate_key_metrics(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate key business metrics"""
        metrics = {}
        
        try:
            # Basic metrics
            metrics['total_transactions'] = len(df)
            metrics['total_spend'] = df['abs_amount'].sum()
            metrics['avg_transaction_amount'] = df['abs_amount'].mean()
            metrics['median_transaction_amount'] = df['abs_amount'].median()
            
            # Vendor metrics
            if 'vendor' in df.columns:
                metrics['unique_vendors'] = df['vendor'].nunique()
                metrics['avg_vendor_spend'] = df.groupby('vendor')['abs_amount'].sum().mean()
            
            # Category metrics
            if 'category' in df.columns:
                metrics['unique_categories'] = df['category'].nunique()
                metrics['top_category_percentage'] = (
                    df['category'].value_counts().iloc[0] / len(df) * 100
                )
            
            # Time-based metrics
            if 'date' in df.columns:
                date_range = df['date'].max() - df['date'].min()
                metrics['analysis_period_days'] = date_range.days
                metrics['transactions_per_day'] = len(df) / max(1, date_range.days)
            
            # Cash flow metrics
            inflows = df[df['amount'] > 0]
            outflows = df[df['amount'] < 0]
            metrics['total_inflow'] = inflows['amount'].sum()
            metrics['total_outflow'] = abs(outflows['amount'].sum())
            metrics['net_cash_flow'] = metrics['total_inflow'] - metrics['total_outflow']
            
        except Exception as e:
            logger.warning(f"Error calculating key metrics: {str(e)}")
        
        return metrics
    
    def _generate_recommendations(self, spending_patterns: List[SpendingPattern], 
                                vendor_relationships: List[VendorRelationship],
                                cash_flow_analysis: CashFlowAnalysis,
                                key_metrics: Dict[str, float]) -> List[str]:
        """Generate actionable recommendations based on analysis"""
        recommendations = []
        
        # Cash flow recommendations
        if cash_flow_analysis.net_cash_flow < 0:
            recommendations.append("Implement cost reduction strategies to improve cash flow")
        
        if cash_flow_analysis.liquidity_ratio < 1.2:
            recommendations.append("Optimize payment terms and cash flow management")
        
        # Vendor concentration recommendations
        high_risk_vendors = [v for v in vendor_relationships if v.risk_level == "High"]
        if len(high_risk_vendors) > 0:
            recommendations.append("Diversify vendor base to reduce concentration risk")
        
        # Spending pattern recommendations
        for pattern in spending_patterns:
            if pattern.pattern_type == "Vendor Concentration" and pattern.impact_score > 0.5:
                recommendations.append("Develop backup vendor relationships for key suppliers")
            
            if pattern.pattern_type == "Seasonal Variation":
                recommendations.append("Implement seasonal budgeting and cash flow planning")
        
        # General recommendations
        if key_metrics.get('total_transactions', 0) > 1000:
            recommendations.append("Consider implementing automated expense management systems")
        
        if key_metrics.get('avg_transaction_amount', 0) > 1000:
            recommendations.append("Review large transaction approval processes")
        
        return recommendations
    
    def _identify_risk_flags(self, spending_patterns: List[SpendingPattern],
                            vendor_relationships: List[VendorRelationship],
                            cash_flow_analysis: CashFlowAnalysis) -> List[str]:
        """Identify potential risk flags"""
        risk_flags = []
        
        # Cash flow risks
        if cash_flow_analysis.net_cash_flow < 0:
            risk_flags.append("Negative net cash flow")
        
        if cash_flow_analysis.liquidity_ratio < 1.0:
            risk_flags.append("Cash outflow exceeds inflow")
        
        if cash_flow_analysis.runway_months and cash_flow_analysis.runway_months < 3:
            risk_flags.append(f"Critical cash runway: {cash_flow_analysis.runway_months:.1f} months")
        
        # Vendor concentration risks
        high_risk_vendors = [v for v in vendor_relationships if v.risk_level == "High"]
        if len(high_risk_vendors) > 2:
            risk_flags.append("High vendor concentration risk")
        
        # Spending pattern risks
        for pattern in spending_patterns:
            if pattern.pattern_type == "Spending Anomalies" and pattern.impact_score > 0.1:
                risk_flags.append("Unusual spending patterns detected")
            
            if pattern.pattern_type == "Vendor Concentration" and pattern.impact_score > 0.7:
                risk_flags.append("Extreme vendor concentration")
        
        return risk_flags
    
    def _identify_opportunities(self, spending_patterns: List[SpendingPattern],
                              vendor_relationships: List[VendorRelationship],
                              cash_flow_analysis: CashFlowAnalysis,
                              key_metrics: Dict[str, float]) -> List[str]:
        """Identify business opportunities"""
        opportunities = []
        
        # Vendor optimization opportunities
        for vendor in vendor_relationships:
            if vendor.transaction_count >= 10 and vendor.relationship_strength > 0.7:
                opportunities.append(f"Negotiate better terms with {vendor.vendor_name}")
        
        # Category optimization opportunities
        if key_metrics.get('unique_categories', 0) > 10:
            opportunities.append("Consolidate spending categories for better tracking")
        
        # Cash flow opportunities
        if cash_flow_analysis.liquidity_ratio > 1.5:
            opportunities.append("Consider investment opportunities with excess cash")
        
        # Process optimization opportunities
        if key_metrics.get('total_transactions', 0) > 500:
            opportunities.append("Implement automated expense categorization")
        
        return opportunities 