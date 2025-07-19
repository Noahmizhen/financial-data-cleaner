# app.py – QuickBooks Data Cleaner (Flask version)
"""Lightweight Flask server that:
1. Performs QuickBooks OAuth2 (development‑only redirect on localhost)
2. Accepts a transaction file upload (CSV or XLSX)
3. Runs deterministic duplicate detection & rule‑based categorisation
4. Uses Claude for still‑blank categories (cost‑controlled)
5. Returns a cleaned CSV or JSON payload for download

Run locally:
$ pip install -r requirements.txt
$ flask --app app.py run --debug

Environment variables (e.g. in .env):
    FLASK_SECRET="super‑secret"
    QB_CLIENT_ID="…"
    QB_CLIENT_SECRET="…"
    QB_REDIRECT_URI="http://localhost:5000/callback"
    QB_REALM_ID="1234567890"
    ANTHROPIC_API_KEY="sk-ant-api03-..."
"""

from __future__ import annotations

import io
import os
import tempfile
import pickle
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd
from flask import (
    Flask,
    jsonify,
    redirect,
    render_template,
    request,
    send_file,
    session,
    url_for,
)

# Add environment variable loading
from dotenv import load_dotenv
load_dotenv('quickbooksdetails.env.txt')

from qb_factory import make_qb
from core.data_cleaner import DataCleaner, SmartDataCleaner  # deterministic engine + smart schema detection
from core.financial_data_cleaner import FinancialDataCleaner  # New Financial Services cleaner
from config import get_settings

import anthropic
import json
from datetime import datetime

################################################################################
# Flask setup
################################################################################

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET", "dev-secret-key")

def get_claude_key():
    """Get Claude API key from environment."""
    return os.environ.get("ANTHROPIC_API_KEY")

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'csv', 'xlsx'}

def read_file_safely(file):
    """Read uploaded file safely."""
    filename = file.filename.lower()
    if filename.endswith('.csv'):
        return pd.read_csv(file)
    elif filename.endswith('.xlsx'):
        return pd.read_excel(file)
    else:
        raise ValueError("Unsupported file format")

################################################################################
# Routes
################################################################################

@app.route('/')
def index():
    """Home page with upload options."""
    msg = request.args.get('msg', '')
    error = request.args.get('error', '')
    return render_template('index_enhanced.html', msg=msg, error=error)

@app.route('/upload', methods=['POST'])
def upload():
    """Standard data cleaning upload."""
    if 'file' not in request.files:
        return redirect(url_for('index', msg='No file part'))
    
    file = request.files['file']
    if file.filename == '':
        return redirect(url_for('index', msg='No file selected'))
    
    if file and allowed_file(file.filename):
        try:
            # Read file
            df = read_file_safely(file)
            print(f"Processing file: {file.filename} ({len(file.read())} bytes)")
            file.seek(0)  # Reset file pointer
            df = read_file_safely(file)
            
            print(f"Successfully read file with {len(df)} rows and {len(df.columns)} columns")
            print(f"Original columns: {list(df.columns)}")
            
            # Detect format and choose cleaner
            detected_format = "quickbooks"  # Simple detection
            print(f"Detected format: {detected_format}")
            
            # Use SmartDataCleaner for intelligent schema detection
            anthropic_key = get_claude_key()
            if anthropic_key:
                cleaner = SmartDataCleaner(df, anthropic_key)
                cleaned_df = cleaner.clean()
                quality_report = cleaner.quality_report() if hasattr(cleaner, 'quality_report') else {"status": "Smart cleaning completed"}
            else:
                # Fallback to basic cleaner
                cleaner = DataCleaner(df, None)
                cleaned_df = cleaner.clean()
                quality_report = cleaner.quality_report() if hasattr(cleaner, 'quality_report') else {"error": "No Claude API key available"}
            
            # Store results in session using the cleaning function
            session['cleaned_data'] = clean_data_for_session(cleaned_df)
            session['quality_report'] = quality_report
            session['original_filename'] = file.filename
            
            # Return results page
            return render_template('results.html',
                                 data=cleaned_df.head(20).to_dict('records'),
                                 columns=list(cleaned_df.columns),
                                 total_rows=len(cleaned_df),
                                 quality_report=quality_report)
            
        except Exception as e:
            print(f"Processing failed: {str(e)}")
            return render_template('index_enhanced.html', 
                                 error=f"Processing failed: {str(e)}")
    
    return redirect(url_for('index', msg='Invalid file type'))

def clean_data_for_session(df: pd.DataFrame) -> List[Dict]:
    """Clean DataFrame for session storage by handling NaT values and other serialization issues."""
    # Convert DataFrame to records, handling NaT values and numpy types
    records = []
    
    # Ensure DataFrame is not empty and has valid shape
    if df.empty:
        return []
    
    # Limit to first 3 rows to reduce session size
    df_limited = df.head(3)
    
    # Ensure all columns exist in the limited DataFrame
    for col in df_limited.columns:
        if col not in df_limited.columns:
            df_limited[col] = None
    
    for _, row in df_limited.iterrows():
        clean_row = {}
        for col, value in row.items():
            # Handle NaN/NaT values more robustly
            if pd.isna(value) or value is None:
                clean_row[col] = None
            # Handle pandas NaT specifically
            elif hasattr(value, 'timetuple') and pd.isna(value):
                clean_row[col] = None
            # Handle datetime objects
            elif hasattr(value, 'strftime'):
                clean_row[col] = value.isoformat() if pd.notna(value) else None
            # Handle numpy types
            elif hasattr(value, 'item'):
                clean_row[col] = value.item()
            # Handle pandas Timestamp
            elif hasattr(value, 'to_pydatetime'):
                clean_row[col] = value.isoformat() if pd.notna(value) else None
            # Handle numpy int64/float64 specifically
            elif hasattr(value, 'dtype') and hasattr(value, 'item'):
                clean_row[col] = value.item()
            # Handle strings - truncate if too long
            elif isinstance(value, str):
                clean_row[col] = value[:100] if len(value) > 100 else value
            # Handle other types
            else:
                clean_row[col] = str(value)[:100] if value else None
        records.append(clean_row)
    return records

@app.route('/upload_financial', methods=['POST'])
def upload_financial():
    """Enhanced upload endpoint using Claude Financial Services."""
    if 'file' not in request.files:
        return redirect(url_for('index', msg='No file part'))
    
    file = request.files['file']
    if file.filename == '':
        return redirect(url_for('index', msg='No file selected'))
    
    if file and allowed_file(file.filename):
        try:
            # Read file
            df = read_file_safely(file)
            print(f"Processing file: {file.filename} ({len(file.read())} bytes)")
            file.seek(0)  # Reset file pointer
            df = read_file_safely(file)
            
            print(f"Successfully read file with {len(df)} rows and {len(df.columns)} columns")
            print(f"Original columns: {list(df.columns)}")
            
            # Get Anthropic API key
            anthropic_key = get_claude_key()
            if not anthropic_key:
                return render_template('index_enhanced.html', 
                                     error="Claude API key not configured. Please check your environment settings.")
            
            # Initialize Financial Services cleaner
            financial_cleaner = FinancialDataCleaner(df, anthropic_key)
            
            # Run enhanced cleaning
            cleaned_df, financial_report = financial_cleaner.clean_with_financial_services()
            
            # Store minimal session data to avoid cookie size issues
            session['cleaned_data'] = clean_data_for_session(cleaned_df.head(3))  # Limit to 3 rows
            session['total_rows'] = len(cleaned_df)
            session['columns'] = list(cleaned_df.columns)
            
            # Clean financial report for session storage with better error handling
            def clean_report_for_session(report):
                """Clean report data for session storage with size limits."""
                if not isinstance(report, dict):
                    return {'error': 'Invalid report format'}
                
                cleaned_report = {}
                for key, value in report.items():
                    if isinstance(value, dict):
                        # Recursively clean nested dictionaries
                        cleaned_report[key] = clean_report_for_session(value)
                    elif isinstance(value, (int, float, str, bool)):
                        # Convert numpy types and limit string length
                        if hasattr(value, 'item'):
                            cleaned_report[key] = value.item()
                        elif isinstance(value, str) and len(value) > 200:
                            cleaned_report[key] = value[:200] + "..."
                        else:
                            cleaned_report[key] = value
                    elif isinstance(value, list):
                        # Handle lists - preserve objects but limit size
                        if key in ['spending_patterns', 'vendor_relationships'] and len(value) > 0:
                            # These are business insight objects - preserve their structure
                            cleaned_list = []
                            for item in value[:10]:  # Limit to 10 items
                                if isinstance(item, dict):
                                    cleaned_list.append(item)
                                else:
                                    # Convert object to dict if it's not already
                                    item_dict = {}
                                    for attr in ['pattern_type', 'description', 'confidence', 'impact_score', 'recommendations']:
                                        if hasattr(item, attr):
                                            item_dict[attr] = getattr(item, attr)
                                    cleaned_list.append(item_dict)
                            cleaned_report[key] = cleaned_list
                        else:
                            # Other lists - convert to strings as before
                            cleaned_report[key] = [str(item)[:50] for item in value[:5]]
                    else:
                        # Convert to string and limit length
                        str_value = str(value) if value else ""
                        cleaned_report[key] = str_value[:100] if len(str_value) > 100 else str_value
                return cleaned_report
            
            try:
                session['financial_report'] = clean_report_for_session(financial_report)
            except Exception as e:
                print(f"⚠️ Failed to store financial report in session: {e}")
                session['financial_report'] = {'error': 'Report storage failed'}
            
            session['original_filename'] = file.filename
            
            # Render results with financial insights
            return render_template('financial_results_enhanced.html',
                                 data=cleaned_df.head(20).to_dict('records'),
                                 columns=list(cleaned_df.columns),
                                 total_rows=len(cleaned_df),
                                 financial_analysis=financial_report.get('financial_analysis', {}),
                                 risk_assessment=financial_report.get('risk_assessment', {}),
                                 business_insights=financial_report.get('business_insights', {}),
                                 processing_stats=financial_report.get('processing_stats', {}))
            
        except Exception as e:
            print(f"Financial processing failed: {str(e)}")
            return render_template('index_enhanced.html', 
                                 error=f"Financial processing failed: {str(e)}")
    
    return redirect(url_for('index', msg='Invalid file type'))

@app.route('/upload_financial_multi', methods=['POST'])
def upload_financial_multi():
    """Multi-file financial analysis endpoint for comprehensive insights."""
    if 'files' not in request.files:
        return redirect(url_for('index', msg='No files uploaded'))
    
    files = request.files.getlist('files')
    if not files or all(file.filename == '' for file in files):
        return redirect(url_for('index', msg='No files selected'))
    
    # Filter valid files
    valid_files = [f for f in files if f.filename and allowed_file(f.filename)]
    if not valid_files:
        return redirect(url_for('index', msg='No valid files found'))
    
    try:
        print(f"📁 Processing {len(valid_files)} files for multi-file analysis...")
        
        # Get Anthropic API key
        anthropic_key = get_claude_key()
        if not anthropic_key:
            return render_template('index_enhanced.html', 
                                 error="Claude API key not configured. Please check your environment settings.")
        
        # Process each file
        datasets = []
        file_info = []
        
        for i, file in enumerate(valid_files):
            try:
                print(f"📊 Processing file {i+1}/{len(valid_files)}: {file.filename}")
                
                # Read file
                df = read_file_safely(file)
                print(f"   Successfully read {len(df)} rows, {len(df.columns)} columns")
                
                # Initialize cleaner for this dataset
                cleaner = FinancialDataCleaner(df, anthropic_key)
                
                # Clean the data
                cleaned_df, financial_report = cleaner.clean_with_financial_services()
                
                # Store dataset info
                datasets.append({
                    'original_df': df,
                    'cleaned_df': cleaned_df,
                    'report': financial_report,
                    'filename': file.filename
                })
                
                file_info.append({
                    'filename': file.filename,
                    'original_rows': len(df),
                    'cleaned_rows': len(cleaned_df),
                    'columns': list(df.columns),
                    'data_type': financial_report.get('financial_analysis', {}).get('data_type', 'unknown')
                })
                
                print(f"   ✅ Processed successfully")
                
            except Exception as e:
                print(f"   ❌ Failed to process {file.filename}: {e}")
                continue
        
        if not datasets:
            return render_template('index_enhanced.html', 
                                 error="Failed to process any files. Please check your file formats.")
        
        # Generate comprehensive multi-file analysis
        multi_analysis = generate_multi_file_analysis(datasets, anthropic_key)
        
        # Store session data
        session['multi_file_data'] = {
            'file_info': file_info,
            'total_datasets': len(datasets),
            'combined_rows': sum(len(d['cleaned_df']) for d in datasets),
            'analysis': multi_analysis
        }
        
        # Store sample data from first dataset for display
        if datasets:
            first_dataset = datasets[0]
            session['cleaned_data'] = clean_data_for_session(first_dataset['cleaned_df'].head(3))
            session['total_rows'] = len(first_dataset['cleaned_df'])
            session['columns'] = list(first_dataset['cleaned_df'].columns)
        
        print(f"✅ Multi-file analysis complete: {len(datasets)} datasets processed")
        
        # Render multi-file results
        return render_template('financial_results_multi.html',
                             datasets=datasets,
                             file_info=file_info,
                             multi_analysis=multi_analysis,
                             total_files=len(datasets))
        
    except Exception as e:
        print(f"Multi-file processing failed: {str(e)}")
        return render_template('index_enhanced.html', 
                             error=f"Multi-file processing failed: {str(e)}")

def generate_multi_file_analysis(datasets, anthropic_key):
    """Generate comprehensive analysis across multiple datasets."""
    try:
        print("🔍 Generating comprehensive multi-file analysis...")
        
        # Prepare dataset summaries
        dataset_summaries = []
        for i, dataset in enumerate(datasets):
            df = dataset['cleaned_df']
            report = dataset['report']
            
            summary = {
                'file_index': i + 1,
                'filename': dataset['filename'],
                'rows': len(df),
                'columns': list(df.columns),
                'data_type': report.get('financial_analysis', {}).get('data_type', 'unknown'),
                'key_insights': report.get('business_insights', {}),
                'risk_assessment': report.get('risk_assessment', {})
            }
            dataset_summaries.append(summary)
        
        # Create comprehensive analysis prompt
        prompt = f"""
As a senior financial analyst, provide a comprehensive analysis of these {len(datasets)} financial datasets:

DATASET SUMMARIES:
{json.dumps(dataset_summaries, indent=2)}

Provide a comprehensive multi-dataset analysis including:

1. **Cross-Dataset Patterns**: Identify patterns across all datasets
2. **Data Quality Assessment**: Overall data quality and consistency
3. **Business Insights**: Combined insights from all datasets
4. **Risk Assessment**: Comprehensive risk analysis
5. **Recommendations**: Strategic recommendations based on all data
6. **Data Integration Opportunities**: How these datasets could be better integrated

Format your response as JSON with these keys:
- cross_dataset_patterns
- overall_data_quality
- combined_business_insights
- comprehensive_risk_assessment
- strategic_recommendations
- integration_opportunities
- summary_statistics
"""
        
        # Get AI analysis
        client = anthropic.Anthropic(api_key=anthropic_key)
        response = client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=3000,
            temperature=0.3,
            messages=[{"role": "user", "content": prompt}]
        )
        
        analysis = json.loads(response.content[0].text)
        print("✅ Multi-file analysis generated successfully")
        return analysis
        
    except Exception as e:
        print(f"⚠️ Multi-file analysis failed: {e}")
        return {
            'error': str(e),
            'cross_dataset_patterns': 'Analysis unavailable',
            'overall_data_quality': 'Analysis unavailable',
            'combined_business_insights': 'Analysis unavailable',
            'comprehensive_risk_assessment': 'Analysis unavailable',
            'strategic_recommendations': ['Analysis unavailable'],
            'integration_opportunities': 'Analysis unavailable',
            'summary_statistics': {}
        }

@app.route('/upload_financial_hybrid', methods=['POST'])
def upload_financial_hybrid():
    """🚀 NEW: Ultra-scale hybrid financial processing for large files (10K-100K+ rows)"""
    if 'file' not in request.files:
        return redirect(url_for('index', msg='No file uploaded'))
    
    file = request.files['file']
    if file.filename == '' or not allowed_file(file.filename):
        return redirect(url_for('index', msg='Invalid file'))
    
    try:
        print(f"🚀 Processing large file with ultra-scale hybrid system: {file.filename}")
        
        # Read file
        if file.filename.endswith('.csv'):
            df = pd.read_csv(file)
        elif file.filename.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(file)
        else:
            return redirect(url_for('index', msg='Unsupported file format'))
        
        print(f"📊 Loaded {len(df):,} rows for hybrid processing")
        
        # Initialize hybrid cleaner  
        anthropic_key = get_claude_key()  # Will use API key if available, otherwise Python-only
        cleaner = FinancialDataCleaner(df, anthropic_key)
        
        print(f"🎯 Auto-detected performance tier: {cleaner.performance_tier}")
        print(f"💰 Estimated cost: {cleaner._get_hybrid_config()['cost_estimate'] if cleaner._should_use_hybrid_processing() else 'FREE'}")
        
        # Run ultra-scale hybrid processing
        final_data, report = cleaner.process_with_hybrid_pipeline()
        
        # Store results in session
        session['cleaned_data'] = final_data.to_dict('records')
        session['last_report'] = report
        session['original_filename'] = file.filename
        
        # Prepare enhanced display data
        insights = report.get('business_insights', {})
        performance = report.get('hybrid_performance', {})
        
        # Create file info for template
        file_info = {
            'original_rows': len(df),
            'final_rows': len(final_data),
            'processing_tier': cleaner.performance_tier,
            'processing_time': performance.get('processing_time', 0),
            'rows_per_second': performance.get('rows_per_second', 0),
            'accuracy_estimate': performance.get('accuracy_estimate', 0),
            'cost_estimate': performance.get('cost_estimate', 'FREE'),
            'cost_efficiency': performance.get('cost_efficiency', 1),
            'ai_samples_analyzed': report.get('executive_summary', {}).get('ai_samples_analyzed', 0),
            'confidence_score': report.get('executive_summary', {}).get('confidence_score', 0)
        }
        
        print(f"✅ Hybrid processing complete: {len(final_data):,} rows in {performance.get('processing_time', 0):.2f}s")
        
        # Use enhanced template if available, otherwise fallback to standard
        try:
            return render_template('financial_results_enhanced.html', 
                                 insights=insights,
                                 file_info=file_info,
                                 hybrid_metadata=report.get('hybrid_metadata', {}))
        except:
            # Fallback to existing template
            return render_template('financial_results.html', 
                                 insights=insights,
                                 file_info=file_info)
        
    except Exception as e:
        print(f"❌ Hybrid processing failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return redirect(url_for('index', msg=f'Hybrid processing failed: {str(e)}'))

@app.route('/download/<format>')
def download(format):
    """Download cleaned data in specified format."""
    if 'cleaned_data' not in session and 'multi_file_data' not in session:
        return redirect(url_for('index', msg='No data to download'))
    
    # Check if we have multi-file data
    if 'multi_file_data' in session:
        return download_multi_file(format)
    
    # Get data from session
    data = session['cleaned_data']
    df = pd.DataFrame(data)
    
    # Get original filename or use default
    original_filename = session.get('original_filename', 'cleaned_data')
    base_name = original_filename.rsplit('.', 1)[0] if '.' in original_filename else original_filename
    
    if format.lower() == 'csv':
        # Create CSV
        output = io.StringIO()
        df.to_csv(output, index=False)
        output_bytes = io.BytesIO(output.getvalue().encode('utf-8'))
        filename = f"{base_name}_cleaned.csv"
        mimetype = 'text/csv'
        
    elif format.lower() == 'excel':
        # Create Excel
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name='Cleaned Data', index=False)
            
            # Add quality report if available
            if 'quality_report' in session or 'financial_report' in session:
                report_data = session.get('financial_report', session.get('quality_report', {}))
                
                # Create summary sheet
                summary_data = []
                if 'processing_stats' in report_data:
                    stats = report_data['processing_stats']
                    for key, value in stats.items():
                        summary_data.append({'Metric': key.replace('_', ' ').title(), 'Value': value})
                
                if summary_data:
                    summary_df = pd.DataFrame(summary_data)
                    summary_df.to_excel(writer, sheet_name='Processing Summary', index=False)
        
        output.seek(0)
        output_bytes = output
        filename = f"{base_name}_cleaned.xlsx"
        mimetype = 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
        
    else:
        return redirect(url_for('index', msg='Invalid download format'))
    
    return send_file(
        output_bytes,
        mimetype=mimetype,
        as_attachment=True,
        download_name=filename
    )

def download_multi_file(format):
    """Download multi-file analysis results."""
    multi_data = session['multi_file_data']
    
    if format.lower() == 'csv':
        # Create combined CSV with all datasets
        output = io.StringIO()
        
        # Write summary information
        output.write("Multi-File Analysis Summary\n")
        output.write("=" * 50 + "\n")
        output.write(f"Total Files: {multi_data['total_datasets']}\n")
        output.write(f"Combined Rows: {multi_data['combined_rows']}\n")
        output.write("\n")
        
        # Write file information
        output.write("File Information\n")
        output.write("-" * 30 + "\n")
        for file_info in multi_data['file_info']:
            output.write(f"File: {file_info['filename']}\n")
            output.write(f"Type: {file_info['data_type']}\n")
            output.write(f"Original Rows: {file_info['original_rows']}\n")
            output.write(f"Cleaned Rows: {file_info['cleaned_rows']}\n")
            output.write(f"Columns: {len(file_info['columns'])}\n")
            output.write("\n")
        
        # Write analysis results
        analysis = multi_data['analysis']
        output.write("Analysis Results\n")
        output.write("-" * 30 + "\n")
        
        if 'cross_dataset_patterns' in analysis:
            output.write(f"Cross-Dataset Patterns: {analysis['cross_dataset_patterns']}\n\n")
        
        if 'combined_business_insights' in analysis:
            output.write(f"Combined Business Insights: {analysis['combined_business_insights']}\n\n")
        
        if 'strategic_recommendations' in analysis:
            output.write("Strategic Recommendations:\n")
            for rec in analysis['strategic_recommendations']:
                output.write(f"- {rec}\n")
            output.write("\n")
        
        output_bytes = io.BytesIO(output.getvalue().encode('utf-8'))
        filename = f"multi_file_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        mimetype = 'text/csv'
        
    elif format.lower() == 'excel':
        # Create Excel with multiple sheets
        output = io.BytesIO()
        
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            # Summary sheet
            summary_data = [
                {'Metric': 'Total Files', 'Value': multi_data['total_datasets']},
                {'Metric': 'Combined Rows', 'Value': multi_data['combined_rows']},
                {'Metric': 'Analysis Date', 'Value': datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
            ]
            
            if 'analysis' in multi_data and 'summary_statistics' in multi_data['analysis']:
                stats = multi_data['analysis']['summary_statistics']
                for key, value in stats.items():
                    summary_data.append({'Metric': key.replace('_', ' ').title(), 'Value': value})
            
            summary_df = pd.DataFrame(summary_data)
            summary_df.to_excel(writer, sheet_name='Summary', index=False)
            
            # File information sheet
            file_info_df = pd.DataFrame(multi_data['file_info'])
            file_info_df.to_excel(writer, sheet_name='File Information', index=False)
            
            # Analysis results sheet
            analysis = multi_data['analysis']
            analysis_data = []
            
            for key, value in analysis.items():
                if isinstance(value, str):
                    analysis_data.append({'Category': key.replace('_', ' ').title(), 'Content': value})
                elif isinstance(value, list):
                    for item in value:
                        analysis_data.append({'Category': key.replace('_', ' ').title(), 'Content': str(item)})
            
            if analysis_data:
                analysis_df = pd.DataFrame(analysis_data)
                analysis_df.to_excel(writer, sheet_name='Analysis Results', index=False)
        
        output.seek(0)
        output_bytes = output
        filename = f"multi_file_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
        mimetype = 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
        
    else:
        return redirect(url_for('index', msg='Invalid download format'))
    
    return send_file(
        output_bytes,
        mimetype=mimetype,
        as_attachment=True,
        download_name=filename
    )

@app.route('/callback')
def oauth_callback():
    """QuickBooks OAuth callback (placeholder)."""
    # This would handle QuickBooks OAuth flow
    return jsonify({"status": "OAuth callback - not implemented yet"})

@app.route('/health')
def health():
    """Health check endpoint."""
    return jsonify({
        "status": "healthy",
        "claude_key_configured": bool(get_claude_key()),
        "version": "1.0.0"
    })

################################################################################
# Error handlers
################################################################################

@app.errorhandler(404)
def not_found(error):
    return render_template('index_enhanced.html', error="Page not found"), 404

@app.errorhandler(500)
def internal_error(error):
    return render_template('index_enhanced.html', error="Internal server error"), 500

################################################################################
# Main
################################################################################

if __name__ == '__main__':
    # Load environment variables
    import os
    from dotenv import load_dotenv
    import atexit
    import signal
    
    # Cleanup function to prevent semaphore leaks
    def cleanup():
        import multiprocessing
        try:
            multiprocessing.current_process()._cleanup()
        except:
            pass
    
    # Register cleanup handlers
    atexit.register(cleanup)
    
    def signal_handler(signum, frame):
        cleanup()
        exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Load environment from file if it exists
    env_file = 'quickbooksdetails.env.txt'
    if os.path.exists(env_file):
        with open(env_file, 'r') as f:
            for line in f:
                if '=' in line and not line.strip().startswith('#'):
                    key, value = line.strip().split('=', 1)
                    os.environ[key] = value
    
    # Check for required API key
    if not get_claude_key():
        print("⚠️ WARNING: ANTHROPIC_API_KEY not found in environment")
        print("   Financial Services features will not be available")
    else:
        print(f"✅ Claude API key loaded successfully")
    
    app.run(debug=True, port=5003) 