# 🧹 Financial Data Cleaner

AI-powered financial data cleaning and categorization engine with QuickBooks integration.

## 🚀 Features

- **🤖 Intelligent Column Mapping**: Automatically maps messy column headers to standard fields
- **🔍 Advanced Duplicate Detection**: AI semantic matching + rule-based duplicate removal  
- **🏷️ Smart Categorization**: Claude 3.5 Sonnet powered transaction categorization
- **📅 Date Standardization**: Handles multiple date formats automatically
- **📝 Memo Generation**: AI-generated memos for blank entries
- **📊 Business Insights**: Advanced financial analysis and risk assessment
- **💼 QuickBooks Integration**: Direct API connectivity for seamless data flow
- **⚡ Performance Optimized**: Multi-tier processing for files up to 10,000+ rows

## 🛠️ Installation

1. **Clone the repository:**
```bash
git clone https://github.com/noahmizhen/financial-data-cleaner.git
cd financial-data-cleaner
```

2. **Create virtual environment:**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

4. **Configure environment:**
```bash
cp .env.example .env
# Edit .env with your API keys
```

5. **Run the application:**
```bash
python app.py
```

Visit `http://localhost:5003` in your browser.

## 📊 Performance Tiers

| File Size | Processing Mode | Features |
|-----------|----------------|----------|
| ≤1,000 rows | Full AI Analysis | Complete feature set |
| ≤5,000 rows | AI + Sampling | Smart analysis with sampling |
| ≤10,000 rows | Python + Basic AI | High performance with basic insights |
| >10,000 rows | Minimal Analysis | Maximum speed processing |

## 🔧 API Endpoints

- `POST /upload` - Standard data cleaning
- `POST /upload_financial` - Advanced financial analysis with business insights
- `GET /` - Web interface

## 🧪 Testing

```bash
python -m pytest tests/
```

## 🏗️ Architecture

```
core/                     # Core processing engines
├── data_cleaner.py       # Smart DataCleaner with duplicate fixes
├── financial_data_cleaner.py  # Advanced financial analysis
└── date_standardizer.py  # Comprehensive date parsing

src/cleaner/              # Structured processing framework
├── llm_client.py         # LLM abstraction layer
├── rules.py              # Data cleaning rules
└── pipeline.yml          # Configuration

integrations/             # External API integrations
├── qb_api.py            # QuickBooks API client
└── ai_column_suggester.py

web/                      # Web interface
├── templates/           # HTML templates
└── static/             # CSS/JS assets
```

## 🔑 Key Features

### ✅ Duplicate Mapping Resolution
- **Smart Column Selection**: Intelligent scoring system for duplicate mappings
- **Conflict Resolution**: Prevents DataFrame vs Series ambiguity errors
- **Best Column Detection**: Automatically selects optimal columns per field

### ✅ Multi-Tier Performance
- **Adaptive Processing**: Automatically adjusts features based on file size
- **Performance Range**: 1,800 to 400,000+ rows/second depending on complexity
- **Memory Efficient**: Optimized for large file processing

### ✅ Advanced AI Integration
- **Claude 3.5 Sonnet**: Latest AI model for categorization and analysis
- **Memo Generation**: AI-powered descriptive memo creation
- **Business Insights**: Financial risk assessment and trend analysis

## 📄 License

MIT License

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📞 Support

For support, please open an issue on GitHub or contact noahmizhen@gmail.com.
