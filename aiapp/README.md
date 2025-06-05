# AI App Backend Documentation

## Overview

The AI App Backend is a Django app that provides comprehensive reporting and analytics capabilities for trained AutoML models. It integrates with Google's Gemini-1.5-Pro for AI-powered insights and uses the LangChain ecosystem for advanced text processing.

## Features

### 🎯 Core Functionality
- **Automated Report Generation**: Generate comprehensive reports for trained models
- **Interactive Charts**: Create pie charts, line charts, bar charts, histograms, and correlation matrices
- **AI-Powered Insights**: Leverage Gemini-1.5-Pro for intelligent data analysis
- **CSV Export**: Export chart data for further analysis
- **Multi-Task Support**: Support for Classification, Regression, Time Series, and Clustering tasks

### 📊 Chart Types
- **Pie Charts**: Class distribution, categorical data visualization
- **Line Charts**: Time series trends, continuous data
- **Bar Charts**: Feature comparisons, missing values analysis
- **Histograms**: Data distribution analysis
- **Correlation Heatmaps**: Feature relationship analysis

## API Endpoints

### Generate Report
```
POST /aiapp/generate/
{
    "model_id": 123,
    "report_type": "analysis"
}
```

### Get Report
```
GET /aiapp/get/?report_id=456
```

### List Reports
```
GET /aiapp/list/?page=1&page_size=10&model_id=123
```

### Export Chart Data
```
GET /aiapp/export-chart-csv/?chart_id=789
```

### Get Chart Statistics
```
GET /aiapp/chart-stats/?chart_id=789
```

## Models

### Report
- Stores report metadata and AI insights
- Links to ModelEntry for data access
- Tracks generation status and timestamps

### ChartData
- Stores chart configurations and generated content
- Supports multiple chart types with flexible data structure
- Includes base64 images and interactive HTML

### DataInsight
- Stores structured insights about the data
- Categorized by insight type and priority
- JSON field for flexible insight data storage

### ReportTemplate
- Configurable templates for different model types
- Defines chart types and insights to generate
- Supports customization per task type

## Services

### ReportGenerationService
- Main service for generating comprehensive reports
- Handles chart creation using Plotly
- Integrates with Gemini API for AI insights
- Task-specific chart generation logic

### ChartExportService
- Handles CSV export functionality
- Provides chart summary statistics
- Supports multiple chart data formats

## Setup and Configuration

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Environment Variables
Create a `.env` file in the project root:
```bash
LLM_API_KEY=your_gemini_api_key_here
LLM_BASE_URL=https://generativelanguage.googleapis.com
LLM_MODEL=gemini-1.5-pro
```

### 3. Database Migration
```bash
python manage.py makemigrations aiapp
python manage.py migrate
```

### 4. Admin Setup
The report models are automatically registered in Django admin for easy management.

## Usage Examples

### Generate a Report
```python
from aiapp.services import ReportGenerationService

service = ReportGenerationService()
report = service.generate_report(model_id=123, report_type='analysis')
```

### Export Chart Data
```python
from aiapp.services import ChartExportService

csv_data = ChartExportService.export_chart_data_to_csv(chart_id=456)
```

## Testing

Run the comprehensive test suite:
```bash
python manage.py test aiapp
```

### Test Coverage
- Model creation and validation
- Service functionality
- API endpoint responses
- Chart generation and export
- Error handling

## Chart Generation Details

### Classification Tasks
- Class distribution pie charts
- Feature importance analysis
- Missing values visualization

### Regression Tasks
- Target variable distribution histograms
- Feature correlation analysis
- Outlier detection charts

### Time Series Tasks
- Trend line charts
- Seasonal decomposition
- Forecast visualization

### Clustering Tasks
- Feature correlation heatmaps
- Cluster distribution analysis
- Dimensionality reduction plots

## AI Insights

The system generates intelligent insights using Gemini-1.5-Pro:
- Data quality assessment
- Model performance analysis
- Recommendations for improvement
- Identification of potential issues

## Performance Optimizations

- Efficient chart generation with Plotly
- Lazy loading of chart images
- Pagination for large report lists
- Caching of generated insights

## Future Enhancements

- Real-time report updates
- Custom chart templates
- Advanced AI analysis features
- Integration with more LLM providers
- Dashboard creation capabilities

## Dependencies

- **Django**: Web framework
- **Plotly**: Interactive chart generation
- **Google Generative AI**: Gemini-1.5-Pro integration
- **LangChain**: LLM orchestration
- **Pandas**: Data manipulation
- **NumPy**: Numerical computations
- **Seaborn**: Statistical visualizations

## Support

For issues and questions, please refer to the project documentation or contact the development team.
