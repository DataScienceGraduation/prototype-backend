import pandas as pd
import numpy as np
import plotly.graph_objects as go
import base64
import io
import logging
import os
from typing import Dict, List, Any
from django.conf import settings
import google.generativeai as genai
from celery import shared_task

from automlapp.models import ModelEntry
from .models import Report, ChartData, DataInsight

logger = logging.getLogger(__name__)

@shared_task(bind=True)
def generate_report_async(self, model_entry_id: int, report_type: str = 'analysis'):
    """
    Celery task for asynchronous report generation

    Args:
        model_entry_id: ID of the ModelEntry
        report_type: Type of report to generate

    Returns:
        Report ID
    """
    try:
        service = ReportGenerationService()
        report = service.generate_report_with_progress(model_entry_id, report_type, self)
        return report.id
    except Exception as e:
        logger.error(f"Celery task failed for model {model_entry_id}: {e}")
        # Update task state to failure
        self.update_state(
            state='FAILURE',
            meta={'error': str(e), 'model_entry_id': model_entry_id}
        )
        raise

class ReportGenerationService:
    """
    Service class for generating comprehensive reports for trained models
    """

    def __init__(self):
        # Configure Gemini API
        self.setup_gemini()

    def setup_gemini(self):
        """Setup Gemini API configuration"""
        try:
            # Use LLM_API_KEY from environment variables
            api_key = os.getenv('LLM_API_KEY')
            if api_key:
                genai.configure(api_key=api_key)
                self.gemini_model = genai.GenerativeModel('gemini-1.5-pro')
            else:
                logger.warning("LLM_API_KEY not found in environment variables. AI insights will be disabled.")
                self.gemini_model = None
        except Exception as e:
            logger.error(f"Error setting up Gemini API: {e}")
            self.gemini_model = None

    def generate_report(self, model_entry_id: int, report_type: str = 'analysis') -> Report:
        """
        Generate a comprehensive report for a trained model (synchronous)

        Args:
            model_entry_id: ID of the ModelEntry
            report_type: Type of report to generate

        Returns:
            Report instance
        """
        return self.generate_report_with_progress(model_entry_id, report_type)

    def generate_report_with_progress(self, model_entry_id: int, report_type: str = 'analysis', task_instance=None) -> Report:
        """
        Generate a comprehensive report for a trained model with progress tracking

        Args:
            model_entry_id: ID of the ModelEntry
            report_type: Type of report to generate
            task_instance: Celery task instance for progress updates

        Returns:
            Report instance
        """
        def update_progress(percentage, step_description):
            """Helper function to update progress"""
            if task_instance:
                task_instance.update_state(
                    state='PROGRESS',
                    meta={'current': percentage, 'total': 100, 'status': step_description}
                )
            if 'report' in locals():
                report.progress_percentage = percentage
                report.current_step = step_description
                report.save()

        try:
            model_entry = ModelEntry.objects.get(id=model_entry_id)

            update_progress(15, 'Creating report instance...')

            # Create report instance
            report = Report.objects.create(
                model_entry=model_entry,
                title=f"Analysis Report for {model_entry.name}",
                description=f"Comprehensive analysis of {model_entry.task} model",
                report_type=report_type,
                status='generating',
                task_id=task_instance.request.id if task_instance else None,
                progress_percentage=15,
                current_step='Loading training data...'
            )

            update_progress(25, 'Loading training data...')
            # Load training data
            data_file_path = f'data/{model_entry.id}.csv'
            if not os.path.exists(data_file_path):
                raise FileNotFoundError(f"Training data file not found: {data_file_path}")
            df = pd.read_csv(data_file_path)

            update_progress(40, 'Generating charts...')
            # Generate different types of charts based on model task
            self._generate_charts_for_task(report, df, model_entry)

            update_progress(65, 'Generating data insights...')
            # Generate data insights
            self._generate_data_insights(report, df, model_entry)

            update_progress(80, 'Generating AI insights...')
            # Generate AI insights using Gemini
            if self.gemini_model:
                ai_insights = self._generate_ai_insights(report, df, model_entry)
                report.ai_insights = ai_insights

            update_progress(95, 'Finalizing report...')
            report.status = 'completed'
            report.progress_percentage = 100
            report.current_step = 'Report generation completed'
            report.save()

            update_progress(100, 'Report generation completed')
            return report

        except Exception as e:
            logger.error(f"Error generating report: {e}")
            if 'report' in locals():
                report.status = 'failed'
                report.current_step = f'Failed: {str(e)}'
                report.progress_percentage = 0
                report.save()
            raise

    def _generate_charts_for_task(self, report: Report, df: pd.DataFrame, model_entry: ModelEntry):
        """Generate charts based on the model task type"""
        task = model_entry.task

        try:
            if task == 'Classification':
                self._generate_classification_charts(report, df, model_entry)
            elif task == 'Regression':
                self._generate_regression_charts(report, df, model_entry)
            elif task == 'TimeSeries':
                self._generate_timeseries_charts(report, df, model_entry)
            elif task == 'Clustering':
                self._generate_clustering_charts(report, df, model_entry)

            # Generate common charts for all tasks
            self._generate_common_charts(report, df, model_entry)
        except Exception as e:
            logger.error(f"Error generating charts for task {task}: {e}")
            # Create a simple error chart instead of failing completely
            ChartData.objects.create(
                report=report,
                chart_type='error',
                title="Chart Generation Error",
                description=f"Error generating charts: {str(e)}",
                chart_data={'error': str(e)},
                chart_image_base64='',
                chart_html=f'<div>Error generating charts: {str(e)}</div>'
            )

    def _generate_classification_charts(self, report: Report, df: pd.DataFrame, model_entry: ModelEntry):
        """Generate charts specific to classification tasks"""
        target_col = model_entry.target_variable

        # Class distribution pie chart
        class_counts = df[target_col].value_counts()
        pie_chart = self._create_pie_chart(
            labels=class_counts.index.tolist(),
            values=class_counts.values.tolist(),
            title=f"Class Distribution - {target_col}"
        )

        ChartData.objects.create(
            report=report,
            chart_type='pie',
            title=f"Class Distribution",
            description=f"Distribution of classes in the target variable '{target_col}'",
            chart_data={
                'labels': class_counts.index.tolist(),
                'values': class_counts.values.tolist()
            },
            chart_image_base64=pie_chart['image'],
            chart_html=pie_chart['html']
        )

    def _generate_regression_charts(self, report: Report, df: pd.DataFrame, model_entry: ModelEntry):
        """Generate charts specific to regression tasks"""
        target_col = model_entry.target_variable

        # Target variable distribution
        hist_chart = self._create_histogram(
            values=df[target_col].dropna().tolist(),
            title=f"Distribution of {target_col}"
        )

        # Generate chart explanation
        hist_values = df[target_col].dropna().tolist()
        hist_explanation = self._generate_chart_explanation(
            chart_type='histogram',
            data={'values': hist_values},
            title=f"Target Variable Distribution",
            context=f"Distribution analysis of {target_col} showing data spread and patterns"
        )

        ChartData.objects.create(
            report=report,
            chart_type='histogram',
            title=f"Target Variable Distribution",
            description=hist_explanation,
            chart_data={'values': [float(x) for x in hist_values]},
            chart_image_base64=hist_chart['image'],
            chart_html=hist_chart['html']
        )

    def _generate_timeseries_charts(self, report: Report, df: pd.DataFrame, model_entry: ModelEntry):
        """Generate charts specific to time series tasks"""
        target_col = model_entry.target_variable
        datetime_col = getattr(model_entry, 'datetime_column', None)

        # If no datetime column is specified, try to find one
        if not datetime_col:
            for col in df.columns:
                if df[col].dtype == 'object':
                    try:
                        pd.to_datetime(df[col], errors='raise')
                        datetime_col = col
                        break
                    except:
                        continue

        if datetime_col and datetime_col in df.columns:
            # Time series line chart
            df_sorted = df.sort_values(datetime_col)
            line_chart = self._create_line_chart(
                x=df_sorted[datetime_col].tolist(),
                y=df_sorted[target_col].tolist(),
                title=f"Time Series - {target_col}"
            )

            # Generate chart explanation
            chart_explanation = self._generate_chart_explanation(
                chart_type='line',
                data={'x': df_sorted[datetime_col], 'y': df_sorted[target_col]},
                title=f"Time Series Trend",
                context=f"Sales data from {df_sorted[datetime_col].min()} to {df_sorted[datetime_col].max()}"
            )

            ChartData.objects.create(
                report=report,
                chart_type='line',
                title=f"Time Series Trend",
                description=chart_explanation,
                chart_data={
                    'x': [str(x) for x in df_sorted[datetime_col].tolist()],
                    'y': [float(x) for x in df_sorted[target_col].tolist()]
                },
                chart_image_base64=line_chart['image'],
                chart_html=line_chart['html']
            )

    def _generate_clustering_charts(self, report: Report, df: pd.DataFrame, model_entry: ModelEntry):
        """Generate charts specific to clustering tasks"""
        # Feature correlation heatmap
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 1:
            corr_matrix = df[numeric_cols].corr()
            heatmap_chart = self._create_correlation_heatmap(
                correlation_matrix=corr_matrix,
                title="Feature Correlation Matrix"
            )

            ChartData.objects.create(
                report=report,
                chart_type='correlation',
                title="Feature Correlation Matrix",
                description="Correlation between numeric features",
                chart_data=corr_matrix.to_dict(),
                chart_image_base64=heatmap_chart['image'],
                chart_html=heatmap_chart['html']
            )

    def _generate_common_charts(self, report: Report, df: pd.DataFrame, model_entry: ModelEntry):
        """Generate common charts for all task types"""
        # Missing values chart
        missing_data = df.isnull().sum()
        missing_data = missing_data[missing_data > 0]

        if len(missing_data) > 0:
            bar_chart = self._create_bar_chart(
                x=missing_data.index.tolist(),
                y=missing_data.values.tolist(),
                title="Missing Values by Feature"
            )

            ChartData.objects.create(
                report=report,
                chart_type='bar',
                title="Missing Values Analysis",
                description="Number of missing values per feature",
                chart_data={
                    'x': missing_data.index.tolist(),
                    'y': [int(x) for x in missing_data.values.tolist()]
                },
                chart_image_base64=bar_chart['image'],
                chart_html=bar_chart['html']
            )

    def _create_pie_chart(self, labels: List[str], values: List[float], title: str) -> Dict[str, str]:
        """Create a pie chart using Plotly"""
        fig = go.Figure(data=[go.Pie(
            labels=labels,
            values=values,
            hole=0.3,
            textinfo='label+percent',
            textposition='outside'
        )])

        fig.update_layout(
            title=title,
            font=dict(size=12),
            showlegend=True,
            width=600,
            height=400
        )

        # Generate base64 image
        img_bytes = fig.to_image(format="png", width=600, height=400)
        img_base64 = base64.b64encode(img_bytes).decode('utf-8')

        # Generate HTML
        html = fig.to_html(include_plotlyjs='cdn', div_id=f"chart_{hash(title)}")

        return {'image': img_base64, 'html': html}

    def _create_line_chart(self, x: List, y: List[float], title: str) -> Dict[str, str]:
        """Create a line chart using Plotly"""
        fig = go.Figure(data=[go.Scatter(
            x=x,
            y=y,
            mode='lines+markers',
            line=dict(width=2),
            marker=dict(size=6)
        )])

        fig.update_layout(
            title=title,
            xaxis_title="Time",
            yaxis_title="Value",
            font=dict(size=12),
            width=800,
            height=400
        )

        img_bytes = fig.to_image(format="png", width=800, height=400)
        img_base64 = base64.b64encode(img_bytes).decode('utf-8')
        html = fig.to_html(include_plotlyjs='cdn', div_id=f"chart_{hash(title)}")

        return {'image': img_base64, 'html': html}

    def _create_histogram(self, values: List[float], title: str) -> Dict[str, str]:
        """Create a histogram using Plotly"""
        fig = go.Figure(data=[go.Histogram(x=values, nbinsx=30)])

        fig.update_layout(
            title=title,
            xaxis_title="Value",
            yaxis_title="Frequency",
            font=dict(size=12),
            width=600,
            height=400
        )

        img_bytes = fig.to_image(format="png", width=600, height=400)
        img_base64 = base64.b64encode(img_bytes).decode('utf-8')
        html = fig.to_html(include_plotlyjs='cdn', div_id=f"chart_{hash(title)}")

        return {'image': img_base64, 'html': html}

    def _create_bar_chart(self, x: List[str], y: List[float], title: str) -> Dict[str, str]:
        """Create a bar chart using Plotly"""
        fig = go.Figure(data=[go.Bar(x=x, y=y)])

        fig.update_layout(
            title=title,
            xaxis_title="Features",
            yaxis_title="Count",
            font=dict(size=12),
            width=600,
            height=400
        )

        img_bytes = fig.to_image(format="png", width=600, height=400)
        img_base64 = base64.b64encode(img_bytes).decode('utf-8')
        html = fig.to_html(include_plotlyjs='cdn', div_id=f"chart_{hash(title)}")

        return {'image': img_base64, 'html': html}

    def _create_correlation_heatmap(self, correlation_matrix: pd.DataFrame, title: str) -> Dict[str, str]:
        """Create a correlation heatmap using Plotly"""
        fig = go.Figure(data=go.Heatmap(
            z=correlation_matrix.values,
            x=correlation_matrix.columns,
            y=correlation_matrix.index,
            colorscale='RdBu',
            zmid=0
        ))

        fig.update_layout(
            title=title,
            font=dict(size=12),
            width=600,
            height=600
        )

        img_bytes = fig.to_image(format="png", width=600, height=600)
        img_base64 = base64.b64encode(img_bytes).decode('utf-8')
        html = fig.to_html(include_plotlyjs='cdn', div_id=f"chart_{hash(title)}")

        return {'image': img_base64, 'html': html}

    def _generate_data_insights(self, report: Report, df: pd.DataFrame, model_entry: ModelEntry):
        """Generate data insights for the report"""
        # Data shape insight
        DataInsight.objects.create(
            report=report,
            insight_type='data_distribution',
            title="Dataset Overview",
            description=f"Dataset contains {df.shape[0]} rows and {df.shape[1]} columns",
            insight_data={
                'rows': int(df.shape[0]),
                'columns': int(df.shape[1]),
                'memory_usage': int(df.memory_usage(deep=True).sum())
            },
            priority=1
        )

        # Missing values insight
        missing_data = df.isnull().sum()
        total_missing = missing_data.sum()
        if total_missing > 0:
            DataInsight.objects.create(
                report=report,
                insight_type='missing_values',
                title="Missing Values Analysis",
                description=f"Found {total_missing} missing values across {(missing_data > 0).sum()} columns",
                insight_data={
                    'total_missing': int(total_missing),
                    'columns_with_missing': int((missing_data > 0).sum()),
                    'missing_percentage': round(float((total_missing / (df.shape[0] * df.shape[1])) * 100), 2)
                },
                priority=2
            )

        # Data types insight
        dtype_counts = df.dtypes.value_counts().to_dict()
        DataInsight.objects.create(
            report=report,
            insight_type='data_distribution',
            title="Data Types Distribution",
            description=f"Dataset contains {len(dtype_counts)} different data types",
            insight_data={
                'data_types': {str(k): int(v) for k, v in dtype_counts.items()},
                'numeric_columns': int(len(df.select_dtypes(include=[np.number]).columns)),
                'categorical_columns': int(len(df.select_dtypes(include=['object']).columns))
            },
            priority=3
        )

    def _generate_ai_insights(self, report: Report, df: pd.DataFrame, model_entry: ModelEntry) -> str:
        """Generate AI-powered insights using Gemini"""
        try:
            # Prepare data summary for AI analysis
            data_summary = {
                'shape': df.shape,
                'columns': df.columns.tolist(),
                'data_types': df.dtypes.to_dict(),
                'missing_values': df.isnull().sum().to_dict(),
                'task_type': model_entry.task,
                'target_variable': model_entry.target_variable,
                'model_performance': {
                    'metric': model_entry.evaluation_metric,
                    'value': model_entry.evaluation_metric_value
                }
            }

            # Create prompt for Gemini
            prompt = f"""
            As a data science expert, analyze the following dataset and model information to provide insights:

            Dataset Information:
            - Shape: {data_summary['shape']}
            - Task Type: {data_summary['task_type']}
            - Target Variable: {data_summary['target_variable']}
            - Model Performance: {data_summary['model_performance']['metric']} = {data_summary['model_performance']['value']}

            Data Types: {data_summary['data_types']}
            Missing Values: {data_summary['missing_values']}

            Please provide:
            1. Key insights about the data quality
            2. Observations about the model performance
            3. Recommendations for improvement
            4. Potential data issues or concerns

            Keep the response concise and actionable.
            """

            response = self.gemini_model.generate_content(prompt)
            return response.text

        except Exception as e:
            logger.error(f"Error generating AI insights: {e}")
            return "AI insights generation failed. Please check the configuration."

    def _generate_chart_explanation(self, chart_type: str, data: dict, title: str, context: str) -> str:
        """Generate AI-powered explanation for specific charts"""
        try:
            if not self.gemini_model:
                return self._get_default_chart_explanation(chart_type, data, title, context)

            # Prepare chart data summary for AI
            if chart_type == 'line':
                y_values = data['y'].dropna()
                trend_direction = "increasing" if y_values.iloc[-1] > y_values.iloc[0] else "decreasing"
                volatility = "high" if y_values.std() > y_values.mean() * 0.5 else "moderate" if y_values.std() > y_values.mean() * 0.2 else "low"

                prompt = f"""
                Analyze this time series chart and provide a clear, concise explanation:

                Chart: {title}
                Context: {context}
                Data points: {len(y_values)}
                Value range: {y_values.min():.1f} to {y_values.max():.1f}
                Average: {y_values.mean():.1f}
                Trend: {trend_direction}
                Volatility: {volatility}

                Provide a 2-3 sentence explanation that describes:
                1. What the chart shows
                2. Key patterns or trends
                3. Notable observations

                Keep it clear and actionable for business users.
                """

                response = self.gemini_model.generate_content(prompt)
                return response.text.strip()

            elif chart_type == 'histogram':
                values = data['values']
                prompt = f"""
                Explain this distribution chart:

                Chart: {title}
                Data points: {len(values)}
                Range: {min(values):.1f} to {max(values):.1f}
                Average: {np.mean(values):.1f}

                Describe the distribution shape and what it means for the business.
                """

                response = self.gemini_model.generate_content(prompt)
                return response.text.strip()

            else:
                return self._get_default_chart_explanation(chart_type, data, title, context)

        except Exception as e:
            logger.error(f"Error generating chart explanation: {e}")
            return self._get_default_chart_explanation(chart_type, data, title, context)

    def _get_default_chart_explanation(self, chart_type: str, data: dict, title: str, context: str) -> str:
        """Provide default explanations when AI is not available"""
        if chart_type == 'line':
            y_values = data['y'].dropna()
            return f"This time series chart shows {title.lower()} over time. The data ranges from {y_values.min():.1f} to {y_values.max():.1f} with an average of {y_values.mean():.1f}. {context}"
        elif chart_type == 'histogram':
            values = data['values']
            return f"This histogram shows the distribution of {title.lower()}. Values range from {min(values):.1f} to {max(values):.1f} with an average of {np.mean(values):.1f}."
        elif chart_type == 'bar':
            return f"This bar chart displays {title.lower()}. {context}"
        elif chart_type == 'pie':
            return f"This pie chart shows the proportional breakdown of {title.lower()}."
        else:
            return f"This {chart_type} chart visualizes {title.lower()}. {context}"


class ChartExportService:
    """
    Service for exporting chart data and generating downloadable files
    """

    @staticmethod
    def export_chart_data_to_csv(chart_id: int) -> str:
        """Export chart data to CSV format"""
        try:
            chart = ChartData.objects.get(id=chart_id)
            chart_data = chart.get_chart_data_as_dict()

            if chart.chart_type == 'pie':
                df = pd.DataFrame({
                    'Label': chart_data.get('labels', []),
                    'Value': chart_data.get('values', [])
                })
            elif chart.chart_type == 'line':
                df = pd.DataFrame({
                    'X': chart_data.get('x', []),
                    'Y': chart_data.get('y', [])
                })
            elif chart.chart_type == 'bar':
                df = pd.DataFrame({
                    'Category': chart_data.get('x', []),
                    'Value': chart_data.get('y', [])
                })
            else:
                # Generic format
                df = pd.DataFrame(chart_data)

            # Convert to CSV string
            csv_buffer = io.StringIO()
            df.to_csv(csv_buffer, index=False)
            return csv_buffer.getvalue()

        except Exception as e:
            logger.error(f"Error exporting chart data: {e}")
            raise

    @staticmethod
    def get_chart_summary_stats(chart_id: int) -> Dict[str, Any]:
        """Get summary statistics for chart data"""
        try:
            chart = ChartData.objects.get(id=chart_id)
            chart_data = chart.get_chart_data_as_dict()

            stats = {
                'chart_type': chart.chart_type,
                'title': chart.title,
                'data_points': 0,
                'summary': {}
            }

            if chart.chart_type == 'pie':
                values = chart_data.get('values', [])
                stats['data_points'] = len(values)
                stats['summary'] = {
                    'total': sum(values),
                    'max_value': max(values) if values else 0,
                    'min_value': min(values) if values else 0
                }
            elif chart.chart_type in ['line', 'bar']:
                y_values = chart_data.get('y', [])
                stats['data_points'] = len(y_values)
                if y_values:
                    # Filter out NaN values for statistics
                    clean_values = [v for v in y_values if v is not None and not (isinstance(v, float) and np.isnan(v))]
                    if clean_values:
                        stats['summary'] = {
                            'mean': float(np.mean(clean_values)),
                            'std': float(np.std(clean_values)),
                            'max': float(np.max(clean_values)),
                            'min': float(np.min(clean_values)),
                            'valid_points': len(clean_values),
                            'missing_points': len(y_values) - len(clean_values)
                        }
                    else:
                        stats['summary'] = {
                            'mean': 0,
                            'std': 0,
                            'max': 0,
                            'min': 0,
                            'valid_points': 0,
                            'missing_points': len(y_values)
                        }

            return stats

        except Exception as e:
            logger.error(f"Error getting chart summary: {e}")
            raise