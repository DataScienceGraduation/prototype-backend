import pandas as pd
import numpy as np
import plotly.graph_objects as go
import base64
import io
import logging
import os
import json
from typing import Dict, List, Any
from django.conf import settings
from django.core.files.storage import default_storage
import google.generativeai as genai
from celery import shared_task
from google.generativeai.types import GenerationConfig
import matplotlib.pyplot as plt
import re  # Add this at the top if not present
from asgiref.sync import async_to_sync
from channels.layers import get_channel_layer

# Try to import kaleido and configure it
try:
    import kaleido
    KALEIDO_AVAILABLE = True
    logger = logging.getLogger(__name__)
    logger.info(f"Kaleido version {kaleido.__version__} loaded successfully")
except ImportError as e:
    KALEIDO_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning(f"Kaleido not available: {e}")

# Configure plotly to use kaleido if available
if KALEIDO_AVAILABLE:
    try:
        import plotly.io as pio
        pio.kaleido.scope.default_format = "png"
        pio.kaleido.scope.default_engine = "kaleido"
    except Exception as e:
        logger.warning(f"Failed to configure Kaleido: {e}")
        KALEIDO_AVAILABLE = False



from automlapp.models import ModelEntry
from .models import Report, ChartData, DataInsight

logger = logging.getLogger(__name__)

def safe_plotly_to_image(fig, format="jpeg", width=800, height=500):
    """Safely convert plotly figure to image with fallback to matplotlib, using JPEG and smaller size for reduced file size"""
    try:
        if KALEIDO_AVAILABLE:
            img_bytes = fig.to_image(format=format, width=width, height=height, engine="kaleido")
            return base64.b64encode(img_bytes).decode('utf-8')
        else:
            logger.warning("Kaleido not available, skipping image generation")
            return None
    except Exception as e:
        logger.error(f"Failed to generate image with Kaleido: {e}")
        try:
            # Try without specifying engine
            img_bytes = fig.to_image(format=format, width=width, height=height)
            return base64.b64encode(img_bytes).decode('utf-8')
        except Exception as e2:
            logger.error(f"Failed to generate image without engine: {e2}")
            # Return None instead of crashing - frontend will handle missing images
            logger.info("Returning None for image - frontend will show chart data without image")
            return None




def ensure_json_serializable(data):
    """
    Ensure data is JSON serializable by handling NaN, inf, and other problematic values
    """
    try:
        if data is None:
            return None
        elif isinstance(data, dict):
            result = {}
            for k, v in data.items():
                key = str(k) if k is not None else 'null'
                result[key] = ensure_json_serializable(v)
            return result
        elif isinstance(data, (list, tuple)):
            return [ensure_json_serializable(item) for item in data]
        elif isinstance(data, (np.integer, int)):
            # Handle potential overflow
            try:
                return int(data)
            except (OverflowError, ValueError):
                return str(data)
        elif isinstance(data, (np.floating, float)):
            if np.isnan(data) or np.isinf(data):
                return None
            try:
                return round(float(data), 2)
            except (OverflowError, ValueError):
                return str(data)
        elif isinstance(data, np.ndarray):
            return ensure_json_serializable(data.tolist())
        elif isinstance(data, pd.Series):
            return ensure_json_serializable(data.tolist())
        elif isinstance(data, pd.DataFrame):
            return ensure_json_serializable(data.to_dict())
        elif pd.isna(data):
            return None
        elif isinstance(data, str):
            # Ensure string is valid UTF-8
            try:
                return str(data).encode('utf-8').decode('utf-8')
            except UnicodeError:
                return str(data).encode('utf-8', errors='replace').decode('utf-8')
        elif isinstance(data, bool):
            return bool(data)
        elif hasattr(data, '__dict__'):
            # Handle objects with attributes
            return str(data)
        else:
            # Try to convert to string as fallback
            try:
                result = str(data)
                # Test if the string representation is reasonable
                if len(result) > 1000:  # Avoid extremely long strings
                    return f"<{type(data).__name__} object>"
                return result
            except Exception:
                logger.warning(f"Could not serialize data of type {type(data)}")
                return f"<{type(data).__name__} object>"
    except Exception as e:
        logger.error(f"Error in ensure_json_serializable: {e}")
        return None


def validate_and_save_chart_data(report, chart_type, title, description, chart_data, chart_image_base64, chart_html):
    """
    Validate chart data and save it safely
    """
    try:
        # Ensure data is JSON serializable
        clean_data = ensure_json_serializable(chart_data)

        # Additional validation - ensure clean_data is not None
        if clean_data is None:
            clean_data = {}

        # Test JSON serialization with detailed error handling
        try:
            json_str = json.dumps(clean_data)
            logger.info(f"Chart data JSON validation passed for {title}. Data size: {len(json_str)} chars")
        except (TypeError, ValueError) as json_error:
            logger.error(f"JSON serialization failed for {title}: {json_error}")
            logger.error(f"Problematic data: {clean_data}")
            # Fallback to empty dict
            clean_data = {'error': 'JSON serialization failed', 'original_error': str(json_error)}

        # Save the chart
        chart = ChartData.objects.create(
            report=report,
            chart_type=chart_type,
            title=title,
            description=description,
            chart_data=clean_data,
            chart_image_base64=chart_image_base64 or '',
            chart_html=chart_html or ''
        )
        logger.info(f"Successfully saved chart: {title} (ID: {chart.id})")
        return chart

    except Exception as e:
        logger.error(f"Error saving chart data for {title}: {e}")
        logger.error(f"Chart data type: {type(chart_data)}")
        logger.error(f"Chart data content: {chart_data}")

        # Save a minimal chart with error info
        try:
            error_chart = ChartData.objects.create(
                report=report,
                chart_type='error',
                title=f"Error: {title}",
                description=f"Chart generation failed: {str(e)}",
                chart_data={'error': str(e), 'chart_type': chart_type},
                chart_image_base64='',
                chart_html=f'<div>Error: {str(e)}</div>'
            )
            logger.info(f"Created error chart for failed {title}")
            return error_chart
        except Exception as fallback_error:
            logger.error(f"Failed to create error chart: {fallback_error}")
            raise e

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
        # By re-raising the exception, we let Celery handle the failure state automatically.
        # This is more robust than manually updating the state. The generate_report_with_progress
        # method already handles updating the Report model's status in the database.
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
                self.gemini_model = genai.GenerativeModel('gemini-2.5-pro')
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

            update_progress(25, 'Loading and preprocessing data...')
            # Load raw training data
            data_file_path = f'data/{model_entry.id}.csv'
            if not default_storage.exists(data_file_path):
                raise FileNotFoundError(f"Training data file not found: {data_file_path}")
            with default_storage.open(data_file_path) as f:
                df_raw = pd.read_csv(f)

            # Load the preprocessing pipeline and transform the data to get the actual model input
            pipeline_path = f'pipelines/{model_entry.id}.pkl'
            if not default_storage.exists(pipeline_path):
                raise FileNotFoundError(f"Pipeline file not found: {pipeline_path}")

            import joblib
            with default_storage.open(pipeline_path) as f:
                pipeline = joblib.load(f)
            df = pipeline.transform(df_raw)

            logger.info(f"Using preprocessed data for insights. Raw shape: {df_raw.shape}, Processed shape: {df.shape}")

            update_progress(40, 'Generating charts with LLM...')
            # Generate charts using the LLM
            self._generate_charts_with_llm(report, df, model_entry)

            update_progress(65, 'Generating data insights...')
            # Generate data insights using both raw and processed data
            self._generate_data_insights(report, df, model_entry, df_raw)

            update_progress(80, 'Generating AI insights...')
            # Generate AI insights using Gemini with both datasets
            if self.gemini_model:
                ai_insights = self._generate_ai_insights(report, df, model_entry, df_raw)
                report.ai_insights = ai_insights

            update_progress(95, 'Finalizing report...')
            report.status = 'completed'
            report.progress_percentage = 100
            report.current_step = 'Report generation completed'
            report.save()

            # WebSocket notification for report completion (if async)
            if task_instance and hasattr(task_instance, 'request') and hasattr(task_instance.request, 'id'):
                channel_layer = get_channel_layer()
                async_to_sync(channel_layer.group_send)(
                    f"report_{task_instance.request.id}",
                    {
                        "type": "report_ready",
                        "report_id": report.id,
                    }
                )

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

    def _generate_charts_with_llm(self, report: Report, df: pd.DataFrame, model_entry: ModelEntry):
        """
        Use an LLM to analyze the dataframe, suggest the most suitable charts based on data characteristics,
        generate sophisticated Python code for them, execute the code, and save the results.
        """
        if not self.gemini_model:
            logger.warning("Gemini model not available. Skipping LLM-based chart generation.")
            return

        try:
            # Prepare comprehensive data analysis for the LLM
            df_head = df.head().to_string()
            df_tail = df.tail().to_string()
            
            with io.StringIO() as buf:
                df.info(buf=buf)
                df_info_str = buf.getvalue()

            # Calculate key data characteristics
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
            datetime_cols = []
            
            # Detect datetime columns
            for col in df.columns:
                if df[col].dtype == 'object':
                    try:
                        pd.to_datetime(df[col], errors='raise')
                        datetime_cols.append(col)
                    except:
                        continue

            # Calculate statistical summaries
            numeric_summary = {}
            if numeric_cols:
                numeric_summary = df[numeric_cols].describe().to_dict()
                # Add correlation analysis if multiple numeric columns
                if len(numeric_cols) > 1:
                    correlations = df[numeric_cols].corr().round(3).to_dict()

            # Categorical analysis
            categorical_summary = {}
            for col in categorical_cols:
                if col in df.columns:
                    value_counts = df[col].value_counts()
                    categorical_summary[col] = {
                        'unique_count': int(value_counts.nunique()),
                        'top_values': value_counts.head(5).to_dict(),
                        'missing_count': int(df[col].isnull().sum())
                    }

            # Target variable analysis
            target_analysis = {}
            if model_entry.target_variable and model_entry.target_variable in df.columns:
                target_col = df[model_entry.target_variable]
                if model_entry.task == 'Classification':
                    target_analysis = {
                        'type': 'classification',
                        'unique_classes': int(target_col.nunique()),
                        'class_distribution': target_col.value_counts().to_dict(),
                        'is_balanced': bool(target_col.value_counts().std() / target_col.value_counts().mean() < 0.5)
                    }
                elif model_entry.task == 'Regression':
                    target_analysis = {
                        'type': 'regression',
                        'mean': float(target_col.mean()),
                        'std': float(target_col.std()),
                        'min': float(target_col.min()),
                        'max': float(target_col.max()),
                        'skewness': float(target_col.skew()),
                        'kurtosis': float(target_col.kurtosis())
                    }
                elif model_entry.task == 'TimeSeries':
                    target_analysis = {
                        'type': 'timeseries',
                        'mean': float(target_col.mean()),
                        'std': float(target_col.std()),
                        'trend': 'increasing' if target_col.iloc[-1] > target_col.iloc[0] else 'decreasing',
                        'volatility': float(target_col.std() / target_col.mean()) if target_col.mean() != 0 else 0
                    }

            # Create sophisticated prompt for data-driven chart recommendations
            prompt = f"""
            You are an expert data scientist and business analyst specializing in creating insightful, sophisticated data visualizations. Your task is to analyze this dataset and recommend the MOST SUITABLE and INSIGHTFUL charts that will provide maximum business value.

            ## DATASET ANALYSIS CONTEXT
            **Model Task**: {model_entry.task}
            **Target Variable**: {model_entry.target_variable}
            **Dataset Shape**: {df.shape[0]} rows × {df.shape[1]} columns

            **Data Types**:
            - Numeric Features: {len(numeric_cols)} ({', '.join(numeric_cols) if len(numeric_cols) <= 5 else ', '.join(numeric_cols[:5]) + '...'})
            - Categorical Features: {len(categorical_cols)} ({', '.join(categorical_cols) if len(categorical_cols) <= 5 else ', '.join(categorical_cols[:5]) + '...'})
            - Datetime Features: {len(datetime_cols)} ({', '.join(datetime_cols) if datetime_cols else 'None'})

            **Target Variable Analysis**:
            {json.dumps(target_analysis, indent=2)}

            **Numeric Features Summary**:
            {json.dumps({k: {sk: round(sv, 3) if isinstance(sv, float) else sv for sk, sv in v.items()} for k, v in numeric_summary.items()}, indent=2)}

            **Categorical Features Summary**:
            {json.dumps(categorical_summary, indent=2)}

            **Dataset Sample**:
            {df_head}

            ## YOUR TASK
            Based on this comprehensive data analysis, recommend 3-5 EXCEPTIONAL charts that will provide the most valuable insights for business stakeholders. 

            **Requirements**:
            1. **Data-Driven Selection**: Choose chart types based on actual data characteristics, not predefined options
            2. **Business Focus**: Each chart should answer a specific business question or reveal actionable insights
            3. **Sophistication**: Use advanced visualization techniques when appropriate (subplots, annotations, custom styling)
            4. **Uniqueness**: Avoid generic charts - create visualizations that are specifically tailored to this dataset
            5. **Storytelling**: Each chart should tell a compelling story about the data
            6. **Findings-Focused Explanation**: For each chart, your explanation MUST describe the actual findings, trends, and patterns visible in the chart/data. Do NOT describe what this chart type generally shows. List at least 2-3 specific findings or trends visible in the chart.
            7. **STRICT**: For the 'detailed_analysis' field, you MUST NOT repeat the business question or chart type. Instead, provide 3-4 specific findings, trends, or patterns that are visible in the chart/data. If there are strong correlations, outliers, clusters, or other notable features, mention them explicitly. If there are no strong patterns, state that clearly. Your answer should be actionable and data-driven.

            **Chart Types to Consider** (but not limited to):
            - Distribution plots (histograms, box plots, violin plots, density plots)
            - Relationship plots (scatter plots, correlation heatmaps, pair plots)
            - Time series plots (line charts, seasonal decomposition, trend analysis)
            - Composition plots (stacked bar charts, waterfall charts, sunburst charts)
            - Comparison plots (grouped bar charts, radar charts, parallel coordinates)
            - Advanced plots (3D scatter plots, contour plots, bubble charts, treemaps)

            For each recommended chart, provide:
            1. **Chart Type**: The specific type of visualization
            2. **Business Question**: What business question does this chart answer?
            3. **Data Justification**: Why is this chart type most suitable for this specific data?
            4. **Sophisticated Python Code**: Complete, production-ready code using plotly.graph_objects
            5. **Detailed Analysis**: Write a comprehensive 4-6 sentence analysis explaining what THIS chart specifically reveals, including key patterns, outliers, trends, and business insights. Focus ONLY on what is actually found in the data and chart, not what this chart type is generally used for. List at least 2-3 specific findings or trends visible in the chart.

            **Code Requirements**:
            - Use plotly.graph_objects (imported as `go`)
            - Include sophisticated styling (colors, fonts, layouts, annotations)
            - Add interactive elements where appropriate
            - Handle edge cases (missing data, outliers, etc.)
            - Use subplots for complex visualizations
            - Include proper titles, axis labels, and legends
            - The DataFrame is available as `df`

            Return your response as a JSON array with this exact structure:
            ```json
            [
                {{
                    "chart_type": "specific_chart_type",
                    "business_question": "What specific business question does this answer?",
                    "data_justification": "Why this chart type is perfect for this data",
                    "python_code": "Complete plotly code that creates a variable named 'fig'",
                    "detailed_analysis": "Comprehensive 4-6 sentence analysis explaining what the chart reveals, key patterns, outliers, trends, and specific business insights. Focus on actionable insights and what stakeholders should learn from this visualization. Do NOT repeat the business question or chart type."
                }}
            ]
            ```

            **IMPORTANT**: Focus on creating unique, sophisticated visualizations that are specifically tailored to this dataset's characteristics. Don't create generic charts - make each one tell a compelling story about the data.
            """

            # Add special instructions for time series problems
            is_time_series = model_entry.task == 'TimeSeries'
            if is_time_series:
                prompt += """
                SPECIAL INSTRUCTIONS FOR TIME SERIES:
                - Use line plots (go.Scatter with mode='lines') for time series data.
                - Consider adding seasonal decomposition or autocorrelation plots if the data is long enough.
                - Do NOT use scatter plots with 'trendline' property.
                - The x-axis should be the datetime or sequential index.
                - Focus on trends, seasonality, and forecast visualization.
                - Do NOT use unsupported properties in plotly.graph_objs.
                - If you want to show a trendline, add a separate line trace for the trend.
                - Do NOT use go.Scatter for time series unless mode='lines' or mode='lines+markers'.
                - Do NOT use seaborn or matplotlib, only plotly.graph_objects as go.
                """

            # Call the LLM with JSON response type
            generation_config = GenerationConfig(response_mime_type="application/json")
            response = self.gemini_model.generate_content(prompt, generation_config=generation_config)
            
            logger.info(f"LLM response received for chart generation. Response length: {len(response.text)}")
            logger.info(f"LLM response preview: {response.text[:500]}...")
            
            chart_suggestions = json.loads(response.text)
            logger.info(f"Successfully parsed {len(chart_suggestions)} chart suggestions from LLM")

            # Track if any chart was successfully generated
            chart_success = False

            # Process each sophisticated chart suggestion
            for suggestion in chart_suggestions:
                chart_type = suggestion.get('chart_type')
                business_question = suggestion.get('business_question')
                data_justification = suggestion.get('data_justification')
                python_code = suggestion.get('python_code')
                detailed_analysis = suggestion.get('detailed_analysis')

                if not all([chart_type, business_question, data_justification, python_code, detailed_analysis]):
                    logger.warning(f"Skipping incomplete chart suggestion: {suggestion}")
                    continue

                max_attempts = 5
                attempt = 0
                success = False
                last_error = None
                current_python_code = python_code
                current_detailed_analysis = detailed_analysis
                while attempt < max_attempts and not success:
                    try:
                        # Remove unsupported 'trendline' property if present
                        if "trendline" in current_python_code:
                            current_python_code = current_python_code.replace("trendline=", "# trendline=")
                            logger.warning("Removed unsupported 'trendline' property from LLM code.")

                        # Remove redundant imports and fig.show() lines using regex (improved)
                        current_python_code = re.sub(
                            r'^\s*(import\s+plotly\\.graph_objects\s+as\s+go|from\s+plotly\\.subplots\s+import\s+make_subplots|import\s+pandas\s+as\s+pd|import\s+numpy\s+as\s+np)\s*$',
                            '',
                            current_python_code,
                            flags=re.MULTILINE
                        )
                        current_python_code = re.sub(r'^\s*fig\\.show\(\)\s*$', '', current_python_code, flags=re.MULTILINE)
                        current_python_code = current_python_code.replace('column_titles=df.columns', 'column_titles=list(df.columns)')
                        current_python_code = current_python_code.replace('row_titles=df.columns', 'row_titles=list(df.columns)')

                        logger.debug(f"Cleaned LLM-generated code for chart '{chart_type}' (attempt {attempt+1}):\n{current_python_code}")

                        from plotly.subplots import make_subplots
                        try:
                            import plotly.graph_objects as go
                        except NameError:
                            import plotly.graph_objects as go
                        local_scope = {'df': df, 'go': go, 'pd': pd, 'np': np, 'make_subplots': make_subplots}
                        import traceback
                        try:
                            exec(current_python_code, local_scope, local_scope)
                        except Exception as e:
                            logger.error(f"Error executing LLM code (attempt {attempt+1}): {e}\n{traceback.format_exc()}\nCode:\n{current_python_code}")
                            raise
                        fig = local_scope.get('fig')
                        if not isinstance(fig, go.Figure):
                            logger.error(f"Generated code for chart '{chart_type}' did not produce a plotly Figure object.")
                            raise ValueError("Generated code did not produce a plotly Figure object.")

                        # Generate high-quality image and HTML from the figure
                        img_base64 = safe_plotly_to_image(fig, format="png", width=1000, height=600)
                        html = fig.to_html(include_plotlyjs='cdn', div_id=f"chart_{hash(chart_type + business_question)}")

                        # Try to extract chart data for explanation
                        chart_data = {}
                        try:
                            if chart_type == 'line':
                                chart_data = {
                                    'x': fig.data[0].x if fig.data else [],
                                    'y': fig.data[0].y if fig.data else []
                                }
                            elif chart_type == 'bar':
                                chart_data = {
                                    'x': fig.data[0].x if fig.data else [],
                                    'y': fig.data[0].y if fig.data else []
                                }
                            elif chart_type == 'histogram':
                                chart_data = {
                                    'values': fig.data[0].x if fig.data else []
                                }
                            elif chart_type == 'pie':
                                chart_data = {
                                    'labels': fig.data[0].labels if fig.data else [],
                                    'values': fig.data[0].values if fig.data else []
                                }
                        except Exception as e:
                            logger.warning(f"Could not extract chart data for explanation: {e}")
                            chart_data = {}

                        use_llm_analysis = False
                        if current_detailed_analysis and len(current_detailed_analysis.strip()) > 0:
                            lower_analysis = current_detailed_analysis.lower()
                            question_in_analysis = business_question.lower()[:30] in lower_analysis
                            chart_type_in_analysis = chart_type.replace('_', ' ').lower() in lower_analysis
                            if not question_in_analysis and not chart_type_in_analysis:
                                chart_description = f"{current_detailed_analysis}\n\n**Data Justification**: {data_justification}"
                                use_llm_analysis = True
                        if not use_llm_analysis:
                            try:
                                chart_explanation = self._generate_chart_explanation(chart_type, chart_data, chart_type.replace('_', ' ').title(), business_question)
                                chart_description = f"{chart_explanation}\n\n**Data Justification**: {data_justification}"
                            except Exception as e:
                                logger.warning(f"Falling back to generic chart description: {e}")
                                chart_description = f"This {chart_type} chart visualizes {chart_type.replace('_', ' ').title()}. {business_question}"

                        chart_title = f"{chart_type.replace('_', ' ').title()}: {business_question}"
                        self._save_llm_chart(
                            report=report,
                            chart_type=chart_type,
                            title=chart_title,
                            description=chart_description,
                            llm_reasoning=data_justification,
                            chart_code=current_python_code,
                            chart_data=chart_data,
                            chart_image_base64=img_base64,
                            chart_html=html
                        )
                        chart_success = True
                        logger.info(f"Successfully generated sophisticated chart: {chart_type} - {business_question}")
                        success = True
                    except Exception as e:
                        last_error = str(e)
                        logger.error(f"Failed to generate sophisticated chart '{chart_type}' (attempt {attempt+1}): {last_error}")
                        if attempt < max_attempts - 1:
                            # Regenerate code with LLM, appending the error message to the prompt
                            retry_prompt = (
                                f"{prompt}\n\nNOTE: The previous code for this chart failed with this error:\n{last_error}\n"
                                "Please generate a new, valid Plotly code for this chart, avoiding the previous mistake. "
                                "Do NOT use any properties or arguments that are not supported by plotly.graph_objects. "
                                "If you are generating a scatter plot matrix (SPLOM), do NOT use 'type' in the diagonal property."
                            )
                            try:
                                generation_config = GenerationConfig(response_mime_type="application/json")
                                response = self.gemini_model.generate_content(retry_prompt, generation_config=generation_config)
                                logger.info(f"LLM retry response received for chart generation. Response length: {len(response.text)}")
                                logger.info(f"LLM retry response preview: {response.text[:500]}...")
                                retry_chart_suggestions = json.loads(response.text)
                                # Use the first suggestion from retry
                                retry_suggestion = retry_chart_suggestions[0] if isinstance(retry_chart_suggestions, list) and len(retry_chart_suggestions) > 0 else None
                                if retry_suggestion:
                                    current_python_code = retry_suggestion.get('python_code', current_python_code)
                                    current_detailed_analysis = retry_suggestion.get('detailed_analysis', current_detailed_analysis)
                                else:
                                    logger.warning("LLM retry did not return a valid suggestion, will fallback to error chart if next attempt fails.")
                            except Exception as retry_e:
                                logger.error(f"LLM retry failed: {retry_e}")
                                # If LLM retry fails, will fallback to error chart on next loop
                        attempt += 1
                if not success:
                    # Save error information for debugging
                    validate_and_save_chart_data(
                        report=report,
                        chart_type='error',
                        title=f"Chart Generation Error: {chart_type}",
                        description=f"Failed to generate sophisticated chart after {max_attempts} attempts: {last_error}\n\nBusiness Question: {business_question}\n\nCode:\n{current_python_code}",
                        chart_data={'error': last_error, 'code': current_python_code, 'business_question': business_question},
                        chart_image_base64='',
                        chart_html=f'<div>Error: {last_error}</div>'
                    )

            # Fallback: If no chart was successfully generated for time series, create a default line plot
            if is_time_series and not chart_success:
                try:
                    logger.warning("No valid LLM chart generated for time series. Falling back to default line plot.")
                    # Try to find a datetime column
                    datetime_col = None
                    for col in df.columns:
                        if df[col].dtype == 'object':
                            try:
                                pd.to_datetime(df[col], errors='raise')
                                datetime_col = col
                                break
                            except:
                                continue
                    target_col = model_entry.target_variable
                    if datetime_col and target_col in df.columns:
                        df_sorted = df.sort_values(datetime_col)
                        import plotly.graph_objects as go
                        fig = go.Figure(data=[go.Scatter(x=df_sorted[datetime_col], y=df_sorted[target_col], mode='lines', name=target_col)])
                        fig.update_layout(title=f"Time Series Line Plot: {target_col}", xaxis_title=datetime_col, yaxis_title=target_col)
                        img_base64 = safe_plotly_to_image(fig, format="png", width=1000, height=600)
                        html = fig.to_html(include_plotlyjs='cdn', div_id=f"chart_default_timeseries_{hash(target_col)}")
                        self._save_llm_chart(
                            report=report,
                            chart_type='line',
                            title=f"Default Time Series Line Plot: {target_col}",
                            description=f"This is a default time series line plot of the target variable '{target_col}' over time (column '{datetime_col}'). The LLM failed to generate a valid chart, so this fallback plot is provided.",
                            llm_reasoning="Fallback: LLM failed to generate a valid time series chart.",
                            chart_code="Default fallback line plot code.",
                            chart_data={},
                            chart_image_base64=img_base64,
                            chart_html=html
                        )
                except Exception as e:
                    logger.error(f"Failed to generate fallback time series line plot: {e}")

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM JSON response: {e}\nLLM Response Text: {response.text}")
        except Exception as e:
            logger.error(f"An error occurred during sophisticated LLM chart generation: {e}")
            DataInsight.objects.create(
                report=report,
                insight_type='data_distribution',
                title="Sophisticated Chart Generation Failed",
                description=f"The process of generating sophisticated charts with the AI failed: {str(e)}",
                insight_data={'error': str(e)},
                priority=1
            )

    def _save_llm_chart(self, report, chart_type, title, description, llm_reasoning, chart_code, chart_data, chart_image_base64, chart_html):
        """
        Saves a chart generated by the LLM.
        """
        try:
            clean_data = ensure_json_serializable(chart_data)
            if clean_data is None:
                clean_data = {}

            chart = ChartData.objects.create(
                report=report,
                chart_type=chart_type,
                title=title,
                description=description,
                llm_reasoning=llm_reasoning,
                chart_code=chart_code,
                chart_data=clean_data,
                chart_image_base64=chart_image_base64 or '',
                chart_html=chart_html or ''
            )
            logger.info(f"Successfully saved LLM-generated chart: {title} (ID: {chart.id})")
            return chart
        except Exception as e:
            logger.error(f"Error saving LLM chart data for {title}: {e}")
            raise e

    def _generate_data_insights(self, report: Report, df: pd.DataFrame, model_entry: ModelEntry, df_raw: pd.DataFrame = None):
        """Generate enhanced data insights for the report using both raw and processed data"""

        # Data preprocessing impact insight
        if df_raw is not None:
            rows_removed = df_raw.shape[0] - df.shape[0]
            cols_removed = df_raw.shape[1] - df.shape[1]

            DataInsight.objects.create(
                report=report,
                insight_type='data_distribution',
                title="Data Preprocessing Impact",
                description=f"Preprocessing pipeline transformed {df_raw.shape[0]} rows × {df_raw.shape[1]} columns to {df.shape[0]} rows × {df.shape[1]} columns. {rows_removed} rows and {cols_removed} columns were removed during preprocessing.",
                insight_data={
                    'raw_rows': int(df_raw.shape[0]),
                    'raw_columns': int(df_raw.shape[1]),
                    'processed_rows': int(df.shape[0]),
                    'processed_columns': int(df.shape[1]),
                    'rows_removed': int(rows_removed),
                    'columns_removed': int(cols_removed),
                    'data_retention_rate': round(float(df.shape[0] / df_raw.shape[0] * 100), 2)
                },
                priority=1
            )

        # Model input data overview
        DataInsight.objects.create(
            report=report,
            insight_type='data_distribution',
            title="Model Input Data Overview",
            description=f"The model was trained on {df.shape[0]} samples with {df.shape[1]} features after preprocessing",
            insight_data={
                'training_samples': int(df.shape[0]),
                'feature_count': int(df.shape[1]),
                'memory_usage_mb': round(float(df.memory_usage(deep=True).sum() / 1024 / 1024), 2),
                'task_type': model_entry.task,
                'target_variable': model_entry.target_variable
            },
            priority=2
        )

        # Data quality insights for processed data
        missing_data = df.isnull().sum()
        total_missing = missing_data.sum()

        if total_missing > 0:
            DataInsight.objects.create(
                report=report,
                insight_type='missing_values',
                title="Data Quality After Preprocessing",
                description=f"After preprocessing, {total_missing} missing values remain across {(missing_data > 0).sum()} features",
                insight_data={
                    'total_missing': int(total_missing),
                    'features_with_missing': int((missing_data > 0).sum()),
                    'missing_percentage': round(float((total_missing / (df.shape[0] * df.shape[1])) * 100), 2),
                    'features_affected': missing_data[missing_data > 0].to_dict()
                },
                priority=3
            )
        else:
            DataInsight.objects.create(
                report=report,
                insight_type='missing_values',
                title="Data Quality After Preprocessing",
                description="Excellent data quality: No missing values found in the processed dataset",
                insight_data={
                    'total_missing': 0,
                    'features_with_missing': 0,
                    'missing_percentage': 0.0,
                    'quality_status': 'excellent'
                },
                priority=3
            )

        # Feature analysis
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        categorical_cols = df.select_dtypes(include=['object']).columns

        feature_analysis = {
            'numeric_features': len(numeric_cols),
            'categorical_features': len(categorical_cols),
            'total_features': df.shape[1]
        }

        # Add target variable analysis if applicable
        if model_entry.target_variable and model_entry.target_variable in df.columns:
            target_col = df[model_entry.target_variable]
            if model_entry.task == 'Classification':
                unique_classes = target_col.nunique()
                class_distribution = target_col.value_counts().to_dict()
                feature_analysis.update({
                    'target_classes': int(unique_classes),
                    'class_distribution': {str(k): int(v) for k, v in class_distribution.items()},
                    'is_balanced': bool(target_col.value_counts().std() / target_col.value_counts().mean() < 0.5)
                })
            elif model_entry.task == 'Regression':
                feature_analysis.update({
                    'target_mean': round(float(target_col.mean()), 2),
                    'target_std': round(float(target_col.std()), 2),
                    'target_range': [round(float(target_col.min()), 2), round(float(target_col.max()), 2)]
                })

        DataInsight.objects.create(
            report=report,
            insight_type='feature_importance',
            title="Feature Analysis",
            description=f"Model uses {feature_analysis['numeric_features']} numeric and {feature_analysis['categorical_features']} categorical features",
            insight_data=feature_analysis,
            priority=4
        )

        # Model performance insight
        DataInsight.objects.create(
            report=report,
            insight_type='performance_metrics',
            title="Model Performance",
            description=f"Model achieved {model_entry.evaluation_metric}: {model_entry.evaluation_metric_value:.4f}",
            insight_data={
                'metric_name': model_entry.evaluation_metric,
                'metric_value': round(float(model_entry.evaluation_metric_value), 4),
                'model_name': model_entry.model_name,
                'task_type': model_entry.task,
                'performance_interpretation': self._interpret_performance(model_entry)
            },
            priority=5
        )

    def _interpret_performance(self, model_entry: ModelEntry) -> str:
        """Interpret model performance based on task and metric value"""
        metric_value = model_entry.evaluation_metric_value
        task = model_entry.task
        metric = model_entry.evaluation_metric

        if task == 'Classification':
            if metric in ['accuracy', 'f1_score', 'precision', 'recall']:
                if metric_value >= 0.9:
                    return "Excellent performance"
                elif metric_value >= 0.8:
                    return "Good performance"
                elif metric_value >= 0.7:
                    return "Fair performance"
                else:
                    return "Needs improvement"
        elif task == 'Regression':
            if metric in ['r2_score']:
                if metric_value >= 0.9:
                    return "Excellent fit"
                elif metric_value >= 0.7:
                    return "Good fit"
                elif metric_value >= 0.5:
                    return "Moderate fit"
                else:
                    return "Poor fit"
            elif metric in ['mean_squared_error', 'mean_absolute_error']:
                return "Lower values indicate better performance"
        elif task == 'TimeSeries':
            return "Lower error values indicate better forecasting accuracy"
        elif task == 'Clustering':
            return "Higher silhouette scores indicate better cluster separation"

        return "Performance varies by context"

    def _generate_ai_insights(self, report: Report, df: pd.DataFrame, model_entry: ModelEntry, df_raw: pd.DataFrame = None) -> str:
        """Generate enhanced AI-powered insights using Gemini with both raw and processed data"""
        original_metric_value = model_entry.evaluation_metric_value
        try:
            # Correct negative error metrics (like scikit-learn's neg_root_mean_squared_error)
            # before passing them to the LLM for analysis.
            metric_name = model_entry.evaluation_metric
            metric_value = model_entry.evaluation_metric_value
            error_metrics = ['rmse', 'mse', 'mae', 'mean_squared_error', 'mean_absolute_error', 'root_mean_squared_error']

            if any(em in metric_name.lower() for em in error_metrics) and metric_value < 0:
                corrected_value = abs(metric_value)
                logger.info(f"Correcting negative metric '{metric_name}' from {metric_value} to {corrected_value} for LLM insights.")
                # Temporarily update the value for the scope of this method
                model_entry.evaluation_metric_value = corrected_value
            
            # Prepare comprehensive data summary for AI analysis
            data_summary = {
                'processed_shape': df.shape,
                'processed_columns': df.columns.tolist(),
                'processed_data_types': df.dtypes.to_dict(),
                'processed_missing_values': df.isnull().sum().to_dict(),
                'task_type': model_entry.task,
                'target_variable': model_entry.target_variable,
                'model_performance': {
                    'metric': model_entry.evaluation_metric,
                    'value': model_entry.evaluation_metric_value,
                    'model_name': model_entry.model_name
                }
            }

            # Add raw data information if available
            if df_raw is not None:
                data_summary.update({
                    'raw_shape': df_raw.shape,
                    'raw_missing_values': df_raw.isnull().sum().to_dict(),
                    'preprocessing_impact': {
                        'rows_removed': df_raw.shape[0] - df.shape[0],
                        'columns_removed': df_raw.shape[1] - df.shape[1],
                        'data_retention_rate': round(df.shape[0] / df_raw.shape[0] * 100, 2)
                    }
                })

            # Add target variable analysis
            if model_entry.target_variable and model_entry.target_variable in df.columns:
                target_col = df[model_entry.target_variable]
                if model_entry.task == 'Classification':
                    data_summary['target_analysis'] = {
                        'unique_classes': target_col.nunique(),
                        'class_distribution': target_col.value_counts().to_dict(),
                        'is_balanced': target_col.value_counts().std() / target_col.value_counts().mean() < 0.5
                    }
                elif model_entry.task == 'Regression':
                    data_summary['target_analysis'] = {
                        'mean': round(target_col.mean(), 2),
                        'std': round(target_col.std(), 2),
                        'range': [round(target_col.min(), 2), round(target_col.max(), 2)]
                    }

            # Add feature analysis
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            categorical_cols = df.select_dtypes(include=['object']).columns.tolist()

            # Calculate feature correlations with target if numeric
            feature_insights = {}
            if model_entry.target_variable in numeric_cols and len(numeric_cols) > 1:
                correlations = df[numeric_cols].corr()[model_entry.target_variable].abs().sort_values(ascending=False)
                top_features = correlations.head(5).to_dict()
                feature_insights['top_correlated_features'] = {k: round(v, 3) for k, v in top_features.items() if k != model_entry.target_variable}

            # Performance interpretation
            performance_context = self._get_performance_context(model_entry)

            # Create enhanced prompt for Gemini
            prompt = f"""
            As a senior data scientist and business analyst, provide a comprehensive analysis of this machine learning model:

            ## Model Overview
            - **Task Type**: {data_summary['task_type']}
            - **Algorithm**: {data_summary['model_performance']['model_name']}
            - **Target Variable**: {data_summary['target_variable']}
            - **Performance Metric**: {data_summary['model_performance']['metric']} = {data_summary['model_performance']['value']:.4f}
            - **Performance Context**: {performance_context}

            ## Dataset Characteristics
            - **Final Dataset**: {data_summary['processed_shape'][0]:,} samples × {data_summary['processed_shape'][1]} features
            - **Feature Composition**: {len(numeric_cols)} numeric features, {len(categorical_cols)} categorical features
            """

            if df_raw is not None:
                prompt += f"""
            - **Original Dataset**: {data_summary['raw_shape'][0]:,} samples × {data_summary['raw_shape'][1]} features
            - **Data Retention Rate**: {data_summary['preprocessing_impact']['data_retention_rate']}%
            - **Preprocessing Impact**: Removed {data_summary['preprocessing_impact']['rows_removed']:,} rows and {data_summary['preprocessing_impact']['columns_removed']} columns
                """

            if 'target_analysis' in data_summary:
                if model_entry.task == 'Classification':
                    prompt += f"""
            - **Target Classes**: {data_summary['target_analysis']['unique_classes']} unique classes
            - **Class Balance**: {'Well-balanced' if data_summary['target_analysis']['is_balanced'] else 'Imbalanced - may require attention'}
                    """
                elif model_entry.task == 'Regression':
                    prompt += f"""
            - **Target Distribution**: Range {data_summary['target_analysis']['range'][0]} to {data_summary['target_analysis']['range'][1]}
            - **Target Statistics**: Mean = {data_summary['target_analysis']['mean']}, Std = {data_summary['target_analysis']['std']}
                    """

            if feature_insights.get('top_correlated_features'):
                top_features_str = ', '.join([f"{k} ({v})" for k, v in list(feature_insights['top_correlated_features'].items())[:3]])
                prompt += f"""
            - **Key Predictive Features**: {top_features_str}
                """

            # Prepare preprocessing context
            preprocessing_context = ""
            if df_raw is not None:
                preprocessing_context = f"Analyze the significant data reduction from {data_summary['raw_shape'][0]} to {data_summary['processed_shape'][0]} samples and its implications."
            else:
                preprocessing_context = "Evaluate the current data preprocessing approach."

            # Prepare feature context
            feature_context = ""
            if feature_insights.get('top_correlated_features'):
                top_features = ', '.join(list(feature_insights['top_correlated_features'].keys())[:3])
                feature_context = f"Highlight the most predictive features: {top_features}."
            else:
                feature_context = "Suggest feature engineering approaches based on the available data."

            # Prepare variables for cleaner prompt formatting
            target_variable = data_summary['target_variable']
            metric_name = data_summary['model_performance']['metric']
            metric_value = data_summary['model_performance']['value']
            task_type = data_summary['task_type']
            sample_count = data_summary['processed_shape'][0]
            missing_values_total = sum(data_summary['processed_missing_values'].values())
            data_completeness = 100 - (missing_values_total / (df.shape[0] * df.shape[1]) * 100)

            prompt += f"""

            ## Data Quality Metrics
            - **Missing Values**: {missing_values_total} total across all features
            - **Data Completeness**: {data_completeness:.1f}%

            IMPORTANT: Please provide your response in EXACTLY this format. Use numbered sections with double hash (##) headers. Do NOT use bold (**) formatting for section headers. Follow this exact structure:

            ## 1. Model Performance Analysis
            Provide a comprehensive evaluation of the model's performance. Explain what the {metric_name} score of {metric_value:.4f} means in practical terms. Compare this to typical benchmarks for {task_type} tasks. Discuss whether this performance level is suitable for production use and what factors might be influencing the results.

            ## 2. Data Quality Assessment
            Analyze the overall quality and characteristics of the training data. Evaluate the impact of missing values, data distribution, and feature composition on model performance. Identify any data quality issues that could be affecting model accuracy. Discuss the relationship between data size ({sample_count:,} samples) and model complexity.

            ## 3. Preprocessing Impact
            Examine how data preprocessing has affected the dataset and model training process. {preprocessing_context} Discuss whether the preprocessing steps are appropriate and if additional preprocessing might improve results.

            ## 4. Feature Engineering Opportunities
            Identify specific opportunities for feature engineering and data enhancement. {feature_context} Recommend techniques for creating new features, handling categorical variables, and improving feature selection.

            ## 5. Actionable Recommendations
            Provide 4-5 specific, prioritized recommendations for improving model performance:
            - **Recommendation 1**: [Specific technical improvement]
            - **Recommendation 2**: [Data quality enhancement]
            - **Recommendation 3**: [Feature engineering suggestion]
            - **Recommendation 4**: [Deployment consideration]
            - **Recommendation 5**: [Monitoring and maintenance]

            ## 6. Business Impact Analysis
            Translate the technical findings into business value and implications. Explain how the model performance translates to real-world outcomes for the '{target_variable}' prediction task. Discuss confidence levels, potential risks, and recommended use cases. Provide guidance on when to retrain the model and what success metrics to monitor.

            FORMATTING RULES:
            - Use ## for section headers (not ** or ***)
            - Each section should be 4-6 sentences
            - Use bullet points with - for lists
            - Use **text** for emphasis within paragraphs only
            - Do NOT add colons after section headers
            - Start each section immediately after the header
            """

            response = self.gemini_model.generate_content(prompt)
            return response.text

        except Exception as e:
            logger.error(f"Error generating AI insights: {e}")
            return "AI insights generation failed. Please check the configuration."
        finally:
            # Restore original value to avoid side effects in other parts of the service
            model_entry.evaluation_metric_value = original_metric_value

    def _get_performance_context(self, model_entry: ModelEntry) -> str:
        """Provide context for model performance based on task type and metric"""
        metric = model_entry.evaluation_metric
        value = model_entry.evaluation_metric_value

        if model_entry.task == 'Classification':
            if metric in ['accuracy', 'f1_score', 'precision', 'recall']:
                if value >= 0.9:
                    return "Excellent performance"
                elif value >= 0.8:
                    return "Good performance"
                elif value >= 0.7:
                    return "Moderate performance"
                else:
                    return "Below average performance"
        elif model_entry.task == 'Regression':
            if metric in ['rmse', 'mae']:
                return "Lower values indicate better performance"
            elif metric in ['r2', 'r2_score']:
                if value >= 0.8:
                    return "Strong predictive power"
                elif value >= 0.6:
                    return "Moderate predictive power"
                else:
                    return "Weak predictive power"

        return "Performance evaluation needed"

    def _generate_chart_explanation(self, chart_type: str, data: dict, title: str, context: str) -> str:
        """Generate AI-powered explanation for specific charts"""
        try:
            if not self.gemini_model:
                return self._get_default_chart_explanation(chart_type, data, title, context)

            # Prepare chart data summary for AI
            if chart_type == 'line':
                y_values = data['y'].dropna()
                start_value = y_values.iloc[0]
                end_value = y_values.iloc[-1]
                min_value = y_values.min()
                max_value = y_values.max()
                mean_value = y_values.mean()
                std_value = y_values.std()

                # Calculate performance metrics
                total_change = ((end_value - start_value) / start_value * 100) if start_value != 0 else 0
                volatility_ratio = std_value / mean_value if mean_value != 0 else 0

                prompt = f"""
                Describe this specific time series chart that is currently displayed to the user.

                CHART DETAILS:
                - Title: {title}
                - Shows: {len(y_values):,} data points over time
                - Overall change: {total_change:+.1f}%
                - Pattern: {trend_desc}

                TASK: Write 2-3 sentences describing what this specific chart shows. Focus on:
                1. The trend pattern visible in the chart
                2. What this pattern means for business understanding
                3. One key insight from the time series

                IMPORTANT:
                - Describe the ACTUAL chart pattern being shown
                - Don't repeat basic statistics (those are shown separately)
                - Focus on the trend and business meaning
                """

                response = self.gemini_model.generate_content(prompt)
                return response.text.strip()

            elif chart_type == 'histogram':
                values = data['values']
                std_dev = np.std(values)
                median = np.median(values)
                q1 = np.percentile(values, 25)
                q3 = np.percentile(values, 75)

                # Determine distribution characteristics
                skewness = "right-skewed" if np.mean(values) > median else "left-skewed" if np.mean(values) < median else "symmetric"
                spread = "high" if std_dev > np.mean(values) * 0.5 else "moderate" if std_dev > np.mean(values) * 0.2 else "low"

                prompt = f"""
                Analyze this specific histogram chart that is currently displayed to the user. Describe what you can see in the actual distribution pattern.

                CHART DETAILS:
                - Title: {title}
                - Data shows: {len(values):,} observations
                - Distribution shape: {skewness}
                - Spread level: {spread} variability

                TASK: Write 2-3 sentences that describe what this specific histogram reveals about the data pattern. Focus on:
                1. The actual shape and pattern visible in the chart
                2. What this pattern means for business understanding
                3. One key insight from the distribution

                IMPORTANT:
                - Describe the ACTUAL chart being shown
                - Don't repeat basic statistics (those are shown separately)
                - Focus on the distribution pattern and business meaning
                - Be specific about what the chart reveals
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

            # Calculate trend characteristics
            start_value = y_values.iloc[0]
            end_value = y_values.iloc[-1]
            min_value = y_values.min()
            max_value = y_values.max()
            mean_value = y_values.mean()

            # Determine overall trend
            if end_value > start_value * 1.1:
                trend_desc = "strong upward trend"
                trend_implication = "indicating growth and positive momentum"
            elif end_value < start_value * 0.9:
                trend_desc = "downward trend"
                trend_implication = "suggesting a decline that may need attention"
            else:
                trend_desc = "relatively stable pattern"
                trend_implication = "showing consistent performance over time"

            # Calculate volatility
            volatility = y_values.std() / mean_value if mean_value != 0 else 0
            if volatility > 0.3:
                volatility_desc = "high volatility with significant fluctuations"
            elif volatility > 0.1:
                volatility_desc = "moderate volatility with some fluctuations"
            else:
                volatility_desc = "low volatility with stable values"

            # Performance change
            total_change = ((end_value - start_value) / start_value * 100) if start_value != 0 else 0

            # Focus on the trend pattern, not basic stats
            if abs(total_change) > 20:
                change_insight = f"The significant {abs(total_change):.0f}% change suggests a major shift in performance that warrants investigation."
            elif abs(total_change) > 5:
                change_insight = f"The moderate {abs(total_change):.0f}% change indicates gradual evolution in the underlying factors."
            else:
                change_insight = f"The stable pattern with minimal change suggests consistent performance over time."

            return f"This time series chart reveals a {trend_desc} pattern in {title.lower()} {trend_implication}. {change_insight} The {volatility_desc} provides insights into the consistency and predictability of the underlying process."
        elif chart_type == 'histogram':
            values = data['values']
            std_dev = np.std(values)
            median = np.median(values)
            mean = np.mean(values)
            q1 = np.percentile(values, 25)
            q3 = np.percentile(values, 75)

            # Determine distribution characteristics
            if mean > median * 1.1:
                shape_desc = "right-skewed, indicating most values are concentrated at the lower end with some high outliers"
                business_implication = "This suggests there are opportunities to understand what drives the higher values"
            elif mean < median * 0.9:
                shape_desc = "left-skewed, indicating most values are concentrated at the higher end with some low outliers"
                business_implication = "This suggests investigating factors that might be causing the lower performance cases"
            else:
                shape_desc = "approximately symmetric, indicating a balanced distribution around the center"
                business_implication = "This suggests a stable, predictable pattern in the data"

            if std_dev > mean * 0.5:
                spread_desc = "high variability, indicating significant differences between data points"
            elif std_dev > mean * 0.2:
                spread_desc = "moderate variability, showing some differences but generally consistent patterns"
            else:
                spread_desc = "low variability, indicating very consistent values across the dataset"

            # Calculate concentration
            iqr = q3 - q1
            concentration = f"50% of values fall between {q1:,.1f} and {q3:,.1f}"

            # Focus on the distribution pattern, not the basic stats
            if mean > median * 1.1:
                pattern_insight = f"The distribution shows most values cluster toward the lower end, with a few high-value cases pulling the average up. This suggests opportunities to understand what drives the top performers."
            elif mean < median * 0.9:
                pattern_insight = f"The distribution shows most values cluster toward the higher end, with some lower-performing cases. This suggests investigating factors that might be limiting performance in certain cases."
            else:
                pattern_insight = f"The distribution shows a balanced pattern around the center, indicating consistent and predictable behavior across the dataset."

            return f"This histogram reveals the distribution pattern of {title.lower()}, showing a {shape_desc}. {pattern_insight} The {spread_desc} indicates {'significant variation' if std_dev > mean * 0.3 else 'moderate consistency'} in the underlying data."
        elif chart_type == 'bar':
            return f"This bar chart displays {title.lower()} across different categories. {context} The chart helps compare values between different groups and identify the highest and lowest performing categories."
        elif chart_type == 'pie':
            return f"This pie chart shows the proportional breakdown of {title.lower()}. Each slice represents the relative contribution of different categories to the total, making it easy to identify the largest and smallest segments."
        else:
            return f"This {chart_type} chart visualizes {title.lower()}. {context} The visualization helps understand patterns and relationships in the data."


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

def generate_data_profile_js_code():
    """
    Generate JavaScript code for customizing data profile HTML reports.
    
    This function returns JavaScript code that:
    1. Hides all section headers and content except overview initially
    2. Adds click event listeners to navigation links to show/hide sections
    3. Replaces all instances of "YData" with "Symplif.ai" branding
    
    Returns:
        str: JavaScript code as a string
    """
    return """
  document.addEventListener('DOMContentLoaded', function() {
    // Hide all section headers and content except overview
    const sectionItems = document.querySelectorAll('.section-items');
    const sectionHeaders = document.querySelectorAll('.section-header');

    // Initially hide non-overview sections
    sectionItems.forEach(function(section) {
        if (!section.previousElementSibling || section.previousElementSibling.id !== 'overview') {
            section.style.display = 'none';
        }
    });

    sectionHeaders.forEach(function(header) {
        if (header.id !== 'overview') {
            header.style.display = 'none';
        }
    });

    // Add click event listeners to nav links
    const navLinks = document.querySelectorAll('.navbar-nav .nav-link');
    navLinks.forEach(function(link) {
        link.addEventListener('click', function(e) {
            // Get the target section ID from the href attribute
            const targetId = this.getAttribute('href').substring(1);
            
            // Hide all sections
            sectionItems.forEach(function(section) {
                section.style.display = 'none';
            });
            
            sectionHeaders.forEach(function(header) {
                header.style.display = 'none';
            });
            
            // Show the target section header and its content
            const targetHeader = document.getElementById(targetId);
            if (targetHeader) {
                // Show the section header
                targetHeader.style.display = 'block';
                
                // Find and show the section items that follow
                const nextElement = targetHeader.nextElementSibling;
                if (nextElement && nextElement.classList.contains('section-items')) {
                    nextElement.style.display = 'block';
                }
            }
        });
    });
     // Replace all instances of "YData" with "Symplif.ai"
    function replaceYDataWithSymplif() {
        // Get all text nodes in the document
        const walker = document.createTreeWalker(
            document.body,
            NodeFilter.SHOW_TEXT,
            null,
            false
        );

        // Array to keep track of text nodes that need modification
        const nodesToModify = [];
        let node;
        
        // Find all text nodes containing "YData"
        while (node = walker.nextNode()) {
            if (node.nodeValue.includes('YData')) {
                nodesToModify.push(node);
            }
        }
        
        // Replace text in all identified nodes
        nodesToModify.forEach(function(node) {
            node.nodeValue = node.nodeValue.replace(/YData/g, 'Symplif.ai');
        });
        
        // Also update attributes like title, alt, etc.
        const elementsWithAttributes = document.querySelectorAll('[title], [alt], [placeholder], [aria-label]');
        elementsWithAttributes.forEach(function(el) {
            if (el.hasAttribute('title')) {
                el.setAttribute('title', el.getAttribute('title').replace(/YData/g, 'Symplif.ai'));
            }
            if (el.hasAttribute('alt')) {
                el.setAttribute('alt', el.getAttribute('alt').replace(/YData/g, 'Symplif.ai'));
            }
            if (el.hasAttribute('placeholder')) {
                el.setAttribute('placeholder', el.getAttribute('placeholder').replace(/YData/g, 'Symplif.ai'));
            }
            if (el.hasAttribute('aria-label')) {
                el.setAttribute('aria-label', el.getAttribute('aria-label').replace(/YData/g, 'Symplif.ai'));
            }
        });
        
        // Update document title
        if (document.title.includes('YData')) {
            document.title = document.title.replace(/YData/g, 'Symplif.ai');
        }

    }
    
    // Run the replacement function
    replaceYDataWithSymplif();
});

function renameNavTags() {
  const navs = document.querySelectorAll('nav');
  navs.forEach(nav => {
    const navv = document.createElement('navv');

    // Copy attributes
    for (let attr of nav.attributes) {
      navv.setAttribute(attr.name, attr.value);
    }

    // Copy children
    while (nav.firstChild) {
      navv.appendChild(nav.firstChild);
    }

    // Replace old nav with new navv
    nav.parentNode.replaceChild(navv, nav);
  });
}
renameNavTags();
"""


def inject_js_into_html(html_content, js_code):
    """
    Inject JavaScript code into HTML content by inserting it before the closing </body> tag.
    
    Args:
        html_content (str): The original HTML content
        js_code (str): JavaScript code to inject
        
    Returns:
        str: HTML content with JavaScript injected
    """
    script_tag = f"\n<script>\n{js_code}\n</script>\n"
    
    # Check if HTML has a closing body tag
    closing_combo = "</body></html>"
    if closing_combo in html_content.replace(" ", "").replace("\n", ""):
        idx = html_content.lower().rfind("</body>")
        if idx != -1:
            html_content = html_content[:idx] + script_tag + html_content[idx:]
        else:
            html_content += script_tag
    else:
        html_content += script_tag
    
    return html_content 