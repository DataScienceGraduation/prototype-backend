from django.shortcuts import render
from django.http import JsonResponse, HttpResponse
from django.views.decorators.http import require_POST, require_GET
from django.views.decorators.csrf import csrf_exempt
from django.core.paginator import Paginator
from automlapp.views import jwt_authenticated
from automlapp.models import ModelEntry, User
from .models import Report, ChartData, DataInsight, Dashboard, DataProfile
from .services import ReportGenerationService, ChartExportService, generate_report_async, generate_data_profile_js_code, inject_js_into_html
from celery.result import AsyncResult
from .tasks import suggest_charts_task
from django.core.files.storage import default_storage
from ydata_profiling import ProfileReport
import json
import logging
import pandas as pd

logger = logging.getLogger(__name__)

@csrf_exempt
@require_POST
@jwt_authenticated
def generate_report(request):
    """
    Generate a comprehensive report for a trained model

    Expected POST data:
    - model_id: ID of the ModelEntry
    - report_type: Type of report (optional, defaults to 'analysis')
    """
    try:
        data = json.loads(request.body)
        model_id = data.get('model_id')
        report_type = data.get('report_type', 'analysis')

        if not model_id:
            return JsonResponse({
                'success': False,
                'message': 'model_id is required'
            }, status=400)

        # Check if user has access to this model
        user = User.objects.get(username=request.jwt_payload['username'])
        if not user.models.filter(id=model_id).exists():
            return JsonResponse({
                'success': False,
                'message': 'Model not found or access denied'
            }, status=404)

        # Check if model is trained
        model_entry = ModelEntry.objects.get(id=model_id)
        if model_entry.status != 'Done':
            return JsonResponse({
                'success': False,
                'message': 'Model must be trained before generating report'
            }, status=400)

        # Check if async parameter is provided
        use_async = data.get('async', True)  # Default to async

        if use_async:
            # Generate report asynchronously using Celery
            task = generate_report_async.delay(model_id, report_type)

            return JsonResponse({
                'success': True,
                'task_id': task.id,
                'message': 'Report generation started',
                'async': True
            }, status=202)  # 202 Accepted for async processing
        else:
            # Generate report synchronously (original behavior)
            service = ReportGenerationService()
            report = service.generate_report(model_id, report_type)

            return JsonResponse({
                'success': True,
                'report_id': report.id,
                'message': 'Report generated successfully',
                'async': False
            }, status=200)

    except ModelEntry.DoesNotExist:
        return JsonResponse({
            'success': False,
            'message': 'Model not found'
        }, status=404)
    except Exception as e:
        logger.error(f"Error generating report: {e}")
        return JsonResponse({
            'success': False,
            'message': 'Error generating report'
        }, status=500)


@csrf_exempt
@require_GET
@jwt_authenticated
def get_report(request):
    """
    Get a specific report with all its charts and insights

    Expected GET parameters:
    - report_id: ID of the report
    """
    try:
        report_id = request.GET.get('report_id')

        if not report_id:
            return JsonResponse({
                'success': False,
                'message': 'report_id is required'
            }, status=400)

        # Check if user has access to this report
        user = User.objects.get(username=request.jwt_payload['username'])
        report = Report.objects.get(id=report_id)

        if not user.models.filter(id=report.model_entry.id).exists():
            return JsonResponse({
                'success': False,
                'message': 'Report not found or access denied'
            }, status=404)

        # Serialize report data
        charts_data = []
        for chart in report.charts.all():
            charts_data.append({
                'id': chart.id,
                'chart_type': chart.chart_type,
                'title': chart.title,
                'description': chart.description,
                'chart_data': chart.get_chart_data_as_dict(),
                'chart_code': chart.chart_code,
                'llm_reasoning': chart.llm_reasoning,
                'chart_image_base64': chart.chart_image_base64,
                'created_at': chart.created_at.isoformat()
            })

        insights_data = []
        for insight in report.insights.all():
            insights_data.append({
                'id': insight.id,
                'insight_type': insight.insight_type,
                'title': insight.title,
                'description': insight.description,
                'insight_data': insight.insight_data,
                'priority': insight.priority,
                'created_at': insight.created_at.isoformat()
            })

        response_data = {
            'success': True,
            'report': {
                'id': report.id,
                'title': report.title,
                'description': report.description,
                'report_type': report.report_type,
                'status': report.status,
                'ai_insights': report.ai_insights,
                'created_at': report.created_at.isoformat(),
                'updated_at': report.updated_at.isoformat(),
                'model_entry': {
                    'id': report.model_entry.id,
                    'name': report.model_entry.name,
                    'task': report.model_entry.task,
                    'target_variable': report.model_entry.target_variable
                }
            },
            'charts': charts_data,
            'insights': insights_data
        }

        return JsonResponse(response_data, status=200)

    except Report.DoesNotExist:
        return JsonResponse({
            'success': False,
            'message': 'Report not found'
        }, status=404)
    except Exception as e:
        logger.error(f"Error getting report: {e}")
        return JsonResponse({
            'success': False,
            'message': 'Error retrieving report'
        }, status=500)


@csrf_exempt
@require_GET
@jwt_authenticated
def list_reports(request):
    """
    List all reports for the authenticated user

    Optional GET parameters:
    - page: Page number for pagination (default: 1)
    - page_size: Number of reports per page (default: 10)
    - model_id: Filter by specific model ID
    """
    try:
        user = User.objects.get(username=request.jwt_payload['username'])
        page = int(request.GET.get('page', 1))
        page_size = int(request.GET.get('page_size', 10))
        model_id = request.GET.get('model_id')

        # Get user's models
        user_models = user.models.all()

        # Filter reports
        reports_query = Report.objects.filter(model_entry__in=user_models)

        if model_id:
            reports_query = reports_query.filter(model_entry_id=model_id)

        # Paginate
        paginator = Paginator(reports_query, page_size)
        reports_page = paginator.get_page(page)

        reports_data = []
        for report in reports_page:
            reports_data.append({
                'id': report.id,
                'title': report.title,
                'description': report.description,
                'report_type': report.report_type,
                'status': report.status,
                'created_at': report.created_at.isoformat(),
                'model_entry': {
                    'id': report.model_entry.id,
                    'name': report.model_entry.name,
                    'task': report.model_entry.task
                },
                'charts_count': report.charts.count(),
                'insights_count': report.insights.count()
            })

        return JsonResponse({
            'success': True,
            'reports': reports_data,
            'pagination': {
                'current_page': page,
                'total_pages': paginator.num_pages,
                'total_reports': paginator.count,
                'has_next': reports_page.has_next(),
                'has_previous': reports_page.has_previous()
            }
        }, status=200)

    except Exception as e:
        logger.error(f"Error listing reports: {e}")
        return JsonResponse({
            'success': False,
            'message': 'Error retrieving reports'
        }, status=500)


@csrf_exempt
@require_GET
@jwt_authenticated
def export_chart_csv(request):
    """
    Export chart data as CSV

    Expected GET parameters:
    - chart_id: ID of the chart to export
    """
    try:
        chart_id = request.GET.get('chart_id')

        if not chart_id:
            return JsonResponse({
                'success': False,
                'message': 'chart_id is required'
            }, status=400)

        # Check if user has access to this chart
        user = User.objects.get(username=request.jwt_payload['username'])
        chart = ChartData.objects.get(id=chart_id)

        if not user.models.filter(id=chart.report.model_entry.id).exists():
            return JsonResponse({
                'success': False,
                'message': 'Chart not found or access denied'
            }, status=404)

        # Export chart data
        csv_data = ChartExportService.export_chart_data_to_csv(chart_id)

        # Create HTTP response with CSV
        response = HttpResponse(csv_data, content_type='text/csv')
        response['Content-Disposition'] = f'attachment; filename="{chart.title.replace(" ", "_")}_data.csv"'

        return response

    except ChartData.DoesNotExist:
        return JsonResponse({
            'success': False,
            'message': 'Chart not found'
        }, status=404)
    except Exception as e:
        logger.error(f"Error exporting chart: {e}")
        return JsonResponse({
            'success': False,
            'message': 'Error exporting chart data'
        }, status=500)


@csrf_exempt
@require_GET
@jwt_authenticated
def get_chart_stats(request):
    """
    Get summary statistics for a chart

    Expected GET parameters:
    - chart_id: ID of the chart
    """
    try:
        chart_id = request.GET.get('chart_id')

        if not chart_id:
            return JsonResponse({
                'success': False,
                'message': 'chart_id is required'
            }, status=400)

        # Check if user has access to this chart
        user = User.objects.get(username=request.jwt_payload['username'])
        chart = ChartData.objects.get(id=chart_id)

        if not user.models.filter(id=chart.report.model_entry.id).exists():
            return JsonResponse({
                'success': False,
                'message': 'Chart not found or access denied'
            }, status=404)

        # Get chart statistics
        stats = ChartExportService.get_chart_summary_stats(chart_id)

        return JsonResponse({
            'success': True,
            'stats': stats
        }, status=200)

    except ChartData.DoesNotExist:
        return JsonResponse({
            'success': False,
            'message': 'Chart not found'
        }, status=404)
    except Exception as e:
        logger.error(f"Error getting chart stats: {e}")
        return JsonResponse({
            'success': False,
            'message': 'Error retrieving chart statistics'
        }, status=500)


@csrf_exempt
@require_GET
@jwt_authenticated
def get_task_status(request):
    """
    Get the status of a Celery task

    Expected GET parameters:
    - task_id: ID of the Celery task
    """
    try:
        task_id = request.GET.get('task_id')

        if not task_id:
            return JsonResponse({
                'success': False,
                'message': 'task_id is required'
            }, status=400)

        # Get task result
        task_result = AsyncResult(task_id)

        response_data = {
            'success': True,
            'task_id': task_id,
            'status': task_result.status,
            'ready': task_result.ready()
        }

        if task_result.status == 'PENDING':
            response_data.update({
                'current': 0,
                'total': 100,
                'status_message': 'Task is waiting to be processed...'
            })
        elif task_result.status == 'PROGRESS':
            response_data.update(task_result.info)
        elif task_result.status == 'SUCCESS':
            result = task_result.result
            response_data.update({
                'current': 100,
                'total': 100,
                'status_message': 'Task completed successfully',
                'result': result
            })
        elif task_result.status == 'FAILURE':
            response_data.update({
                'current': 0,
                'total': 100,
                'status_message': f'Task failed: {str(task_result.info)}',
                'error': str(task_result.info)
            })

        return JsonResponse(response_data, status=200)

    except Exception as e:
        logger.error(f"Error getting task status: {e}")
        return JsonResponse({
            'success': False,
            'message': 'Error retrieving task status'
        }, status=500)


@csrf_exempt
@require_GET
@jwt_authenticated
def get_report_by_task(request):
    """
    Get a report by its task ID

    Expected GET parameters:
    - task_id: ID of the Celery task
    """
    try:
        task_id = request.GET.get('task_id')

        if not task_id:
            return JsonResponse({
                'success': False,
                'message': 'task_id is required'
            }, status=400)

        # Check if user has access to this task's report
        user = User.objects.get(username=request.jwt_payload['username'])

        try:
            report = Report.objects.get(task_id=task_id)

            # Verify user has access to this report
            if not user.models.filter(id=report.model_entry.id).exists():
                return JsonResponse({
                    'success': False,
                    'message': 'Report not found or access denied'
                }, status=404)

            # If report is still generating, return progress info
            if report.status == 'generating':
                return JsonResponse({
                    'success': True,
                    'status': 'generating',
                    'progress_percentage': report.progress_percentage,
                    'current_step': report.current_step,
                    'report_id': report.id
                }, status=200)

            # If completed, return full report data (reuse existing get_report logic)
            elif report.status == 'completed':
                # Serialize report data (same as get_report view)
                charts_data = []
                for chart in report.charts.all():
                    charts_data.append({
                        'id': chart.id,
                        'chart_type': chart.chart_type,
                        'title': chart.title,
                        'description': chart.description,
                        'chart_data': chart.get_chart_data_as_dict(),
                        'chart_code': chart.chart_code,
                        'llm_reasoning': chart.llm_reasoning,
                        'chart_image_base64': chart.chart_image_base64,
                        'created_at': chart.created_at.isoformat()
                    })

                insights_data = []
                for insight in report.insights.all():
                    insights_data.append({
                        'id': insight.id,
                        'insight_type': insight.insight_type,
                        'title': insight.title,
                        'description': insight.description,
                        'insight_data': insight.insight_data,
                        'priority': insight.priority,
                        'created_at': insight.created_at.isoformat()
                    })

                return JsonResponse({
                    'success': True,
                    'status': 'completed',
                    'report': {
                        'id': report.id,
                        'title': report.title,
                        'description': report.description,
                        'report_type': report.report_type,
                        'status': report.status,
                        'ai_insights': report.ai_insights,
                        'progress_percentage': report.progress_percentage,
                        'current_step': report.current_step,
                        'created_at': report.created_at.isoformat(),
                        'updated_at': report.updated_at.isoformat(),
                        'model_entry': {
                            'id': report.model_entry.id,
                            'name': report.model_entry.name,
                            'task': report.model_entry.task,
                            'target_variable': report.model_entry.target_variable
                        }
                    },
                    'charts': charts_data,
                    'insights': insights_data
                }, status=200)

            # If failed
            elif report.status == 'failed':
                return JsonResponse({
                    'success': False,
                    'status': 'failed',
                    'message': f'Report generation failed: {report.current_step}',
                    'report_id': report.id
                }, status=500)

        except Report.DoesNotExist:
            return JsonResponse({
                'success': False,
                'message': 'Report not found for this task'
            }, status=404)

    except Exception as e:
        logger.error(f"Error getting report by task: {e}")
        return JsonResponse({
            'success': False,
            'message': 'Error retrieving report'
        }, status=500)


@csrf_exempt
@require_POST
@jwt_authenticated
def start_suggest_charts(request):
    """
    Start async chart suggestion for a model. POST with JSON: { "model_id": ... }
    Returns: { "success": true, "task_id": ... }
    """
    data = json.loads(request.body)
    model_id = data.get("model_id")
    if not model_id:
        return JsonResponse({"success": False, "message": "model_id required"}, status=400)
    task = suggest_charts_task.delay(model_id)
    return JsonResponse({"success": True, "task_id": task.id})


@csrf_exempt
@require_POST
@jwt_authenticated
def get_suggest_charts_result(request):
    """
    Get result of async chart suggestion. POST with JSON: { "task_id": ... }
    Returns: { "success": true, "status": ..., "result": ... }
    """
    data = json.loads(request.body)
    task_id = data.get("task_id")
    if not task_id:
        return JsonResponse({"success": False, "message": "task_id required"}, status=400)
    result = AsyncResult(task_id)
    if result.state == 'PENDING':
        return JsonResponse({"success": True, "status": "PENDING"})
    elif result.state == 'SUCCESS':
        return JsonResponse({"success": True, "status": "SUCCESS", "result": result.result})
    elif result.state == 'FAILURE':
        return JsonResponse({"success": False, "status": "FAILURE", "error": str(result.info)})
    else:
        return JsonResponse({"success": True, "status": result.state})


@csrf_exempt
@require_GET
@jwt_authenticated
def get_dashboard_by_model(request):
    """
    Get the dashboard (charts) for a given model. GET with ?model_id=... Returns charts or error.
    """
    model_id = request.GET.get('model_id')
    if not model_id:
        return JsonResponse({'success': False, 'message': 'model_id is required.'}, status=400)
    try:
        model_entry = ModelEntry.objects.get(id=model_id)
        dashboard = Dashboard.objects.get(model_entry=model_entry)
        return JsonResponse({
            'success': True,
            'charts': dashboard.charts,
            'title': model_entry.name,
            'description': model_entry.description
        })
    except Dashboard.DoesNotExist:
        return JsonResponse({'success': False, 'message': 'Dashboard not found.'}, status=404)
    except ModelEntry.DoesNotExist:
        return JsonResponse({'success': False, 'message': 'Model not found.'}, status=404)


@csrf_exempt
@require_GET
@jwt_authenticated
def dataprofile(request):
    """
    Generate a minimal data profile for a model's dataset using ydata-profiling
    """
    try:
        model_id = request.GET.get('id')
        if not model_id:
            return JsonResponse({'success': False, 'message': 'Model ID is required'}, status=400)
        
        # Check if user has access to this model
        user = User.objects.get(username=request.jwt_payload['username'])
        if not user.models.filter(id=model_id).exists():
            return JsonResponse({'success': False, 'message': 'Model not found or access denied'}, status=404)
        
        # Get the model entry
        model_entry = ModelEntry.objects.get(id=model_id)
        
        # Check if a data profile already exists for this model
        try:
            existing_profile = DataProfile.objects.get(model_entry=model_entry)
            # Return existing profile
            return JsonResponse({
                'success': True,
                'html': existing_profile.html_content,
                'model_id': model_id,
                'dataset_shape': existing_profile.dataset_shape,
                'columns': json.loads(existing_profile.columns),
                'cached': True,
                'created_at': existing_profile.created_at.isoformat()
            }, status=200)
        except DataProfile.DoesNotExist:
            # Profile doesn't exist, generate new one
            pass
        
        # Check if dataset file exists
        dataset_path = f'data/{model_id}.csv'
        if not default_storage.exists(dataset_path):
            return JsonResponse({'success': False, 'message': 'Dataset not found'}, status=404)
        
        # Read the dataset into a DataFrame
        with default_storage.open(dataset_path) as f:
            df = pd.read_csv(f)
        
        # Generate minimal profile report
        profile = ProfileReport(
            df, 
            title=f"Data Profile for Model {model_id}",
            sample=None
        )
        
        # Generate HTML report
        html_content = profile.to_html()

        # Inject JavaScript code into the HTML
        js_code = generate_data_profile_js_code()
        html_content = inject_js_into_html(html_content, js_code)

        # Save the profile to database
        data_profile = DataProfile.objects.create(
            model_entry=model_entry,
            html_content=html_content,
            dataset_shape=f"{df.shape[0]}, {df.shape[1]}",
            columns=json.dumps(list(df.columns))
        )
        
        return JsonResponse({
            'success': True,
            'html': html_content,
            'model_id': model_id,
            'dataset_shape': f"{df.shape[0]}, {df.shape[1]}",
            'columns': list(df.columns),
            'cached': False,
            'created_at': data_profile.created_at.isoformat()
        }, status=200)
        
    except Exception as e:
        logger.error(f"Error in dataprofile endpoint: {e}")
        return JsonResponse({
            'success': False,
            'message': f'Error generating data profile: {str(e)}'
        }, status=500)


@csrf_exempt
@require_POST
@jwt_authenticated
def delete_dataprofile(request):
    """
    Delete a data profile for a model
    """
    try:
        data = json.loads(request.body)
        model_id = data.get('model_id')
        
        if not model_id:
            return JsonResponse({'success': False, 'message': 'Model ID is required'}, status=400)
        
        # Check if user has access to this model
        user = User.objects.get(username=request.jwt_payload['username'])
        if not user.models.filter(id=model_id).exists():
            return JsonResponse({'success': False, 'message': 'Model not found or access denied'}, status=404)
        
        # Get the model entry
        model_entry = ModelEntry.objects.get(id=model_id)
        
        # Try to delete the existing profile
        try:
            existing_profile = DataProfile.objects.get(model_entry=model_entry)
            existing_profile.delete()
            return JsonResponse({
                'success': True,
                'message': 'Data profile deleted successfully'
            }, status=200)
        except DataProfile.DoesNotExist:
            return JsonResponse({
                'success': False,
                'message': 'No data profile found for this model'
            }, status=404)
            
    except ModelEntry.DoesNotExist:
        return JsonResponse({'success': False, 'message': 'Model not found'}, status=404)
    except Exception as e:
        logger.error(f"Error deleting data profile: {e}")
        return JsonResponse({
            'success': False,
            'message': f'Error deleting data profile: {str(e)}'
        }, status=500)


@csrf_exempt
@require_GET
@jwt_authenticated
def get_latest_report_for_model(request):
    """
    Get the latest report for a given model (if any) for the authenticated user.
    GET with ?model_id=... Returns the latest report or 404 if none exists.
    """
    try:
        model_id = request.GET.get('model_id')
        if not model_id:
            return JsonResponse({'success': False, 'message': 'model_id is required'}, status=400)
        user = User.objects.get(username=request.jwt_payload['username'])
        # Check user has access to this model
        if not user.models.filter(id=model_id).exists():
            return JsonResponse({'success': False, 'message': 'Model not found or access denied'}, status=404)
        # Get latest report for this model
        report = Report.objects.filter(model_entry_id=model_id).order_by('-created_at').first()
        if not report:
            return JsonResponse({'success': False, 'message': 'No report found for this model'}, status=404)
        # Return minimal info (id, status, created_at, title, etc.)
        return JsonResponse({
            'success': True,
            'report': {
                'id': report.id,
                'title': report.title,
                'status': report.status,
                'created_at': report.created_at.isoformat(),
                'updated_at': report.updated_at.isoformat(),
                'report_type': report.report_type,
            }
        }, status=200)
    except Exception as e:
        logger.error(f"Error getting latest report for model: {e}")
        return JsonResponse({
            'success': False,
            'message': 'Error retrieving latest report for model'
        }, status=500)


@csrf_exempt
@require_POST
@jwt_authenticated
def share_model(request):
    """
    Shares a model with another user by adding it to their accessible models.

    Expected POST data:
    - model_id: ID of the ModelEntry to share
    - email: Email of the user to share the model with
    """
    try:
        data = json.loads(request.body)
        model_id = data.get('model_id')
        target_email = data.get('email')

        if not model_id or not target_email:
            return JsonResponse({
                'success': False,
                'message': 'model_id and email are required'
            }, status=400)

        # Get the current user (sharer)
        sharer_user = User.objects.get(username=request.jwt_payload['username'])

        # Verify the sharer has access to the model
        try:
            model_entry = ModelEntry.objects.get(id=model_id)
            if not sharer_user.models.filter(id=model_id).exists():
                return JsonResponse({
                    'success': False,
                    'message': 'Model not found or access denied for sharing'
                }, status=403) # 403 Forbidden if sharer doesn't own or have access
        except ModelEntry.DoesNotExist:
            return JsonResponse({
                'success': False,
                'message': 'Model not found'
            }, status=404)

        # Find the target user by email
        try:
            target_user = User.objects.get(email=target_email)
        except User.DoesNotExist:
            return JsonResponse({
                'success': False,
                'message': 'Target user with this email does not exist'
            }, status=404)

        # Prevent sharing with self
        if sharer_user.id == target_user.id:
            return JsonResponse({
                'success': False,
                'message': 'Cannot share a model with yourself'
            }, status=400)

        # Add the model to the target user's accessible models
        # Check if the model is already shared with the target user
        if target_user.models.filter(id=model_id).exists():
            return JsonResponse({
                'success': False,
                'message': 'Model already shared with this user'
            }, status=409) # 409 Conflict if already shared

        target_user.models.add(model_entry)
        target_user.save()

        return JsonResponse({
            'success': True,
            'message': 'Model shared successfully!'
        }, status=200)

    except User.DoesNotExist:
        return JsonResponse({
            'success': False,
            'message': 'Authentication error: User not found.'
        }, status=401)
    except Exception as e:
        logger.error(f"Error sharing model: {e}")
        return JsonResponse({
            'success': False,
            'message': f'An unexpected error occurred: {str(e)}'
        }, status=500)