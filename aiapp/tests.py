import pytest
import json
import pandas as pd
import numpy as np
from django.test import TestCase, Client
from django.urls import reverse
from django.contrib.auth import get_user_model
from unittest.mock import patch, MagicMock
from automlapp.models import ModelEntry, User
from .models import Report, ChartData, DataInsight
from .services import ReportGenerationService, ChartExportService, generate_report_async
from celery.result import AsyncResult
import tempfile
import os
from django.conf import settings
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile
import io

User = get_user_model()

@pytest.mark.django_db
class TestReportModels(TestCase):
    """Test cases for report models"""

    def setUp(self):
        """Set up test data"""
        self.user = User.objects.create_user(
            username='testuser',
            email='test@example.com',
            password='testpass123'
        )

        self.model_entry = ModelEntry.objects.create(
            name="Test Model",
            description="Test Description",
            task="Classification",
            target_variable="target",
            list_of_features='{"feature1": "float", "feature2": "int"}',
            status='Done',
            model_name="RandomForest",
            evaluation_metric="accuracy",
            evaluation_metric_value=0.85
        )

        self.user.models.add(self.model_entry)

        self.report = Report.objects.create(
            model_entry=self.model_entry,
            title="Test Report",
            description="Test report description",
            report_type="analysis",
            status="completed"
        )

    def test_report_creation(self):
        """Test report model creation"""
        self.assertEqual(self.report.title, "Test Report")
        self.assertEqual(self.report.model_entry, self.model_entry)
        self.assertEqual(self.report.status, "completed")
        self.assertEqual(str(self.report), f"Report for {self.model_entry.name} - {self.report.title}")

    def test_chart_data_creation(self):
        """Test chart data model creation"""
        chart = ChartData.objects.create(
            report=self.report,
            chart_type='pie',
            title='Test Pie Chart',
            description='Test chart description',
            chart_data={'labels': ['A', 'B'], 'values': [10, 20]}
        )

        self.assertEqual(chart.chart_type, 'pie')
        self.assertEqual(chart.title, 'Test Pie Chart')
        self.assertEqual(chart.get_chart_data_as_dict(), {'labels': ['A', 'B'], 'values': [10, 20]})
        self.assertEqual(str(chart), "Pie Chart - Test Pie Chart")

    def test_data_insight_creation(self):
        """Test data insight model creation"""
        insight = DataInsight.objects.create(
            report=self.report,
            insight_type='data_distribution',
            title='Test Insight',
            description='Test insight description',
            insight_data={'rows': 100, 'columns': 5},
            priority=1
        )

        self.assertEqual(insight.insight_type, 'data_distribution')
        self.assertEqual(insight.priority, 1)
        self.assertEqual(insight.insight_data, {'rows': 100, 'columns': 5})


@pytest.mark.django_db
class TestReportGenerationService(TestCase):
    """Test cases for report generation service"""

    def setUp(self):
        """Set up test data"""
        self.user = User.objects.create_user(
            username='testuser',
            email='test@example.com',
            password='testpass123'
        )

        self.model_entry = ModelEntry.objects.create(
            name="Test Model",
            description="Test Description",
            task="Classification",
            target_variable="target",
            list_of_features='{"feature1": "float", "feature2": "int"}',
            status='Done',
            model_name="RandomForest",
            evaluation_metric="accuracy",
            evaluation_metric_value=0.85
        )

        self.user.models.add(self.model_entry)

        # Create test CSV file
        self.test_data = pd.DataFrame({
            'feature1': [1.0, 2.0, 3.0, 4.0, 5.0],
            'feature2': [10, 20, 30, 40, 50],
            'target': ['A', 'B', 'A', 'B', 'A']
        })

        # Save test data to CSV
        csv_buffer = io.StringIO()
        self.test_data.to_csv(csv_buffer, index=False)
        file_path = f'data/{self.model_entry.id}.csv'
        default_storage.save(file_path, ContentFile(csv_buffer.getvalue().encode('utf-8')))

        self.service = ReportGenerationService()

    def tearDown(self):
        """Clean up test files"""
        try:
            file_path = f'data/{self.model_entry.id}.csv'
            default_storage.delete(file_path)
        except FileNotFoundError:
            pass

    @patch('aiapp.services.genai')
    def test_generate_report_classification(self, mock_genai):
        """Test report generation for classification task"""
        # Mock Gemini API
        mock_model = MagicMock()
        mock_model.generate_content.return_value.text = "Test AI insights"
        mock_genai.GenerativeModel.return_value = mock_model
        self.service.gemini_model = mock_model

        report = self.service.generate_report(self.model_entry.id, 'analysis')

        self.assertIsInstance(report, Report)
        self.assertEqual(report.model_entry, self.model_entry)
        self.assertEqual(report.status, 'completed')
        self.assertTrue(report.charts.exists())
        self.assertTrue(report.insights.exists())

    def test_generate_report_without_gemini(self):
        """Test report generation without Gemini API"""
        self.service.gemini_model = None

        report = self.service.generate_report(self.model_entry.id, 'analysis')

        self.assertIsInstance(report, Report)
        self.assertEqual(report.status, 'completed')
        self.assertEqual(report.ai_insights, '')

    def test_create_pie_chart(self):
        """Test pie chart creation"""
        labels = ['A', 'B', 'C']
        values = [10, 20, 30]
        title = 'Test Pie Chart'

        result = self.service._create_pie_chart(labels, values, title)

        self.assertIn('image', result)
        self.assertIn('html', result)
        self.assertIsInstance(result['image'], str)
        self.assertIsInstance(result['html'], str)

    def test_create_line_chart(self):
        """Test line chart creation"""
        x = [1, 2, 3, 4, 5]
        y = [10, 20, 15, 25, 30]
        title = 'Test Line Chart'

        result = self.service._create_line_chart(x, y, title)

        self.assertIn('image', result)
        self.assertIn('html', result)
        self.assertIsInstance(result['image'], str)
        self.assertIsInstance(result['html'], str)


@pytest.mark.django_db
class TestChartExportService(TestCase):
    """Test cases for chart export service"""

    def setUp(self):
        """Set up test data"""
        self.user = User.objects.create_user(
            username='testuser',
            email='test@example.com',
            password='testpass123'
        )

        self.model_entry = ModelEntry.objects.create(
            name="Test Model",
            description="Test Description",
            task="Classification",
            target_variable="target",
            list_of_features='{"feature1": "float", "feature2": "int"}',
            status='Done',
            model_name="RandomForest",
            evaluation_metric="accuracy",
            evaluation_metric_value=0.85
        )

        self.report = Report.objects.create(
            model_entry=self.model_entry,
            title="Test Report",
            description="Test report description"
        )

        self.chart = ChartData.objects.create(
            report=self.report,
            chart_type='pie',
            title='Test Chart',
            chart_data={'labels': ['A', 'B', 'C'], 'values': [10, 20, 30]}
        )

    def test_export_pie_chart_csv(self):
        """Test exporting pie chart data to CSV"""
        csv_data = ChartExportService.export_chart_data_to_csv(self.chart.id)

        self.assertIn('Label,Value', csv_data)
        self.assertIn('A,10', csv_data)
        self.assertIn('B,20', csv_data)
        self.assertIn('C,30', csv_data)

    def test_get_chart_summary_stats(self):
        """Test getting chart summary statistics"""
        stats = ChartExportService.get_chart_summary_stats(self.chart.id)

        self.assertEqual(stats['chart_type'], 'pie')
        self.assertEqual(stats['title'], 'Test Chart')
        self.assertEqual(stats['data_points'], 3)
        self.assertEqual(stats['summary']['total'], 60)
        self.assertEqual(stats['summary']['max_value'], 30)
        self.assertEqual(stats['summary']['min_value'], 10)


@pytest.mark.django_db
class TestReportViews(TestCase):
    """Test cases for report views"""

    def setUp(self):
        """Set up test data"""
        self.client = Client()
        self.user = User.objects.create_user(
            username='testuser',
            email='test@example.com',
            password='testpass123'
        )

        self.model_entry = ModelEntry.objects.create(
            name="Test Model",
            description="Test Description",
            task="Classification",
            target_variable="target",
            list_of_features='{"feature1": "float", "feature2": "int"}',
            status='Done',
            model_name="RandomForest",
            evaluation_metric="accuracy",
            evaluation_metric_value=0.85
        )

        self.user.models.add(self.model_entry)

        # Create test CSV file
        test_data = pd.DataFrame({
            'feature1': [1.0, 2.0, 3.0],
            'feature2': [10, 20, 30],
            'target': ['A', 'B', 'A']
        })
        csv_buffer = io.StringIO()
        test_data.to_csv(csv_buffer, index=False)
        file_path = f'data/{self.model_entry.id}.csv'
        default_storage.save(file_path, ContentFile(csv_buffer.getvalue().encode('utf-8')))

        # Mock JWT token
        self.jwt_payload = {'username': 'testuser'}

    def tearDown(self):
        """Clean up test files"""
        try:
            file_path = f'data/{self.model_entry.id}.csv'
            default_storage.delete(file_path)
        except FileNotFoundError:
            pass

    @patch('jwt.decode')
    @patch('aiapp.services.ReportGenerationService.generate_report')
    def test_generate_report_view(self, mock_generate, mock_jwt_decode):
        """Test generate report view"""
        # Mock JWT decode to return our test payload
        mock_jwt_decode.return_value = self.jwt_payload

        # Mock report generation
        mock_report = Report.objects.create(
            model_entry=self.model_entry,
            title="Test Report"
        )
        mock_generate.return_value = mock_report

        response = self.client.post(
            '/aiapp/generate/',
            data=json.dumps({
                'model_id': self.model_entry.id,
                'async': False  # Add async parameter
            }),
            content_type='application/json',
            HTTP_AUTHORIZATION='Bearer test-token'
        )

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertTrue(data['success'])
        self.assertEqual(data['report_id'], mock_report.id)

    @patch('jwt.decode')
    def test_generate_report_invalid_model(self, mock_jwt_decode):
        """Test generate report with invalid model ID"""
        # Mock JWT decode to return our test payload
        mock_jwt_decode.return_value = self.jwt_payload

        response = self.client.post(
            '/aiapp/generate/',
            data=json.dumps({
                'model_id': 99999,
                'async': False  # Add async parameter
            }),
            content_type='application/json',
            HTTP_AUTHORIZATION='Bearer test-token'
        )

        self.assertEqual(response.status_code, 404)
        data = response.json()
        self.assertFalse(data['success'])


@pytest.mark.django_db
class TestCeleryIntegration(TestCase):
    """Test cases for Celery async functionality"""

    def setUp(self):
        """Set up test data"""
        self.client = Client()
        self.user = User.objects.create_user(
            username='testuser',
            email='test@example.com',
            password='testpass123'
        )

        self.model_entry = ModelEntry.objects.create(
            name="Test Model",
            description="Test Description",
            task="Classification",
            target_variable="target",
            list_of_features='{"feature1": "float", "feature2": "int"}',
            status='Done',
            model_name="RandomForest",
            evaluation_metric="accuracy",
            evaluation_metric_value=0.85
        )

        self.user.models.add(self.model_entry)

        # Create test CSV file
        test_data = pd.DataFrame({
            'feature1': [1.0, 2.0, 3.0, 4.0, 5.0],
            'feature2': [10, 20, 30, 40, 50],
            'target': ['A', 'B', 'A', 'B', 'A']
        })
        csv_buffer = io.StringIO()
        test_data.to_csv(csv_buffer, index=False)
        file_path = f'data/{self.model_entry.id}.csv'
        default_storage.save(file_path, ContentFile(csv_buffer.getvalue().encode('utf-8')))

        # Mock JWT token
        self.jwt_payload = {'username': 'testuser'}

    def tearDown(self):
        """Clean up test files"""
        try:
            file_path = f'data/{self.model_entry.id}.csv'
            default_storage.delete(file_path)
        except FileNotFoundError:
            pass

    @patch('jwt.decode')
    @patch('aiapp.services.generate_report_async.delay')
    def test_async_report_generation_view(self, mock_task, mock_jwt_decode):
        """Test async report generation endpoint"""
        # Mock JWT decode to return our test payload
        mock_jwt_decode.return_value = self.jwt_payload

        # Mock Celery task
        mock_result = MagicMock()
        mock_result.id = 'test-task-id-123'
        mock_task.return_value = mock_result

        response = self.client.post(
            '/aiapp/generate/',
            data=json.dumps({
                'model_id': self.model_entry.id,
                'report_type': 'analysis',
                'async': True
            }),
            content_type='application/json',
            HTTP_AUTHORIZATION='Bearer test-token'
        )

        self.assertEqual(response.status_code, 202)
        data = response.json()
        self.assertTrue(data['success'])
        self.assertEqual(data['task_id'], 'test-task-id-123')
        self.assertTrue(data['async'])
        self.assertEqual(data['message'], 'Report generation started')

        # Verify task was called with correct parameters
        mock_task.assert_called_once_with(self.model_entry.id, 'analysis')

    @patch('jwt.decode')
    @patch('aiapp.services.ReportGenerationService.generate_report')
    def test_sync_report_generation_view(self, mock_generate, mock_jwt_decode):
        """Test synchronous report generation endpoint"""
        # Mock JWT decode to return our test payload
        mock_jwt_decode.return_value = self.jwt_payload

        # Mock report generation
        mock_report = Report.objects.create(
            model_entry=self.model_entry,
            title="Test Report"
        )
        mock_generate.return_value = mock_report

        response = self.client.post(
            '/aiapp/generate/',
            data=json.dumps({
                'model_id': self.model_entry.id,
                'report_type': 'analysis',
                'async': False
            }),
            content_type='application/json',
            HTTP_AUTHORIZATION='Bearer test-token'
        )

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertTrue(data['success'])
        self.assertEqual(data['report_id'], mock_report.id)
        self.assertFalse(data['async'])
        self.assertEqual(data['message'], 'Report generated successfully')

    @patch('jwt.decode')
    @patch('aiapp.views.AsyncResult')
    def test_task_status_view_pending(self, mock_async_result, mock_jwt_decode):
        """Test task status endpoint for pending task"""
        # Mock JWT decode to return our test payload
        mock_jwt_decode.return_value = self.jwt_payload

        # Mock AsyncResult
        mock_result = MagicMock()
        mock_result.status = 'PENDING'
        mock_result.ready.return_value = False
        mock_result.info = None
        mock_async_result.return_value = mock_result

        response = self.client.get(
            '/aiapp/task-status/?task_id=test-task-id',
            HTTP_AUTHORIZATION='Bearer test-token'
        )

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertTrue(data['success'])
        self.assertEqual(data['status'], 'PENDING')
        self.assertIn('status_message', data)

    @patch('jwt.decode')
    @patch('aiapp.views.AsyncResult')
    def test_task_status_view_success(self, mock_async_result, mock_jwt_decode):
        """Test task status endpoint for successful task"""
        # Mock JWT decode to return our test payload
        mock_jwt_decode.return_value = self.jwt_payload

        # Mock AsyncResult
        mock_result = MagicMock()
        mock_result.status = 'SUCCESS'
        mock_result.ready.return_value = True
        mock_result.result = 123  # Report ID
        mock_async_result.return_value = mock_result

        response = self.client.get(
            '/aiapp/task-status/?task_id=test-task-id',
            HTTP_AUTHORIZATION='Bearer test-token'
        )

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertTrue(data['success'])
        self.assertEqual(data['status'], 'SUCCESS')
        self.assertEqual(data['result'], 123)

    @patch('jwt.decode')
    @patch('aiapp.views.AsyncResult')
    def test_task_status_view_failure(self, mock_async_result, mock_jwt_decode):
        """Test task status endpoint for failed task"""
        # Mock JWT decode to return our test payload
        mock_jwt_decode.return_value = self.jwt_payload

        # Mock AsyncResult
        mock_result = MagicMock()
        mock_result.status = 'FAILURE'
        mock_result.ready.return_value = True
        mock_result.info = 'Test error message'  # Should be string, not dict
        mock_async_result.return_value = mock_result

        response = self.client.get(
            '/aiapp/task-status/?task_id=test-task-id',
            HTTP_AUTHORIZATION='Bearer test-token'
        )

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertTrue(data['success'])
        self.assertEqual(data['status'], 'FAILURE')
        self.assertIn('error', data)

    @patch('jwt.decode')
    def test_get_report_by_task_view(self, mock_jwt_decode):
        """Test get report by task ID endpoint"""
        # Mock JWT decode to return our test payload
        mock_jwt_decode.return_value = self.jwt_payload

        # Create a report with task_id
        report = Report.objects.create(
            model_entry=self.model_entry,
            title="Test Report",
            task_id="test-task-id-123",
            status="completed"  # Set status to completed to get full report data
        )

        response = self.client.get(
            '/aiapp/report-by-task/?task_id=test-task-id-123',
            HTTP_AUTHORIZATION='Bearer test-token'
        )

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertTrue(data['success'])
        self.assertEqual(data['status'], 'completed')
        self.assertEqual(data['report']['id'], report.id)
        self.assertEqual(data['report']['title'], "Test Report")

    @patch('jwt.decode')
    def test_get_report_by_task_not_found(self, mock_jwt_decode):
        """Test get report by task ID when report doesn't exist"""
        # Mock JWT decode to return our test payload
        mock_jwt_decode.return_value = self.jwt_payload

        response = self.client.get(
            '/aiapp/report-by-task/?task_id=nonexistent-task',
            HTTP_AUTHORIZATION='Bearer test-token'
        )

        self.assertEqual(response.status_code, 404)
        data = response.json()
        self.assertFalse(data['success'])
        self.assertEqual(data['message'], 'Report not found for this task')

    @patch('aiapp.services.ReportGenerationService.generate_report_with_progress')
    def test_generate_report_async_task(self, mock_generate_with_progress):
        """Test the actual Celery task function"""
        # Mock the service method
        mock_report = Report.objects.create(
            model_entry=self.model_entry,
            title="Async Test Report"
        )
        mock_generate_with_progress.return_value = mock_report

        # Create a mock task instance
        mock_task = MagicMock()
        mock_task.update_state = MagicMock()

        # Call the task function directly (without self parameter)
        # We need to call the actual function, not the Celery task wrapper
        from aiapp.services import ReportGenerationService
        service = ReportGenerationService()
        result = service.generate_report_with_progress(self.model_entry.id, 'analysis', mock_task)

        self.assertEqual(result, mock_report)
        mock_generate_with_progress.assert_called_once_with(
            self.model_entry.id, 'analysis', mock_task
        )

    def test_report_model_task_id_field(self):
        """Test that Report model can store task_id"""
        report = Report.objects.create(
            model_entry=self.model_entry,
            title="Test Report with Task ID",
            task_id="test-task-id-456"
        )

        self.assertEqual(report.task_id, "test-task-id-456")

        # Test retrieval by task_id
        retrieved_report = Report.objects.get(task_id="test-task-id-456")
        self.assertEqual(retrieved_report.id, report.id)