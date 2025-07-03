
from rest_framework import viewsets, status
from rest_framework.response import Response
from rest_framework.decorators import action
from sqlalchemy import create_engine, inspect
import pandas as pd
from django.core.files.base import ContentFile
from django.core.files.storage import default_storage
import io
from .views import loadData 

class DBConnectionViewSet(viewsets.ViewSet):
    @action(detail=False, methods=['post'])
    def test_connection(self, request):
        db_details = request.data
        try:
            engine = create_engine(
                f"postgresql://{db_details['user']}:{db_details['password']}@{db_details['host']}:{db_details['port']}/{db_details['database']}"
            )
            connection = engine.connect()
            connection.close()
            return Response({'success': True, 'message': 'Connection successful'})
        except Exception as e:
            return Response({'success': False, 'message': str(e)}, status=status.HTTP_400_BAD_REQUEST)

    @action(detail=False, methods=['post'])
    def list_tables(self, request):
        db_details = request.data
        try:
            engine = create_engine(
                f"postgresql://{db_details['user']}:{db_details['password']}@{db_details['host']}:{db_details['port']}/{db_details['database']}"
            )
            inspector = inspect(engine)
            tables = inspector.get_table_names()
            return Response({'success': True, 'tables': tables})
        except Exception as e:
            return Response({'success': False, 'message': str(e)}, status=status.HTTP_400_BAD_REQUEST)

    @action(detail=False, methods=['post'])
    def preview_table(self, request):
        db_details = request.data
        table_name = request.data.get('table')
        if not table_name:
            return Response({'success': False, 'message': 'Table name not provided'}, status=status.HTTP_400_BAD_REQUEST)
        try:
            engine = create_engine(
                f"postgresql://{db_details['user']}:{db_details['password']}@{db_details['host']}:{db_details['port']}/{db_details['database']}"
            )
            df = pd.read_sql(f'SELECT * FROM "{table_name}" LIMIT 5', engine)
            csv_buffer = io.StringIO()
            df.to_csv(csv_buffer, index=False)
            return Response({'success': True, 'preview': csv_buffer.getvalue()})
        except Exception as e:
            return Response({'success': False, 'message': str(e)}, status=status.HTTP_400_BAD_REQUEST)

    @action(detail=False, methods=['post'])
    def load_table(self, request):
        db_details = request.data
        table_name = request.data.get('table')
        if not table_name:
            return Response({'success': False, 'message': 'Table name not provided'}, status=status.HTTP_400_BAD_REQUEST)
        try:
            engine = create_engine(
                f"postgresql://{db_details['user']}:{db_details['password']}@{db_details['host']}:{db_details['port']}/{db_details['database']}"
            )
            df = pd.read_sql(f'SELECT * FROM "{table_name}"', engine)
            
            csv_buffer = io.StringIO()
            df.to_csv(csv_buffer, index=False)
            
            # Create a ContentFile to simulate a file upload
            csv_file = ContentFile(csv_buffer.getvalue().encode('utf-8'), name=f"{table_name}.csv")
            
            # Create a mock request object for loadData
            from django.http import HttpRequest
            from django.core.files.uploadedfile import InMemoryUploadedFile

            mock_request = HttpRequest()
            mock_request.method = 'POST'
            mock_request.FILES['file'] = InMemoryUploadedFile(csv_file, 'file', f"{table_name}.csv", 'text/csv', len(csv_buffer.getvalue()), None)
            mock_request.jwt_payload = request.jwt_payload

            # Call the existing loadData view
            response = loadData(mock_request)
            return response

        except Exception as e:
            return Response({'success': False, 'message': str(e)}, status=status.HTTP_400_BAD_REQUEST)
