
from rest_framework import viewsets, status
from rest_framework.response import Response
from rest_framework.decorators import action
from sqlalchemy import create_engine, inspect
import pandas as pd
from django.core.files.base import ContentFile
import io
from .views import _process_uploaded_data
from .authentication import JWTAuthentication

class DBConnectionViewSet(viewsets.ViewSet):
    authentication_classes = [JWTAuthentication]

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
            csv_buffer.seek(0)
            
            # Use the refactored function
            return _process_uploaded_data(csv_buffer, request.user)

        except Exception as e:
            return Response({'success': False, 'message': str(e)}, status=status.HTTP_400_BAD_REQUEST)
