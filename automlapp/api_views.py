from rest_framework import viewsets, permissions, status
from rest_framework.response import Response
from rest_framework.decorators import api_view, permission_classes, authentication_classes
from rest_framework.permissions import IsAuthenticated
from .models import APIKey, ModelEntry, User
from .serializers import APIKeySerializer
from .authentication import APIKeyAuthentication
import joblib
import pandas as pd
import json
from django.core.files.storage import default_storage
import logging

logger = logging.getLogger(__name__)

class APIKeyViewSet(viewsets.ModelViewSet):
    serializer_class = APIKeySerializer
    permission_classes = [permissions.IsAuthenticated]

    def get_queryset(self):
        return self.request.user.api_keys.all()

    def perform_create(self, serializer):
        serializer.save(user=self.request.user)

@api_view(['POST'])
@authentication_classes([APIKeyAuthentication])
@permission_classes([IsAuthenticated])
def api_infer(request):
    """
    Perform inference using a trained model, authenticated via API key.
    """
    logger.info(f"Received API inference request from user: {request.user.username}")
    
    try:
        data = request.data
        model_id = data.get('id')

        if not model_id:
            logger.warning("Missing model ID in API inference request")
            return Response({'success': False, 'message': 'Model ID is required'}, status=status.HTTP_400_BAD_REQUEST)

        logger.info(f"Initiating API inference for model ID: {model_id}")

        # Verify user has access to the model
        if not request.user.models.filter(id=model_id).exists():
            logger.warning(f"User {request.user.username} attempted to access unauthorized model ID: {model_id}")
            return Response({'success': False, 'message': 'Model not found or access denied'}, status=status.HTTP_404_NOT_FOUND)

        entry = ModelEntry.objects.get(id=model_id)
        logger.info(f"Found model '{entry.name}' (Task: {entry.task}) for API inference")

        try:
            model_path = f'models/{entry.id}.pkl'
            pipeline_path = f'pipelines/{entry.id}.pkl'
            
            with default_storage.open(model_path) as f:
                model = joblib.load(f)
            with default_storage.open(pipeline_path) as f:
                pl = joblib.load(f)
            
            logger.info(f"Successfully loaded model and pipeline for model ID: {model_id}")

        except FileNotFoundError:
            logger.error(f"Model or pipeline file not found for model ID: {model_id}")
            return Response({'success': False, 'message': 'Model files not found. Please retrain the model.'}, status=status.HTTP_404_NOT_FOUND)

        if entry.task == 'TimeSeries':
            # TimeSeries forecasting via API is not yet supported in this example.
            # A full implementation would need to decide how to handle forecasting horizons.
            logger.warning(f"API inference for TimeSeries model {model_id} is not supported.")
            return Response({'success': False, 'message': 'TimeSeries forecasting via API is not currently supported.'}, status=status.HTTP_400_BAD_REQUEST)
        else:
            input_data = data.get('data', {})
            if not input_data:
                logger.warning("No data provided for API inference")
                return Response({'success': False, 'message': 'No data provided for inference'}, status=status.HTTP_400_BAD_REQUEST)

            logger.info(f"Received input data for API inference: {input_data}")
            
            df = pd.DataFrame([input_data])
            
            try:
                processed_df = pl.transform(df)
                logger.info("Successfully transformed input data with pipeline")
            except Exception as e:
                logger.error(f"Error transforming data for model ID {model_id}: {e}")
                return Response({'success': False, 'message': f'Error in data transformation: {str(e)}'}, status=status.HTTP_400_BAD_REQUEST)

            try:
                prediction = model.predict(processed_df)
                final_prediction = prediction[0]
                
                if hasattr(final_prediction, 'item'):
                    final_prediction = final_prediction.item()
                
                logger.info(f"Successfully made API prediction: {final_prediction}")
                return Response({'success': True, 'prediction': final_prediction}, status=status.HTTP_200_OK)

            except Exception as e:
                logger.error(f"Error during API prediction for model ID {model_id}: {e}")
                return Response({'success': False, 'message': f'Error during prediction: {str(e)}'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

    except Exception as e:
        logger.exception(f"A critical unexpected error occurred in the api_infer view: {e}")
        return Response({'success': False, 'message': 'An unexpected server error occurred'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
