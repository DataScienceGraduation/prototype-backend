from django.shortcuts import render
from django.conf import settings
from django.views.decorators.http import require_POST, require_GET
from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse
from functools import wraps
from automlapp.models import ModelEntry, User
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile
import pandas as pd
from automlapp.tasks import train_model_task
import joblib
import json
import ast
import jwt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import base64
import io
import os
import logging
from django.http import HttpResponse


logger = logging.getLogger(__name__)

JWT_ALGORITHM = 'HS256'
JWT_SECRET = settings.SECRET_KEY

def generate_timeseries_plot(predictions, forecast_horizon, target_variable):
    """
    Generate a time series forecast plot for multiple timesteps and return it as a base64 encoded string
    """
    plt.figure(figsize=(12, 6))

    # Multiple predictions - show as a line plot
    x_values = list(range(1, len(predictions) + 1))
    plt.plot(x_values, predictions, marker='o', linewidth=2.5, markersize=8,
            color='#2E86AB', markerfacecolor='#A23B72', markeredgecolor='white', markeredgewidth=2)
    plt.title(f'Time Series Forecast - Next {forecast_horizon} Timesteps', fontsize=14, fontweight='bold')
    plt.xlabel('Future Timestep', fontsize=12)
    plt.ylabel(target_variable, fontsize=12)
    plt.grid(True, alpha=0.3)

    # Add value annotations on points
    for i, value in enumerate(predictions):
        plt.annotate(f'{value:.2f}', (x_values[i], value),
                    textcoords="offset points", xytext=(0,15), ha='center',
                    fontsize=10, bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

    # Set x-axis to show integer ticks only
    plt.xticks(x_values)

    plt.tight_layout()

    # Save plot to base64 string
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight', facecolor='white')
    buffer.seek(0)
    plot_data = buffer.getvalue()
    buffer.close()
    plt.close()  # Important: close the figure to free memory

    # Encode to base64
    plot_base64 = base64.b64encode(plot_data).decode('utf-8')
    return plot_base64

# Create your views here.
def jwt_authenticated(view_func):
    @wraps(view_func)
    def wrapped_view(request, *args, **kwargs):
        auth_header = request.headers.get('Authorization')
        if not auth_header or not auth_header.startswith('Bearer '):
            return JsonResponse({'success': False, 'message': 'Authorization header missing or invalid'}, status=401)

        token = auth_header.split(' ')[1]
        try:
            payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
            request.jwt_payload = payload
        except jwt.ExpiredSignatureError:
            return JsonResponse({'success': False, 'message': 'Token expired'}, status=401)
        except jwt.InvalidTokenError:
            return JsonResponse({'success': False, 'message': 'Invalid token'}, status=401)

        return view_func(request, *args, **kwargs)

    return wrapped_view

@csrf_exempt
@require_POST
def login(request):
    try:
        # Handle both JSON and form data
        if request.content_type == 'application/json':
            data = json.loads(request.body)
        else:
            data = request.POST

        user = User.objects.get(username=data['username'])
        if user.check_password(data['password']):
            token = jwt.encode({'username': user.username}, JWT_SECRET, algorithm=JWT_ALGORITHM)
            return JsonResponse({'success': True, 'token': token}, status=200)
        else:
            return JsonResponse({'success': False, 'message': 'Invalid credentials'}, status=400)
    except Exception as e:
        print(f"Error in login: {e}")
        return JsonResponse({'success': False, 'message': 'There was an error'}, status=500)



@csrf_exempt
@require_POST
def register(request):
    try:
        data = request.POST
        username = data['username']
        first_name = data['first_name']
        last_name = data['last_name']
        email = data['email']
        password = data['password']
        if not username or not password or not email:
            return JsonResponse({'success': False, 'message': 'Username or password not provided'}, status=400)
        if User.objects.filter(username=username).exists():
            return JsonResponse({'success': False, 'message': 'Username already exists'}, status=400)

        user = User.objects.create_user(
            username=username,
            password=password,
            email=email,
            first_name=first_name,
            last_name=last_name
        )
        return JsonResponse({'success': True, 'message': 'User registered successfully'}, status=201)
    except Exception as e:
        print(f"Error in register: {e}")
        return JsonResponse({'success': False, 'message': 'There was an error'}, status=500)



def _process_uploaded_data(file_obj, user):
    """
    Reusable function to process uploaded data, whether from a file or a database.
    """
    logger.info(f"Processing data for user: {user.username}")
    
    try:
        df = pd.read_csv(file_obj)
        if df.empty:
            logger.error("Uploaded data is empty")
            return JsonResponse({'success': False, 'message': 'The uploaded data is empty'}, status=400)
    except Exception as e:
        logger.error(f"Failed to parse data: {str(e)}")
        return JsonResponse({'success': False, 'message': f'Error parsing your data: {str(e)}'}, status=400)
    
    # Analyze columns (same logic as before)
    columns = list(df.columns)
    suggested_target_variables = []
    features = {}
    datetime_columns = []
    
    logger.info(f"Analyzing {len(columns)} columns from dataset with {len(df)} rows")
    
    for column in columns:
        if df[column].dtype == 'object':
            try:
                parsed_col = pd.to_datetime(df[column], errors='coerce')
                valid_ratio = parsed_col.notna().sum() / len(df[column])
                if valid_ratio > 0.8:
                    datetime_columns.append(column)
                    features[column] = 'datetime'
            except Exception:
                pass
    
    for column in columns:
        if column in datetime_columns:
            continue
        unique_values = df[column].nunique()
        total_rows = len(df)
        suggested_target_variables.append(column)
        
        if unique_values == total_rows and unique_values > 1:
            features[column] = 'unique'
        elif unique_values == 1:
            features[column] = 'constant'
        elif unique_values < 10:
            features[column] = [str(x) for x in df[column].unique() if pd.notna(x)]
        elif df[column].dtype == 'object':
            features[column] = 'categorical'
        elif df[column].dtype in ['int64', 'float64']:
            features[column] = 'numerical'
        else:
            features[column] = 'unknown'

    # Create model entry
    try:
        entry = ModelEntry.objects.create(
            name="",
            description="",
            task="",
            target_variable="",
            list_of_features=features,
            status='Data Loaded',
            model_name="",
            evaluation_metric="",
            evaluation_metric_value=0
        )
        user.models.add(entry)
        
        file_path = f'data/{entry.id}.csv'
        csv_buffer = io.StringIO()
        df.to_csv(csv_buffer, index=False)
        default_storage.save(file_path, ContentFile(csv_buffer.getvalue().encode('utf-8')))
        
        entry.save()
        logger.info(f"Model entry created with ID: {entry.id}")
        
        return JsonResponse({
            'success': True, 
            'features': list(features.keys()), 
            'list_of_features': features,
            'data': suggested_target_variables, 
            'datetime_columns': datetime_columns, 
            'id': entry.id
        }, status=200)

    except Exception as e:
        logger.error(f"Failed to create model entry: {str(e)}")
        return JsonResponse({'success': False, 'message': f'Database error: {str(e)}'}, status=500)


@csrf_exempt
@require_POST
@jwt_authenticated
def loadData(request):
    """
    Load data from a CSV file, analyze columns, and create a model entry.
    """
    if 'file' not in request.FILES:
        return JsonResponse({'success': False, 'message': 'No file provided'}, status=400)
    
    file_obj = request.FILES['file']
    username = request.jwt_payload['username']
    user = User.objects.get(username=username)
    
    return _process_uploaded_data(file_obj, user)


@csrf_exempt
@require_POST
@jwt_authenticated
def trainModel(request):
    """
    Process a model training request, validate parameters, and queue the training task.
    """
    logger.info(f"Received model training request from user: {request.jwt_payload['username']}")
    
    try:
        # Extract and validate required fields
        model_id = request.POST.get('id')
        name = request.POST.get('name', '').strip()
        description = request.POST.get('description', '').strip()
        task = request.POST.get('task', '').strip()
        target_variable = request.POST.get('target_variable', '').strip()
        
        # Validate basic required fields
        if not name:
            logger.warning("Missing required field: name")
            return JsonResponse({'success': False, 'message': 'Model name is required'}, status=400)
        
        if not description:
            logger.warning("Missing required field: description")
            return JsonResponse({'success': False, 'message': 'Model description is required'}, status=400)
            
        if not task:
            logger.warning("Missing required field: task")
            return JsonResponse({'success': False, 'message': 'Task type is required'}, status=400)
            
        if not model_id:
            logger.warning("Missing required field: id")
            return JsonResponse({'success': False, 'message': 'Model ID is required'}, status=400)
        
        # Validate task-specific required fields
        if task != 'Clustering' and not target_variable:
            logger.warning(f"Missing target_variable for {task} task")
            return JsonResponse({
                'success': False, 
                'message': 'Target variable is required for non-clustering tasks'
            }, status=400)
        
        # Validate time series specific parameters
        if task == 'TimeSeries':
            datetime_column = request.POST.get('datetime_column', '').strip()
            date_format = request.POST.get('date_format', '').strip()
            
            if not datetime_column:
                logger.warning("Missing datetime_column for TimeSeries task")
                return JsonResponse({
                    'success': False, 
                    'message': 'Datetime column is required for time series tasks'
                }, status=400)
                
            if not date_format:
                logger.warning("Missing date_format for TimeSeries task")
                return JsonResponse({
                    'success': False, 
                    'message': 'Date format is required for time series tasks'
                }, status=400)
        
        # Verify model exists and user has access
        try:
            username = request.jwt_payload['username']
            user = User.objects.get(username=username)
            
            if not user.models.filter(id=model_id).exists():
                logger.warning(f"User {username} attempted to access non-existent or unauthorized model ID: {model_id}")
                return JsonResponse({'success': False, 'message': 'Model not found or access denied'}, status=404)
                
            entry = ModelEntry.objects.get(id=model_id)
            logger.info(f"Found model entry with ID {model_id}, updating with new parameters")
            
        except User.DoesNotExist:
            logger.error(f"User {request.jwt_payload['username']} not found in database")
            return JsonResponse({'success': False, 'message': 'User not found'}, status=404)
        except ModelEntry.DoesNotExist:
            logger.error(f"Model with ID {model_id} not found in database")
            return JsonResponse({'success': False, 'message': 'Model not found'}, status=404)
        
        # Update model entry with new parameters
        try:
            entry.name = name
            entry.description = description
            entry.task = task
            entry.target_variable = target_variable
            
            if task == 'TimeSeries':
                entry.datetime_column = datetime_column
                entry.date_format = date_format
                
            entry.status = 'Model Training'
            entry.save()
            logger.info(f"Successfully updated model parameters for ID {model_id}")
            
        except Exception as e:
            logger.error(f"Database error while updating model entry: {str(e)}")
            return JsonResponse({'success': False, 'message': f'Error updating model: {str(e)}'}, status=500)
        
        # Queue the training task
        try:
            logger.info(f"Queuing training task for model ID {model_id}")
            task_id = train_model_task.delay(entry.id)
            logger.info(f"Training task queued successfully with task ID: {task_id}")
            
            return JsonResponse({
                'success': True, 
                'message': 'Model training has been queued',
                'task_id': str(task_id)
            }, status=200)
            
        except Exception as e:
            logger.error(f"Failed to queue training task: {str(e)}")
            entry.status = 'Error'
            entry.save()
            return JsonResponse({'success': False, 'message': f'Failed to queue training task: {str(e)}'}, status=500)
            
    except Exception as e:
        logger.exception(f"Unexpected error in trainModel: {str(e)}")
        return JsonResponse({'success': False, 'message': f'There was an error: {str(e)}'}, status=500)


@csrf_exempt
@require_GET
@jwt_authenticated
def getAllModels(request):
    try:
        entries = User.objects.get(username=request.jwt_payload['username']).models.all()
        data = []
        for entry in entries:
            data.append({
                'id': entry.id,
                'name': entry.name,
                'description': entry.description,
                'task': entry.task,
                'target_variable': entry.target_variable,
                'list_of_features': entry.list_of_features,
                'status': entry.status,
                'model_name': entry.model_name,
                'evaluation_metric': entry.evaluation_metric,
                'evaluation_metric_value': entry.evaluation_metric_value
            })
        response = JsonResponse({'success': True, 'data': data }, status=200)
        response["Access-Control-Allow-Origin"] = "http://localhost:3000"
        return response
    except Exception as e:
        print(f"Error in getting all models: {e}")
        return JsonResponse({'success': False, 'message': 'There was an error'}, status=500)

@csrf_exempt
@require_GET
def getModel(request):
    try:
        entries = ModelEntry.objects.get(id=request.GET['id'])
        
        # Get dataset information
        dataset_info = {}
        try:
            import pandas as pd
            file_path = f'data/{entries.id}.csv'
            with default_storage.open(file_path) as f:
                df = pd.read_csv(f)
            dataset_info = {
                'rows': len(df),
                'columns': len(df.columns),
                'column_names': list(df.columns),
                'missing_values': df.isnull().sum().to_dict(),
                'data_types': df.dtypes.astype(str).to_dict(),
                'sample_data': df.head(5).to_dict('records')
            }
        except Exception as e:
            print(f"Error reading dataset: {e}")
            dataset_info = {'error': 'Could not load dataset information'}
        
        data = {
            'id': entries.id,
            'name': entries.name,
            'description': entries.description,
            'task': entries.task,
            'target_variable': entries.target_variable,
            'list_of_features': entries.list_of_features,
            'status': entries.status,
            'model_name': entries.model_name,
            'evaluation_metric': entries.evaluation_metric,
            'evaluation_metric_value': entries.evaluation_metric_value,
            'dataset': dataset_info
        }
        response = JsonResponse({'success': True, 'data': data }, status=200)
        response["Access-Control-Allow-Origin"] = "http://localhost:3000"
        return response
    except Exception as e:
        print(f"Error in getting model: {e}")
        return JsonResponse({'success': False, 'message': 'There was an error'}, status=500)

@csrf_exempt
@require_POST
@jwt_authenticated
def infer(request):
    """
    Perform inference using a trained model.
    This function handles both single predictions and time series forecasting.
    """
    logger.info(f"Received inference request from user: {request.jwt_payload['username']}")
    
    try:
        data = request.POST
        model_id = data.get('id')

        if not model_id:
            logger.warning("Missing model ID in inference request")
            return JsonResponse({'success': False, 'message': 'Model ID is required'}, status=400)

        logger.info(f"Initiating inference for model ID: {model_id}")

        # Verify user has access to the model
        username = request.jwt_payload['username']
        user = User.objects.get(username=username)
        if not user.models.filter(id=model_id).exists():
            logger.warning(f"User {username} attempted to access unauthorized model ID: {model_id}")
            return JsonResponse({'success': False, 'message': 'Model not found or access denied'}, status=404)

        # Fetch model entry
        entry = ModelEntry.objects.get(id=model_id)
        logger.info(f"Found model '{entry.name}' (Task: {entry.task})")

        # Load model and pipeline
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
            return JsonResponse({'success': False, 'message': 'Model files not found. Please retrain the model.'}, status=404)

        # Handle Time Series Forecasting
        if entry.task == 'TimeSeries':
            try:
                forecast_horizon = int(data.get('forecast_horizon', 1))
                logger.info(f"Performing time series forecast with horizon: {forecast_horizon}")
                
                prediction = model.forecast(steps=forecast_horizon)

                if isinstance(prediction, pd.Series):
                    predictions_list = prediction.tolist()
                else:
                    predictions_list = prediction.iloc[:, 0].tolist() if len(prediction.shape) > 1 else prediction.tolist()

                predictions_list = [round(pred, 2) for pred in predictions_list]
                
                logger.info(f"Successfully generated {len(predictions_list)} forecasts")

                if forecast_horizon == 1:
                    return JsonResponse({'success': True, 'prediction': str(predictions_list[0])}, status=200)
                else:
                    plot_base64 = generate_timeseries_plot(predictions_list, forecast_horizon, entry.target_variable)
                    return JsonResponse({
                        'success': True,
                        'prediction': predictions_list,
                        'forecast_horizon': forecast_horizon,
                        'plot': plot_base64
                    }, status=200)

            except Exception as e:
                logger.exception(f"Error during time series forecasting for model ID {model_id}: {e}")
                return JsonResponse({'success': False, 'message': f'Error in forecasting: {str(e)}'}, status=500)

        # Handle other task types (Classification, Regression, Clustering)
        else:
            try:
                input_data = json.loads(data.get('data', '{}'))
                if not input_data:
                    logger.warning("No data provided for inference")
                    return JsonResponse({'success': False, 'message': 'No data provided for inference'}, status=400)

                logger.info(f"Received input data for inference: {input_data}")
                
                df = pd.DataFrame([input_data])
                
                # Preprocess data using the pipeline
                try:
                    processed_df = pl.transform(df)
                    logger.info("Successfully transformed input data with pipeline")
                except Exception as e:
                    logger.error(f"Error transforming data for model ID {model_id}: {e}")
                    return JsonResponse({'success': False, 'message': f'Error in data transformation: {str(e)}'}, status=400)

                # Make prediction
                try:
                    prediction = model.predict(processed_df)
                    final_prediction = prediction[0]
                    
                    if hasattr(final_prediction, 'item'):
                        final_prediction = final_prediction.item()
                    
                    logger.info(f"Successfully made prediction: {final_prediction}")
                    return JsonResponse({'success': True, 'prediction': final_prediction}, status=200)

                except Exception as e:
                    logger.error(f"Error during prediction for model ID {model_id}: {e}")
                    return JsonResponse({'success': False, 'message': f'Error during prediction: {str(e)}'}, status=500)

            except json.JSONDecodeError:
                logger.error("Invalid JSON format in 'data' field")
                return JsonResponse({'success': False, 'message': 'Invalid JSON format in input data'}, status=400)
            except Exception as e:
                logger.exception(f"An unexpected error occurred during inference for model ID {model_id}: {e}")
                return JsonResponse({'success': False, 'message': f'An unexpected error occurred: {str(e)}'}, status=500)

    except User.DoesNotExist:
        logger.error(f"Authenticated user not found in database: {request.jwt_payload.get('username')}")
        return JsonResponse({'success': False, 'message': 'User not found'}, status=404)
    except ModelEntry.DoesNotExist:
        logger.error(f"Model with ID {model_id} not found in database")
        return JsonResponse({'success': False, 'message': 'Model not found'}, status=404)
    except Exception as e:
        logger.exception(f"A critical unexpected error occurred in the infer view: {e}")
        return JsonResponse({'success': False, 'message': 'An unexpected server error occurred'}, status=500)


@csrf_exempt
@require_POST
def is_valid_token(token):
    try:
        jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        return True
    except jwt.ExpiredSignatureError:
        return False
    except jwt.InvalidTokenError:
        return False
    except Exception:
        return False

@csrf_exempt
@require_GET
@jwt_authenticated
def getModelDataset(request):
    """Get the actual dataset for a model"""
    try:
        model_id = request.GET.get('id')
        if not model_id:
            return JsonResponse({'success': False, 'message': 'Model ID is required'}, status=400)
        
        # Check if user has access to this model
        user = User.objects.get(username=request.jwt_payload['username'])
        if not user.models.filter(id=model_id).exists():
            return JsonResponse({'success': False, 'message': 'Model not found or access denied'}, status=404)
        
        # Check if dataset file exists
        dataset_path = f'data/{model_id}.csv'
        if not default_storage.exists(dataset_path):
            return JsonResponse({'success': False, 'message': 'Dataset not found'}, status=404)
        
        # Read and return the dataset
        with default_storage.open(dataset_path) as f:
            df = pd.read_csv(f)
        dataset_data = df.to_dict('records')  # Convert to list of dictionaries
        
        response = JsonResponse({'success': True, 'data': dataset_data}, status=200)
        response["Access-Control-Allow-Origin"] = "http://localhost:3000"
        return response
        
    except Exception as e:
        print(f"Error in getting model dataset: {e}")
        return JsonResponse({'success': False, 'message': 'There was an error'}, status=500)
    
@csrf_exempt
@require_POST
@jwt_authenticated
def batchPredict(request):
    """
    Batch prediction endpoint. Accepts a CSV file and model id, returns the same CSV with a prediction column as a downloadable file.
    """
    try:
        # Validate file and model id
        if 'file' not in request.FILES:
            return JsonResponse({'success': False, 'message': 'No file provided'}, status=400)
        file_obj = request.FILES['file']
        model_id = request.POST.get('model_id') or request.POST.get('id')
        if not model_id:
            return JsonResponse({'success': False, 'message': 'Model ID is required'}, status=400)

        # Check user access
        user = User.objects.get(username=request.jwt_payload['username'])
        if not user.models.filter(id=model_id).exists():
            return JsonResponse({'success': False, 'message': 'Model not found or access denied'}, status=404)
        entry = ModelEntry.objects.get(id=model_id)

        # Disallow batch prediction for TimeSeries models
        if entry.task == 'TimeSeries':
            return JsonResponse({'success': False, 'message': 'Batch prediction is not supported for TimeSeries models. Please use the single prediction form.'}, status=400)

        # Read CSV
        try:
            df = pd.read_csv(file_obj)
            if df.empty:
                return JsonResponse({'success': False, 'message': 'The uploaded file is empty'}, status=400)
        except Exception as e:
            return JsonResponse({'success': False, 'message': f'Error parsing your file: {str(e)}'}, status=400)

        # Load model and pipeline
        model_path = f'models/{entry.id}.pkl'
        pipeline_path = f'pipelines/{entry.id}.pkl'
        with default_storage.open(model_path) as f:
            model = joblib.load(f)
        with default_storage.open(pipeline_path) as f:
            pl = joblib.load(f)

        # Non-time series: preprocess and predict for each row
        try:
            list_of_features = ast.literal_eval(entry.list_of_features)
        except Exception:
            list_of_features = entry.list_of_features
        # Fill missing target variable if needed
        file_path = f'data/{entry.id}.csv'
        with default_storage.open(file_path) as f:
            df2 = pd.read_csv(f)
        target_variable_first_value = df2[entry.target_variable].iloc[0] if entry.target_variable in df2 else None
        # Prepare input
        input_df = df.copy()
        if entry.target_variable and entry.target_variable not in input_df.columns and target_variable_first_value is not None:
            input_df[entry.target_variable] = target_variable_first_value
        # Type conversion
        for key in input_df.columns:
            if key in list_of_features:
                if list_of_features[key] == 'integer':
                    input_df[key] = input_df[key].astype('Int64')
                elif list_of_features[key] == 'float':
                    input_df[key] = input_df[key].astype(float)
        # Preprocess
        try:
            processed = pl.transform(input_df)
            if entry.target_variable in processed.columns:
                processed = processed.drop(entry.target_variable, axis=1)
        except Exception as e:
            return JsonResponse({'success': False, 'message': f'Error in preprocessing: {str(e)}'}, status=400)
        # Predict
        try:
            if hasattr(model, 'labels_'):  # likely DBSCAN or other unsupervised clustering
                preds = model.labels_
            elif hasattr(model, 'predict'):
                preds = model.predict(processed)
            # Always return the raw model output
            df['prediction'] = preds
        except Exception as e:
            return JsonResponse({'success': False, 'message': f'Error in prediction: {str(e)}'}, status=400)

        # Return the CSV file with predictions
        csv_buffer = io.StringIO()
        df.to_csv(csv_buffer, index=False)
        csv_buffer.seek(0)
        response = HttpResponse(csv_buffer.getvalue(), content_type='text/csv')
        response['Content-Disposition'] = 'attachment; filename="batch_predictions.csv"'
        return response
    except Exception as e:
        logger.exception(f"Unexpected error in batch_predict: {str(e)}")
        return JsonResponse({'success': False, 'message': f'There was an error: {str(e)}'}, status=500)