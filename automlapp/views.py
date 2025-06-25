from django.shortcuts import render
from django.conf import settings
from django.views.decorators.http import require_POST, require_GET
from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse
from functools import wraps
from automlapp.models import ModelEntry, User
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



@csrf_exempt
@require_POST
@jwt_authenticated
def loadData(request):
    try:
        request_data = request.FILES
        try:
            df = pd.read_csv(request_data['file'])
        except:
            return JsonResponse({'success': False, 'message': 'Error parsing your file'}, status=400)
        columns = list(df.columns)
        suggested_target_variables = []
        features = {}
        datetime_columns = []


        for column in columns:
            if df[column].dtype == 'object':
                try:
                    parsed_col = pd.to_datetime(df[column], errors='coerce')
                    if parsed_col.notna().sum() / len(df[column]) > 0.8:
                        datetime_columns.append(column)
                        features[column] = 'datetime'
                except Exception as e:
                    print(f"Error parsing column {column} as datetime: {e}")
                    continue

        for column in columns:
            suggested_target_variables.append(column)
            print(column, len(df[column].unique()), len(df), df[column].dtype)
            if len(df[column].unique()) != len(df) and len(df[column].unique()) > 1 and df[column].dtype in ['int64', 'object', 'bool', 'float64']:
                pass
            if len(df[column].unique()) == len(df) and len(df[column].unique()) > 1:
                features[column] = 'unique'
            elif len(df[column].unique()) == 1:
                features[column] = 'constant'
            elif len(df[column].unique()) < 10:
                features[column] = [x for x in df[column].unique()]
            elif df[column].dtype == 'object':
                features[column] = 'categorical'
            elif df[column].dtype == 'int64':
                features[column] = 'integer'
            elif df[column].dtype == 'float64':
                features[column] = 'float'
            else:
                features[column] = 'unknown'




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

        User.objects.get(username=request.jwt_payload['username']).models.add(entry)

        df.to_csv(f'data/{entry.id}.csv', index=False)

        entry.save()
        print(datetime_columns)

        feature_names = list(features.keys())
        return JsonResponse({'success': True, 'features': feature_names, 'data': suggested_target_variables, "datetime_columns":datetime_columns, 'id': entry.id}, status=200)
    except Exception as e:
        print(f"Error in loading data: {e}")
        return JsonResponse({'success': False, 'message': 'There was an error'}, status=500)


@csrf_exempt
@require_POST
@jwt_authenticated
def trainModel(request):
    try:
        if not request.POST['name'] or not request.POST['description'] or not request.POST['task']:
            return JsonResponse({'success': False, 'message': 'Missing required fields'}, status=400)
        # Only require target_variable for non-clustering tasks
        if request.POST['task'] != 'Clustering' and not request.POST['target_variable']:
            return JsonResponse({'success': False, 'message': 'Target variable is required for non-clustering tasks'}, status=400)
        # For time series tasks, require datetime_column and date_format
        if request.POST['task'] == 'TimeSeries':
            if not request.POST.get('datetime_column'):
                return JsonResponse({'success': False, 'message': 'Datetime column is required for time series tasks'}, status=400)
            if not request.POST.get('date_format'):
                return JsonResponse({'success': False, 'message': 'Date format is required for time series tasks'}, status=400)
        if not User.objects.filter(models=request.POST['id']).exists():
            return JsonResponse({'success': False, 'message': 'Model not found'}, status=404)
        data = request.POST
        entry = ModelEntry.objects.get(id=data['id'])
        entry.name = data['name']
        entry.description = data['description']
        entry.task = data['task']
        entry.target_variable = data['target_variable'] if 'target_variable' in data else ''
        if entry.task == 'TimeSeries':
            entry.datetime_column = data['datetime_column']
            entry.date_format = data['date_format']
        entry.status = 'Model Training'
        entry.save()
        train_model_task.delay(entry.id)
        print("done")
        return JsonResponse({'success': True}, status=200)
    except Exception as e:
        print(f"Error in training model: {e}")
        return JsonResponse({'success': False, 'message': 'There was an error'}, status=500)


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
            df = pd.read_csv(f'data/{entries.id}.csv')
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
    try:
        data = request.POST
        if not User.objects.filter(models=data['id']).exists():
            return JsonResponse({'success': False, 'message': 'Model not found'}, status=404)
        entry = ModelEntry.objects.get(id=data['id'])
        model = joblib.load(f'models/{entry.id}.pkl')
        pl = joblib.load(f'pipelines/{entry.id}.pkl')
        data = json.loads(data['data'])
        print(data)

        if entry.task == 'TimeSeries':
            # For time series, we don't need input data - just forecast horizon
            forecast_horizon = int(request.POST.get('forecast_horizon', 1))
            try:
                prediction = model.forecast(steps=forecast_horizon)

                # Convert prediction to list format
                if isinstance(prediction, pd.Series):
                    predictions_list = prediction.tolist()
                else:
                    predictions_list = prediction.iloc[:, 0].tolist() if len(prediction.shape) > 1 else prediction.tolist()

                # Round predictions to 2 decimal places
                predictions_list = [round(pred, 2) for pred in predictions_list]

                # Return response - only include plot for multiple timesteps
                if forecast_horizon == 1:
                    finalPrediction = predictions_list[0]
                    return JsonResponse({
                        'success': True,
                        'prediction': str(finalPrediction)
                    }, status=200)
                else:
                    # Generate plot only for multiple timesteps
                    plot_base64 = generate_timeseries_plot(predictions_list, forecast_horizon, entry.target_variable)
                    return JsonResponse({
                        'success': True,
                        'prediction': predictions_list,
                        'forecast_horizon': forecast_horizon,
                        'plot': plot_base64
                    }, status=200)
            except Exception as e:
                print(f"Error in forecasting: {e}")
                return JsonResponse({
                    'success': False,
                    'message': f'Error in forecasting: {str(e)}'
                }, status=400)
        else:
            # For non-time series models, process the input data
            list_of_features = ast.literal_eval(entry.list_of_features)
            print(list_of_features)
            df2 = pd.read_csv(f'data/{entry.id}.csv')
            target_variable_first_value = df2[entry.target_variable].iloc[0]

            for key, value in data.items():
                if key == entry.target_variable:
                    data[key] = target_variable_first_value
                    print(data[key])
                print(key, data[key])
                if key in list_of_features:
                    if list_of_features[key] == 'integer':
                        data[key] = int(value)
                    elif list_of_features[key] == 'float':
                        data[key] = float(value)

            df = pd.DataFrame([data])
            df[entry.target_variable] = target_variable_first_value
            df = pl.transform(df)

            df.drop(entry.target_variable, axis=1, inplace=True)
            prediction = model.predict(df)
            print(f"Prediction: {prediction}")
            finalPrediction = prediction[0]
            if entry.task == 'Classification' and len(df2[entry.target_variable].unique()) == 2:
                finalPrediction = 'True' if finalPrediction == 1 else 'False'

        return JsonResponse({'success': True, 'prediction': str(finalPrediction)}, status=200)
    except Exception as e:
        if '0 sample' in str(e):
            return JsonResponse({'success': False, 'message': 'Provided Row is an outlier'}, status=400)
        print(f"Error in inference: {e}")
        return JsonResponse({'success': False, 'message': 'There was an error'}, status=500)

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