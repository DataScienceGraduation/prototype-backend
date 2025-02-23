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

JWT_ALGORITHM = 'HS256'
JWT_SECRET = settings.SECRET_KEY

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
        password = data['password']
        email = data['email']
        if not username or not password or not email:
            return JsonResponse({'success': False, 'message': 'Username or password not provided'}, status=400)
        if User.objects.filter(username=username).exists():
            return JsonResponse({'success': False, 'message': 'Username already exists'}, status=400)
        
        user = User.objects.create_user(username=username, password=password)
        user.set_password(password)
        user.save()
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
        
        for column in columns:
            suggested_target_variables.append(column)
            print(column, len(df[column].unique()), len(df), df[column].dtype)
            if len(df[column].unique()) != len(df) and len(df[column].unique()) > 1 and df[column].dtype in ['int64', 'object', 'bool', 'float64']:
                #suggested_target_variables.append(column)
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

        feature_names = list(features.keys())
        return JsonResponse({'success': True, 'features': feature_names, 'data': suggested_target_variables, 'id': entry.id}, status=200)         
    except Exception as e:
        print(f"Error in loading data: {e}")
        return JsonResponse({'success': False, 'message': 'There was an error'}, status=500)
    

@csrf_exempt
@require_POST
@jwt_authenticated
def trainModel(request):
    try:
        if not request.POST['name'] or not request.POST['description'] or not request.POST['task'] or not request.POST['target_variable']:
            return JsonResponse({'success': False, 'message': 'Missing required fields'}, status=400)
        if not User.objects.filter(models=request.POST['id']).exists():
            return JsonResponse({'success': False, 'message': 'Model not found'}, status=404)
        data = request.POST
        entry = ModelEntry.objects.get(id=data['id'])
        entry.name = data['name']
        entry.description = data['description']
        entry.task = data['task']
        entry.target_variable = data['target_variable']
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
            'evaluation_metric_value': entries.evaluation_metric_value
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
        list_of_features = ast.literal_eval(entry.list_of_features)
        print(list_of_features)
        df2 = pd.read_csv(f'data/{entry.id}.csv')
        target_variable_first_value = df2[entry.target_variable].iloc[0]
        for key, value in data.items():
            if key == entry.target_variable:
                data[key] = target_variable_first_value
                print(data[key])
            print(key, data[key])
            if list_of_features[key] == 'integer':
                data[key] = int(data[key])
            elif list_of_features[key] == 'float':
                data[key] = float(data[key])
        df = pd.DataFrame([data])
        df[entry.target_variable] = target_variable_first_value
        df = pl.transform(df)
        df.drop(entry.target_variable, axis=1, inplace=True)
        print("Transformed Data")
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
    except Exception as e:
        return False