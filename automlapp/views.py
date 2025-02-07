from django.shortcuts import render
from django.views.decorators.http import require_POST, require_GET
from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse
from automlapp.models import ModelEntry
import pandas as pd
from automlapp.tasks import train_model_task
import joblib
import json
import ast

# Create your views here.
@csrf_exempt
@require_POST
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

        df.to_csv(f'data/{entry.id}.csv', index=False)

        return JsonResponse({'success': True, 'data': suggested_target_variables, 'id': entry.id}, status=200)         
    except Exception as e:
        print(f"Error in loading data: {e}")
        return JsonResponse({'success': False, 'message': 'There was an error'}, status=500)
    

@csrf_exempt
@require_POST
def trainModel(request):
    try:
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
def getAllModels(request):
    try:
        entries = ModelEntry.objects.all()
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
def infer(request):
    try:
        data = request.POST
        entry = ModelEntry.objects.get(id=data['id'])
        model = joblib.load(f'models/{entry.id}.pkl')
        pl = joblib.load(f'pipelines/{entry.id}.pkl')
        data = json.loads(data['data'])
        print(data)
        list_of_features = ast.literal_eval(entry.list_of_features)
        print(list_of_features)
        print("TEST")
        for key, value in data.items():
            print(key, value)
            if list_of_features[key] == 'integer':
                data[key] = int(value)
            elif list_of_features[key] == 'float':
                data[key] = float(value)
        # add outcome to data
        df = pd.DataFrame([data])
        df2 = pd.read_csv(f'data/{entry.id}.csv')
        # get value of target variable
        target_variable_first_value = df2[entry.target_variable].iloc[0]
        df[entry.target_variable] = target_variable_first_value
        print(df)
        df = pl.transform(df)
        df.drop(entry.target_variable, axis=1, inplace=True)
        print("Transformed Data")
        prediction = model.predict(df)
        print(f"Prediction: {prediction}")
        finalPrediction = prediction[0]
        return JsonResponse({'success': True, 'prediction': str(finalPrediction)}, status=200)
    except Exception as e:
        if '0 sample' in str(e):
            return JsonResponse({'success': False, 'message': 'Provided Row is an outlier'}, status=400)
        print(f"Error in inference: {e}")
        return JsonResponse({'success': False, 'message': 'There was an error'}, status=500)#