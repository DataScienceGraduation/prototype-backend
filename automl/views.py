from django.shortcuts import render
from django.views.decorators.http import require_POST, require_GET
from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse
from automl.models import ModelEntry
import pandas as pd
from automl.tasks import train_model_task
import joblib
import json

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
        
        for column in columns:
            print(column, len(df[column].unique()), len(df), df[column].dtype)
            if len(df[column].unique()) != len(df) and len(df[column].unique()) > 1 and df[column].dtype in ['int64', 'object', 'bool']:
                suggested_target_variables.append(column)

        entry = ModelEntry.objects.create(
            name="", 
            description="", 
            task="", 
            target_variable="",
            list_of_features=columns, 
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
@require_POST
def infer(request):
    try:
        data = request.POST
        entry = ModelEntry.objects.get(id=data['id'])
        model = joblib.load(f'models/{entry.id}.pkl')
        pl = joblib.load(f'pipelines/{entry.id}.pkl')
        data = json.loads(data['data'])
        # add outcome to data
        df = pd.DataFrame([data])
        df = df.reset_index(drop=True)
        print(df)
        df = pl.transform(df)

        print("Transformed Data")
        prediction = model.predict(df)
        print(f"Prediction: {prediction}")
        finalPrediction = prediction[0]
        return JsonResponse({'success': True, 'prediction': str(finalPrediction)}, status=200)
    except Exception as e:
        print(f"Error in inference: {e}")
        return JsonResponse({'success': False, 'message': 'There was an error'}, status=500)