from django.shortcuts import render
from django.views.decorators.http import require_POST
from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse
from automl.models import ModelEntry
import pandas as pd

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
        return JsonResponse({'success': True}, status=200)
    except Exception as e:
        print(f"Error in training model: {e}")
        return JsonResponse({'success': False, 'message': 'There was an error'}, status=500)