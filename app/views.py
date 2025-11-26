from django.shortcuts import render
from pathlib import Path
import joblib

MODEL = None

def _model_path():
    base = Path(__file__).resolve().parent.parent
    return base / 'AI' / 'random_forest_model.pkl'

def _load_model():
    global MODEL
    if MODEL is None:
        MODEL = joblib.load(_model_path())
    return MODEL

def _map_work_type(valor):
    pt_to_en = {
        'Público': 'Govt_job',
        'Privado': 'Private',
        'Empreendedor': 'Self-employed',
        'Nunca trabalhou': 'Never_worked',
    }
    en = pt_to_en.get(valor, 'Private')
    mapping = {
        'Govt_job': 0,
        'Never_worked': 1,
        'Private': 2,
        'Self-employed': 3,
        'children': 4,
    }
    return mapping.get(en, 2)

def _map_smoking_status(valor):
    pt_to_en = {
        'Fuma atualmente': 'smokes',
        'Fuma': 'smokes',
        'Fumava no passado': 'formerly smoked',
        'Nunca fumou': 'never smoked',
        'Não sei': 'Unknown',
    }
    en = pt_to_en.get(valor, 'Unknown')
    mapping = {
        'Unknown': 0,
        'formerly smoked': 1,
        'never smoked': 2,
        'smokes': 3,
    }
    return mapping.get(en, 0)

def _map_residence(valor):
    return 0 if valor == 'Urbana' else 1

def _map_gender(valor):
    return 0 if valor == 'Masculino' else 1

def index(request):
    if request.method == 'POST':
        genero = request.POST.get('genero')
        idade = request.POST.get('idade')
        hipertensao = request.POST.get('hipertensao')
        doenca = request.POST.get('doenca_cardiaca')
        trabalho = request.POST.get('tipo_trabalho')
        residencia = request.POST.get('tipo_residencia')
        glicose = request.POST.get('nivel_glicose')
        imc = request.POST.get('imc')
        fumo = request.POST.get('status_fumo')

        try:
            features = [
                _map_gender(genero),
                float(idade),
                1 if hipertensao == 'Sim' else 0,
                1 if doenca == 'Sim' else 0,
                _map_work_type(trabalho),
                _map_residence(residencia),
                float(glicose),
                float(imc),
                _map_smoking_status(fumo),
            ]
            model = _load_model()
            pred = float(model.predict([features])[0])
            pred = min(max(pred, 0.0), 1.0)
            risco = pred * 100.0
            return render(request, 'resultado.html', {'risco_percent': risco})
        except Exception:
            return render(request, 'resultado.html', {'risco_percent': None, 'erro_msg': 'Não foi possível calcular o risco.'})
    return render(request, 'form.html')
