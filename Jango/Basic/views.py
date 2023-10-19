
from django.shortcuts import render
from django.http import HttpResponse
from fp.predict import predict_cyberbullying


def index(request):
    context = {
        "is_predicted": False,
        "prediction_safe": False,
    }
    if request.method == "POST" and request.POST.get('message'):
        message = request.POST.get('message')
        prediction = predict_cyberbullying(message)
        context["prediction_safe"] = prediction == 0 
        context["is_predicted"] = True

    
    return render(request, 'index.html', context)



# prediction = predict_cyberbullying(input_sentence)

