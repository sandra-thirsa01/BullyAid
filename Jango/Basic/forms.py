from django import forms

from manger.models import message

from django.forms.widgets import NumberInput, RadioSelect, Select





class CyberCreateForm(forms.ModelForm):
  

    class Meta:
        model = message
        fields =[
              "text","name,"email" 
        ]
          

    def _init_(self, *args, **kwargs):
        super()._init_(*args,**kwargs)