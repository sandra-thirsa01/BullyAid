from django.db import models

# Create your models here.

class message(models.Model):
    text =  models.CharField(max_length=200, null=True)
    name =  models.CharField(max_length=200, null=True)
    email = models.CharField(max_length=200, null=True)
   

