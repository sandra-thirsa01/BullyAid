# Generated by Django 4.2.1 on 2023-06-18 11:56

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('basic', '0001_initial'),
    ]

    operations = [
        migrations.AddField(
            model_name='message',
            name='email',
            field=models.CharField(max_length=200, null=True),
        ),
        migrations.AddField(
            model_name='message',
            name='name',
            field=models.CharField(max_length=200, null=True),
        ),
    ]
