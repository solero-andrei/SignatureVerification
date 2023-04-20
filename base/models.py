from django.db import models
import uuid

# Create your models here.
class SignatureData(models.Model):
    Firstname = models.CharField(max_length=100)
    Middlename = models.CharField(max_length=100)
    Lastname = models.CharField(max_length=100)
    Email = models.CharField(max_length=100)
    ContactNumber = models.CharField(max_length=100)
    Address = models.CharField(max_length=100)

class Signatures(models.Model):
    user = models.ForeignKey(SignatureData, on_delete=models.SET_NULL, null=True, blank=True)
    Signature = models.ImageField(null=True, blank=True)