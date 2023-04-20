from django import forms

from .models import SignatureData, Signatures

class RegisterUserForm(forms.ModelForm):
    class Meta:
        model = SignatureData
        fields = '__all__'

    Firstname = forms.CharField(label='', widget=forms.TextInput(attrs={
        'class': 'form-control',
    }))
    Middlename = forms.CharField(label='', widget=forms.TextInput(attrs={
        'class': 'form-control',
    }))
    Lastname = forms.CharField(label='', widget=forms.TextInput(attrs={
        'class': 'form-control',
    }))
    Email = forms.CharField(label='', widget=forms.EmailInput(attrs={
        'class': 'form-control',
    }))
    ContactNumber = forms.CharField(label='', widget=forms.TextInput(attrs={
        'class': 'form-control',
    }))
    Address = forms.CharField(label='', widget=forms.TextInput(attrs={
        'class': 'form-control',
    }))