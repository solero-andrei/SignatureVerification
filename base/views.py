# ML Imports
import tensorflow as tf
from tensorflow import keras
from keras.applications import ResNet152
from tensorflow.python.keras.backend import set_session
from keras.models import load_model, Sequential
from keras.layers import Dense, Flatten
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator

import numpy as np

# Required imports
from django.shortcuts import render, redirect, HttpResponse
import os

from django.conf import settings
from django.core.files.storage import FileSystemStorage, DefaultStorage

# My Forms
from .forms import RegisterUserForm

# My Models
from .models import SignatureData, Signatures


img_height, img_width = 180, 180
datasets_dir = os.path.join(settings.BASE_DIR, 'images', 'signatures', 'datasets')

def get_class_names():
    data_dir = os.path.join(settings.BASE_DIR, 'images', 'signatures', 'datasets')

    if not os.listdir(data_dir):
        print('no current dataset exist')
        class_names = None
    else:
        class_names = tf.keras.preprocessing.image_dataset_from_directory(
            data_dir,
            label_mode='categorical',
            batch_size=1
        ).class_names
    
    return class_names

classes = get_class_names()

def TrainModel():
    get_class_names()

    batch_size = 8
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        datasets_dir,
        validation_split=0.2,
        subset='training',
        seed=123,
        label_mode='categorical',
        image_size=(img_height, img_width),
        batch_size=batch_size,
    )

    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        datasets_dir,
        validation_split=0.2,
        subset='validation',
        seed=123,
        label_mode='categorical',
        image_size=(img_height, img_width),
        batch_size=batch_size,
    )

    class_names = train_ds.class_names
    print(class_names)

    model = Sequential()

    pretrained_model = ResNet152(
        include_top=False,
        weights='imagenet',
        input_shape=(img_height, img_width, 3),
        classes=len(class_names),
        pooling='avg'
    )

    for layer in pretrained_model.layers:
        layer.trainable = False

    model.add(pretrained_model)

    model.add(Flatten())

    model.add(Dense(512, activation='relu'))
    model.add(Dense(len(class_names), activation='softmax'))

    model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

    model.fit(train_ds, validation_data=val_ds, epochs=5)    
    model.save(os.path.join('models', 'signature-verifier.h5'))

def GetGroundTruthLabels():
    batch_size = 8

    test_datagen = ImageDataGenerator(rescale=1./255)

    test_generator = test_datagen.flow_from_directory(
        datasets_dir,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False
    )

    ground_truth = test_generator.classes
    class_names = list(test_generator.class_indices.keys())

    context = {
        'labels': ground_truth,
    }

    return context

def ScanImage(signature_file):
    classes = get_class_names()

    image = keras.preprocessing.image.load_img(signature_file, target_size=(180, 180))
    image = keras.preprocessing.image.img_to_array(image)
    image = keras.applications.resnet.preprocess_input(image)

    image = tf.expand_dims(image, axis=0)

    model = load_model(settings.MODEL)
    prediction = model.predict(image)

    predicted_class = classes[np.argmax(prediction)]
    print(prediction)

    context = {
        'predicted_class': predicted_class,
    }

    return context
    
def RetrainModel(request):
    if request.method == 'POST':
        TrainModel()
        return redirect('scan_signature')
    return render(request, 'base/retrain_model.html')

def RegisterUser(request):
    get_class_names()

    if request.method == 'POST':
        form = RegisterUserForm(request.POST)

        new_data = SignatureData.objects.create(
            Firstname=request.POST['Firstname'],
            Middlename=request.POST['Middlename'],
            Lastname=request.POST['Lastname'],
            Email=request.POST['Email'],
            ContactNumber=request.POST['ContactNumber'],
            Address=request.POST['Address'],
        )

        if form.is_valid():
            new_data.save()
            new_data_id = new_data.id

            return redirect('register_signature', id=new_data_id)

    else:
        form = RegisterUserForm()

    context = {'form': form}
    return render(request, 'base/register_user.html', context)


def RegisterSignature(request, id):
    get_class_names()

    current_user = SignatureData.objects.get(pk=id)
    if request.method == 'POST':
        signatures = request.FILES.getlist('signatures')

        dir_name = f'{current_user.Lastname}-{current_user.Firstname}-{current_user.Middlename}-{current_user.id}'
        signature_dir = os.path.join(settings.MEDIA_ROOT, 'datasets', dir_name)
        if os.path.isdir(signature_dir) == False:
            os.makedirs(signature_dir)

        for signature in signatures:
            fs = FileSystemStorage(location=signature_dir)
            fs.save(signature.name, signature)

            Signatures.objects.create(
                user=current_user,
                Signature = signature   
            ) 

        TrainModel()
        
        return render(request, 'base/index.html')

    context = {
        'user': current_user,
    }

    return render(request, 'base/register_signature.html', context)

def ScanSignature(request):
    class_names = get_class_names()
    num_class = 0

    if class_names != None:
        num_class = len(class_names)

    context = {
        'classes': num_class
    }

    if request.method == 'POST':
        signature_file = request.FILES['signature']
        scanned_signature_dir = os.path.join(settings.MEDIA_ROOT, 'scanned-signature')

        fs = FileSystemStorage(location=scanned_signature_dir)
        fs.save(signature_file.name, signature_file)

        saved_scanned_signature = os.path.join(scanned_signature_dir, signature_file.name)
        
        prediction = ScanImage(saved_scanned_signature)
        predicted_class = prediction['predicted_class']   
        p_class = {
            'predicted_class': predicted_class
        }
        return render(request, 'base/scan_signature.html', p_class)


    return render(request, 'base/scan_signature.html', context)


def index(request):
    return render(request, 'base/index.html')