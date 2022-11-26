from django.shortcuts import render
from django.core.files.storage import FileSystemStorage

import tensorflow as tf


from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model

import uuid



def home(request):
    return render(request, "home.html")

def predict(request):
    fileObj = request.FILES["filePath"]
    fs = FileSystemStorage()
    fileName = fs.save(fileObj.name,fileObj)
    filePathName = fs.url(fileName)
    
    testimage = '.' + filePathName
    img = image.load_img(testimage, target_size = (224,224))

    fs.delete(fileName)

    img_path = "/media/" + str(uuid.uuid4().hex) + '.jpg'
    img.save('.' + img_path)
    x = image.img_to_array(img)
    x /= 255
    x = x.reshape(1,224, 224, 3)
    model=load_model('./models/trained_fakedetector.h5')
    pred_prob = model.predict(x)
    pred_class = (pred_prob>0.5).astype("int32")[0][0]
    if pred_class == 0:
        pred_prob = 1 - pred_prob[0][0]
    else:
        pred_prob = pred_prob[0][0]

    return render(request, 'home.html', {'filePathName':img_path, 'predictedLabel': str(pred_class), 'predictedProb':pred_prob})

