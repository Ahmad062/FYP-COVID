from django.shortcuts import render,HttpResponse, redirect
from django.contrib.auth.models import User
from django.contrib.auth import logout
from django.contrib.auth import authenticate
# Create your views here.
# packages and Libraries
from scipy.spatial import distance as dist
import numpy as np
import argparse
import cv2
import os
import winsound
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import time
#password or user is Ali@1234

from .forms import Video_form
from .models import Video

# def vidtest(request):
#     all_video = Video.objects.latest()
#     if request.method == "POST":
#         form=Video_form(data=request.POST,files=request.FILES)
#         if form.is_valid():
#             form.save()
#             return HttpResponse("<h1> Uploaded successfully </h1>")
#     else:
#         form=Video_form()
#     return render(request,'index.html',{"form":form,"all":all_video})


def index(request):
    if request.user.is_anonymous:
        return redirect("/login")
    else:    
        return render(request,'index.html')
 
def login(request):
    if request.method=="POST":
        username = request.POST.get('username')
        password = request.POST.get('password')
        user = authenticate(username=username, password=password)

        if user is not None:
            if request.method == "POST":
                form=Video_form(data=request.POST,files=request.FILES)
                if form.is_valid():
                    form.save()
                    return HttpResponse("<h1> Uploaded successfully </h1>")
            else:
                form=Video_form()
            return render(request,'index.html', {"form": form})
            # return render(request,'index.html')

        else:
            return render(request,'login.html')
    else:
        return render(request,'login.html')

    
def about(request):
    return render(request,'about.html' )

def contact(request):
    return render(request,'contact.html')

def sops(request):
    return render(request,'sops.html')

def faqs(request):
    return render(request,'faqs.html')


def logoutuser(request): 
    logout(request)
    return render(request,'login.html')


def selection(request):

    MIN_CONF = 0.3
    NMS_THRESH = 0.3
    MIN_DISTANCE = 250

        ################################  Person Detection Function ##############################

    def detect_people(frame, net, ln, personIdx=0):
        (H, W) = frame.shape[:2]
        results = []

        blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416),
            swapRB=True)
        net.setInput(blob)
        layerOutputs = net.forward(ln)

        boxes = []
        centroids = []
        confidences = []

        for output in layerOutputs:
            for detection in output:
                scores = detection[5:]
                classID = np.argmax(scores)
                confidence = scores[classID]

                if classID == personIdx and confidence > MIN_CONF:

                    box = detection[0:4] * np.array([W, H, W, H])
                    (centerX, centerY, width, height) = box.astype("int")

                    x = int(centerX - (width / 2))
                    y = int(centerY - (height / 2))

                    boxes.append([x, y, int(width), int(height)])
                    centroids.append((centerX, centerY))
                    confidences.append(float(confidence))

        idxs = cv2.dnn.NMSBoxes(boxes, confidences, MIN_CONF, NMS_THRESH)

        if len(idxs) > 0:
            for i in idxs.flatten():
                (x, y) = (boxes[i][0], boxes[i][1])
                (w, h) = (boxes[i][2], boxes[i][3])

                r = (confidences[i], (x, y, x + w, y + h), centroids[i])
                results.append(r)

        return results


        ########################   Mask Detection Function ##############################

    def detect_and_predict_mask(frame, faceNet, maskNet):
        (h, w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 1.0, (224, 224),
            (104.0, 177.0, 123.0))

        faceNet.setInput(blob)
        detections = faceNet.forward()

        faces = []
        locs = []
        preds = []

        for i in range(0, detections.shape[2]):

            confidence = detections[0, 0, i, 2]

            if confidence > 0.5:

                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")

                (startX, startY) = (max(0, startX), max(0, startY))
                (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

                face = frame[startY:endY, startX:endX]
                face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                face = cv2.resize(face, (224, 224))
                face = img_to_array(face)
                face = preprocess_input(face)

                faces.append(face)
                locs.append((startX, startY, endX, endY))


        if len(faces) > 0:
            faces = np.array(faces, dtype="float32")
            preds = maskNet.predict(faces, batch_size=32)
        return (locs, preds)


    #################################### Main Code #########################################
    p_count = 0
    frequency = 2000
    duration = 500
    status = "No Mask"
    count = 0

    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input", type=str, default="")

    ap.add_argument("-d", "--display", type=int, default=1)


    if request.method == "POST":
        form=Video_form(data=request.POST,files=request.FILES)
        if form.is_valid():
            form.save()
            # return HttpResponse("<h1> Uploaded successfully </h1>")
    else:
        form=Video_form()

    args = vars(ap.parse_args(["--input","C:/Users/Imahm/Desktop/django/fyp","--display","1"]))
    # args = vars(ap.parse_args(["--input",Video,"--display","1"]))

    labelsPath = os.path.sep.join(["yolo-coco/coco.names"])
    LABELS = open(labelsPath).read().strip().split("\n")

    weightsPath = os.path.sep.join(["yolo-coco/yolov3.weights"])
    configPath = os.path.sep.join(["yolo-coco/yolov3.cfg"])

    net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)

    ln = net.getLayerNames()
    ln = [ln[i - 1] for i in net.getUnconnectedOutLayers()]

    prototxtPath = os.path.sep.join(["face_detector/deploy.prototxt"])
    weightPath = os.path.sep.join(["face_detector/res10_300x300_ssd_iter_140000.caffemodel"])
    faceNet = cv2.dnn.readNet(prototxtPath, weightPath)

    maskNet = load_model("mask_detector.model")

  
    all_video = Video.objects.latest()
    #######  Streaming Type Selection ##########
    vs = cv2.VideoCapture(0) if request.POST.get('status') == "True" else cv2.VideoCapture(args["input"] + all_video.video.url)
    # vs = cv2.VideoCapture(0) if request.POST.get('status') == "True" else cv2.VideoCapture(all_video.video.url)


    while True:
        (grabbed, frame) = vs.read()

        if not grabbed:
            break

        frame = cv2.resize(frame,(1280,720))

        results = detect_people(frame, net, ln, personIdx=LABELS.index("person"))
        
        p_count = len(results)
        
        p_txt = "Total Persons: {}".format(p_count)
        cv2.putText(frame, p_txt, (5, 650), cv2.FONT_HERSHEY_SIMPLEX, 1, (57, 198, 41), 3)
        
        if p_count > 15:
            winsound.Beep(frequency, duration)

        violate = set()

        if len(results) >= 2:
            centroids = np.array([r[2] for r in results])
            D = dist.cdist(centroids, centroids, metric="euclidean")

            for i in range(0, D.shape[0]):
                for j in range(i + 1, D.shape[1]):
                    if D[i, j] < MIN_DISTANCE:
                        violate.add(i)
                        violate.add(j)

        for (i, (prob, bbox, centroid)) in enumerate(results):
            (startX, startY, endX, endY) = bbox
            (cX, cY) = centroid
            color = (0, 255, 0)

            if i in violate:
                color = (0, 0, 255)

            cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
            cv2.circle(frame, (cX, cY), 5, color, 1)

        text = "Social Distancing Violations: {}".format(len(violate))
        cv2.putText(frame, text, (10, frame.shape[0] - 25),
            cv2.FONT_HERSHEY_SIMPLEX, 0.85, (57, 198, 41), 3)

        (locs, preds) = detect_and_predict_mask(frame, faceNet, maskNet)

        for (box, pred) in zip(locs, preds):
            (startX, startY, endX, endY) = box
            (mask, withoutMask) = pred

            label = "Mask" if mask > withoutMask else "No Mask"
            color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
            
            label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)

            cv2.putText(frame, label, (startX, startY - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
            cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
            
            mask_count = "Mask" if mask > withoutMask else "No Mask"
            
            mask_txt = "Mask Violations: {}".format(mask_count.count(status))
            cv2.putText(frame, mask_txt, (5, 600), cv2.FONT_HERSHEY_SIMPLEX, 1, (57, 198, 41), 3)

        if args["display"] > 0:
            cv2.imshow("Results", frame)
            key = cv2.waitKey(1)

            if key == ord("q"):
                break                         
    cv2.destroyAllWindows()

    # return render(request,'index.html')
    return render(request,'index.html',{"form":form})

def temperature(request):
    print("Measuring Temperature")
    return render(request,'index.html' )