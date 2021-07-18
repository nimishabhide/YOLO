import cv2
import numpy as np
import matplotlib.pyplot as plt
yolo=cv2.dnn.readNet("./yolov3-tiny.weights","./yolo3-tiny.cfg")
classes=[]
with open("./coco.names","r") as f:
  classes=f.read().splitlines()
img=cv2.imread("./bhoo.jpg")
blob=cv2.dnn.blobFromImage(img,1/255,(320,320),(0,0,0),swapRB=True,crop=False)
i=blob[0].reshape(320,320,3)
plt.imshow(i)
yolo.setInput(blob)
output_layer_name=yolo.getUnconnectedOutLayersNames()
layeroutput=yolo.forward(output_layer_name)
boxes=[]
confidences=[]
class_ids=[]
for output in layeroutput:
  for detection in output:
    score=detection[5:]
    class_id=np.argmax(score)
    confidence=score[class_ids]
    if confidence>0.7:
      center_x=int(detection[0]*width)
      center_y=int(detection[0]*height)
      w=int(detection[0]*width)
      h=int(detection[0]*height)

      x=int(center_x-w/2)
      y=int(center_y-h/2)

      boxes.append([x,y,w,h])
      confidences.append(float(confidence))
      class_ids.append(class_id)
indexes=cv2.dnn.NMSBoxes(boxes,confidences,0.5,0.4)
font=cv2.FONT_HERSHEY_PLAIN 
colors=np.random.uniform(0,255,size=(len(boxes),3))
for i in indexes.flatten():
  x,y,w.h=boxes[i]
  label=str(classes[class_ids[i]])
  confi=str(round(confidences[i],2))
  color=colors[i]

  cv2.rectangle(img,(x,y),(x+w,y+h),color,1)
  cv2.putText(img,label+" "+config,(x,y+20),font,2,(255,255,255),1)
plt.imshow(img)
