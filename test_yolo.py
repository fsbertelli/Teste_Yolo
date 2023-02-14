import cv2
import time
import datetime
import numpy as np
from PIL import Image, ImageGrab

#CORES DAS CLASSES
COLORS = [(0, 255, 255), (255, 255, 0), (0, 255, 0), (255, 0, 0)]

#CARREGA AS CLASSES
class_names = []
with open('coco.names', 'r') as f:
  class_names = [cname.strip() for cname in f.readlines()]

#CAPTURA DO VIDEO
#cap = cv2.VideoCapture('drone.mp4')

#CARREGA OS PESOS DA REDE NEURAL
net = cv2.dnn.readNet('yolov4-tiny.weights', 'yolov4-tiny.cfg')

#SETANDO OS PARAMETROS DA REDE NEURAL
model = cv2.dnn_DetectionModel(net)
model.setInputParams(size=(320,320), scale=1/255)

#LOOP PARA LER OS FRAMES

while True:
    screen = np.array(ImageGrab.grab(bbox=(0,0,800,600)))
    #CAP DO FRAME
    #_, frame = cap.read()

    #CONTA MS (FPS)
    start = time.time()

    #DETECÇÃO
    classes, scores, boxes = model.detect(screen, 0.1, 0.2)

    #FIM MS
    end = time.time()

    #DATAHORA
    now = datetime.datetime.now()
    t = now.strftime("%H:%M:%S")
    
    #PERCORRE TODAS AS DETECÇÕES
    for (classid, score, box) in zip(classes, scores, boxes):

        #GERANDO UMA COR PARA A CLASSE
        color = COLORS[int(classid) % len(COLORS)]

        #PEGANDO O NOME DA CLASSE PELO ID E O SEU SCORE DE ACURACIA
        label = f"{class_names[classid]} : {score}"

        #DESENHANDO A BOX DE DETECÇÃO
        cv2.rectangle(screen, box, color, 2)

        #ESCREVENDO O NOME DA CLASSE EM CIMA DO BOX DO OBJETO
        cv2.putText(screen, label, (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        if class_names[classid] == "person" and score > 0.6:
          print(t + " "+ class_names[classid] + " detectado! " + str(score))
        if class_names[classid] == "person" and score < 0.6:
          print(t + " Não tenho ctz se é " +  class_names[classid] + " ou et bilu")
    
    #CALCULANDO O FPS
    fps_label = f"FPS: {round((1.0/(end - start)),2)}"

    #ESCREVENDO O FPS NA IMAGEM
    cv2.putText(screen, fps_label, (0, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 5)
    cv2.putText(screen, fps_label, (0, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255, 0), 3)

    #MOSTRA A IMAGEM
    cv2.imshow('robot', screen)

    #QUIT
    if cv2.waitKey(1) & 0xFF == ord('q'):
      break
    
#cap.release()
cv2.destroyAllWindows()



