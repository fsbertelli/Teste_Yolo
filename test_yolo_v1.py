import cv2
import time
import datetime
import numpy as np
import pyrealsense2 as rs

# CORES DAS CLASSES
COLORS = [(0, 255, 255), (255, 255, 0), (0, 255, 0), (255, 0, 0)]

# CARREGA AS CLASSES
class_names = []
with open('coco.names', 'r') as f:
  class_names = [cname.strip() for cname in f.readlines()]

# CARREGA OS PESOS DA REDE NEURAL
net = cv2.dnn.readNet('yolov4-tiny.weights', 'yolov4-tiny.cfg')

# SETANDO OS PARAMETROS DA REDE NEURAL
model = cv2.dnn_DetectionModel(net)
model.setInputParams(size=(320,320), scale=1/255)

# INICIALIZA A CÂMERA
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 15)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 15)
pipeline.start(config)

# LOOP PARA LER OS FRAMES
while True:
    # CAPTURA DO FRAME
    frames = pipeline.wait_for_frames()
    depth_frame = frames.get_depth_frame()
    color_frame = frames.get_color_frame()

    if not depth_frame or not color_frame:
        print("Error")

    # CONVERTENDO O FRAME EM ARRAY DO OPENCV
    color_image = np.asanyarray(color_frame.get_data())

    #CONTA MS (FPS)
    start = time.time()

    # DETECÇÃO
    classes, scores, boxes = model.detect(color_image, 0.1, 0.2)

    #FIM MS
    end = time.time()

    #DATAHORA
    now = datetime.datetime.now()
    t = now.strftime("%H:%M:%S")

    # PERCORRE TODAS AS DETECÇÕES
    for (classid, score, box) in zip(classes, scores, boxes):
        # GERANDO UMA COR PARA A CLASSE
        color = COLORS[int(classid) % len(COLORS)]

        # PEGANDO O NOME DA CLASSE PELO ID E O SEU SCORE DE ACURACIA
        label = f"{class_names[classid]} : {score}"

        # DESENHANDO A BOX DE DETECÇÃO
        cv2.rectangle(color_image, box, color, 2)

        # ESCREVENDO O NOME DA CLASSE EM CIMA DO BOX DO OBJETO
        cv2.putText(color_image, label, (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # MEDINDO A DISTÂNCIA
        x, y, w, h = box
        depth = depth_frame.get_distance(int(x + w / 2), int(y + h / 2))
        distance = depth * 100  # CONVERTENDO PARA CM
        distance_label = f"{distance:.2f} cm"

        #CALCULANDO O FPS
        fps_label = f"FPS: {round((1.0/(end - start)),2)}"

        #ESCREVENDO O FPS E A DISTÂNCIA NA IMAGEM
        #cv2.putText(color_image, fps_label, (0, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 5)
        cv2.putText(color_image, fps_label, (0, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255, 0), 3)
        cv2.putText(color_image, distance_label, (box[0], box[1] - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        #ENTRA NO LOOP DE CONDIÇÕES
        if distance != 0:
            if class_names[classid] == "chair" and score > 0.2 and distance < 500:
                print(t + " "+ class_names[classid] + " detectado a " + f"{distance/100:.2f} m de distancia" + " com " + f"{100*score:.2f} % de assertividade")
            if class_names[classid] == "person" and score < 0.6:
                print(t + " Não tenho ctz se é " +  class_names[classid] + " ou et bilu")
                

    # MOSTRA A IMAGEM
    cv2.imshow('Frame', color_image)

    # QUIT
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# FINALIZA A CÂMERA E O OPENCV
pipeline.stop()
cv2.destroyAllWindows()
