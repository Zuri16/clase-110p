# Importar la biblioteca OpenCV
import cv2
import numpy as np
import tensorflow as tf
  
model=tf.keras.models.load_model("keras_model.h5")

# Definir un objeto de captura de video
vid = cv2.VideoCapture(0)
  
while(True):
      
    # Capturar el video fotograma por fotograma
    ret, frame = vid.read()
  
    img=cv2.resize(frame,(224,224))
    test_image=np.array(img,dtype=np.float32)
    test_image=np.expand_dims(test_image,axis=0)
    imagen_normalizada=test_image/255.0

    #Ejecutar el modelo para dar la predicción
    prediccion=model.predict(imagen_normalizada)
    print("prediccion: ",prediccion)

    # Mostrar el fotograma resultante
    cv2.imshow('Fotograma', frame)
      
    # Salir de la ventana con la barra espaciadora
    key = cv2.waitKey(1)
    
    if key == 32:
        break
  
# Después del bucle, liberar al objeto de captura
vid.release()

# Destruir todas las ventanas
#cv2.destroyAllWindows()
