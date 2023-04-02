from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import load_img,img_to_array
from skimage.transform import resize
import numpy as np

from PIL import Image
import time
from keras.models import load_model
from keras.models import model_from_json
from keras.applications import imagenet_utils
from prediction import predict_type



#define main classes that we have
classes = [ "Rotten Banana", "Good Banana", "Rotten Orange", "Good Orange" ] 
# load json and create model
json_file = open('model3.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model1 = model_from_json(loaded_model_json)
# load weights into new model
loaded_model1.load_weights("model.h5")

#Main Function of prediction
def predict_type(location):
    img=load_img(location,target_size=(224,224,3))
    #plt.imshow(img)
    img=img_to_array(img)
    
    img=img/255
    #plt.imshow(img)
    img=np.expand_dims(img,[0])
    
    answer=loaded_model1.predict(img)
    #To print prediction list
    #print(answer)

    y_class = answer.argmax(axis=-1)
    y = " ".join(str(x) for x in y_class)
    y = int(y)
    res = classes[y]
    return res
