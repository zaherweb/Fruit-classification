

#importing the libraries
import streamlit as st
import joblib
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import load_img,img_to_array

from PIL import Image
from skimage.transform import resize
import numpy as np
import time
from keras.models import load_model
from keras.models import model_from_json
from keras.applications import imagenet_utils
import numpy as np
import os
#from prediction import predict_type

# Download file model file
#% wget --no-check-certificate  -O "model.h5" "https://drive.google.com/uc?export=download&id=1Eph_kEy5vJDg79CORGhh4qUPHIkgSjzN"
import requests

url = "https://drive.google.com/uc?export=download&id=1Eph_kEy5vJDg79CORGhh4qUPHIkgSjzN"
filename = "model.h5"


# Check if the file already exists
if os.path.exists(filename):
    pass
else:
    # Download the file
    response = requests.get(url)
    response.raise_for_status() # raise an exception if response status is not OK (i.e. 200)
    with open(filename, "wb") as f:
        f.write(response.content)


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



# Designing the interface
#st.title(' :blue[   Agrocommercial ] ')
#st.header("Fruit Image Classification App")

st.markdown("<h1 style='text-align: center; color: blue;'>Agrocommercial</h1>", unsafe_allow_html=True)

st.markdown("<h3 style='text-align: center; color: black;'>Fruit Image Classification App</h2>", unsafe_allow_html=True)

# For newline
st.write('\n')

image = Image.open('1675237883685.jpg')
show = st.image(image, use_column_width=True)

st.sidebar.title("Upload Image")

#Disabling warning
st.set_option('deprecation.showfileUploaderEncoding', False)
#Choose your own image
uploaded_file = st.sidebar.file_uploader(" ",type=['png', 'jpg', 'jpeg'] )

if uploaded_file is not None:
 #########################################################   
    u_img = Image.open(uploaded_file)
    show.image(u_img, 'Uploaded Image', use_column_width=True)
    # We preprocess the image to fit in algorithm.
    #image = np.asarray(u_img)/255
    
    #my_image= resize(image, (64,64)).reshape((1, 64*64*3)).T

# For newline
st.sidebar.write('\n')
    
if st.sidebar.button("Click Here to Classify"):
    
    if uploaded_file is None:
        
        st.sidebar.write("Please upload an Image to Classify")
    
    else:
        
        with st.spinner('Classifying ...'):

            prediction =predict_type(uploaded_file)
            time.sleep(2)
            st.success('Done!')
            
        st.sidebar.header("Algorithm Predicts: ")
        
        #Formatted probability value to 3 decimal places
        #probability = "{:.3f}".format(float(prediction*100))
        
        # Classify cat being present in the picture if prediction > 0.5
        
        
        st.markdown(f"<h2 style='text-align: center; color: blue;'>I think this is a {prediction} </h1>", unsafe_allow_html=True)
        st.sidebar.write("It's a ", prediction,'\n' )
            
