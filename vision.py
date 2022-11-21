from face.analysis import Analysis
import cv2
import time
import streamlit as st
import numpy as np
# import matplotlib.pyplot as plt

# st.markdown("""
#                 <style> #MainMenu {visibility: hidden;}
#                 footer {visibility: hidden;}
#                 </style>
#             """, unsafe_allow_html=True)

start_time = time.time()

def my_func(name=None, Image=None, Image_name=None):
    # st.write('name: ',name,'Image:',Image)
    if name is None:
        img=Image
        name_save = "IMG/out_" + Image_name
    else:
        img=cv2.imread("IMG/"+name)
        name_save = "IMG/out_" + name
    # st.write(name)
    app = Analysis(providers=['CUDAExecutionProvider'])
    app.prepare(ctx_id=0, det_size=(640, 640))

    if img is None:
        raise ValueError('Could not read the intended image!')

    faces = app.get(img)
    out_json = app.out_json(faces)
    rimg = app.draw_on(img, faces)
    # print(out_json)
    # plt.plot(rimg)
    cv2.imwrite(name_save, rimg)

    return rimg, time.time() - start_time, out_json


st.set_page_config(layout="wide")

st.image("IMG\hero-banner.gif", use_column_width = True)

menu = ['Only face Detection','Face land-marks','All face features']
st.sidebar.header('Mode Selection')
choice = st.sidebar.selectbox('How would you like to interpret the image ?', menu)



# Create the Home page
Image1 = st.file_uploader('Upload your portrait here, please!',type=['jpg','jpeg','png'])
if Image1 is not None:
    col3, col4 = st.columns(2)
    Image = Image1.read()
    if Image is None:
        raise ValueError('No Image!!!')

    with col3:
        st.image(Image)
        st.write("---Execution time: in seconds: ")
        st.write("---Number of faces: ")
        Autocrop = st.checkbox('Show JSON of land-marks',value=False)
        st.write('******************************************')
    with col4:
        # st.write(Image)
        if Image == None:
            raise ValueError('Image variable is empty!')
        # else:
        #     st.write(Image)
        # TMP = np.array(Image).reshape(1,Image1.shape[0], Image1.shape[1], Image1.shape[2])
        file_bytes = np.asarray(bytearray(Image), dtype=np.uint8)
        opencv_image = cv2.imdecode(file_bytes, 1)
        IMG, execution_time, out_json = my_func(Image=opencv_image, Image_name=Image1.name)
        # st.image("IMG/out_" + Image+ ".jpg")
        st.image(IMG)
        st.write(str(execution_time))
        st.write(str(out_json['face_number']))
        if Autocrop:
            st.json(out_json)
        else:
            st.write(' ')
            st.write(' ')
        st.write('******************************************')

st.write('===================================================================================================================')


names = [str(i+1) for i in range(21)]

col1, col2 = st.columns(2)

for i in names:
    start_time = time.time()
    with col1:
        st.image('IMG/'+i+'.jpg')
        st.write("---Execution time: in seconds: ")
        st.write("---Number of faces: ")
        Autocrop1 = st.checkbox('Show JSON of land-marks',value=False, key=i)
        st.write('******************************************')
    with col2:
        IMG, execution_time, out_json = my_func(i+'.jpg')
        st.image('IMG/out_'+i+'.jpg')
        st.write(str(execution_time))
        st.write(str(out_json['face_number']))
        if Autocrop1:
            st.json(out_json)
        else:
            st.write('None')

        st.write('******************************************')

st.write('The End!')