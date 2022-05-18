
import streamlit as st
import pydicom
import numpy as np
import cv2
from tensorflow import keras

#pydicom.config.convert_wrong_length_to_UN = True
#pydicom.config.Settings.infer_sq_for_un_vr = True

st.set_option('deprecation.showfileUploaderEncoding', False)
from PIL import Image
image = Image.open('Capture.PNG')

st.sidebar.image(image,channels="RGB",width =250)
class WrongFileType(ValueError):
    pass

def main():
    st.title("CLASSIFICATION MEDICAL DICOM")
    st.sidebar.title("Configuration")
    st.sidebar.text("YOU SHOULD CHOOSE LUNG IMAGE")
    
    
    def load_image(img):
                im = Image.open(img).convert('RGB')
                image = np.array(im)
                return image
            
            
    mode1 = st.sidebar.radio(
        "Select type of input",
        ('Image', 'Dicom'))
    if mode1 == 'Dicom':
        dicom_bytes = st.sidebar.file_uploader("Upload file", type=["dcm","dicom"])
        # Config
        print(dicom_bytes)
        classes = ['AORTIC ENLARGEMENT ', 'COVID 19', 'OPACITY', 'NORMAL']
        model = keras.models.load_model('RES_128_proposed.h5')

        mode = st.sidebar.radio(
            "Select input source",
            ('View Image', 'View information'))

        if not dicom_bytes:
            raise st.stop()

        try:
            dicom_header = pydicom.read_file(dicom_bytes, force=True)
            image_dicom = dicom_header.pixel_array/4095
        except:
            st.write(WrongFileType("Does not appear to be a DICOM file"))

        if mode == 'View Image':

            if st.sidebar.button("Load Image"):
                st.image(image_dicom,width =500)

            if st.sidebar.button("Predicted"):
                st.image(image_dicom,width =500)
                #print(image_dicom.shape)
                my_data2 = cv2.resize(image_dicom, (128, 128))
                a = my_data2.reshape(-1, 128, 128, 1)
                # pass the image through the network to obtain our predictions
                preds = model.predict(a)
                print(max(preds[0]))
                if max(preds[0]) <= 0.96:
                    st.text("THIS IS NOT A FILE OF LUNG DICOM")

                else:
                    label = classes[np.argmax(preds)]
                    st.text("THE RESULT OF DICOM IS: " + label + " WITH ACCURACY " + str(max(preds[0])*100) + " %")

        if mode == 'View information' and st.sidebar.button("Load Information"):
            view = dicom_header.__repr__().split("\n")
            view = "\n".join(view)
            f"""
                ```
                {view}
                ```
                """

    if mode1 == 'Image':
        image_bytes = st.sidebar.file_uploader("Upload file", type=["jpg", "png"])
        # Config
        #classes = ['COVID 19', 'NORMAL', 'PNEUMONIA']
        #model = keras.models.load_model('RESNET50_224_image.h5')
        classes = ['AORTIC ENLARGEMENT ', 'COVID 19', 'OPACITY', 'NORMAL']
        model = keras.models.load_model('VGG16_128_image.h5')

        if st.sidebar.button("Load Image"):
            st.image(image_bytes, width=500)
            image = load_image(image_bytes)
            
            #print(image.shape)
            #print(image)



        if st.sidebar.button("Predicted"):
                    image = load_image(image_bytes)
                    st.image(image_bytes, width=500)
                    #print(image_bytes.shape)
                    image = image/255
                    image = cv2.resize(image, (128, 128))
                    a = np.expand_dims(image, axis = 0)
                    #st.text(a.shape)
                    # pass the image through the network to obtain our predictions
                    preds = model.predict(a)
                    print(preds)
                    if max(preds[0]) <= 0.96:
                        st.text("THIS IS NOT A FILE OF LUNG IMAGE")

                    else:
                        label = classes[np.argmax(preds)]
                        st.text("THE RESULT OF IMAGE IS: " + label + " WITH ACCURACY " + str(max(preds[0])*100) + " %")

if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        pass
