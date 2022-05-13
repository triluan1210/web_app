import re
from statistics import mode
import streamlit as st
import pydicom
import numpy as np
import cv2
from tensorflow import keras

#pydicom.config.convert_wrong_length_to_UN = True
#pydicom.config.Settings.infer_sq_for_un_vr = True

st.set_option('deprecation.showfileUploaderEncoding', False)

class WrongFileType(ValueError):
    pass

def main():
    st.title("CLASSIFICATION MEDICAL DICOM")
    st.sidebar.title("Configuration")

    dicom_bytes = st.sidebar.file_uploader("Upload DICOM file", type=["dcm","dicom"])

    # Config
    classes = ['Aortic enlargement ', 'Covid', 'Opacity', 'Normal']


    #model_chose= st.sidebar.radio(
        #"Model Selection",
        #('VGG16_224_standard', 'VGG16_224_proposed', 'RES_224_standard','RES_224_proposed'))


    model = keras.models.load_model("RES_224_proposed.h5")

    mode = st.sidebar.radio(
        "Select input source",
        ('View Image', 'View information'))

    if not dicom_bytes:
        raise st.stop()

    try:
        dicom_header = pydicom.read_file(dicom_bytes, force=True)

        image_dicom = dicom_header.pixel_array / 4095
    except:
        st.write(WrongFileType("Does not appear to be a DICOM file"))

    if mode == 'View Image':

        if st.sidebar.button("Load Image"):
            st.image(image_dicom,width =400)

        if st.sidebar.button("Predicted"):
            st.image(image_dicom,width =400)
            my_data2 = cv2.resize(image_dicom, (224, 224))
            a = my_data2.reshape(-1, 224, 224, 1)
            # pass the image through the network to obtain our predictions
            preds = model.predict(a)
            label = classes[np.argmax(preds)]
            st.text("RESULT : " + label)

    if mode == 'View information' and st.sidebar.button("Load Information"):
        view = dicom_header.__repr__().split("\n")
        view = "\n".join(view)
        f"""
            ```
            {view}
            ```
            """


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        pass