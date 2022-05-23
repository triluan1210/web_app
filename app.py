import streamlit as st
import pydicom
import numpy as np
import cv2,csv
from tensorflow import keras

# pydicom.config.convert_wrong_length_to_UN = True
# pydicom.config.Settings.infer_sq_for_un_vr = True

st.set_option('deprecation.showfileUploaderEncoding', False)
from PIL import Image

image = Image.open('Capture.PNG')

st.sidebar.image(image, channels="RGB", width=250)


class WrongFileType(ValueError):
    pass


def main():
    st.title("CLASSIFICATION MEDICAL DICOM")

    # st.sidebar.title("Configuration")
    # st.sidebar.text("YOU SHOULD CHOOSE LUNG IMAGE")

    def load_image(img):
        im = Image.open(img).convert('RGB')
        image = np.array(im)
        return image

    mode1 = st.sidebar.radio(
        "Select type of input",
        ('Image', 'Dicom'))
    if mode1 == 'Dicom':
        dicom_bytes = st.sidebar.file_uploader("Upload file", type=["dcm", "dicom"])
        # Config
        # print(dicom_bytes)
        classes = ['AORTIC ENLARGEMENT ', 'COVID 19', 'OPACITY', 'NORMAL']
        model = keras.models.load_model('RES_128_proposed.h5')

        mode = st.sidebar.radio(
            "Select input source",
            ('View Image', 'View information'))

        if not dicom_bytes:
            raise st.stop()

        try:
            dicom_header = pydicom.read_file(dicom_bytes, force=True)
            image_dicom = dicom_header.pixel_array / 4095
            # st.text(image_dicom)
        except:
            st.write(WrongFileType("Does not appear to be a DICOM file"))
            # pass
        if mode == 'View Image':

            if st.sidebar.button("Load Image"):
                st.image(image_dicom, width=500)

            if st.sidebar.button("Predict"):
                st.image(image_dicom, width=500)
                # print(image_dicom.shape)
                my_data2 = cv2.resize(image_dicom, (128, 128))
                a = my_data2.reshape(-1, 128, 128, 1)
                # pass the image through the network to obtain our predictions
                preds = model.predict(a)
                print(max(preds[0]))
                if max(preds[0]) <= 0.962:
                    st.text("THIS IS NOT A FILE OF LUNG DICOM")

                else:
                    label = classes[np.argmax(preds)]
                    st.text("THE RESULT OF DICOM IS: " + label + " WITH ACCURACY " + str(max(preds[0]) * 100) + " %")

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
        # classes = ['COVID 19', 'NORMAL', 'PNEUMONIA']
        # model = keras.models.load_model('RESNET50_224_image.h5')
        classes = ['AORTIC ENLARGEMENT ', 'COVID 19', 'OPACITY', 'NORMAL']
        model = keras.models.load_model('VGG16_224_image.h5')
        mode = st.sidebar.radio(
            "Select input source",
            ('View Image', 'View information'))

        if mode == 'View Image':
            if st.sidebar.button("Load Image"):
                st.image(image_bytes, width=500)
                image = load_image(image_bytes)

            if st.sidebar.button("Predicted"):
                image = load_image(image_bytes)
                st.image(image_bytes, width=500)
                # print(image_bytes.shape)
                image = image / 255
                image = cv2.resize(image, (224, 224))
                a = np.expand_dims(image, axis=0)
                # st.text(a.shape)
                # pass the image through the network to obtain our predictions
                preds = model.predict(a)
                print(preds)
                if max(preds[0]) <= 0.98:
                    st.text("THIS IS NOT A FILE OF LUNG IMAGE")

                else:
                    label = classes[np.argmax(preds)]
                    st.text("THE RESULT OF IMAGE IS: " + label + " WITH ACCURACY " + str(max(preds[0]) * 100) + " %")
        if mode == 'View information' and st.sidebar.button("Load Information"):
                image = load_image(image_bytes)
                e = image[:, :, 0]
                save_infor_read = []
                for a1 in e[0]:
                    infor_read = '{0:08b}'.format(a1)
                    save_infor_read.append(infor_read[-1:])
                for a2 in e[1]:
                    infor_read = '{0:08b}'.format(a2)
                    save_infor_read.append(infor_read[-1:])
                for a3 in e[2]:
                    infor_read = '{0:08b}'.format(a3)
                    save_infor_read.append(infor_read[-1:])
                for a4 in e[3]:
                  infor_read = '{0:08b}'.format(a4)
                  save_infor_read.append(infor_read[-1:])
                for a5 in e[4]:
                  infor_read = '{0:08b}'.format(a5)
                  save_infor_read.append(infor_read[-1:])
                # print(len(save_infor_read))
                k = 0
                save_bit_read = []
                for a in range(len(save_infor_read) + 1):
                    n = 8 * k
                    save_bit_read.append(
                        save_infor_read[n] + save_infor_read[n + 1] + save_infor_read[n + 2] + save_infor_read[n + 3] +
                        save_infor_read[n + 4] + save_infor_read[n + 5] + save_infor_read[n + 6] + save_infor_read[
                            n + 7])
                    k += 1
                    if k == len(save_infor_read) / 8:
                        break
                # print(len(save_bit_read))
                infor_read_final = []
                for a in save_bit_read:
                    with open('protagonist.csv', newline='') as csvfile:
                        spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
                        for row in spamreader:
                            if row[0] == str(int(a, 2)):
                                # print(row[1])
                                if row[1] == '(':
                                    infor_read_final.append('\n')
                                    infor_read_final.append(row[1])
                                else:
                                    infor_read_final.append(row[1])

                                if row[1] == ')':
                                    infor_read_final.append('   ')
                  
                final = []
                for a in infor_read_final:
                  if a == '@':
                    break
                  final.append(a)
                st.text(''.join(str(e) for e in final))


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        pass
