import os
import streamlit as st
import streamlit.components.v1 as components
import time
from video import getvideo
from modelDownloader import downloader
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import tempfile
from pathlib import PurePosixPath

def detect_objects(our_image):
    
    # st.set_option('deprecation.showPyplotGlobalUse', False)

    col1, col2 = st.columns(2)
    new_img = np.array(our_image.convert('RGB'))
    img = cv2.cvtColor(new_img,1)
    height,width = img.shape[:2] 
    
 
    img_blob = cv2.dnn.blobFromImage(img, 0.003922, (416, 416), swapRB=True, crop=False)

# only single label 
    class_labels = ["corona_virus"]

    #Declare only a single color
    class_colors = ["0,255,0"]
    class_colors = [np.array(every_color.split(",")).astype("int") for every_color in class_colors]
    class_colors = np.array(class_colors)
    class_colors = np.tile(class_colors,(1,1))

    cfgpath=os.path.join(os.path.dirname( __file__ ),'model','cov_yolov4.cfg')
    modelpath=os.path.join(os.path.dirname( __file__ ),'model','cov_yolov4_best.weights')

    #if weight file doesn't exist download it int the model folder
    if not os.path.exists(modelpath):
        loc=os.path.join(os.path.dirname( __file__ ),'model')#model folder
        d=downloader()
        
        with st.spinner('Downloading weights..'):
            d.downloadFile("https://dl.dropbox.com/s/909wlai4r3y4uz1/cov_yolov4_best.weights?dl=1",loc)
    
    # Loading the coronavirus custom model 
    # input preprocessed blob into model and pass through the model
    # obtain the detection predictions by the model using forward() method
    
    yolo_model = cv2.dnn.readNetFromDarknet(cfgpath,modelpath)

    # Get all layers from the yolo network
    # Loop and find the last layer (output layer) of the yolo network 
    yolo_layers = yolo_model.getLayerNames()
    #yolo_output_layer = [yolo_layers[yolo_layer[0] - 1] for yolo_layer in yolo_model.getUnconnectedOutLayers()]
    yolo_output_layer=yolo_model.getUnconnectedOutLayersNames()
    # input preprocessed blob into model and pass through the model
    yolo_model.setInput(img_blob)
    # obtain the detection layers by forwarding through till the output layer
    obj_detection_layers = yolo_model.forward(yolo_output_layer)


    ############## NMS Change 1 ###############
    # initialization for non-max suppression (NMS)
    # declare list for [class id], [box center, width & height[], [confidences]
    class_ids_list = []
    boxes_list = []
    confidences_list = []
    pclass = []
    count = 0
    ############## NMS Change 1 END ###########


    # loop over each of the layer outputs
    for object_detection_layer in obj_detection_layers:
        # loop over the detections
        for object_detection in object_detection_layer:
            
            # obj_detections[1 to 4] => will have the two center points, box width and box height
            # obj_detections[5] => will have scores for all objects within bounding box
            all_scores = object_detection[5:]
            predicted_class_id = np.argmax(all_scores)
            prediction_confidence = all_scores[predicted_class_id]
        
            # take only predictions with confidence more than 20%
            if prediction_confidence > 0.20:
                #get the predicted label
                predicted_class_label = class_labels[predicted_class_id]
                #obtain the bounding box co-oridnates for actual image from resized image size
                bounding_box = object_detection[0:4] * np.array([width,height,width,height])
                (box_center_x_pt, box_center_y_pt, box_width, box_height) = bounding_box.astype("int")
                start_x_pt = int(box_center_x_pt - (box_width / 2))
                start_y_pt = int(box_center_y_pt - (box_height / 2))
                
                ############## NMS Change 2 ###############
                #save class id, start x, y, width & height, confidences in a list for nms processing
                #make sure to pass confidence as float and width and height as integers
                class_ids_list.append(predicted_class_id)
                confidences_list.append(float(prediction_confidence))
                boxes_list.append([start_x_pt, start_y_pt, int(box_width), int(box_height)])
                ############## NMS Change 2 END ###########
                
    ############## NMS Change 3 ###############
    # Applying the NMS will return only the selected max value ids while suppressing the non maximum (weak) overlapping bounding boxes      
    # Non-Maxima Suppression confidence set as 0.5 & max_suppression threhold for NMS as 0.4 (adjust and try for better perfomance)
    # score_threshold = st.sidebar.slider("Confidence Threshold", 0.00,1.00,0.5,0.01)
    # nms_threshold = st.sidebar.slider("NMS Threshold", 0.00, 1.00, 0.4, 0.01)
    
   
    with st.sidebar:
        score_threshold=st.slider("Confidence Threshold",0.00,1.00,0.5,0.01)
        nms_threshold =st.slider("NMS Threshold",0.00, 1.00,0.4, 0.01)
       
        st.markdown("""
        <style>
        div.Heading h3{
            margin: auto;
            width: 50%;
            color: #dc3545;
            padding: 10px;
        }
       
        </style>
        """,unsafe_allow_html=True)
        
        st.markdown("""
        <br><br>
        <div className="Heading"><h3>Presentation</h3></div>
        <iframe src="https://onedrive.live.com/embed?cid=CC721E0634E29AC8&amp;resid=CC721E0634E29AC8%2113676&amp;authkey=AJqlhggJ3vIp8MA&amp;em=2&amp;wdAr=1.7777777777777777&amp;wdEaaCheck=0" width="300px" height="288px" frameborder="0">This is an embedded <a target="_blank" href="https://office.com">Microsoft Office</a> presentation, powered by <a target="_blank" href="https://office.com/webapps">Office</a>.</iframe>
        """,unsafe_allow_html=True)
        
    max_value_ids = cv2.dnn.NMSBoxes(boxes_list, confidences_list,score_threshold,nms_threshold )

    # loop through the final set of detections remaining after NMS and draw bounding box and write text
    for max_valueid in max_value_ids:
        #max_class_id = max_valueid[0]
        max_class_id = max_valueid
        box = boxes_list[max_class_id]
        start_x_pt = box[0]
        start_y_pt = box[1]
        box_width = box[2]
        box_height = box[3]
        
        #get the predicted class id and label
        predicted_class_id = class_ids_list[max_class_id]
        predicted_class_label = class_labels[predicted_class_id]
        prediction_confidence = confidences_list[max_class_id]
    ############## NMS Change 3 END ########### 
            
        end_x_pt = start_x_pt + box_width
        end_y_pt = start_y_pt + box_height
        
        #get a random mask color from the numpy array of colors
        box_color = class_colors[predicted_class_id]
        
        #convert the color numpy array as a list and apply to text and box
        box_color = [int(c) for c in box_color]
        
        # print the prediction in console
        predicted_class_label = "{}: {:.2f}%".format(predicted_class_label, prediction_confidence * 100)
        pclass.append(predicted_class_label)
        #print("predicted object {}".format(predicted_class_label))
        count+=1
        # draw rectangle and text in the image
        cv2.rectangle(img, (start_x_pt, start_y_pt), (end_x_pt, end_y_pt), box_color, 1)
        cv2.putText(img, predicted_class_label, (start_x_pt, start_y_pt-5), cv2.FONT_HERSHEY_COMPLEX, 0.7, box_color, 2)
        
        
        
        
    with col1:
        st.header("Original image")
        st.image(our_image,width=15,use_column_width='auto')

    with col2:
        st.header("Detected objects in the image")
        st.image(img,width=15,use_column_width='auto')

        
        
        
        
    st.info("Zoom in the image to see the confidence scores of the objects detected")    
    if len(pclass)==1:
        st.success("Coronovirus detected")
    elif len(pclass)>=2:
        st.success("Detected {} coronaviruses.".format(count))
    else:
        st.error("No coronavirus detected. Make sure you have uploaded the correct grayscale electron microscopic image of coronavirus. If you see this error even after uploading the correct image for coronavirus then the model requires further training")        

def object_main():
    """OBJECT DETECTION APP"""
    #Favicon
    favpath=os.path.join(os.path.dirname( __file__ ),'images','icons8-coronavirus-16.png')
    img1=Image.open(favpath)
    
    path=os.path.abspath(os.path.join(os.path.dirname( __file__ ), '.', 'images','coronavirus.jpg'))
    
    
    #st.set_page_config(layout='wide')
    st.set_page_config(layout='wide',page_title='Object detection',page_icon=img1,initial_sidebar_state = 'auto')
    #components.iframe("https://docs.streamlit.io/en/latest")
    # st.set_page_config(layout='wide',page_title='Object detection',initial_sidebar_state = 'auto')
    hide_streamlit_style = """
            <style>
            footer {visibility: hidden;}
            </style>
            """
    st.markdown(hide_streamlit_style,unsafe_allow_html=True)       
    # Overriding some default styles
    st.markdown("""
    <style>
    div.css-18e3th9{
    position: relative;
    padding-bottom: 0px;
    padding-top: 0px;
    }
    </style>
    """,unsafe_allow_html=True)
    
    #Creating navbar
    #Declaring styles for navbar
    st.markdown("""
    <style>
    nav{
	position: relative;
	display: flex;
	width: 640px;
	margin: 4rem auto;
       }

    nav.navbar a{
        display: block;
        width: 20%;
        padding: .65rem 0;
        color: #dc3545;
        text-decoration: none;
        text-align: center;
        text-transform: uppercase;
    }

    .nav-underline, .nav-underline2{
        position: absolute;
        left: 0;
        bottom: -1px;
        width: 18%;
        height: 2px;
        background: #fff;
        transition: all .3s ease-in-out;
    }

    .nav-underline2{
        top: -1px !important;
    }

    nav a:hover{
        font-size: 20px;
        font-weight: 900;
        transition: font-size .1s linear,
                    font-weight .1s linear;
        color: rgb(220,20,60);            
    }

    nav a:nth-child(1).current ~ .nav-underline{
        left: 0;
    }


    nav a:nth-child(2).current ~ .nav-underline{
        left: 20%;
    }


    nav a:nth-child(3).current ~ .nav-underline{
        left: 40%;
    }


    nav a:nth-child(4).current ~ .nav-underline{
        left: 60%;
    }


    nav a:nth-child(5).current ~ .nav-underline{
        left: 80%;
    }

    nav a:nth-child(1):hover ~ .nav-underline{
        left: 0;
    }


    nav a:nth-child(2):hover ~ .nav-underline{
        left: 20%;
    }


    nav a:nth-child(3):hover ~ .nav-underline{
        left: 40%;
    }


    nav a:nth-child(4):hover ~ .nav-underline{
        left: 60%;
    }


    nav a:nth-child(5):hover ~ .nav-underline{
        left: 80%;
    }

    nav a:nth-child(1).current ~ .nav-underline2{
        left: 0;
    }


    nav a:nth-child(2).current ~ .nav-underline2{
        left: 20%;
    }


    nav a:nth-child(3).current ~ .nav-underline2{
        left: 40%;
    }


    nav a:nth-child(4).current ~ .nav-underline2{
        left: 60%;
    }


    nav a:nth-child(5).current ~ .nav-underline2{
        left: 80%;
    }

    nav a:nth-child(1):hover ~ .nav-underline2{
        left: 0;
    }


    nav a:nth-child(2):hover ~ .nav-underline2{
        left: 20%;
    }


    nav a:nth-child(3):hover ~ .nav-underline2{
        left: 40%;
    }


    nav a:nth-child(4):hover ~ .nav-underline2{
        left: 60%;
    }


    nav a:nth-child(5):hover ~ .nav-underline2{
        left: 80%;
    }
    </style>
    """,unsafe_allow_html=True)
    
    #Navbar links modify it accordint to needs
    st.markdown("""
    <nav class="navbar">
		<a href="#">Home</a>
		<a href="https://github.com/Kaushal000/Streamlit-coronavirus-detection-app/wiki" target="_blank">Docs</a>
		<a href="#" class="current">Thesis</a>
		<a href="https://colab.research.google.com/drive/1KszU9b3t-T_Ia5GNjiy_uuktOnydlEID" target="_blank">Training</a>
		<a href="https://github.com/Kaushal000/Streamlit-coronavirus-detection-app" target="_blank">Code</a>
		<div class="nav-underline"></div>
		<div class="nav-underline2"></div>
	</nav>""",unsafe_allow_html=True)

  
    st.title("Novel Coronavirus detection")
    st.write("The app detects coronaviruses from both images and video. It employs the YOLO algorithm (You Only Look Once), a state-of-the-art algorithm trained to identify thousands of objects types. It extracts objects from images and identifies them using OpenCV and Yolo. This task involves Deep Neural Networks(DNN), yolo trained model, yolo configuration and a custom dataset to detect coronaviruses.")

    choice = st.radio("", ("Show Demo", "Browse an Image","Show stats","Upload video"))
    
    # choice = st.radio("", ("Browse an Image" ,"Upload video"))
    st.write()

    if choice == "Browse an Image":
        st.set_option('deprecation.showfileUploaderEncoding', False)
        image_file = st.file_uploader("Upload Image", type=['jpg','png','jpeg'])

        if image_file is not None:
            our_image = Image.open(image_file)  
            st.info('Image uploaded')
            
            with st.spinner('Detecting objects and generating confidence scores..'):
                time.sleep(5)
                detect_objects(our_image)
     
    elif choice== "Upload video" :
        st.write()
        f=st.file_uploader("Upload Video",type='mp4')
        col1, col2, col3 = st.columns([10,20,1])

        
        
        if f is not None:
            tfile = tempfile.NamedTemporaryFile(delete=False) 
            tfile.write(f.read())

            nm=tfile.name
            with col1:
                st.write("")

            with col2:
                getvideo(nm)

            with col3:
                st.write("")
        
    elif choice=="Show stats":
        st.title("Graph containing the average loass as well as mAP%")
        location=os.path.join(os.path.dirname( __file__ ), 'images','cov.png')
        stats_image = Image.open(location)
        st.image(stats_image,width=20,use_column_width='auto')
        st.markdown("""
        <style>
        .Blue{
            color:rgb(12,23,138)
        }
        .Red{
            color:rgb(245,66,81)
        }
        </style>
        <p>
        <strong>The&nbsp;<span class="Blue">blue&nbsp;</span>line depicts average loss while the&nbsp;<span class="Red">red&nbsp;</span>line depicts mAP%</strong></p>
        """,unsafe_allow_html=True)
       
    
    else :
        iloc=os.path.join(os.path.dirname( __file__ ), 'images','coronavirus.jpg')
        our_image = Image.open(iloc)
        detect_objects(our_image)
# embed streamlit docs in a streamlit app

if __name__ == '__main__':
    object_main()
