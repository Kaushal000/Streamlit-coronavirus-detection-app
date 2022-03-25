#An utitlity app to detect coronavirus from video. Can be run as a standalone app .Type streamlit run video,py on the terminal to run it as standalone app

import os
import streamlit as st
import cv2 
import numpy as np
import tempfile
from modelDownloader import downloader


def getvideo(name):
    #st.markdown('')
  
    vf = cv2.VideoCapture(name)
    img_height, img_width = None, None


  
    if 'key' not in st.session_state:
        st.session_state.key = 'Detecting confidences'
    
    class_labels = ["corona_virus"]


    cfgpath=os.path.join(os.path.dirname( __file__ ),'model','cov_yolov4.cfg')
    modelpath=os.path.join(os.path.dirname( __file__ ),'model','cov_yolov4_best.weights')
      #if weight file doesn't exist download it int the model folder
    if not os.path.exists(modelpath):
        loc=os.path.join(os.path.dirname( __file__ ),'model')#model folder
        d=downloader()
        
        with st.spinner('Downloading weights..'):
            d.downloadFile("https://dl.dropbox.com/s/909wlai4r3y4uz1/cov_yolov4_best.weights?dl=1",loc)
    
    network = cv2.dnn.readNetFromDarknet(cfgpath,modelpath)
    layers_names_output = network.getUnconnectedOutLayersNames()
    #colors = np.random.randint(0, 255, size=(len(labels), 3), dtype='uint8')
    colors = ["0,255,0"]
    colors = [np.array(every_color.split(",")).astype("int") for every_color in colors]
    colors = np.array(colors)
    colors = np.tile(colors,(1,1))


  
    stframe = st.empty()
    # Defining loop for catching frames    
    while vf.isOpened():
        #stf=st.empty()
        ret, frame = vf.read()
        # if frame is read correctly ret is True
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        if img_width is None or img_height is None:
            # Slicing from tuple only first two elements
            img_height, img_width = frame.shape[:2]

     
        blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416),
                                    swapRB=True, crop=False)

       
        
        # Implementing forward pass with our blob and only through output layers
        # Calculating at the same time, needed time for forward pass
        network.setInput(blob)  # setting blob as input to the network
        
        output_from_network = network.forward(layers_names_output)
        
        class_ids_list = []
        boxes_list = []
        confidences_list = []
        
        
        for object_detection_layer in output_from_network :
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
                    bounding_box = object_detection[0:4] * np.array([img_width, img_height, img_width, img_height])
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
        max_value_ids = cv2.dnn.NMSBoxes(boxes_list, confidences_list, 0.5, 0.4)
        
        # loop through the final set of detections remaining after NMS and draw bounding box and write text
        for max_valueid in max_value_ids:
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
            box_color = colors[predicted_class_id]
            
            #convert the color numpy array as a list and apply to text and box
            box_color = [int(c) for c in box_color]
            
            # print the prediction in console
            predicted_class_label = "{}: {:.2f}%".format(predicted_class_label, prediction_confidence * 100)
            
            
            # draw rectangle and text in the image
            cv2.rectangle(frame, (start_x_pt, start_y_pt), (end_x_pt, end_y_pt), box_color, 1)
            cv2.putText(frame, predicted_class_label, (start_x_pt, start_y_pt-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, box_color, 2)
            # Preparing lists for detected bounding boxes,
            # obtained confidences and class's number
        
            
            #checkpoint 
            #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
         
                
            stframe.image(frame,use_column_width='yes')
            # stf.success("predicted object {}".format(predicted_class_label))    

    if not ret:
        st.info("Video ended")        
                
def object_main():
    f = st.file_uploader("Upload file")
    if f is not None:

        tfile = tempfile.NamedTemporaryFile(delete=False) 
        tfile.write(f.read())

        nm=tfile.name
        
        getvideo(nm)
        
if __name__ == '__main__':
    object_main()
