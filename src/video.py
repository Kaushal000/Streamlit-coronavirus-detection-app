import os
import streamlit as st
import cv2 
import numpy as np
import tempfile
import time
from modelDownloader import downloader

def getvideo(name,score_threshold,nms_threshold):
    #st.markdown('')
  
    vf = cv2.VideoCapture(name)
    img_height, img_width = None, None
    writer=None

  
    if 'key' not in st.session_state:
        st.session_state.key = 'Detecting confidences'
    
    class_labels = ["corona_virus"]

    videopath=os.path.join(os.path.dirname( __file__ ),'video','result.mp4')
    cfgpath=os.path.join(os.path.dirname( __file__ ),'model','cov_yolov4.cfg')
    modelpath=os.path.join(os.path.dirname( __file__ ),'model','cov_yolov4_best.weights')

    if not os.path.exists(modelpath):
        loc=os.path.join(os.path.dirname( __file__ ),'model')
        d=downloader()
        
        with st.spinner('Downloading weights...'):
            d.downloadFile("https://dl.dropbox.com/s/909wlai4r3y4uz1/cov_yolov4_best.weights?dl=1",loc)

    network = cv2.dnn.readNetFromDarknet(cfgpath,modelpath)

    layers_names_output = network.getUnconnectedOutLayersNames()
    
    #colors = np.random.randint(0, 255, size=(len(labels), 3), dtype='uint8')
    colors = ["0,255,0"]
    colors = [np.array(every_color.split(",")).astype("int") for every_color in colors]
    colors = np.array(colors)
    colors = np.tile(colors,(1,1))


    
    # Defining variable for counting frames
    # At the end we will show total amount of processed frames
    f = 0

    # Defining variable for counting total time
    # At the end we will show time spent for processing all frames
    t = 0
    # Defining loop for catching frames  

    stinfo = st.empty()
    stframe = st.empty()
    while vf.isOpened():
        stf=st.empty()
        ret, frame = vf.read()
        # if frame is read correctly ret is True
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        
            # Slicing from tuple only first two elements
        img_height, img_width = frame.shape[:2]

    
        blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416),
                                    swapRB=True, crop=False)

    
        
        # Implementing forward pass with our blob and only through output layers
        # Calculating at the same time, needed time for forward pass
        network.setInput(blob)  # setting blob as input to the network
        
        with st.spinner("Detecting from frames and writing down the individual detected frames to video"):
            start = time.time()
            output_from_network = network.forward(layers_names_output)
            end = time.time()

            # Increasing counters for frames and total time
            f += 1
            t += end - start

            # Showing spent time for single current frame
            # print('Frame number {0} took {1:.5f} seconds'.format(f, end - start))
            stinfo.info('Frame number {0} took {1:.5f} seconds'.format(f, end - start))
            
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
            max_value_ids = cv2.dnn.NMSBoxes(boxes_list, confidences_list,score_threshold,nms_threshold)
            
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

            if writer is None:
            # Constructing code of the codec
            # to be used in the function VideoWriter
            # fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                fourcc = 0x00000021
                # fourcc = cv2.VideoWriter_fourcc(*'vp80')

                # Writing current processed frame into the video file
                # Pay attention! If you're using Windows, yours path might looks like:
                # r'videos\result-traffic-cars.mp4'
                # or:
                # 'videos\\result-traffic-cars.mp4'
                writer = cv2.VideoWriter(videopath, fourcc, 30,
                                        (frame.shape[1], frame.shape[0]), True)

        # Write processed current frame to the file
            writer.write(frame)
            # stf.success("predicted object {}".format(predicted_class_label))    
        
    stinfo.info("End of video. Writing the detected frames and showing the resultant video...")
    writer.release()
    stinfo.empty()
    stframe.empty()                
    
    with st.spinner("Optimizing and encoding the video for web compability"):
        time.sleep(5)
        video_file = open(videopath, 'rb')
        video_bytes = video_file.read()


    st.video(video_bytes)
    st.success(f'Total number of frames: {f}')
    st.success('Total amount of time {:.5f} seconds'.format(t))
    fps=round((f/t),1)
    st.success(f'FPS:{fps}')

def object_main():
    f = st.file_uploader("Upload file")
    if f is not None:

        tfile = tempfile.NamedTemporaryFile(delete=False) 
        tfile.write(f.read())

        nm=tfile.name
        
        getvideo(nm)
        
if __name__ == '__main__':
    object_main()