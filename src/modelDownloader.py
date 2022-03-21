#Model downloading utility app using wget

import os
from wget import download

#
class downloader:

    def downloadFile(self, url, location=""):
            # Download file and with a custom progress bar
        download(url, out = location)

# downloadObj = downloader()
# loc=os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..', 'src\model'))

# if __name__=="__main__":
#     downloadObj = downloader()   
#     loc=os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..', 'src\model'))   
#     downloadObj.downloadFile("https://dl.dropbox.com/s/909wlai4r3y4uz1/cov_yolov4_best.weights?dl=1",loc)    