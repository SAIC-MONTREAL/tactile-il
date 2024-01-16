import cv2
import pandas as pd
import json

# find all resolutions supported by camera 0
# output as a config file 
# Note: This must be run on a machine with access to the internet

url = "https://en.wikipedia.org/wiki/List_of_common_resolutions"
table = pd.read_html(url)[0]
table.columns = table.columns.droplevel()
cap = cv2.VideoCapture(3, cv2.CAP_V4L2)
resolutions = {}
for index, row in table[["W", "H"]].iterrows():
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, row["W"])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, row["H"])
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    resolutions[str(width)+"x"+str(height)] = "OK"
config = {
        "default": "640x480",
        "resolutions" : resolutions
}
print(json.dumps(config, indent=4))
