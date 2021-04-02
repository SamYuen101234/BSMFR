import cv2
import pickle
import base64
from scipy import optimize as op
import numpy as np

def urlsafe_base64_encode(s):
    """
    Encode a bytestring to a base64 string for use in URLs. Strip any trailing
    equal signs.
    """
    return base64.urlsafe_b64encode(s).rstrip(b'\n=').decode('ascii')

def embedding_to_base64_string(mask_embedding_ndarray):
    np_bytes = pickle.dumps(mask_embedding_ndarray)
    return urlsafe_base64_encode(np_bytes)

def add_bounding_box(img, box, mask, recognition):
    if recognition is not None:
        if mask == 0:
            # recognized and wearing a mask, rendering green box
            img = cv2.rectangle(img, (box[0],box[1]), (box[0]+box[2],box[1]+box[3]), (0,255,0), 5)
        elif mask == 1:
            # recognized and not wearing a maks, rendering yellow box
            img = cv2.rectangle(img, (box[0],box[1]), (box[0]+box[2],box[1]+box[3]), (255,255,0), 5)
        cv2.putText(img, recognition, (box[0],box[1]-5), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 3)
    else:
        if mask == 0:
            # not recognized and wearing a mask, rendering black box
            img = cv2.rectangle(img, (box[0],box[1]), (box[0]+box[2],box[1]+box[3]), (0,0,0), 5)
        elif mask == 1:
            # not recognized and not wearing a mask, rendering white box
            img = cv2.rectangle(img, (box[0],box[1]), (box[0]+box[2],box[1]+box[3]), (255,255,255), 5)
        cv2.putText(img, "invalid", (box[0],box[1]-5), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 3)
    return img

# function to fit depth image to distinguish real person or paper
def anti_spoofing_curve(x, A, B, C, D, E):
    return A * np.power(x, 4) + B * np.power(x, 3) + C * np.square(x) + D * x + E

def check_depth(depth):
    y_group = np.squeeze(depth)
    x_group = np.arange(y_group.shape[0])
    A, B, C, D, E = op.curve_fit(anti_spoofing_curve, x_group, y_group)[0]
    x = np.arange(x_group.min(), x_group.max(), 1)
    y = A * np.power(x, 4) + B * np.power(x, 3) + C * np.square(x) + D * x + E
    MSE = (((y - y_group[0:-1])**2)/y.shape[0]).sum()
    return MSE
