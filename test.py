from door import *
from pymata import *

if __name__ == "__main__":
    # create a PyMata instance
    # set the COM port string specifically for your platform
    Arduino = PyMata("/dev/cu.usbmodem14101")
    # create an entrance
    ENTRANCE = door(Arduino)
    
    # turn on the blue LED and play the sound1
    #ENTRANCE.open()
    # turn on the red LED and play the sound2
    #ENTRANCE.reject()

    Arduino.close()   

    