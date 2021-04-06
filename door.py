import sys
import signal
from pymata import *
from multiprocessing import Process


class door:
    def __init__(self, board):
        self.board = board
        # listen for SIGINT
        self.signal = signal.signal(signal.SIGINT, self.signal_handler)
        
    # signal handler function called when Control-C occurs
    def signal_handler(self, signal, frame):
        print('You pressed Ctrl+C!!!!')
        if self.board != None:
            self.board.reset()
        sys.exit(0)

    def sound1(self):
        BEEPER = 5  # pin that piezo device is attached to
        start = time.time()
        counter = 0.3
        while (time.time() - start <= 10):
            self.board.play_tone(BEEPER, self.board.TONE_TONE, 523, 100)
            time.sleep(counter)
            self.board.play_tone(BEEPER, self.board.TONE_TONE, 587, 100)
            time.sleep(counter)
            self.board.play_tone(BEEPER, self.board.TONE_TONE, 659, 100)
            time.sleep(counter)
            self.board.play_tone(BEEPER, self.board.TONE_TONE, 698, 100)
            time.sleep(counter)
            self.board.play_tone(BEEPER, self.board.TONE_TONE, 784, 100)
            time.sleep(counter)
            self.board.play_tone(BEEPER, self.board.TONE_TONE, 880, 100)
            time.sleep(counter)
            self.board.play_tone(BEEPER, self.board.TONE_TONE, 988, 100)
            time.sleep(counter)
            self.board.play_tone(BEEPER, self.board.TONE_TONE, 1046, 100)
            time.sleep(0.5)
            counter -= 0.05

    # alarm for reject access
    def sound2(self):
        BEEPER = 5
        start = time.time()
        while (time.time() - start <= 3):
            self.board.play_tone(BEEPER, self.board.TONE_TONE, 650, 230)
            time.sleep(0.05)
            self.board.play_tone(BEEPER, self.board.TONE_TONE, 2000, 100)
            time.sleep(0.05)

    # open door function
    def blue_LED(self):
        # pin3 on the arduino board
        '''RED_LED = 4
        self.board.set_pin_mode(RED_LED, self.board.OUTPUT, self.board.DIGITAL)
        self.board.digital_write(RED_LED, 0)'''
        print('hello')

        BLUE_LED = 3
        self.board.set_pin_mode(BLUE_LED, self.board.OUTPUT, self.board.DIGITAL)
        

        counter = 1    
        # the timer for the LED
        start = time.time()
        ### the LED light will turn on and off
        while True:
            ### Stop the red LED first
            
            # the first five second
            if (time.time() - start) < 4:
                self.board.digital_write(BLUE_LED, counter)
                time.sleep(0.2)
            # the first 5 - 9 sec
            elif (time.time() - start) <= 8:
                if counter == 1:
                    self.board.digital_write(BLUE_LED, counter)
                    counter = 0
                else:
                    self.board.digital_write(BLUE_LED, counter)
                    counter = 1
                time.sleep(0.5)
            # the last 3 sec
            elif (time.time() - start) <= 10:
                if counter == 1:
                    self.board.digital_write(BLUE_LED, counter)
                    counter = 0
                else:
                    self.board.digital_write(BLUE_LED, counter)
                    counter = 1
                time.sleep(0.1)
            # turn off the LED/close the door
            else:
                self.board.digital_write(BLUE_LED, 0)
                time.sleep(3)
                return


    # reject signal
    def red_LED(self):
        # pin4 on the arduino board
        BOARD_LED = 4
        self.board.set_pin_mode(BOARD_LED, self.board.OUTPUT, self.board.DIGITAL)
        counter = 1
        # the timer for the LED
        start = time.time()
        ### the LED light will turn on and off
        while True:
            if (time.time() - start) <= 3:
                if counter == 1:
                    self.board.digital_write(BOARD_LED, counter)
                    counter = 0
                else:
                    self.board.digital_write(BOARD_LED, counter)
                    counter = 1
                time.sleep(0.5)
            else:
                self.board.digital_write(BOARD_LED, 0)
                time.sleep(3)
                return

    def open(self, lock):
        lock.value = 1.0
        p1 = Process(target=self.sound1)
        p1.start()
        p2 = Process(target=self.blue_LED)
        p2.start()
        # wait the two processes
        p1.join()
        p2.join()
        lock.value = 0.0

    def reject(self, lock):
        lock.value = 1.0
        # parallel program to turn on the red LED and the buzzer at the same time
        p1 = Process(target=self.sound2)
        p1.start()
        p2 = Process(target=self.red_LED)
        p2.start()
        # wait the two processes
        p1.join()
        p2.join()
        lock.value = 0.0






