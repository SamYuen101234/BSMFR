import sys
from PyQt5 import uic
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QApplication, QWidget, QDialog, QMainWindow, QLineEdit
from PasswordEdit import *
import copy

# main window (login page)
class App(QMainWindow):
    def __init__(self):
        super().__init__()
        uic.loadUi('login.ui', self)
        self.setWindowTitle('Masked Facial Recognition')
        self.button.clicked.connect(self.printValue)  # login button
        self.register_2.mousePressEvent = self.register_page   # New user button
        self.forgetPW.mousePressEvent = self.forgetPW_page # forget password button
        self.password = PasswordEdit(self.password) # make the password bar to customized password bar (override)

    def printValue(self):
        print(self.account.text())
        print(self.password.text())
    
    # go to sign up page
    def register_page(self, event):
        self.widget1 = register()  # initialize register page
        self.widget1.show()  # show register page
        self.hide() # hide the login page
    
    # go to finding password page
    def forgetPW_page(self, event):
        self.widget2 = forgetPW()
        self.widget2.show()  # show page for finding pw
        self.hide()  # hide the login page

# sign up page
class register(QWidget):
    def __init__(self):
        super().__init__()
        uic.loadUi('register.ui', self)
        self.verticalScrollBar.setMaximum(255) # max value of the scroll bar
        self.verticalScrollBar.sliderMoved.connect(self.sliderval) # event triggered when move the bar
        
    def sliderval(self):
        print(self.verticalScrollBar.value())

# a page for users to find their password
class forgetPW(QWidget):
    def __init__(self):
        super().__init__()
        uic.loadUi('forgetPW.ui', self)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    demo = App()
    demo.show()
    try:
        sys.exit(app.exec_())
    except SystemExit:
        print('Closing Window...')