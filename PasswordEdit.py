from PyQt5 import QtGui
from PyQt5.QtWidgets import QLineEdit
import copy

class PasswordEdit(QLineEdit):
    """
    A LineEdit with icons to show/hide password entries
    """
    CSS = '''QLineEdit {
        border-radius: 0px;
        height: 30px;
        width: 280px;
        margin: 0px 0px 0px 0px;
    }
    '''

    def __init__(self, parent):
        self.parent = parent
        super().__init__(self.parent)
        # Set styles
        self.setStyleSheet(self.CSS)

        self.visibleIcon = QtGui.QIcon("./img/eye_visible.svg")
        self.hiddenIcon = QtGui.QIcon("./img/eye_hidden.svg")

        self.setEchoMode(QLineEdit.Password)
        self.togglepasswordAction = self.addAction(self.visibleIcon, QLineEdit.TrailingPosition)
        self.togglepasswordAction.triggered.connect(self.on_toggle_password_Action)
        self.password_shown = False

    def on_toggle_password_Action(self):
        if not self.password_shown:
            self.setEchoMode(QLineEdit.Normal)
            self.password_shown = True
            self.togglepasswordAction.setIcon(self.hiddenIcon)
        else:
            self.setEchoMode(QLineEdit.Password)
            self.password_shown = False
            self.togglepasswordAction.setIcon(self.visibleIcon)