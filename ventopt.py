# -*- coding: utf-8 -*-
"""
Created on Tue Aug 29 00:58:47 2023

@author: MiloPC
"""

import torch
import os 
from PyQt5.QtCore import Qt, QTimer, QRectF
from PyQt5.QtWidgets import QApplication, QHBoxLayout, QLabel, QSizePolicy, QSlider, QSpacerItem, QVBoxLayout, QWidget, QPushButton
from PyQt5.QtGui import QPainter, QColor, QBrush, QPen
import time

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
torch.set_default_device('cuda')

class Slider(QWidget):
    def __init__(self, minimum, maximum, name, parent=None):
        super(Slider, self).__init__(parent=parent)
        self.verticalLayout = QVBoxLayout(self)
        self.label = QLabel(self)
        self.verticalLayout.addWidget(self.label)
        self.horizontalLayout = QHBoxLayout()
        spacerItem = QSpacerItem(0, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)
        self.horizontalLayout.addItem(spacerItem)
        self.slider = QSlider(self)
        self.slider.setOrientation(Qt.Horizontal)
        self.horizontalLayout.addWidget(self.slider)
        spacerItem1 = QSpacerItem(0, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)
        self.horizontalLayout.addItem(spacerItem1)
        self.verticalLayout.addLayout(self.horizontalLayout)
        self.resize(self.sizeHint())

        self.minimum = minimum
        self.maximum = maximum
        self.name = name
        self.slider.valueChanged.connect(self.setLabelValue)
        self.slider.setValue(50)
        self.x = None
  
        self.setLabelValue(self.slider.value())
        
    def setLabelValue(self, value):

        self.x = self.minimum + (float(value) / (self.slider.maximum() - self.slider.minimum())) * (
        self.maximum - self.minimum)
        self.label.setText(f"{self.name}\n{self.x:.5f}")

class SimpleToggle(QWidget):
    def __init__(self, parent=None):
        super(SimpleToggle, self).__init__(parent)
        self.setFixedSize(60, 30)
        self._margin = 3
        self._bg_color = QColor(100, 100, 100)
        self._circle_color = QColor(255, 255, 255)
        self._is_toggled = True
        self._update_position()

    def _update_position(self):
        if self._is_toggled:
            self._circle_position = self.width() - self.height() + self._margin
        else:
            self._circle_position = self._margin

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setPen(QPen(Qt.NoPen))
        
        # Background
        painter.setBrush(QBrush(self._bg_color))
        painter.drawRoundedRect(0, 0, self.width(), self.height(), 15, 15)

        # Circle
        painter.setBrush(QBrush(self._circle_color))
        circle_rect = QRectF(self._circle_position, self._margin, 
                             self.height() - 2 * self._margin, 
                             self.height() - 2 * self._margin)
        painter.drawEllipse(circle_rect)

    def mousePressEvent(self, event):
        self._is_toggled = not self._is_toggled
        self._update_position()
        self.update()

class Tuner(QWidget):
    def __init__(self, tunable, parent=None):
        super(Tuner, self).__init__(parent=parent)
        
        # self.TunableEntity = SimplePlane()
        self.TunableEntity = tunable
        self.TunableEntity.forward(self.TunableEntity.vars)
        self.UIsetup()
        self.TunerSetup()
        self.startFunction()
        
    def createButtons(self):
        
        btn1 = QPushButton('Stop', self)
        btn1.move(0, 0)
        btn1.clicked.connect(self.stopFunction)
        
        btn2 = QPushButton('Reset', self)
        btn2.move(100, 0)
        btn2.clicked.connect(self.resetFunction)
        
    def TunerSetup(self):
        self.gamma2 = 1.0
        
    def setInitialUIvals(self):
        for i in range(self.num_sliders):
            self.w[2*i+1].x = (self.ranges[i,0] + self.ranges[i,1])/2.
            
    def createTimers(self):
        self.timer.timeout.connect(self.AppUpdate)

    def startFunction(self):
        # Start the timer to call the connected function every 1000 milliseconds (1 second)
        self.timer.start(0)

    def resetFunction(self):
        self.TunableEntity.reset()
            
    def stopFunction(self):
        QApplication.quit()
        
    def layoutBuilder(self):
        self.mainLayout = QVBoxLayout(self)  # This is the main layout
        
        self.timer = QTimer(self)
        
        self.num_sliders = self.idx_ctrl_ranges.shape[0]
        self.w = []
        
        # I assume the number of toggles is the same as the number of sliders.
        
        for i in range(self.num_sliders):
            # Create a horizontal layout for each pair of slider and toggle.
            h_layout = QHBoxLayout()
            
            toggle = SimpleToggle()
            self.w.append(toggle)
            h_layout.addWidget(toggle)
            
            slider = Slider(self.ranges[i,0], self.ranges[i,1], self.name[i])
            
            self.w.append(slider)
            h_layout.addWidget(slider)
            
            # Add the horizontal layout (containing slider and toggle) to the main layout.
            self.mainLayout.addLayout(h_layout)
            
            
    def ForwardUpdate(self,s):
                
            for i in range(self.ctrl.shape[0]):
                    self.TunableEntity.vars[self.ctrl[i]] = torch.tensor(s[i],requires_grad=True)
            
            self.R = self.TunableEntity.forward(self.TunableEntity.vars)
        
    def NewtonUpdate(self,s):
            

        
        
        
        if self.ctrl.shape[0] == 0:
            for i in range(self.TunableEntity.vars.shape[0]):
                self.TunableEntity.vars[i] = torch.tensor(self.TunableEntity.vars[i],requires_grad=True)
            
        else:
            for i in range(self.ctrl.shape[0]):
                self.TunableEntity.vars[self.ctrl[i]] = torch.tensor(s[i],requires_grad=True)
        
        self.R = self.TunableEntity.forward(self.TunableEntity.vars)
        
     
        
        self.J = torch.autograd.grad(self.R, self.TunableEntity.vars,  torch.eye(self.R.numel()), is_grads_batched= True, allow_unused=False)[0].view(self.R.numel(),-1)
        
     
        
        # self.J_reduced = self.J[:,self.idx]
        self.J_reduced = self.J[:-1,self.idx]
        self.grads = self.J[-1,self.idx]
        self.vars_reduced = self.TunableEntity.vars[self.idx]
        self.n_reduced = self.vars_reduced.shape[0]
  
        # self.Im = torch.eye(self.R.shape[0])
        self.Im = torch.eye(self.R.shape[0]-1)
        self.J_Jt = torch.matmul(self.J_reduced,self.J_reduced.permute([1,0]))
        
        # self.M2 = torch.linalg.solve(self.J_Jt + self.gamma2*self.Im,self.R)
        self.M2 = torch.linalg.solve(self.J_Jt + self.gamma2*self.Im,self.R[0:-1].reshape(-1,1))

        self.M1 = torch.linalg.solve(self.J_Jt,self.J_reduced)
        self.Z = (torch.eye(self.n_reduced)- torch.matmul(self.J_reduced.permute([1,0]),self.M1))
        self.DX_P = torch.matmul(self.J_reduced.permute([1,0]),self.M2)
        
        
        
        
        self.DX_S2 = torch.matmul(self.Z, 0.1*self.grads.reshape(-1,1))
        
        self.DX_S = torch.matmul(self.Z, self.weights[self.idx].reshape(-1,1)*self.vars_reduced)


        self.TunableEntity.vars[self.idx] -= self.DX_P + self.DX_S + self.DX_S2
        
        
        
    def UIsetup(self):
        
        self.idx_ctrl_ranges_list = []
        self.name = []
        
        for i in range(len(self.TunableEntity.cfg)):
            self.idx_ctrl_ranges_list += [self.TunableEntity.cfg[i][1:]]
            self.name += [self.TunableEntity.cfg[i][0]]
            
        self.idx_ctrl_ranges = torch.tensor(self.idx_ctrl_ranges_list)
        self.weights = self.idx_ctrl_ranges[:,0].unsqueeze(dim=1)
        self.ranges = self.idx_ctrl_ranges[:,1:].detach().cpu().numpy()
        
        self.layoutBuilder()
        self.setInitialUIvals()    
        self.createTimers()
        self.createButtons()
        
        self.TunableEntity.plot_init()
        
    def UIhandler(self):
        
        self.s = []
        
        ctrl_bool = torch.zeros(self.num_sliders)
        for i in range(self.num_sliders):
            
           
            if self.w[2*i]._is_toggled:
               self.s += [self.w[2*i+1].x]
               ctrl_bool[i] = 1
            else:
                scaledVal = (self.TunableEntity.vars[i]-self.ranges[i,0])/(self.ranges[i,1] - self.ranges[i,0])
                self.w[2*i+1].slider.setValue(min(100,max(int(100*scaledVal),0)))
               
        self.ctrl = ctrl_bool.nonzero().reshape([-1])  
        self.idx = (1-ctrl_bool).nonzero().reshape([-1])
        
    
    def AppUpdate(self):

        now = time.perf_counter()
        
        self.UIhandler()
        
        if len(self.s) == self.num_sliders:
            self.ForwardUpdate(self.s)
         
        else:
          
            self.NewtonUpdate(self.s)
        self.TunableEntity.plot_update()
        
        print(self.TunableEntity.res)
 
        
        
        print(time.perf_counter() - now)