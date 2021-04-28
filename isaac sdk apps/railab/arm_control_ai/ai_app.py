'''
Copyright (c) 2021, Railab srl. All rights reserved.
'''

import numpy as np
import argparse
import json

from isaac import Application, Cask, Composite
from packages.pyalice import Codelet

from jetcam.usb_camera import USBCamera
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
import cv2
import PIL.Image
import os

import threading
import time

import math

'''
Moves a robot arm based on joint waypoints according to gesture recognition.
It accepts 2 different robotic arms:
* Universal Robots UR10 --> can be only moved in simulation (Omniverse Kit Isaac Sim)
* Denso Cobotta --> can be only moved in real

Fingers gestures are as follows:
* gesture "1" (middle and index fingers open) --> move the robot to home position
* gesture "2" (middle and index fingers close and swipe towards right) --> move the robot to position#1
* gesture "3" (middle and index fingers close and swipe towards left) --> move the robot to position#2
* gesture "4" (middle and index fingers close and swipe towards up) --> move the robot to position#3
* gesture "5" (middle and index fingers close and swipe towards down) --> move the robot to position#4

'''

# A Python codelet for a robotic arm control based on hand gesture recognition
#
# Fingers gesture is classified using a resnet18 neural network properly trained.
# The output of the neural network is used to command a pre-stored position
# (in joint space) to the robotic arm
class ImageRegression(Codelet):
    def start(self):
        # This part will be run once in the beginning of the program
        
        # Load the regression network
        self._model_path = self.config.model_path
        self._device = torch.device('cuda')
        self._output_dim = 4 # x, y coordinate for middle finger and index finger

        # RESNET 18
        self._model = torchvision.models.resnet18(pretrained=True)
        self._model.fc = torch.nn.Linear(512, self._output_dim)
        self._model = self._model.to(self._device)
        self._model.load_state_dict(torch.load(self._model_path))
        
        self._mean = torch.Tensor([0.485, 0.456, 0.406]).cuda()
        self._std = torch.Tensor([0.229, 0.224, 0.225]).cuda()

        # Create video source
        # NOTE: confirm the capture_device number
        self._camera = USBCamera(width=224, height=224, capture_device=0)
        self._camera.running = True

        # Codelet configuration      
             
        self._measure = "position"
        self._entities = self.config.joints
        num_entities = len(self._entities)
        self._quantities = [[x, self._measure, 1] for x in self._entities]
        print(self._quantities)
        self._joint_targets = None
        
        self._centroid_x_prev = None
        self._centroid_y_prev = None
        self._open_threshold = 30
        self._move_threshold = 15
        self._move_right = False
        self._move_left = False
        self._move_up = False
        self._move_down = False
        self._movement_x = 0
        self._movement_y = 0
        
        # Robotic arm position definition
        # PLEASE NOTE that UR10 and Cobotta robots differ on the setting of 0 value
        # for joint number 2 !!
        if self.config.arm == "ur10":
            self._position_1 = np.array([1.5, -1.5, 1.5, 0, 0, 0])
            self._position_2 = np.array([-1.5, -1.5, 1.5, 0, 0, 0])
            self._position_3 = np.array([0, -1.5, 0, 0, 0, 0])
            self._position_4 = np.array([1.5, -1.5, 0, 0, 0, 0])
            self._position_home = np.array([-1.5, -1.5, 0, 0, 0, 0])
        elif self.config.arm == "denso_cobotta":
            self._position_1 = np.array([1.5, 0, 1.5, 0, 0, 0])
            self._position_2 = np.array([-1.5, 0.7, 1.5, 0, 0, 0])
            self._position_3 = np.array([0, 0, 1.5, 0, 0, 0])
            self._position_4 = np.array([1.5, 0.7, 1.5, 0, 0, 0])
            self._position_home = np.array([-1.5, 0, 1.5, 0, 0, 0])
        else:
            raise ValueError("Robot model not recognized !!")

        # Input and output messages for the Codelet
        self.rx = self.isaac_proto_rx("CompositeProto", "state")
        self.tx = self.isaac_proto_tx("CompositeProto", "command")

        # Print some information
        print("Starting robot control node ... ")

        # Tick periodically, on every message
        self.tick_periodically(0.01)

    def tick(self):
        # This part will be run at every tick. We are ticking periodically.
        # Capture next image
        img = self._camera.value

        # Preprocess the image
        preprocessed = self.preprocess(img)
        
        # Find the position of the 2 fingers
        output = self._model(preprocessed).detach().cpu().numpy().flatten()
        middle_finger_x = output[0]
        middle_finger_y = output[1]
        index_finger_x = output[2]
        index_finger_y = output[3]
                
        middle_finger_x = int(self._camera.width * (middle_finger_x / 2.0 + 0.5))
        middle_finger_y = int(self._camera.height * (middle_finger_y / 2.0 + 0.5))
        index_finger_x = int(self._camera.width * (index_finger_x / 2.0 + 0.5))
        index_finger_y = int(self._camera.height * (index_finger_y / 2.0 + 0.5))
        
        centroid_x = (middle_finger_x + index_finger_x) / 2.0
        centroid_y = (middle_finger_y + index_finger_y) / 2.0
        fingers_distance = math.sqrt((middle_finger_x - index_finger_x)**2 + (middle_finger_y - index_finger_y)**2)

        if ((self._centroid_x_prev == None) or (self._centroid_y_prev == None)):
            self._centroid_x_prev = centroid_x
            self._centroid_y_prev = centroid_y

        increment_x = centroid_x - self._centroid_x_prev
        increment_y = centroid_y - self._centroid_y_prev

        self._movement_x = self._movement_x + increment_x
        self._movement_y = self._movement_y + increment_y
        
        self._centroid_x_prev = centroid_x
        self._centroid_y_prev = centroid_y

        if ((fingers_distance >= self._open_threshold) or ((abs(increment_x) <= 2.0) and (abs(increment_y) <= 2.0))):
            self._movement_x = 0
            self._movement_y = 0
                                              
        if(abs(self._movement_x) >= self._move_threshold):
            if(self._movement_x >= 0):
                self._move_right = True
                self._move_left = False
                self._move_up = False
                self._move_down = False
            else:
                self._move_right = False
                self._move_left = True
                self._move_up = False
                self._move_down = False
        elif(abs(self._movement_y) >= self._move_threshold):
            if(self._movement_y >= 0):
                self._move_right = False
                self._move_left = False
                self._move_up = True
                self._move_down = False
            else:
                self._move_right = False
                self._move_left = False
                self._move_up = False
                self._move_down = True
   
        # if arm state data are not available, do nothing (return)
        state_msg = self.rx.message
        if state_msg is None:
            return
            
        # read arm state message received
        state_values = Composite.parse_composite_message(state_msg, self._quantities)
        if len(self._entities) != len(state_values):
            raise ValueError("Size of state doesn't match number of joints")
         
        if (fingers_distance >= self._open_threshold):
            # gesture "open fingers" --> robot going home
            self._move_right = False
            self._move_left = False
            self._move_up = False
            self._move_down = False
            self._movement_x = 0
            self._movement_y = 0
            if (not np.all(self._joint_targets == self._position_home)):
                print("Robot going home... ")
            self._joint_targets = self._position_home
        elif (self._move_right):
            # gesture "swipe right"
            if (not np.all(self._joint_targets == self._position_1)):
                print("Moving robot to position #1... ")
            self._joint_targets = self._position_1
        elif (self._move_left):
            # gesture "swipe left"
            if (not np.all(self._joint_targets == self._position_2)):
                print("Moving robot to position #2... ")
            self._joint_targets = self._position_2
        elif (self._move_up):
            # gesture "swipe up"
            if (not np.all(self._joint_targets == self._position_3)):
                print("Moving robot to position #3... ")
            self._joint_targets = self._position_3
        elif (self._move_down):
            # gesture "swipe down"
            if (not np.all(self._joint_targets == self._position_4)):
                print("Moving robot to position #4... ")
            self._joint_targets = self._position_4
        else:
            # no gesture detected
            self._joint_targets = None
                
        # if you want to print out debug information, just comment out the following lines
        # print(" Detected fingers positions : middle x {:05.2f} --- middle y {:05.2f} --- index x {:05.2f} --- index y {:05.2f}".format(middle_finger_x, middle_finger_y, index_finger_x, index_finger_y))
        # print(" Detected fingers positions : centroid x {:05.2f} --- centroid y {:05.2f} --- distance {:05.2f}".format(centroid_x, centroid_y, fingers_distance))
        # if self._joint_targets is not None:
        # if joint_targets is not None:
            # print(" Joint targets : ", *self._joint_targets, sep=' ; ')
        # if state_values is not None:
            # print(" Arm state values : ", *state_values, sep=' ; ')

        # send message with robot target position
        if (self._joint_targets is not None):
            cmd_values = np.array([x for x in self._joint_targets.tolist()], dtype=np.float64)
            self.tx._msg = Composite.create_composite_message(self._quantities, cmd_values)
            if (self.tx._msg is not None):
                self.tx.publish()


    def preprocess(self, image):
        device = torch.device('cuda')
        image = PIL.Image.fromarray(image)
        image = transforms.functional.to_tensor(image).to(device)
        image.sub_(self._mean[:, None, None]).div_(self._std[:, None, None])
        return image[None, ...]

    
if __name__ == '__main__':
    # Parse arguments
    parser = argparse.ArgumentParser(description="AI control Demo")
    parser.add_argument("--arm", help="Type of arm used.", choices=["ur10", "denso_cobotta"], default="ur10")
    parser.add_argument("--ip", help="IP address of the real robot (only for denso_cobotta robot arm).", default="10.22.255.13")
    parser.add_argument("--model", help="Path top trained model.", default='/home/railab/nvdli-data/regression/fingers_control_model.pth')
    args = parser.parse_args()

    # get kinematic file and joints
    kinematic_file = "apps/assets/kinematic_trees/{}.kinematic.json".format(args.arm)
    joints = []
    with open(kinematic_file, 'r') as fd:
        kt = json.load(fd)
        for link in kt['links']:
            if 'motor' in link and link['motor']['type'] != 'constant':
                joints.append(link['name'])

    # Create and start the app
    app = Application(name="Arm_Control")

    regression_network = app.add("AI_Regression").add(ImageRegression)
    regression_network.config.kinematic_file = kinematic_file
    regression_network.config.arm = args.arm
    regression_network.config.joints = joints
    regression_network.config.model_path = args.model
  
    app.load("packages/planner/apps/multi_joint_lqr_control.subgraph.json", prefix="lqr")
    # Load multi joint lqr control subgraph and configure parameters
    lqr_interface = app.nodes["lqr.subgraph"]["interface"]
    kinematic_tree = app.nodes["lqr.kinematic_tree"]["KinematicTree"]
    lqr_planner = app.nodes["lqr.local_plan"]["MultiJointLqrPlanner"]

    app.nodes["lqr.controller"]["MultiJointController"].config.control_mode = "position"

    kinematic_tree.config.kinematic_file = kinematic_file
    lqr_planner.config.speed_min = [-0.35] * len(joints) 
    lqr_planner.config.speed_max = [0.35] * len(joints)
    lqr_planner.config.acceleration_min = [-0.35] * len(joints)
    lqr_planner.config.acceleration_max = [0.35] * len(joints)
    
    if args.arm == "denso_cobotta":
        lqr_interface.config.tick_period = "8ms"
        lqr_planner.config.tick_period = "8ms"

        # Load denso cobotta driver codelet
        app.load_module("denso_robot")
        driver = app.add("driver").add(app.registry.isaac.denso_robot.DensoRobot)
      
        # Configure edges
        app.connect(driver, "arm_state", lqr_interface, "joint_state")
        app.connect(lqr_interface, "joint_command", driver, "arm_command")
     
        app.connect(driver, "arm_state", regression_network, "state")
        app.connect(regression_network, "command", lqr_interface, "joint_target")
        
        # Configure parameters
        driver.config.kinematic_tree = "lqr.kinematic_tree"
        driver.config.denso_robot_ip = args.ip
        driver.config.tick_period = "8ms"
        driver.config.control_mode = "joint position"

    else:
        # Load sim tsubgraph for tcp connection with Isaac-Sim
        app.load("packages/navsim/apps/navsim_tcp.subgraph.json", prefix="simulation")
        sim_in = app.nodes["simulation.interface"]["input"]
        sim_out = app.nodes["simulation.interface"]["output"]

        app.connect(sim_out, "joint_state", lqr_interface, "joint_state")
        app.connect(lqr_interface, "joint_command", sim_in, "joint_position")

        app.connect(sim_out, "joint_state", regression_network, "state")
        app.connect(regression_network, "command", lqr_interface, "joint_target")

    viewers = app.add("viewers")

    # Run app
    app.run()
    print("exiting ...")
    os._exit(00)
