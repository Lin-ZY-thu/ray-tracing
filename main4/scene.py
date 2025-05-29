from config import device
import torch
import textures

from objs import Sphere, Plane,Cylinder

def create_scene():
    scene = [
        Sphere(center=[0., 0.1, 0.4], radius=0.1, diffuse=[0.2, 0.8, 0.3], mirror=0.3),
        Plane(center=[0, -0.3, 0.4], normal=[0, 1, 0], diffuse=[0.6, 0.2, 0.0], mirror=0.4,texture=None),
        #Plane(center=[-0.5, 0.1, 0.4], normal=[0.8, 0.6, 0], diffuse=[0.6, 0.0, 0.5], mirror=0.4,texture=None),
        #Plane(center=[0.5, 0.1, 0.4], normal=[0.8, -0.6, 0], diffuse=[0.6, 0.0, 0.5], mirror=0.4,texture=None),
        Cylinder(center=[0,0.1,0.0],normal=[0,0.8,0.6],radius=0.03,height=0.2,diffuse=[0.7,0,0.5])
    ]
    return scene


