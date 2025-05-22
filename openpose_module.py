import sys
sys.path.append('/home/ical/PenguinChuan/openpose/build/python/')
from openpose import pyopenpose as op

def initialize_openpose():
    params = dict()
    params["model_folder"] = "/home/ical/PenguinChuan/openpose/models"
    opWrapper = op.WrapperPython()
    opWrapper.configure(params)
    opWrapper.start()
    return opWrapper

def get_keypoints_from_frame(opWrapper, frame):
    datum = op.Datum()
    datum.cvInputData = frame
    opWrapper.emplaceAndPop(op.VectorDatum([datum]))
    return datum.poseKeypoints, datum.cvOutputData
