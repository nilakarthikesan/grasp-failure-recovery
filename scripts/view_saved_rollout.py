"""Interactive geometry replay of a saved MuJoCo model and full-qpos trace.

Run using mjpython on macOS. Space pauses/resumes; R restarts. This performs
forward kinematics for display and does not rerun inference or contact dynamics.
"""
import argparse
import time

import mujoco
import mujoco.viewer
import numpy as np


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scene',required=True)
    parser.add_argument('--trace',required=True)
    args=parser.parse_args()
    model=mujoco.MjModel.from_xml_path(args.scene)
    data=mujoco.MjData(model)
    saved=np.load(args.trace)
    if saved['qpos'].shape[1]!=model.nq:
        raise ValueError('Saved trace does not match the model')
    state={'paused':False,'index':0}

    def keyboard(key):
        if key==32:state['paused']=not state['paused']
        if key in (82,114):state['index']=0

    print('Saved rollout replay: Space pauses/resumes, R restarts. Close window to exit.',flush=True)
    with mujoco.viewer.launch_passive(model,data,key_callback=keyboard) as viewer:
        viewer.cam.lookat[:]=[0,0,.12]
        viewer.cam.distance=.7
        viewer.cam.azimuth=130
        viewer.cam.elevation=-25
        while viewer.is_running():
            start=time.monotonic()
            with viewer.lock():
                data.qpos[:]=saved['qpos'][state['index']]
                data.qvel[:]=0
                data.time=float(saved['time_s'][state['index']])
                mujoco.mj_forward(model,data)
            viewer.sync()
            if not state['paused']:state['index']=(state['index']+1)%len(saved['time_s'])
            time.sleep(max(0,.04-(time.monotonic()-start)))


if __name__=='__main__':main()
