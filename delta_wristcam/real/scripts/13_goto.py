"""Move the tool to X Y Z with the policy's orientation (gripper vertical, theta = 180, 0, 97.5) at 5 cm/s.
Asks for an empty Enter first; q quits. Keep a hand on the e-stop. Used to set the start pose / check the start view.

    python real/scripts/13_goto.py                    # stay at the current xyz, only rotate to the policy orientation
    python real/scripts/13_goto.py 0.247 0.052 0.416  # go to that point (lifts straight up first if coming from below)
"""
from __future__ import annotations

import sys
import threading
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from kinova_common import Connection, tool_pose  # noqa: E402

THETA = (180.0, 0.0, 97.5)
LO, HI = np.array([0.10, -0.25, 0.25]), np.array([0.55, 0.30, 0.55])


def main():
    from kortex_api.autogen.client_stubs.BaseClientRpc import BaseClient
    from kortex_api.autogen.client_stubs.BaseCyclicClientRpc import BaseCyclicClient
    from kortex_api.autogen.messages import Base_pb2
    with Connection() as router:
        base, cyc = BaseClient(router), BaseCyclicClient(router)
        mode = Base_pb2.ServoingModeInformation(); mode.servoing_mode = Base_pb2.SINGLE_LEVEL_SERVOING
        base.SetServoingMode(mode)
        state = Base_pb2.ArmState.Name(base.GetArmState().active_state)
        if state != 'ARMSTATE_SERVOING_READY':
            sys.exit(f'arm not ready ({state}): release manual control / clear faults first')
        cur = tool_pose(cyc)
        tgt = np.array([float(v) for v in sys.argv[1:4]]) if len(sys.argv) >= 4 else cur[:3].copy()
        if np.any(tgt < LO) or np.any(tgt > HI):
            sys.exit(f'target {tgt.round(3)} outside X[.10,.55] Y[-.25,.30] Z[.25,.55]')

        def move(xyz):
            act = Base_pb2.Action(); act.name = 'goto'
            p = act.reach_pose.target_pose
            p.x, p.y, p.z = map(float, xyz); p.theta_x, p.theta_y, p.theta_z = THETA
            act.reach_pose.constraint.speed.translation = 0.05
            act.reach_pose.constraint.speed.orientation = 20.0
            done = threading.Event(); res = {}

            def cb(n):
                if n.action_event in (Base_pb2.ACTION_END, Base_pb2.ACTION_ABORT):
                    res['ev'] = Base_pb2.ActionEvent.Name(n.action_event); done.set()
            h = base.OnNotificationActionTopic(cb, Base_pb2.NotificationOptions())
            try:
                base.ExecuteAction(act)
                if not done.wait(60):
                    base.Stop(); print('timeout -> Stop')
            except KeyboardInterrupt:
                base.Stop(); print('Ctrl-C -> Stop'); raise
            finally:
                base.Unsubscribe(h)
            print('result:', res.get('ev'))

        print(f'now  xyz ({cur[0]:.3f}, {cur[1]:.3f}, {cur[2]:.3f})  theta ({cur[3]:.1f}, {cur[4]:.1f}, {cur[5]:.1f})')
        steps = []
        if cur[2] < tgt[2] - 0.01 and np.linalg.norm(cur[:2] - tgt[:2]) > 0.01:
            steps.append(('lift straight up', np.array([cur[0], cur[1], tgt[2]])))
        steps.append(('go to', tgt))
        for what, xyz in steps:
            ans = input(f'[{what}] ({xyz[0]:.3f}, {xyz[1]:.3f}, {xyz[2]:.3f}) theta {THETA}  Enter = go | q = quit > ').strip().lower()
            if ans != '':
                sys.exit('stopped by operator (nothing sent)')
            move(xyz)
        p = tool_pose(cyc)
        print(f'done xyz ({p[0]:.3f}, {p[1]:.3f}, {p[2]:.3f})  theta ({p[3]:.1f}, {p[4]:.1f}, {p[5]:.1f})')


if __name__ == '__main__':
    main()
