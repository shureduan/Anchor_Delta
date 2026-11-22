"""Move the arm to its built-in "Retract" pose (before power-off). Waits for Enter; keep a hand on the e-stop.

    python real/scripts/12_retract.py            # or:  --pose Home
"""
import argparse
import sys
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from kinova_common import Connection, tool_pose  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pose', default='Retract')
    a = ap.parse_args()
    from kortex_api.autogen.client_stubs.BaseClientRpc import BaseClient
    from kortex_api.autogen.client_stubs.BaseCyclicClientRpc import BaseCyclicClient
    from kortex_api.autogen.messages import Base_pb2
    with Connection() as router:
        base, cyc = BaseClient(router), BaseCyclicClient(router)
        state = Base_pb2.ArmState.Name(base.GetArmState().active_state)
        print('arm state:', state)
        if state != 'ARMSTATE_SERVOING_READY':
            sys.exit('arm not ready (manual control active, fault or e-stop)')
        mode = Base_pb2.ServoingModeInformation(); mode.servoing_mode = Base_pb2.SINGLE_LEVEL_SERVOING
        base.SetServoingMode(mode)
        req = Base_pb2.RequestedActionType(); req.action_type = Base_pb2.REACH_JOINT_ANGLES
        actions = {x.name: x.handle for x in base.ReadAllActions(req).action_list}
        if a.pose not in actions:
            sys.exit(f'no stored action "{a.pose}"; available: {sorted(actions)}')
        p = tool_pose(cyc)
        if input(f'move from tool ({p[0]:.3f}, {p[1]:.3f}, {p[2]:.3f}) to the stored "{a.pose}" pose (joint motion, the arm '
                 f'folds over its base -- clear the area). Enter (empty) = go, anything else = abort > ').strip() != '':
            sys.exit('aborted')
        done = threading.Event(); res = {}

        def cb(n):
            if n.action_event in (Base_pb2.ACTION_END, Base_pb2.ACTION_ABORT):
                res['ev'] = Base_pb2.ActionEvent.Name(n.action_event); done.set()
        h = base.OnNotificationActionTopic(cb, Base_pb2.NotificationOptions())
        try:
            base.ExecuteActionFromReference(actions[a.pose])
            if not done.wait(60):
                base.Stop(); print('timeout -> Stop')
        except KeyboardInterrupt:
            base.Stop(); print('Ctrl-C -> Stop'); raise
        finally:
            base.Unsubscribe(h)
        print(res.get('ev'), 'tool now', tool_pose(cyc)[:3].round(3))


if __name__ == '__main__':
    main()
