import time
from pathlib import Path

import glfw
import mujoco as mj
import mujoco.viewer

from utils import _parse_float_list, empty, mjx_cable


class DLOSim:
    def __init__(self):
        self.m, self.d = self._build_model()

        self._post_init()

    def _build_model(self) -> tuple[mj.MjModel, mj.MjData]:

        scene = empty()

        twist = 60000.0 * 5
        bend = 10000000.0 * 5

        cable = mjx_cable(
            twist=twist,
            bend=bend,
            size=0.4,
            initial="free",
        )

        _c = scene.worldbody.add_camera(
            name="cam",
            pos=[1.2, 0.234, 0.156],
            xyaxes=[-0.037, 0.999, 0.000, -0.001, -0.000, 1.000],
            resolution=[640, 480],
        )

        gripper_spawn = [0.0, 0.4, 0.4]

        mocap = scene.worldbody.add_body(
            name="mocap", mocap=True, pos=gripper_spawn, euler=[0, 0, 0]
        )
        mocap.add_geom(
            name="mocap",
            type=mj.mjtGeom.mjGEOM_BOX,
            size=[0.02, 0.02, 0.02],
            contype=0,
            conaffinity=0,
            # rgba=[1, 1, 1, 0.2],
        )
        s_mocap_1 = mocap.add_site(name="mocap_site_1", pos=[0, -0.01, 0])
        s_mocap_2 = mocap.add_site(name="mocap_site_2", pos=[0, 0.01, 0])
        s_mocap_3 = mocap.add_site(name="mocap_site_3", pos=[0.01, 0, 0])
        s_mocap_4 = mocap.add_site(name="mocap_site_4", pos=[-0.01, 0, 0])

        cable_root = cable.worldbody.first_body()
        scene.worldbody.add_frame(pos=[0, 0.2, 0.1], euler=[0, 0, 0]).attach_body(
            cable_root
        )

        b = None
        b0 = cable_root
        for i in range(10):
            b = b0.first_body()
            b0 = b

        b.add_site(
            name="cable_weld_site_1",
            pos=[0.0, -0.01, 0.0],  # adjust if you want an offset on that segment
            euler=[0.0, 0.0, 1.57],
            group=1,
            rgba=[0, 1, 1, 1],
        )
        b.add_site(
            name="cable_weld_site_2",
            pos=[0.0, 0.01, 0.0],  # adjust if you want an offset on that segment
            euler=[0.0, 0.0, 1.57],
            group=1,
            rgba=[0, 1, 1, 1],
        )
        b.add_site(
            name="cable_weld_site_3",
            pos=[0.01, 0, 0.0],  # adjust if you want an offset on that segment
            euler=[0.0, 0.0, 1.57],
            group=1,
            rgba=[0, 1, 1, 1],
        )
        b.add_site(
            name="cable_weld_site_4",
            pos=[-0.01, 0, 0.0],  # adjust if you want an offset on that segment
            euler=[0.0, 0.0, 1.57],
            group=1,
            rgba=[0, 1, 1, 1],
        )

        scene.add_equality(
            name="mocap_cable_weld_1",
            type=mj.mjtEq.mjEQ_WELD,
            objtype=mj.mjtObj.mjOBJ_SITE,
            name1="mocap_site_1",
            name2="cable_weld_site_1",
            # Keep the current relative pose at creation time.
            solref=[0.00000001, 1],
            solimp=[0.95, 0.99, 0.001, 0.1, 6],
        )
        scene.add_equality(
            name="mocap_cable_weld_2",
            type=mj.mjtEq.mjEQ_WELD,
            objtype=mj.mjtObj.mjOBJ_SITE,
            name1="mocap_site_2",
            name2="cable_weld_site_2",
            # Keep the current relative pose at creation time.
            solref=[0.00000001, 1],
            solimp=[0.95, 0.99, 0.001, 0.1, 6],
        )
        scene.add_equality(
            name="mocap_cable_weld_3",
            type=mj.mjtEq.mjEQ_WELD,
            objtype=mj.mjtObj.mjOBJ_SITE,
            name1="mocap_site_3",
            name2="cable_weld_site_3",
            # Keep the current relative pose at creation time.
            solref=[0.00000001, 1],
            solimp=[0.95, 0.99, 0.001, 0.1, 6],
        )
        scene.add_equality(
            name="mocap_cable_weld_4",
            type=mj.mjtEq.mjEQ_WELD,
            objtype=mj.mjtObj.mjOBJ_SITE,
            name1="mocap_site_4",
            name2="cable_weld_site_4",
            # Keep the current relative pose at creation time.
            solref=[0.00000001, 1],
            solimp=[0.95, 0.99, 0.001, 0.1, 6],
        )

        self._init_key_name = "init"

        scene.add_key(
            name=self._init_key_name,
            time=0,
            qpos=_parse_float_list(
                "-0.00178983 0.265561 0.0928174 0.755305 0.655325 0.00531633 0.00594599 0.999999 -0.0011596 -1.50973e-07 -2.58863e-05 0.999989 -0.00468051 -2.6357e-06 -0.000102329 0.99994 -0.0109141 -1.68553e-05 -0.000250369 0.999782 -0.0208733 -6.92685e-05 -0.000541302 0.999326 -0.0367043 -0.000226958 -0.0011333 0.998046 -0.0624409 -0.00066422 -0.00232277 0.994444 -0.105151 -0.00184404 -0.00453926 0.98436 -0.175926 -0.00491532 -0.00784959 0.957916 -0.28665 -0.0119359 -0.00926809"
            ),
            qvel=_parse_float_list(
                "-0.000999776 2.95313e-05 -4.62839e-05 -0.000177692 -0.0170984 -0.00223061 8.27751e-06 8.36807e-08 2.11904e-05 2.06721e-05 1.48882e-06 8.25273e-05 3.06709e-05 9.66566e-06 0.000195584 4.77091e-05 3.86553e-05 0.000403653 9.17848e-05 0.000117716 0.000802283 0.000176729 0.000305613 0.00156849 0.000261393 0.000725882 0.00297842 0.000114886 0.001673 0.00529442 -0.000944918 0.00409848 0.00808618"
            ),
            mpos=_parse_float_list("0 0.4 0.4"),
            mquat=_parse_float_list("1 0 0 0"),
        )

        m = scene.compile()
        d = mj.MjData(m)

        return m, d

    def _post_init(self) -> None:
        # initialize the simulation in a stable state
        self._init_key_id = self.m.key(self._init_key_name).id
        mj.mj_resetDataKeyframe(self.m, self.d, self._init_key_id)
        mj.mj_forward(self.m, self.d)

    def cb(self, key: int) -> None:
        if key is glfw.KEY_SPACE:
            print("Here you can but some of your code...")

    def run(self):

        with mujoco.viewer.launch_passive(
            model=self.m,
            data=self.d,
            key_callback=self.cb,
            show_left_ui=True,
            show_right_ui=True,
        ) as viewer:
            while viewer.is_running():
                step_start = time.time()

                mj.mj_step(self.m, self.d)
                viewer.sync()

                # Rudimentary time keeping, will drift relative to wall clock.
                time_until_next_step = self.m.opt.timestep - (time.time() - step_start)
                if time_until_next_step > 0:
                    time.sleep(time_until_next_step)


if __name__ == "__main__":
    sim = DLOSim()
    sim.run()
