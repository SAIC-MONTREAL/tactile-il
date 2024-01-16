# built-ins
import sys
import os
import queue

# ros
import rospy
import tf2_ros
import tf.transformations as tf_trans
from std_msgs.msg import Float64MultiArray
from nav_msgs.msg import Path

# other
from PyQt5 import QtWidgets, QtGui, QtCore, uic
import numpy as np
from numpy.linalg import norm

# own
import ros_np_tools as rnt

# for type hint, otherwise creates circular import
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from thing_gym_ros.envs.thing_ros_generic import ThingRosEnv

path = os.path.dirname(os.path.abspath(__file__))
MainWindowUI, MainWindowBase = uic.loadUiType(os.path.join(path, 'thing_ros_generic.ui'))


class ThingRosGenericGui(MainWindowBase, MainWindowUI):
    img_ready_sig = QtCore.pyqtSignal(object)

    def __init__(self, env_to_gui_q, gui_to_env_q, env_obj):
        super().__init__()
        self.setupUi(self)

        # queues
        self.env_to_gui_q = env_to_gui_q
        self.gui_to_env_q = gui_to_env_q

        # play pause button
        self.play_pause_env.toggled.connect(self.handle_play_pause_env)
        self.play_pause_env.setChecked(True)

        self.testing_group_box.hide()

        # ros tf
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        self.tf_static_broadcaster = tf2_ros.StaticTransformBroadcaster()

        # frames to keep track of -- mirrors main env class
        self.tf_odom_base = rnt.tf_msg.TransformWithUpdate(self.tf_buffer, 'odom', 'base_link')
        self.tf_base_tool = rnt.tf_msg.TransformWithUpdate(self.tf_buffer, 'base_link', 'thing_tool')
        self.tf_odom_tool = rnt.tf_msg.TransformWithUpdate(self.tf_buffer, 'odom', 'thing_tool')
        self.tf_odom_cam = rnt.tf_msg.TransformWithUpdate(self.tf_buffer, 'odom', 'camera_link')
        tf_base_cam = rnt.tf_msg.TransformWithUpdate(self.tf_buffer, 'base_link', 'camera_link')
        self._base_cam_mat = tf_base_cam.as_mat()

        # ros publishers
        self.pub_servo = rospy.Publisher('/servo/command', Float64MultiArray, queue_size=1)
        self.pub_base_traj = rospy.Publisher('goal_ridgeback', Path, queue_size=1)
        self.pub_arm_traj = rospy.Publisher('goal_ur10', Path, queue_size=1)

        # set initial values
        self._moving_base = env_obj._moving_base
        if self._moving_base:
            self._main_odom_base_mat = env_obj._main_odom_base_mat
            self._main_odom_base_pos_eul = env_obj._main_odom_base_pos_eul
            self._reset_base_vel_trans = env_obj._reset_base_vel_trans
            self._reset_base_vel_rot = env_obj._reset_base_vel_rot
            self.workspace_center_tf_msg = env_obj.workspace_center_tf_msg
            self.ep_odom_base_mat = env_obj.ep_odom_base_mat
            self.cam_workspace_dist.setValue(env_obj._cam_workspace_distance)
            self._cam_forward_axis = env_obj._cam_forward_axis
        self._time_between_poses_tc = env_obj._time_between_poses_tc
        self.sim = env_obj.sim

        # base buttons
        if self._moving_base:
            self.T_main_base_before_adjustment = None
            self.move_to_main_base_pose.clicked.connect(self.handle_move_to_main_base_pose)
            self.move_to_prev_base_pose.clicked.connect(self.handle_move_to_prev_base_pose)
            self.main_base_to_current.clicked.connect(self.handle_main_base_to_current)
            self.x_base_main_spin_box.setValue(self._main_odom_base_pos_eul[0])
            self.y_base_main_spin_box.setValue(self._main_odom_base_pos_eul[1])
            self.theta_base_main_spin_box.setValue(self._main_odom_base_pos_eul[5])
            self.x_base_main_spin_box.valueChanged.connect(self.handle_main_base_spin_boxes)
            self.y_base_main_spin_box.valueChanged.connect(self.handle_main_base_spin_boxes)
            self.theta_base_main_spin_box.valueChanged.connect(self.handle_main_base_spin_boxes)

        self.set_base_spinbox_status(False)
        self.set_base_adjustment_status(False)

        # cam workspace dist
        self.cam_workspace_dist.valueChanged.connect(self.handle_cam_workspace_dist)

        # self.randomize_base_pose.clicked.connect(self.handle_randomize_base_pose)
        # self.test_theta_high.clicked.connect(self.handle_test_theta_high)
        # self.test_theta_low.clicked.connect(self.handle_test_theta_low)
        # self.base_trans_vel.setValue(self.env_obj._reset_base_vel_trans)
        # self.base_rot_vel.setValue(self.env_obj._reset_base_vel_rot)

        # images
        self.img_ready_sig.connect(self.set_rgb_img)

        self.update_timer = QtCore.QTimer()
        self.update_timer.timeout.connect(self.gui_update)
        self.update_timer.start(100)

    def gui_update(self):
        # loop though q to only take latest data
        q_empty = False
        gui_data_dict = None
        latest_img = None
        while not q_empty:
            try:
                gui_data_dict = self.env_to_gui_q.get_nowait()
                if 'env_img' in gui_data_dict:
                    latest_img = gui_data_dict['env_img']
                for k in gui_data_dict:
                    if k != 'env_img':
                        setattr(self, k, gui_data_dict[k])

            except queue.Empty:
                q_empty = True
        if latest_img is not None:
            self.img_ready_sig.emit(latest_img)

        # update base pose
        self.tf_odom_base.update()
        odom_base_pos_eul = self.tf_odom_base.as_pos_eul()
        self.x_base_actual_line.setText("%3.4f" % odom_base_pos_eul[0])
        self.y_base_actual_line.setText("%3.4f" % odom_base_pos_eul[1])
        self.theta_base_actual_line.setText("%3.4f" % odom_base_pos_eul[5])

        # update statuses for setting main base pose
        if self.play_pause_env.isChecked():
            self.set_base_spinbox_status(False)
            self.set_base_adjustment_status(False)
        else:
            self.set_base_spinbox_status(True)
            self.set_base_adjustment_status(True)
        if self._moving_base:
            dist_base_to_init = norm(odom_base_pos_eul[:2] - self._main_odom_base_pos_eul[:2])
            rot_dist_base_to_main = np.abs(odom_base_pos_eul[5] - self._main_odom_base_pos_eul[5])
            if dist_base_to_init < 0.02 and rot_dist_base_to_main < 0.02 and not self.play_pause_env.isChecked():
                self.cam_workspace_dist.setEnabled(True)
                self.base_at_main_label.setText("YES")
                self.base_at_main_label.setStyleSheet("QLabel { background-color : green; }")
                if self.T_main_base_before_adjustment is not None:
                    self.move_to_prev_base_pose.setEnabled(True)
            else:
                self.cam_workspace_dist.setEnabled(False)
                self.base_at_main_label.setText("NO")
                self.base_at_main_label.setStyleSheet("QLabel { background-color : red; }")
                self.move_to_prev_base_pose.setEnabled(False)

    def handle_play_pause_env(self, checked):
        if checked:
            self.play_pause_env.setStyleSheet("QPushButton { background-color : green; color : black; }")
            self.gui_to_env_q.put(dict(play_pause_env=True))
        else:
            self.play_pause_env.setStyleSheet("QPushButton { background-color : white; color : black; }")
            self.gui_to_env_q.put(dict(play_pause_env=False))

    def set_base_adjustment_status(self, allow):
        self.move_to_main_base_pose.setEnabled(allow)

    def set_base_spinbox_status(self, allow):
        self.x_base_main_spin_box.setReadOnly(not allow)
        self.y_base_main_spin_box.setReadOnly(not allow)
        self.theta_base_main_spin_box.setReadOnly(not allow)

    def handle_move_to_main_base_pose(self):
        """ Move the robot base to the current "main" base pose. """
        self.tf_odom_base.update()
        self.tf_base_tool.update()

        if self.base_at_main_label.text() == 'NO':
            self.T_main_base_before_adjustment = \
                rnt.tf_mat.invert_transform(self.tf_odom_base.as_mat()).dot(self._main_odom_base_mat)

        self.ep_odom_base_mat = self._main_odom_base_mat
        self.gui_to_env_q.put(dict(ep_odom_base_mat=self.ep_odom_base_mat))

        base_path = rnt.path.generate_smooth_path(
            first_frame=self.tf_odom_base.transform,
            last_frame=rnt.tf_msg.mat_to_tf_msg(self._main_odom_base_mat),
            trans_velocity=self._reset_base_vel_trans,
            rot_velocity=self._reset_base_vel_rot,
            time_btwn_poses=self._time_between_poses_tc )

        # ensure that arm always maintains same pose relative to base
        arm_path = rnt.thing.gen_no_movement_arm_path(base_path, self.tf_base_tool.as_mat())

        self.statusBar().showMessage("Publishing robot trajectory to return base to main pose.")
        self.pub_arm_traj.publish(arm_path)
        self.pub_base_traj.publish(base_path)

    def handle_move_to_prev_base_pose(self):
        """ After fixing the "main" base pose, move the base back to the original relative pose. """
        self.tf_odom_base.update()
        self.tf_base_tool.update()

        T_prev_base_new_ref = self.tf_odom_base.as_mat().dot(self.T_main_base_before_adjustment)
        self.ep_odom_base_mat = T_prev_base_new_ref

        self.gui_to_env_q.put(dict(ep_odom_base_mat=self.ep_odom_base_mat))

        base_path = rnt.path.generate_smooth_path(
            first_frame=self.tf_odom_base.transform,
            last_frame=rnt.tf_msg.mat_to_tf_msg(T_prev_base_new_ref),
            trans_velocity=self._reset_base_vel_trans,
            rot_velocity=self._reset_base_vel_rot,
            time_btwn_poses=self._time_between_poses_tc)
        arm_path = rnt.thing.gen_no_movement_arm_path(base_path, self.tf_base_tool.as_mat())
        self.statusBar().showMessage("Publishing robot trajectory to return base to prev pose.")
        self.pub_arm_traj.publish(arm_path)
        self.pub_base_traj.publish(base_path)

    def handle_main_base_spin_boxes(self):
        T_prev_base_ep = rnt.tf_mat.invert_transform(self.ep_odom_base_mat).dot(self._main_odom_base_mat)
        self._main_odom_base_pos_eul[0] = self.x_base_main_spin_box.value()
        self._main_odom_base_pos_eul[1] = self.y_base_main_spin_box.value()
        self._main_odom_base_pos_eul[5] = self.theta_base_main_spin_box.value()
        self._main_odom_base_mat = rnt.tf_mat.pos_eul_to_mat(self._main_odom_base_pos_eul)
        self.ep_odom_base_mat = self._main_odom_base_mat.dot(T_prev_base_ep)

        self.handle_cam_workspace_dist()
        self.gui_to_env_q.put(dict(
            _main_odom_base_pos_eul=self._main_odom_base_pos_eul,
            _main_odom_base_mat=self._main_odom_base_mat,
            ep_odom_base_mat=self.ep_odom_base_mat
        ))

    def handle_cam_workspace_dist(self):
        # self.tf_odom_cam.update()

        T_update = np.eye(4)

        update_axis_dict = dict(x=0, y=1, z=2)
        T_update[update_axis_dict[self._cam_forward_axis], 3] = self.cam_workspace_dist.value()
        # T_workspace_center = self.tf_odom_cam.as_mat().dot(T_update)

        T_workspace_center = self._main_odom_base_mat.dot(self._base_cam_mat).dot(T_update)
        self.workspace_center_tf_msg.transform = rnt.tf_msg.mat_to_tf_msg(T_workspace_center)
        self.workspace_center_tf_msg.header.stamp = rospy.Time.now()
        self.tf_static_broadcaster.sendTransform(self.workspace_center_tf_msg)

        self.gui_to_env_q.put(dict(
            _workspace_center_tf_msg=self.workspace_center_tf_msg,
            _cam_workspace_distance=self.cam_workspace_dist.value()))

    def handle_main_base_to_current(self):
        self.x_base_main_spin_box.setValue(float(self.x_base_actual_line.text()))
        self.y_base_main_spin_box.setValue(float(self.y_base_actual_line.text()))
        self.z_base_main_spin_box.setValue(float(self.z_base_actual_line.text()))

    def closeEvent(self, e):
        self.update_timer.stop()
        self.gui_to_env_q.put('close')
        print('Gui closing.')

        e.accept()

    @QtCore.pyqtSlot(object)
    def set_rgb_img(self, image):
        height, width, channel = image.shape
        bytes_per_line = 3 * width
        q_img = QtGui.QImage(image, width, height, bytes_per_line, QtGui.QImage.Format_RGB888)
        img = QtGui.QPixmap(q_img)
        self.rgb_img_label.setPixmap(img)