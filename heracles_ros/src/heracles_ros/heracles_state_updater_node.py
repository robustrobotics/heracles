#!/usr/bin/env python3
import rclpy
import tf2_ros
from dsg_updater.dsg_state_utils import robot_hold_obj, robot_unhold_obj, set_obj_center
from geometry_msgs.msg import TransformStamped
from heracles.query_interface import Neo4jWrapper
from heracles_agents.dsg_interfaces import HeraclesDsgInterface
from heracles_ros_interfaces.srv import UpdateHoldingState
from rclpy.node import Node

from heracles_ros.map_state import bump_map_version


class HeraclesStateUpdater(Node):
    def __init__(self):
        super().__init__("heracles_state_updater")

        self.declare_parameter("heracles_ip", "")
        self.declare_parameter("heracles_port", -1)
        self.declare_parameter("map_frame", "map")
        self.declare_parameter("robot_name", "hamilton")
        self.declare_parameter("publish_rate", 1.0)
        self.map_frame = self.get_parameter("map_frame").value
        self.robot_name = self.get_parameter("robot_name").value
        self.publish_rate = self.get_parameter("publish_rate").value

        ip = self.get_parameter("heracles_ip").get_parameter_value().string_value
        port = self.get_parameter("heracles_port").get_parameter_value().integer_value
        self.get_logger().info(f"Port: {port}")

        assert ip != "", "Please set database IP"
        assert port > 0, "Please set database port"

        self.dsgdb_conf = HeraclesDsgInterface(
            dsg_interface_type="heracles",
            uri=f"neo4j://{ip}:{port}",
        )

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.timer = self.create_timer(1.0 / self.publish_rate, self.timer_callback)

        self.holding_srv = self.create_service(
            UpdateHoldingState,
            "~/update_holding_state",
            self.update_holding_state_callback,
        )

    def _get_robot_pose(self):
        target_frame = f"{self.robot_name}/base_link"
        try:
            transform: TransformStamped = self.tf_buffer.lookup_transform(
                self.map_frame, target_frame, rclpy.time.Time()
            )
        except Exception as e:
            self.get_logger().warn(f"TF not available yet: {e}")
            return None

        t = transform.transform.translation
        q = transform.transform.rotation
        return t.x, t.y, t.z, q.w, q.x, q.y, q.z

    def update_holding_state_callback(self, request, response):
        object_id = request.id
        is_holding = request.is_holding

        with Neo4jWrapper(
            self.dsgdb_conf.uri,
            (
                self.dsgdb_conf.username.get_secret_value(),
                self.dsgdb_conf.password.get_secret_value(),
            ),
            atomic_queries=True,
            print_profiles=False,
        ) as db:
            if is_holding:
                response.success = robot_hold_obj(db, self.robot_name, object_id)
            else:
                if request.has_position:
                    p = request.position
                    put_at = (p.x, p.y, p.z)
                else:
                    robot_pose = self._get_robot_pose()
                    put_at = None if robot_pose is None else robot_pose[:3]
                if put_at is None:
                    response.success = False
                else:
                    # Release first. While HOLDS exists, the holding rule in
                    # dsg_updater keeps snapping the object back onto the
                    # robot, and could undo a position written before it.
                    unhold_success = robot_unhold_obj(db, self.robot_name, object_id)
                    last_pos_success = set_obj_center(db, object_id, *put_at)
                    response.success = last_pos_success and unhold_success
            if response.success:
                bump_map_version(db)

        if response.success:
            self.get_logger().info(
                f"Successfully set holding state: robot={self.robot_name}, "
                f"object={object_id}, is_holding={is_holding}"
            )
        else:
            self.get_logger().error(
                f"Failed to set holding state: robot={self.robot_name}, "
                f"object={object_id}, is_holding={is_holding}"
            )

        return response

    def timer_callback(self):
        robot_pose = self._get_robot_pose()
        if robot_pose is None:
            return

        x, y, z, qw, qx, qy, qz = robot_pose

        query = f"""
            MERGE (r:Robot {{name: '{self.robot_name}'}})
            SET r.position = point({{x: {x}, y: {y}, z: {z}}}),
                r.qw = {qw},
                r.qx = {qx},
                r.qy = {qy},
                r.qz = {qz}
            RETURN r
        """
        with Neo4jWrapper(
            self.dsgdb_conf.uri,
            (
                self.dsgdb_conf.username.get_secret_value(),
                self.dsgdb_conf.password.get_secret_value(),
            ),
            atomic_queries=True,
            print_profiles=False,
        ) as db:
            db.query(query)

            self.get_logger().debug(
                f"Updating DB: {self.robot_name} pos=({x:.2f},{y:.2f},{z:.2f})"
            )


def main(args=None):
    rclpy.init(args=args)
    node = HeraclesStateUpdater()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
