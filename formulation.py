HANDLING_SENSITIVITY = 0.2
THROTTLE_SENSITIVITY = 0.3
MAX_ALTITUDE = 600
MAX_VELOCITY = 130


OBS = ["throttle",
       "altitude",
       "velocity_x",
       "velocity_y",
       "velocity_z",]
ACTIONS = [
           "throttle_up",
           "throttle_down",]

"""
OBS = ["throttle",
       "altitude",
       "velocity_x",
       "velocity_y",
       "velocity_z",
       "rotation_x",
       "rotation_y",
       "rotation_z",
       "rotation_w"]
ACTIONS = ["yaw_up",
           "yaw_down",
           "pitch_up",
           "pitch_down",
           "roll_up",
           "roll_down",
           "throttle_up",
           "throttle_down",
           "do_nothing"]
"""


# observations = ["altitude",
#                 "fuel",
#                 "angular_velocity_x",
#                 "angular_velocity_y",
#                 "angular_velocity_z",
#                 "velocity_x",
#                 "velocity_y",
#                 "velocity_z",
#                 "rotation_x",
#                 "rotation_y",
#                 "rotation_z",
#                 "rotation_w"]
# actions = ["throttle_up",
#            "throttle_down"]
