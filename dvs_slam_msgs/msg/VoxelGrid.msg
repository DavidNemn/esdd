# This message contains a stamped voxel grid (3D array containing
# the number of votes for each voxel in a non-uniform, projective grid)

geometry_msgs/Pose T_w_ref             # 坐标变换， 参考帧（体素网格->世界坐标系）
std_msgs/Float32MultiArray voxel_grid  # 包含体素投票的三维数组
