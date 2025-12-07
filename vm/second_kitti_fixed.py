_base_ = '../configs/second/second_hv_secfpn_8xb6-80e_kitti-3d-3class.py'

# Fix data path
test_dataloader = dict(
    dataset=dict(
        data_prefix=dict(pts='training/velodyne')))  # Change from velodyne_reduced

val_dataloader = dict(
    dataset=dict(
        data_prefix=dict(pts='training/velodyne')))

load_from = 'checkpoints/second_hv_secfpn_8xb6-80e_kitti-3d-3class-b086d0a3.pth'
