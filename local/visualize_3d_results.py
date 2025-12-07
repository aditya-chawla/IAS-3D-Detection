import json
from pathlib import Path

import numpy as np
import open3d as o3d

RESULTS_DIR = Path(r"C:\Users\adity\mmdet3d_project\results_unified")


def points_in_box_mask(points_xyz, box):
    """Check which points fall inside a 3D box."""
    cx, cy, cz, l, w, h, yaw, score = box
    x_min, x_max = cx - l / 2, cx + l / 2
    y_min, y_max = cy - w / 2, cy + w / 2
    z_min, z_max = cz - h / 2, cz + h / 2
    x, y, z = points_xyz[:, 0], points_xyz[:, 1], points_xyz[:, 2]
    return (x >= x_min) & (x <= x_max) & (y >= y_min) & (y <= y_max) & (z >= z_min) & (z <= z_max)


def show_3d_colorized(ply_path, json_path, window_title):
    """Display 3D point cloud with red-colored detected objects."""
    if not ply_path.exists() or not json_path.exists():
        print(f"[{window_title}] Missing files:")
        print(f"  PLY:  {ply_path}")
        print(f"  JSON: {json_path}")
        return

    pcd = o3d.io.read_point_cloud(str(ply_path))
    points = np.asarray(pcd.points, dtype=np.float32)
    if len(points) == 0:
        print(f"[{window_title}] Empty point cloud.")
        return

    with open(json_path) as f:
        data = json.load(f)
    boxes = data.get("boxes", [])

    print(f"\n{'='*60}")
    print(f"{window_title}")
    print(f"{'='*60}")
    print(f"  Points: {len(points)}")
    print(f"  Boxes:  {len(boxes)}")

    # Color all points green by default
    colors = np.zeros_like(points)
    colors[:, 1] = 1.0  # green

    # Color points inside boxes as red
    colored_count = 0
    for i, box in enumerate(boxes):
        box = [float(v) for v in box]
        mask = points_in_box_mask(points[:, :3], box)
        num_colored = mask.sum()
        colors[mask] = np.array([1.0, 0.0, 0.0])  # red
        colored_count += num_colored
        print(f"  Box {i}: colored {num_colored} points")

    print(f"  Total red (detected) points: {colored_count}")
    pcd.colors = o3d.utility.Vector3dVector(colors)

    # Create Open3D visualizer window
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=window_title, width=1280, height=720)
    vis.add_geometry(pcd)
    
    # Set view angle
    ctr = vis.get_view_control()
    ctr.set_zoom(0.5)
    ctr.set_front([0.5, -0.3, -0.8])
    ctr.set_lookat([0, 0, 0])
    ctr.set_up([0, 0, 1])
    
    print(f"→ 3D window opened: {window_title}")
    print("   (rotate/zoom with mouse, close window to continue)\n")
    
    vis.run()
    vis.destroy_window()


def show_kitti_2d_example(frame_id="000006"):
    """Show path to KITTI raw camera image (no OpenCV imshow)."""
    img_path = RESULTS_DIR / "kitti_pointpillars" / "raw_images" / f"{frame_id}.png"
    
    print(f"\n{'='*60}")
    print(f"2D – KITTI-{frame_id} – raw camera")
    print(f"{'='*60}")
    
    if not img_path.exists():
        print(f"  ⚠ No KITTI raw image found at: {img_path}")
        print("    (Make sure KITTI processing ran successfully)")
    else:
        print(f"  ✓ Raw camera image location:")
        print(f"    {img_path}")
        print(f"  → Open this file manually in Windows Explorer / Photos app")
        print(f"    to view the 2D reference image.\n")


def main():
    print("=" * 70)
    print("3D VISUALIZATION OF SAVED RESULTS")
    print("=" * 70)
    print("\nThis script shows:")
    print("  1. Path to raw KITTI camera image (2D reference)")
    print("  2. Interactive 3D LiDAR views with detected objects in RED")
    print("=" * 70)

    # ===== KITTI VISUALIZATIONS =====
    
    # 1) 2D – KITTI – raw camera image (just show path, no cv2.imshow)
    show_kitti_2d_example("000006")

    # 2) 3D – KITTI – PointPillars
    kitti_pp_ply = RESULTS_DIR / "kitti_pointpillars" / "plys" / "000006.ply"
    kitti_pp_json = RESULTS_DIR / "kitti_pointpillars" / "jsons" / "000006.json"
    show_3d_colorized(kitti_pp_ply, kitti_pp_json, "3D – KITTI-01 – PointPillars")

    # 3) 3D – KITTI – VoxelNet
    kitti_vn_ply = RESULTS_DIR / "kitti_voxelnet" / "plys" / "000000.ply"
    kitti_vn_json = RESULTS_DIR / "kitti_voxelnet" / "jsons" / "000000.json"
    show_3d_colorized(kitti_vn_ply, kitti_vn_json, "3D – KITTI-01 – VoxelNet")

    # ===== NUSCENES VISUALIZATIONS =====
    
    # Find first nuScenes frame
    nusc_pp_dir = RESULTS_DIR / "nuscenes_pointpillars" / "plys"
    nusc_files = sorted(nusc_pp_dir.glob("*.ply"))
    
    if nusc_files:
        nusc_name = nusc_files[0].stem

        # 4) 3D – nuScenes – PointPillars
        nusc_pp_ply = RESULTS_DIR / "nuscenes_pointpillars" / "plys" / f"{nusc_name}.ply"
        nusc_pp_json = RESULTS_DIR / "nuscenes_pointpillars" / "jsons" / f"{nusc_name}.json"
        show_3d_colorized(nusc_pp_ply, nusc_pp_json, "3D – nuScenes-01 – PointPillars")

        # 5) 3D – nuScenes – VoxelNet
        nusc_vn_ply = RESULTS_DIR / "nuscenes_voxelnet" / "plys" / f"{nusc_name}.ply"
        nusc_vn_json = RESULTS_DIR / "nuscenes_voxelnet" / "jsons" / f"{nusc_name}.json"
        show_3d_colorized(nusc_vn_ply, nusc_vn_json, "3D – nuScenes-01 – VoxelNet")
    else:
        print("\n⚠ No nuScenes PLYs found under nuscenes_pointpillars/plys/")
        print("  Make sure you ran run_3d_processing.py successfully.")

    print("\n" + "=" * 70)
    print("DONE - All visualizations complete")
    print("=" * 70)
    print("\nFor your report:")
    print("  • Use the KITTI raw image (open manually from the path shown)")
    print("  • Take screenshots of the 4 Open3D windows (green=background, red=objects)")
    print("  • Reference the metrics table from run_3d_processing.py output")
    print("=" * 70)


if __name__ == "__main__":
    main()
