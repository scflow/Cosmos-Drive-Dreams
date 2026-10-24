# Cosmos-Drive-Dreams Toolkits
This sub-repo provides toolkits for:

* A rendering script that converts RDS-HQ datasets into input videos (LiDAR and HDMAP) compatible with [**Cosmos-Transfer1-7B-Sample-AV**](https://github.com/nvidia-cosmos/cosmos-transfer1/blob/main/examples/inference_cosmos_transfer1_7b_sample_av.md) and [**Cosmos-Transfer1-7B-Sample-AV-Multiview**](https://github.com/nvidia-cosmos/cosmos-transfer1/blob/main/examples/inference_cosmos_transfer1_7b_sample_av_single2multiview.md).

* A conversion script that converts open-source datasets (e.g., Waymo Open Dataset) into RDS-HQ format, so you can reuse the rendering script to render the input videos.

* A conversion [script](convert_lidar_pointcloud_to_rangemap.py) that converts raw lidar data into range map format using Nvidia Ncore libray. Note that this only serves as a reference script for nvidia lidar data, you will need to modify for your own use case. 

* An interactive visualization tool to visualize the RDS-HQ dataset and generate novel ego trajectories.


<div align="center">
  <img src="../assets/av_example.gif" alt=""  width="1100" />
</div>

## Visualize Dataset
You can use `visualize_rds_hq.py` to visualize the RDS-HQ dataset. It would be helpful to see our example data first.
```bash
usage: visualize_rds_hq.py -i RDS_HQ_FOLDER -c CLIP_ID
                          [-np NOVEL_POSE_FOLDER] 
                          [-d {rds_hq_mv,...}] 

required arguments:
  -i RDS_HQ_FOLDER, --input_root RDS_HQ_FOLDER
                        The root folder of the RDS-HQ dataset, 
                        or other datasets converted to RDS-HQ format.

  -c CLIP_ID, --clip_id CLIP_ID
                        Specific clip id to visualize.

optional arguments:
  -np NOVEL_POSE_FOLDER, --novel_pose_folder NOVEL_POSE_FOLDER
                        Folder name for saving novel ego trajectories.
                        Default: novel_pose
                        
  -d {rds_hq_mv,...}, --dataset {rds_hq_mv,...}
                        dataset config name, refer to `config/dataset_<dataset_name>.json`
                        We will generate pose for 'CAMERAS' in `config/dataset_<dataset_name>.json`
                        Default: rds_hq_mv
                      
```
This python script will launch a [viser](https://github.com/nerfstudio-project/viser) server to visualize the 3D HD map world with dynamic bounding boxes. You can use 
- `w a s d` to control camera's position
- `q e` to control camera's z-axis coordinate

![viser](../assets/viser.png)

> [!NOTE]
> You can run this script on a **remote server** with VS Code + remote-ssh plugin. VS Code will automatically forward the port of viser to the local host. 


## Rendering Input Control Videos for Cosmos
You can use `render_from_rds_hq.py` to render the HD map + bounding box / LiDAR / World Scenario condition videos from RDS-HQ dataset. GPU is required for rendering LiDAR and World Scenario.
```bash
usage: render_from_rds_hq.py [-h] -i INPUT_ROOT -o OUTPUT_ROOT
                            [-cj CLIP_ID_JSON] 
                            [-d {rds_hq,rds_hd_mv,waymo,waymo_mv}]
                            [-c {pinhole,ftheta}] 
                            [-s hdmap] [-s lidar] [-s world_scenario]
                            [-p] 
                            [-np NOVEL_POSE_FOLDER]

required arguments:
  -i RDS_HQ_FOLDER, --input_root RDS_HQ_FOLDER
                        The root folder of the RDS-HQ dataset, 
                        or other datasets converted to RDS-HQ format.
  
  -o OUTPUT_ROOT, --output_root OUTPUT_ROOT
                        Output directory

optional arguments:
  -cj CLIP_ID_JSON, --clip_id_json CLIP_ID_JSON
                        Specific clip(s) to process:
                        Choice 1: path to a json file containing expected clip ids
                        Choice 2: clip id string (render only one clip)
                        Choice 3: None (render all clips in the dataset)
                        Default: None
  
  -d {rds_hq,rds_hd_mv,waymo,waymo_mv}, --dataset {rds_hq,rds_hd_mv,waymo,waymo_mv}
                        Dataset configuration (default: rds_hq), refer to `config/dataset_<dataset_name>.json`
                        
  -c {pinhole,ftheta}, --camera_type {pinhole,ftheta}
                        Camera model type:
                        RDS-HQ dataset use ftheta model while waymo dataset use pinhole model
                        Default: ftheta
  
  -s hdmap, --skip hdmap
                        Skip HD Map rendering
  
  -s lidar, --skip lidar
                        Skip LiDAR rendering
  
  -s world_scenario, --skip world_scenario
                        Skip World Scenario rendering
  
  -p POST_TRAINING, --post_training POST_TRAINING
                        Also generate RGB video clip for post-training
                        Default: False
  
  -np NOVEL_POSE_FOLDER, --novel_pose_folder NOVEL_POSE_FOLDER
                        If provided, we will use pose in this folder as the novel ego trajectory.
                        If None, we will use the default pose in `pose` folder.
                        Default: None
```
This will automatically launch multiple jobs based on [Ray](https://docs.ray.io/en/releases-2.4.0/index.html). If you want to use single process (e.g. for debugging), you can set `USE_RAY=False` in `render_from_rds_hq.py`. You can add `--skip hdmap`, `--skip lidar`, or `--skip world_scenario` to skip the rendering of HD map, LiDAR, or World Scenario respectively.

### World Scenario Rendering (New in Cosmos-Transfer2.5-2B/auto/multiview)
World Scenario rendering is an improved version of the HDMap condition, introduced in Cosmos-Transfer 2.5. Compared to traditional HDMap rendering, it offers several enhancements:

**Key Differences from HDMap:**
- GPU-enabled rendering with moderngl
- Accurate 3D occlusion and heading for bbox rendering
- Fine-grained laneline style rendering with different patterns (solid, dashed, dotted) 


**RDS-HQ Rendering Results**
<div align="center">
  <img src="../assets/rds_hq_render.png" alt="RDS-HQ Rendering Results" width="800" />

  HDMap Rendering & LiDAR Depth Rendering
</div>

<div align="center">
  <img src="../assets/rds_hq_render_world_scenario.png" alt="RDS-HQ World Scenario Rendering Results" width="800" />
  
  HDMap Rendering v.s. World Scenario Rendering
</div>



> [!NOTE]
> If you're interested, we offer [documentation](../assets/ftheta.pdf) that explains the NVIDIA f-theta camera in detail.


The output folder structure will be like this. Note that `videos` will only be generated when setting `--post_training true`.
```bash
<OUTPUT_FOLDER>
├── hdmap
│   └── {camera_type}_{camera_name}
│       ├── <CLIP_ID>_0.mp4
│       ├── <CLIP_ID>_1.mp4
│       └── ...
│
├── world_scenario
│   └── {camera_type}_{camera_name}
│       ├── <CLIP_ID>_0.mp4
│       ├── <CLIP_ID>_1.mp4
│       └── ...
│
├── lidar
│   └── {camera_type}_{camera_name}
│       ├── <CLIP_ID>_0.mp4
│       ├── <CLIP_ID>_1.mp4
│       └── ...
│
└── videos
    └── {camera_type}_{camera_name}
        ├── <CLIP_ID>_0.mp4
        ├── <CLIP_ID>_1.mp4
        └── ...
```


## Generate Novel Ego Trajectory
You can use `visualize_rds_hq.py` to generate novel trajectories.
```bash
visualize_rds_hq.py -i RDS_HQ_FOLDER -c CLIP_ID
                    [-np NOVEL_POSE_FOLDER] 
                    [-d {rds_hq_mv,...}] 
```
A detailed explanation of each argument can be found above. Using the panel on the right, you record keyframe poses (make sure include the first frame and the last frame), and the script will interpolate all intermediate poses and save them as a `.tar` file in the `NOVEL_POSE_FOLDER` folder at `RDS_HQ_FOLDER`. Then you can pass `--novel_pose_folder <NOVEL_POSE_FOLDER>` to the rendering script `render_from_rds_hq.py` to use the novel ego trajectory.
```bash
python render_from_rds_hq.py -i RDS_HQ_FOLDER -o OUTPUT_FOLDER -cj CLIP_ID -np NOVEL_POSE_FOLDER
```

## Convert Public Datasets

We provide a conversion and rendering script for the Waymo Open Dataset as an example of how information from another AV source can interface with the model. 

[Convert Waymo Open Dataset](./docs/convert_public_dataset.md)

### nuPlan RDS conversion and image export

`convert_nuplan_and_render_images.py` runs conversion before rendering and saves
World Scenario plus an untitled aligned RGB/HD-map/World-Scenario/fusion 2x2
grid. Output names contain location, Beijing timestamp (`BJT`), and camera name.

```bash
python convert_nuplan_and_render_images.py \
  --db_path /path/to/nuplan.db \
  --sensor_root /path/to/nuplan-v1.1/sensor_blobs \
  --map_root /path/to/maps/maps \
  --output_root /path/to/output \
  --all_cameras \
  --start_index 0 \
  --frame_count 200
```

The output is separated into `rds/`, `rendered/`, and `images/`. Images are
organized as `images/<location>/<Beijing-date>/<camera>/<image-type>/*.jpg`.
Only `world_scenario` and `four_grid` are stored as standalone JPEG products;
RGB, RDS HD map, and RGB/RDS fusion remain visible inside the four-grid without
being duplicated as separate files. Temporary HD-map and World Scenario render
videos are deleted after those JPEGs are extracted; the source videos and all
other RDS data under `rds/` are retained.

For a complete nuPlan release, use the restartable multi-process batch driver:

```bash
python batch_convert_nuplan_and_render_images.py \
  --db_root /path/to/nuplan-v1.1/splits \
  --sensor_root /path/to/nuplan-v1.1/sensor_blobs \
  --map_root /path/to/maps/maps \
  --output_root /path/to/nuplan_rds_export \
  --all_cameras \
  --clip_frames 200 \
  --workers 32 \
  --threads_per_worker 2 \
  --max_in_flight 64
```

The batch driver recursively discovers databases, removes duplicate log stems,
splits each log into non-overlapping clips, and records per-clip state under
`status/` and logs under `logs/`. Successful clips are skipped on the next run.
Use `--dry_run` to inspect the plan before starting. Use `--convert_only` when
only RDS data is required. Full batch visualization always exports every source
camera frame; it does not expose sampling or maximum-frame options.

`--max_in_flight` bounds queued plus running clips so a full-release plan does
not create hundreds of thousands of `Future` objects at once. Zero (the
default) selects `2 * workers`. Increasing `--workers` beyond the storage
system's useful I/O concurrency may reduce throughput. Estimate output size on
a representative clip before a full 8-camera image export: each timestamp
produces 16 JPEGs (two image products for each of eight cameras), in addition
to the RDS files.

While running, the terminal displays a clip progress bar with ETA and
success/failure counts. Global events are appended to `batch.log`, live numeric
progress is stored in `batch_progress.json`, and each clip has its own detailed
`logs/<task-id>.log` and `status/<task-id>.json` files.

The nuPlan World Scenario conversion includes painted lane boundaries, road
boundaries, crosswalk polygons, stop/yield lines, physical traffic-light heads
with per-frame state, and tracked-object cuboids. Lane centerlines, inferred
signal poles, and map layers without a reliable nuPlan counterpart are not
fabricated. Filled geometry is clipped at the camera near plane, calibrated
projection domain, and image bounds before rendering.

To add the new crosswalk, wait-line, and traffic-light layers to existing RDS
clips without rewriting camera videos, poses, intrinsics, or object labels:

```bash
python update_nuplan_rds_world_scenario.py \
  --rds_root /path/to/existing/export/rds \
  --workers 16
```

The updater is restartable. It records `nuplan-world-scenario-v2` in each
updated manifest and skips those clips on the next run. Use `--force` to
regenerate the layers. World Scenario and four-grid images must be rendered
again after updating because existing JPEGs still contain the old geometry.

**Waymo Rendering Results (use ftheta intrinsics in RDS-HQ)**
![Waymo Rendering Results](../assets/waymo_render_ftheta.png)


**Waymo Rendering Results (use pinhole intrinsics in Waymo Open Dataset)**
![Waymo Rendering Results](../assets/waymo_render_pinhole.png)


**Our Model with Waymo Inputs**
<div align="center">
  <img src="../assets/waymo_example.gif" alt=""  width="1100" />
</div>


## Rectify f-theta Camera to Pinhole Camera
Pinhole camera model is a more widely used camera model, and we can rectify the $f$-theta camera to it. This requires `ftheta_intrinsic` and `pinhole_intrinsic` from RDS-HQ dataset. Then you can convert a cosmos-generated video or real-world RDS-HQ video to pinhole camera model.

```bash
python rectify_ftheta_to_pinhole.py -i INPUT_VIDEO_PATH -o OUTPUT_ROOT -r RDS_HQ_FOLDER -c CLIP_ID [-n CAMERA_NAME]


required arguments:
  -i INPUT_VIDEO_PATH, --input_video_path INPUT_VIDEO_PATH
                        Path to source ftheta video file requiring rectification.
                        Can be a cosmos-generated video or a real-world RDS-HQ video.
  
  -o OUTPUT_ROOT, --output_root OUTPUT_ROOT
                        Directory for rectified outputs.
  
  -r RDS_HQ_FOLDER, --rds_hq_folder RDS_HQ_FOLDER
                        Path to RDS-HQ dataset root.
  
  -c CLIP_ID, --clip_id CLIP_ID
                        Clip id (to query the ftheta_intrinsic and pinhole_intrinsic)
                        

optional arguments:
  -n CAMERA_NAME, --camera_name CAMERA_NAME
                        Camera name to use (default: camera_front_wide_120fov)

```

After rectification, we will have some quality degradation and pixel loss. Typically, the field of view (FOV) will also be reduced. See the difference below:
![rectification_example](../assets/rectify.png)

## Prompting During Inference
We provide a captioning modification example to help users reproduce our results. Please refer to [prompt rewriter](https://github.com/nv-tlabs/Cosmos-Drive-Dreams/tree/main?tab=readme-ov-file#2-prompt-rewriting).
