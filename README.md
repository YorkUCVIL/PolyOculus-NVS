# PolyOculus: Simultaneous Multi-view Image-based Novel View Synthesis
[Project Page](https://yorkucvil.github.io/PolyOculus-NVS)

## Python environment
Please install the included environments in the root of this repo:
```
conda env create -f environment.yaml
```
Training requires Torchvision with video_reader support, requiring the library to be built from source.
This can be done by first activating the training conda environment `polyoculus`, and cloning the Torchvision repo somewhere on your system: [torchvision](https://github.com/pytorch/vision/tree/release/0.15).
Checkout the `release/0.15` branch of Torchvision, and run:
```
python setup.py install
```
This should detect the ffmpeg installation in the environment and install Torchvision with video_reader enabled.

## Directory structure
```
├── environment.yaml
├── dataset-data-realestate
│   ├── data
│   │   ├── test
│   │   │   ├── videos                       // videos for this split
│   │   │   └── poses.npy                    // converted camera poses
│   │   ├── train
│   │   │   ├── videos                       // videos for this split
│   │   │   └── poses.npy                    // converted camera poses
│   │   ├── RealEstate10K-original           // original data from RealEstate10K dataset
│   │   │   ├── test                         // txt files for test camera poses
│   │   │   └── train                        // txt files for test camera poses
│   └──  extract-poses.py                    // camera pose conversion script
├── instance-data-realestate-multiview_ldm   // contains data from training and sampling
│   └── checkpoints                          // model checkpoints
│   ├── set-all-cyclic                       // an example sampling spec for cyclic trajectories generating all frames at once
│   ├── set-all-gt-traj                      // an example sampling spec for sequential trajectories generating all frames at once
│   ├── set-grouped-stereo                   // an example sampling spec for grouped generation of stereo views
│   ├── set-keyframed-cyclic                 // an example sampling spec for cyclic trajectories using our keyframed approach
│   ├── set-keyframed-gt-traj                // an example sampling spec for sequential trajectories using our keyframed approach
│   ├── set-keyframed-large                  // an example sampling spec for a larget set of orbital views using our keyframed approach
│   ├── std-auto-cyclic                      // an example sampling spec for cyclic trajectories using standard autoregressive sampling
│   ├── std-auto-gt-traj                     // an example sampling spec for sequential trajectories using standard autoregressive sampling
│   └── std-auto-stereo                      // an example sampling spec for stereo views using standard autoregressive sampling
├── instance-data-realestate-vqgan           // contains data for vqgan weights
│   └── checkpoints                          // model checkpoints
└── src
    ├── configs          // yaml files that configure the models
    ├── datasets         // data input pipelines
    ├── launch-scripts   // shell scripts for launching slurm jobs
    ├── models           // model definitions
    ├── scripts          // python scripts for training and sampling
    └── utils            // various utilities for QOL
```

## Data preparation
[RealEstate10K](https://google.github.io/realestate10k) is a dataset consisting of real estate videos scraped from YouTube. Camera poses are recovered using SLAM.
Videos in the dataset are provided as YouTube URLs, and need to be downloaded manually using tools such as [yt-dlp](https://github.com/yt-dlp/yt-dlp).
The included data pipeline directly reads frames from the videos downloaded at 360p.
The camera poses provided by the dataset are provided using the camera extrinsics. We preprocess the camera poses into world transformations of a canonical camera, specifically the same camera and coordinate system as Blender.
Navigate to the `dataset-data-realestate` directory and place the downloaded Realestate files under `dataset-data-realestate/data/RealEstate10K-original`.
Please also populate the `dataset-data-realestate/data/test/videos` and `dataset-data-realestate/data/train/videos` directories with the downloaded videos.
To convert the poses run:
```
python extract-poses.py test
python extract-poses.py train
```

## Training
Please train with the `polyoculus` environment.
Training uses PyTorch Lightning. An example slurm script is provided under `src/launch-scripts/train-deploy.sh`.

## Pretrained weights
RealEstate10K VQGAN weights: [Google Drive](https://drive.google.com/file/d/1OdoBd6ChbusRc4gSsvw16XKyqOmfwB1p/view?usp=drive_link)

RealEstate10K diffusion model weights: [Google Drive](https://drive.google.com/file/d/1Og_USg-8uzfakVC41TJ3SD-eSzJ-JSUH/view?usp=drive_link)

Please place the first stage VQGAN weights under `instance-data-realestate-vqgan/checkpoints/` and the diffusion model weights under `instance-data-realestate-multiview_ldm/checkpoints/`.

## Sampling
Please use the `polyoculus` environment.
Sampling requires a specific directory structure per sequence to specify the desired camera pose and the given source image.
The directory will also contain the generated samples.
An examples for the various sampling methods are provided under `instance-data-realestate-multiview_ldm/`:
```
└── instance-data-realestate-multiview_ldm
    ├── set-all-cyclic                   // an example sampling spec for cyclic trajectories generating all frames at once
    │    └── scenes                      // a directory of multiple sceness
    │       ├── 584f2fc6d686aebc         // directory for one scene
    │       │   ├── observed             // contains observed image(s)
    │       │   │   ├── images
    │       │   │   │   └── 0000.png
    │       │   │   └── latents          // the latent encoding of the images
    │       │   │       └── 0000.npy
    │       │   ├── samples              // contains sampled images, this directory is not created by the sampling script
    │       │   └── sampling-spec.json   // specifies trajectory of poses
    │       └── ...
    ├── set-all-gt-traj
    │   └── ...
    ├── set-grouped-stereo
    │   └── ...
    ├── set-keyframed-cyclic
    │   └── ...
    ├── set-keyframed-gt-traj
    │   └── ...
    ├── set-keyframed-large
    │   └── ...
    ├── std-auto-cyclic
    │   └── ...
    ├── std-auto-gt-traj
    │   └── ...
    └── std-auto-stereo
        └── ...
```
When creating sampling specs, the focal length should be adjusted differently for each scene depending on how the observed image(s) was captured.
Before sampling you must place images under the `observed/images` directory of a sampling spec for a scene.
Once this is done for all scenes, they must be encoded using the VQGAN using:
```
python scripts/encode-latent.py -c realestate-multiview_ldm.yaml -o set-keyframed-cyclic
```
Sampling is performed by running:
```
python scripts/sample-from-spec.py -c realestate-multiview_ldm.yaml -o set-keyframed-cyclic -s 0
```
which will sample the novel views for scene 0 (defined my alphanumeric ordering) under the `set-keyframed-cyclic` directory.
