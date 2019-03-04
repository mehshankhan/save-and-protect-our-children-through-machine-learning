# save-and-protect-our-children-through-machine-learning

c3d model link
https://drive.google.com/file/d/19NWziHWh1LgCcHU34geoKwYezAogv9fX/view?usp=sharing

Put your own videos in "dataset_videos" folder in the following hiearchy:
  ```
  dataset_videos
  ├── Dragging
  │   ├── dragging_1.avi
  │   └── ...
  ├── Kindnapping
  │   ├── kidnapping_1.avi
  │   └── ...
  └── Normal
  │   ├── normal_1.avi
  │   └── ...
  ```
Use image_frm_video.py to extract frames from video put them in the "VAR/dataset_images" folder in following order:
  ```
  VAR/dataset_images
  ├── Dragging
  │   ├── dragging_1
  │   │   ├── 00001.jpg
  │   │   └── ...
  │   └── ...
  ├── Kindnapping
  │   ├── Kindnapping
  │   │   ├── 00001.jpg
  │   │   └── ...
  │   └── ...
  └── Normal
  │   ├── Normal
  │   │   ├── 00001.jpg
  │   │   └── ...
  │   └── ...
  ```
  
  run "python train.py" to train data
  
  
  run "python inference.py" to test the model data on any video
  
  Results of our work is here in YouTube video:https://www.youtube.com/watch?v=i4EloWzGLAA
  
  run the code and enjoy
  
