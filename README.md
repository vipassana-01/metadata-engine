# Metadata Engine

Python scripts to build a Metadata Engine which takes a large amount of raw HD video footage, generate useful and well-defined metadata for each video file, each distinct moment called frame in each video file. 


## What it does?

This system uses the YOLOv5 object detection model to detect people in a given video and save the metadata for each frame in a JSON file. It also draws bounding boxes around the detected people in real-time for visualization purposes. With huge amounts of raw HD video footage, it generates a variety of metadata, such as per-frame face tracking, unique face detection, and person tokenization per video.


## System Overview

The system takes as input a video file and a directory path to save the metadata files. It uses OpenCV to read the frames from the video and passes each frame through the YOLOv5 model to detect people. The bounding box coordinates for the detected people are saved in a JSON file along with the frame number and the number of detected people in the frame.

For visualization purposes, the system draws the bounding boxes around the detected people on each frame in real-time using OpenCV.

## Metadata Generated

* `frame_number`: the frame number of the current frame

* `boxes`: a list of bounding boxes for each detected person in the current frame. Each bounding box is represented as a list of four values: [xmin, ymin, xmax, ymax], 

    where:

    xmin and ymin: coordinates of the top-left corner of the bounding box

    xmax and ymax: coordinates of the bottom-right corner of the bounding box.

* `num_people`: the number of people detected in the current frame.
 
 The frame number is included to keep track of the metadata for each frame. The bounding box coordinates are necessary to locate the detected people in the frame, and the number of detected people provides an overview of the person density in the video.

 ## Why did we chose to display this metadata?

 We chose these metadata categories because they provide a concise summary of the person detection results and are sufficient for further analysis and processing.

## Perfomance

The performance of the Engine can be evaluated in terms of its [real-time factor](https://openvoice-tech.net/index.php/Real-time-factor#:~:text=If%20it%20takes%20time%20f%20%28d%29%20to%20process,1%2C%20the%20processing%20is%20done%20%22in%20real%20time%22.). The Engine processes each frame of the video and generates metadata in the form of a JSON file for each frame. It also displays the video with bounding boxes around the detected people.

The performance of the Engine is highly dependent on the hardware specs of the machine it is running on. In general, the [real-time factor](https://openvoice-tech.net/index.php/Real-time-factor#:~:text=If%20it%20takes%20time%20f%20%28d%29%20to%20process,1%2C%20the%20processing%20is%20done%20%22in%20real%20time%22.) of the Engine can be improved by using a more powerful machine with a faster CPU and GPU.

However, the YOLOv5 model, which is known for its fast inference time and high accuracy. This means that the Engine should be able to process the video in near [real-time](https://openvoice-tech.net/index.php/Real-time-factor#:~:text=If%20it%20takes%20time%20f%20%28d%29%20to%20process,1%2C%20the%20processing%20is%20done%20%22in%20real%20time%22.) on a mid-range machine with a decent GPU.

To further improve the performance of the Engine, it may be helpful to experiment with different YOLOv5 models with varying sizes and architectures, as well as adjusting the input image size for inference. Additionally, optimizing the code for parallel processing and utilizing multi-threading could also improve the real-time factor of the Engine.

Overall, the given code has the potential to achieve real-time performance on mid-range machines with a decent GPU. However, to achieve better performance on weaker machines or to scale the Engine to process larger videos, further optimization may be necessary.

## Perfomance Benchmarks

The given code performs person detection using the YOLOv5 model and saves the metadata for each frame in a JSON file. It also displays the frames with bounding boxes around the detected persons in real-time.

The hardware specs of the device used for testing include:
* AMD Ryzen 5 4600H processor  
* 8GB RAM
* Nvidia 1650 Graphic Card
* 64-bit operating system with no pen or touch input.

After testing the code on a video, the **[real-time](https://openvoice-tech.net/index.php/Real-time-factor#:~:text=If%20it%20takes%20time%20f%20%28d%29%20to%20process,1%2C%20the%20processing%20is%20done%20%22in%20real%20time%22.) factor was found to be >1**, indicating that the code takes longer to process each frame than the actual time of the video.

## Performance Considerations

To optimize the performance of the code, we can consider the following suggestions:

* To reduce the processing time and memory usage, the frames are resized to a smaller size (640x640) before passing them through the model. This tradeoff between accuracy and speed was made to ensure real-time processing of the video.

* On using a smaller YOLOv5 model, such as the yolov5s, instead of the default model used in the code, which is the yolov5x. The smaller model will have fewer parameters and hence faster processing times.

* Use multi-processing to parallelize the processing of each frame. This will utilize the multiple cores of the CPU and speed up the processing time.

* Use a GPU to perform the person detection instead of the CPU. GPUs are optimized for parallel processing and can perform the computations faster than CPUs.

By implementing these suggestions, we can improve the performance of the code and reduce the real-time factor.


## Resources Used

The system uses the following libraries and resources:

1. [OpenCV](https://pypi.org/project/opencv-python/) for video processing and visualization
2. [PyTorch](https://pypi.org/project/torch/) for loading the YOLOv5 model
3. [Ultralytics' YOLOv5](https://github.com/ultralytics/yolov5) repository for accessing the YOLOv5 model

4. [tqdm](https://pypi.org/project/tqdm/) for progress tracking
5. [PIL](https://pypi.org/project/Pillow/) for image preprocessing
6. [NumPy](https://pypi.org/project/numpy/) for array manipulation

7. [JSON](https://docs.python.org/3/library/json.html): for saving metadata



## Requirements

The following libraries are required to run this code:

* OpenCV (cv2)
* PyTorch
* tqdm
* Pillow (PIL)
* NumPy
* json

The YOLOv5 model is loaded using the PyTorch `torch.hub.load` method.

## Usage

To use this code, simply instantiate an Engine object with the path to the video file and the directory where the metadata files should be saved. Then call the run or detect_people method to start the person detection process.

```python
video_path = "video1.mp4"
metadata_dir = "/path/to/metadata/directory"


engine = Engine(video_path, metadata_dir)
engine.run()
```
The run method will display the video frames with the detected bounding boxes in a window. The detect_people method performs the same detection process but does not display the frames in a window.


## Setup Instructions

To set up the environment using pip, the user can run the following command in the terminal:

```
pip install -r requirements.txt
```
To set up the environment using conda,run the following command in the terminal:

```bash
conda env create -f environment.yml
conda activate yolov5
```
After activating the environment, then run the Python script using the command:

```
python engine.py
```

> **Not**e: You should replace engine.py with the actual name of the Python script containing the code. Additionally, you should make sure that the video file and metadata directory exist and are accessible.
