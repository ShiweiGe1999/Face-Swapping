# CIS 581 Final Project Face Swapping



> Contributor: Shiwei Ge & Jiawei He

> Here is a link to our final presentation video [Presentation Video](https://drive.google.com/file/d/1mIwyoQ0Jw0kGeevDIudSKS6MQDG9W6VN/view?usp=share_link)

> [Report PDF](FaceSwapping_mid.pdf)

> [PPT Link](Final-Demo.pptx)

## Package Requirements

Make sure you installed the following dependencys

```
pip install -r requirements.txt
```
## Instructions on how to run the project

### The project divided into several parts

> Face Detector Class

> Face Segmentation Class

> Face Swapping Class

> Main Class (integrated with absl) 


### Command line based application

Since it's a face-swapping application, which consists of swapping face in image and swapping face in a video, the user is required to have either two images which are `source image` and `destination image` or `source image` and `destination video (The video you want to put the source image face on)`

- Face Swapping between images

```
python3 Main.py --i <source-image-path> --o <destination-image-path>
```
- Swapping image into a video
```
python3 Main.py --i <source-image-path> --video <path-to-video>
```
> After running the command above, there's will be `output.avi` file in `images` folder, which is the video after face swapping



