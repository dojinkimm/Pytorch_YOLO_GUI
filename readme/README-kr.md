# Pytorch-YOLO GUI 

### 같은 프로그램을 Tensorflow에서 작동시키고 싶으면 tensorflow_gui 브랜치에 가세요!

해당 프로그램을 구현하면서 밑의 repository를 참고했습니다: 

* Pytorch-YOLO <br/>
https://github.com/eriklindernoren/PyTorch-YOLOv3.git<br/>
* GUI <br/>
https://github.com/PySimpleGUI/PySimpleGUI/tree/master/YoloObjectDetection<br/> 

## 요구사항

* Python 3.6
* imutils 0.5.2<br> 
```pip install opencv-python imutils```
* torch 1.0
* torchvision 0.3.0<br>
```pip install torch==1.0 torchvision```


## YOLO v3 Pre-trained Models 다운 받는 방법
Weight 파일을 다운 받는 방법은 두가지가 있습니다. 첫 째, [Darknet](https://pjreddie.com/darknet/yolo/) 사이트에 가서 직접 다운 받고 weight 디렉토리에 넣어주는 것입니다.
둘 째, 밑의 명령을 작성해주세요:
```Shell
git clone https://github.com/dojinkimm/Pytorch_YOLO_GUI
cd Pytorch_YOLO_GUI/weight
wget https://pjreddie.com/media/files/yolov3.weights
```
명령 실행이 끝나면 yolov3.weights 파일이 weight 디렉토리 안에 다운받아질 것입니다. 

## Demo 실행
3개의 Python 파일이 있는데 각각 비디오를 실행시키는 포맷이 다릅니다.
하지만 처음에 나오는 GUI는 다 같습니다. `pytorch_yolo_gui_window.py` 와 `pytorch_yolo_gui_faster_window.py`에서는
confidence nmsthreshold 값을 실시간으로 변경할 수 있습니다. 

<div align="center">
    <img src="yolo_player.png" width="600px"/>
</div>

각 파트를 설명하자면: 

* Path to input video - 비디오 파일이 있는 경로
* Path to cfg File - cfg 파일이 있는 경로 (yolov3.cfg)
* Path to weight File - weight 파일이 있는 경로 (yolov3.weight)
* Path to label names - label names 가 있는 경로 (coco.names) 
* confidence - confidence threshold 값
* NMSThreshold - nms threshold 값
* Resolution - 416, 일반적으로 쓰이는 값 (낮추면 빨라지되 정확도 떨어지고, 높이면 정확도가 높아지되 속도가 느려진다)
* Classes to detect - default로 coco.names의 모든 클래스들이 선택되어 있다. 만약 어떤 클래스가 detect되는 것을 원치 않으면 클릭하면 됩니다. <br/>
Ex) person을 클릭하면 비디오에서 person은 detect되지 않습니다<br/>
* Use webcam - 체크되면, 웹캠이 켜질 것입니다<br/>

#### pytorch_yolo_gui.py

![Screenshot](yolo_no_window.png)

#### pytorch_yolo_gui_window.py

![Screenshot](yolo_window.png)

#### pytorch_yolo_gui_faster_window.py

![Screenshot](yolo_faster_window.png)







