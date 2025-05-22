# Action-Recognition-of-OpenPose
使適應足球各比賽場景導入深度學習技術，應用結合物件偵測(YOLOv8)與動作辨識技術(OpenPose)。

# 建立虛擬環境 (For Linux)※Windows變因很多暫不使用


- Ubuntu 20.04(Python 3.8.10, gcc 9.4.0)我使用環境。
- 此教學為Linux中設置虛擬環境和運行OpenPose。
- 確認有NVIDIA顯卡(指令:nvidia-smi)再進行OpenPose安裝。
- https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/doc/installation/1_prerequisites.md#ubuntu-prerequisites   OpenPose詳細的安裝資訊
## Step.0 安裝基本套件與虛擬環境啟動

```
sudo apt install python3-venv
python3 -m venv username
source username/bin/activate
```
## Step.1 Clone OpenPose

```
git clone https://github.com/CMU-Perceptual-Computing-Lab/openpose
cd openpose/
git submodule update --init --recursive --remote
```
## Step.2 Prerequisites

```
sudo apt-get install cmake-qt-gui  （若有安裝CMAKE則此行可以省略）
sudo bash ./scripts/ubuntu/install_cuda.sh   (建議自行先安裝好CUDA)
sudo apt-get install libopencv-dev  (也可以利用cmake自行安裝)
sudo bash ./scripts/ubuntu/install_deps.sh   （安裝caffe相依套件的部份 , 一定要在安裝cuda之後再執行）
sudo apt install protobuf-compiler libgoogle-glog-dev libgflags-dev （CMake config會用到的套件）
sudo apt install libboost-all-dev libhdf5-dev libatlas-base-dev  
sudo pip3 install numpy opencv-python  （使用python語法會用到）
```

## Step.3 CMake Configuration

```
mkdir build/
cd build/
cmake-gui ..  (將 BUILD_PYTHON 勾選, 按下 Configure, 按 Generate)
```
## Step.4 Compilation

```
nproc
make -j8   (將nproc顯示的數字填入8的位置)
5.Running OpenPose
cd ..  (回到 openpose目錄)
```
## Step.5 DEMO
1.於Roboflow平台使用YOLOv8訓練足球員，使影像抓到觀眾以外之分類球員。

![yolo001](https://github.com/user-attachments/assets/81496706-34bc-417f-a75c-d04d982d2bfe)

2.使影像球員背景顏色過濾，並使用K-means隊球衣顏色進行聚類，再更改球員邊界框顏色以區分隊伍。

![kmeans](https://github.com/user-attachments/assets/3e6aa7fd-f1b5-465e-851c-5f068b5ae11b)
![obj00](https://github.com/user-attachments/assets/36fcda30-5b06-4e00-87d8-8bff47527e11)

3.對球員對做進行訓練(不同資料集訓練80%與測試20%)與動作分類(跑步、站立、走路、強踢、傳球)，但影像整體呈現繁雜，缺乏關鍵時刻資訊，使不必要運算量增加，後續步驟將其改進。

![objact01](https://github.com/user-attachments/assets/5b30653f-61d4-40c2-8a75-5226f397aef5)

4.以足球分類中心周圍特定區域進行動作辨識(OpenPose)，全畫面關節點擷取縮減至畫面的1/6，並降低影像輸入以減少系統運算量。
以足球偵測框周圍放大至右上角，蒐集球員關節點並持續輸出分類動作，並新增動作比例與鳥瞰圖於畫面上。

![act02](https://github.com/user-attachments/assets/9cdbe58e-cba1-4129-ad35-6fcfc0ced917)



