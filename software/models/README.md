# Model Weights

Large binary model files are **not tracked in git** (see `.gitignore`).

Download them manually and place them in this directory.

| File | Size | Download |
|------|------|----------|
| `shape_predictor_68_face_landmarks.dat` | ~99 MB | [dlib models](http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2) |
| `mmod_human_face_detector.dat` | ~729 KB | [dlib models](http://dlib.net/files/mmod_human_face_detector.dat.bz2) |
| `haarcascade_frontalface_default.xml` | ~930 KB | [OpenCV GitHub](https://github.com/opencv/opencv/tree/master/data/haarcascades) |
| `fer_model.hdf5` | ~873 KB | Trained locally — see `vision/m_model.py` |
| `model_weights.pkl` | ~94 MB | Trained locally — see `vision/m_model.py` |
| `yolov3-tiny.cfg` | ~2 KB | [YOLO configs](https://github.com/pjreddie/darknet/blob/master/cfg/yolov3-tiny.cfg) |

After downloading, run:
```bash
bzip2 -d shape_predictor_68_face_landmarks.dat.bz2
bzip2 -d mmod_human_face_detector.dat.bz2
```
