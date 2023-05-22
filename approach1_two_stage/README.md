Location for 2 stage approach code and notebooks.

## Approach 1: 2 Stage, Face detection then CNN from scratch


### Code location

All notebooks are accompanied by a pdf containing their outputs.

Requirements are saved in `requirements.txt`.

* FaceDetection
	+ Face detection algorithms to crop out faces
	+ Face detection algorithm comparison
* FaceMaskDetection_12k
	+ Cropping out faces for the "FaceMaskDetection" dataset with 12k instances
	+ Processing crops into np arrays for training on this dataset
* files_to_data
	+ Turning image crops into numpy arrays for training/testing
* FMD_dataset
	+ Code for extracting face crops from other dataset called "FaceMaskDetection"
* Results
	+ Results and model weights in pickle files or np save files
* Webcam
	+ Code for processing the webcam footage
	+ Webcam output footage saved in `Webcam\videos\processed_fps_corrected.avi`
* Evaluating model trained on multiple datasets.ipynb
	+ Comparing the models trained on different classifiers
* Model mlfw only.ipynb
	+ Code for model trained on mlfw dataset only
* Model multi dataset.ipynb
	+ Model trained on all datasets
* Model WWMR only.ipynb
	+ Model trained on wwmr dataset only
	