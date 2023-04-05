# AMLS_II_assignment22_23

### Environment(conda):

python==3.8.15
pytorch==1.13.0
pillow==9.2.0
h5py==3.7.0
numpy==1.23.4

### Files

##### main.py

Program entry, calling dataset preprocessing, training, and testing.

##### utils.py

Tool class.

##### datasets.py

Defined dataset classes.

##### models.py

Defined a network model.

##### prepare.py

Encapsulated dataset preprocessing functions.

##### train.py

Encapsulated training functions.

##### test.py

Encapsulated testing functions.

### Organization

Datasets: (1) Contains high resolution images of DIV2K dataset in folder "DIV2K_train_HR" and "DIV2K_valid_HR", will be preprocessed by prepare.py to generate 6 files, "train(or valid)_X2(or 3, 4).h5" in Datasets folder. So, Please make sure "Datasets/DIV2K_train_HR/xxx.png" and "Datasets/DIV2K_valid_HR/xxx.png"exist. (2) Contains "Set5", "Set14", "BSD100" to generate test results. It will looks like:

Set5
	HR
		xxx.png
		...
	FSRCNN
		X2
			xxx.png
			...
		X3
			xxx.png
			...
		X4
			xxx.png
			...
	OFSRCNN
		X2
			xxx.png
			...
		X3
			xxx.png
			...
		X4
			xxx.png
			...

Please make sure "Datasets/Set5/HR/xxx.png" exist.

### Others

You can modify parameters, as detailed in the parameters passed in main.py.