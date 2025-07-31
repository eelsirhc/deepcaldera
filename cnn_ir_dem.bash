#!/bin/bash

#cnn on IR and DEM
start_index=$1
padded_index=`printf %.05i $start_index`

root=/mnt/export/lee/1-Projects/deepcaldera/data
remove_processed() {
	rm -f $root/data/processed/sys_cal_craters_${padded_index}.hdf5
	rm -f $root/data/processed/sys_cal_images_${padded_index}.hdf5 
}

template_match_ir () {
	python ./predict_model.py make-prediction --index=$start_index --prefix=sys_cal --dataset=IR
#	rm -f $root/data/predictions/IR/sys_mars_craterdist_${padded_index}.hdf5
#	rm -f $root/data/predictions/IR/sys_cal_preds_${padded_index}.hdf5
} 

template_match_dem () {
	python ./predict_model.py make-prediction --index=$start_index --prefix=sys_cal --dataset=DEM
#	rm -f $root/data/predictions/DEM/sys_cal_craterdist_${padded_index}.hdf5
#	rm -f $root/data/predictions/DEM/sys_cal_preds_${padded_index}.hdf5
}

cnn_dem() {
	python ./predict_model.py cnn-prediction --index=$start_index --prefix=sys_cal --dataset=DEM
	template_match_dem
}

cnn_ir() {
	python ./predict_model.py cnn-prediction --index=$start_index --prefix=sys_cal --dataset=IR
	template_match_ir &
	cnn_dem
	remove_processed
}

cnn_ir
