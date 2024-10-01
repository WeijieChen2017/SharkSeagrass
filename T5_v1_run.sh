tar -xzvf SharkSeagrass.tar.gz
rm SharkSeagrass.zip

tar -xzvf ind_axial.tar.gz
tar -xzvf ind_coronal.tar.gz
tar -xzvf ind_sagittal.tar.gz

rm ind_axial.tar.gz
rm ind_coronal.tar.gz
rm ind_sagittal.tar.gz
mv ind_axial ./SharkSeagrass/
mv ind_coronal ./SharkSeagrass/
mv ind_sagittal ./SharkSeagrass/

cd SharkSeagrass
mkdir results
echo "============================================"
ls
echo "============================================"
python T5_v1_train_nowandb.py --cross_validation $1 --pretrain $2 --model_architecture $3 --model_scale $4 --batch_size $5
echo "============================================"
du -lh -d 1
tar -czvf T5v1f8_cv$1.tar.gz results
echo "============================================"
ls
echo "============================================"
mv T5v1f8_cv$1.tar.gz ../
