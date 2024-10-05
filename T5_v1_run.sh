tar -xzvf SharkSeagrass.tar.gz
rm SharkSeagrass.tar.gz

tar -xzvf ind_axial.tar.gz
tar -xzvf ind_coronal.tar.gz
tar -xzvf ind_sagittal.tar.gz

rm ind_axial.tar.gz
rm ind_coronal.tar.gz
rm ind_sagittal.tar.gz

mkdir TC256_v2_vq_f8
mv ind_axial ./TC256_v2_vq_f8/
mv ind_coronal ./TC256_v2_vq_f8/
mv ind_sagittal ./TC256_v2_vq_f8/
mv TC256_v2_vq_f8 ./SharkSeagrass/

cd SharkSeagrass
mkdir results
echo "============================================"
ls
echo "============================================"
python3 T5_v1_train.py --cross_validation $1 --pretrain $2 --model_architecture $3 --model_scale $4 --batch_size $5
echo "============================================"
du -lh -d 1
tar -czvf T5v1f8_cv$1.tar.gz results
echo "============================================"
ls
echo "============================================"
mv T5v1f8_cv$1.tar.gz ../
