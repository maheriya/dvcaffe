#!/bin/csh -f

#set image = data/dvia/ineg_01.png
#set image = data/dvia/stairs_up/stairs_up_001.png
#set image = data/dvia/stairs_dn/stairs_dn_011.png
set image = data/dvia/stairs_dn/stairs_dn_021.png

set result = data/dvia/result.txt

echo "Running classification of image $image"
echo "$image 0" > data/dvia/test_list.txt
caffe test -model examples/dvia/dvia_lenet_h5_test_imagedata.prototxt \
	-weights examples/dvia/dvia_lenet_h5_iter_150.caffemodel \
	-iterations 1
	#-iterations 1 >& $result

exit
@ l = 0
echo "Probabilities of each label: "
rm -f /tmp/res; touch /tmp/res
foreach p ( `tail -n3 $result | sed 's#^.*\] prob = ##'` )
  perl -e 'printf(''"%9.5f%\tLabel %d\n"'", 100*$p, $l);" | tee -a /tmp/res
  @ l = $l + 1
end
echo ""
echo "Predicted Label: "
sort -rn /tmp/res | head -n1
