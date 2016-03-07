#!/bin/csh -f

#set image = data/mnist/i3_01.png
#set image = data/mnist/i9_01.png
set image = data/mnist/i9_02.png

## Negative tests
#set image = data/mnist/ineg_01.png
#set image = data/mnist/ineg_02.png

set result = data/mnist/result.txt

echo "Running classification of image $image"
echo "$image 0" > data/mnist/test_list.txt
caffe test -model examples/mnist/lenet_test_imagedata.prototxt \
	-weights examples/mnist/lenet_pretrained.caffemodel \
	-iterations 1 >& $result
@ l = 0 
echo "Probabilities of each label: " 
rm -f /tmp/res; touch /tmp/res
foreach p ( `tail -n10 $result | sed 's#^.*\] prob = ##'` )
  perl -e 'printf(''"%9.5f%\tLabel %d\n"'", 100*$p, $l);" | tee -a /tmp/res
  @ l = $l + 1
end
echo ""
echo "Predicted Label: "
sort -rn /tmp/res | head -n1

