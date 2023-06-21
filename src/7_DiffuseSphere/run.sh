if [[ $1 == image ]]; then
	make image0.run TYPE=0
	LD_LIBRARY_PATH=../../lib ./image.run > outputimage0.ppm
	make image1.run TYPE=1
	LD_LIBRARY_PATH=../../lib ./image.run > outputimage1.ppm
	make image2.run TYPE=2
	LD_LIBRARY_PATH=../../lib ./image.run > outputimage2.ppm
fi