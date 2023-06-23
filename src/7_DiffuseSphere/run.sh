if [[ $1 == image ]]; then
	make image0.run
	LD_LIBRARY_PATH=../../lib ./image.run
	make image1.run
	LD_LIBRARY_PATH=../../lib ./image.run
	make image2.run
	LD_LIBRARY_PATH=../../lib ./image.run
fi