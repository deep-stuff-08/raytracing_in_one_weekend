if [[ $1 == image ]]; then
	make image.run
	LD_LIBRARY_PATH=../../lib ./image.run
fi