SRC:=$(wildcard *.cpp)
all: $(SRC)
	@for FILE in $? ; do \
	mex -R2018a -I../../../src $${FILE} -I../../../../dependencies/eigen -lut -lmwservices ; \
	done
