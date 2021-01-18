SRC:=$(wildcard *.cpp)
all: $(SRC)
	@for FILE in $? ; do \
	mex -R2018a -I../../../src -I../../../../dependencies/eigen $${FILE} ; \
	done
