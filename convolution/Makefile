CC = gcc
CC_FLAGS = -mavx2 -mfma -O3 -std=c99

default: all assemble

all: kernel 

kernel: kernel_driver.x
	./kernel_driver.x

kernel_driver.x:
# 	${CC} ${CC_FLAGS} -o kernel_driver.x kernel_vertical_tbf.c 
# 	${CC} ${CC_FLAGS} -o kernel_driver.x kernel_vertical.c 
	${CC} ${CC_FLAGS} -o kernel_driver.x kernel_driver.c kernel.c

assemble:
	objdump -s -d -f --source ./kernel_driver.x > kernel_driver.S

clean:
	rm -rf *.x *.S